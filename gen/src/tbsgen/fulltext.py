"""S3 full-text acquisition: a configurable cascade of channels behind one download ledger.

Channels (order = `fulltext.cascade`; each one has its own `enabled` switch):
  cache     read-only caches (`fulltext.cache_dirs`) and the write cache (`fulltext.write_cache_dir`); no network
  oa        open-access copies: Europe PMC JATS (REST API, by PMCID), then PDF links from the retrieval record
            (oa_url / pdf_url / openAccessPdf) and, if `oa.unpaywall_email` is set, Unpaywall OA locations
  elsevier  Elsevier Article Retrieval API (TDM, full-text XML), DOI prefixes `elsevier.prefixes`, key from env
  wiley     Wiley TDM API (PDF), DOI prefixes `wiley.prefixes`, token from env
  inst      plain HTTP to publisher PDF URLs that the institution's IP range is entitled to (`inst.routes`,
            empty by default)
  browser   a logged-in local browser (ego-browser, see browser_ego.py); off by default and, even when on, not
            run inline in S3 unless `browser.inline` is true: S3 queues the paper and `tbsgen fetch` drains
            the queue
A channel is skipped when it is disabled, its secret is missing, the DOI does not match its prefixes, the
channel answered for this DOI with metadata / abstract only before (`not_entitled`, see below), or the ledger
refuses (cap, cooldown, breaker).  Nothing here bypasses access control: every channel only fetches what the
user's own subscriptions / API terms allow, and bot-wall pages are never clicked through.

Skipped is not failed: a paper whose channels were only refused by the ledger (rec["limited"], with the time
the refusal lifts) or by the egress check made no request, so the queue keeps it due (until the earliest lift
time) instead of backing it off.  Only a paper with at least one real request (rec["tried"]) and no refusal
backs off.

Not entitled: a TDM API (elsevier / wiley) that answers 200 with too little text for a full article is
returning metadata / abstract only (volume not covered by the subscription).  That (channel, DOI) pair is booked
in the ledger table `not_entitled` and the channel never asks for that DOI again (`fulltext.not_entitled`).

Every download is booked in dlledger.DownloadLedger.  Successful texts go to the write cache as
<doi_key>.txt plus <doi_key>.json (provenance: channel, licence class, url, time, bytes, sha256, title match);
raw bytes go to <write_cache_dir>/raw/ when `keep_raw` is true.  Read-only caches are never written.
`licence` (LICENCES) records the terms a text was obtained under; `tbsgen fulltext purge` deletes the texts of
given channels (TDM terms: delete at the end of the project).

Network: urllib with ProxyHandler({}) (environment proxies ignored) unless `use_env_proxy = true`.  Optional
egress self-check (`egress_check.enabled`): before the first download in a process, ask `egress_check.urls` for
the public IP and refuse every network channel if it is outside `egress_check.allow_cidrs`.
"""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import contextlib
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .dlledger import COOLDOWN_429_S, DownloadLedger, LimitReached, block_signal, challenge_page

CHANNELS = ("cache", "oa", "elsevier", "wiley", "inst", "browser")
# licence class of a text, by the channel that fetched it (provenance `licence`)
LICENCES = {"elsevier": "elsevier_tdm", "wiley": "wiley_tdm", "oa": "oa", "inst": "inst", "browser": "browser"}
TDM_CHANNELS = ("elsevier", "wiley")
LIMIT_RECHECK_S = 900          # a ledger refusal with no known end: look again after this


def licence_of(channel: str | None) -> str | None:
    return LICENCES.get(channel or "")
UA_BROWSERISH = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                 "Chrome/140.0 Safari/537.36")


def doi_key(doi: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", (doi or "").lower().strip())


def norm_doi(doi: str | None) -> str:
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", (doi or "").strip(), flags=re.I).lower()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def title_match(title: str, text: str) -> float:
    """share of title words (len>=4) found in the first 6000 chars of the full text"""
    words = [w for w in _norm(title).split() if len(w) >= 4]
    if not words:
        return 0.0
    head = _norm(text[:6000])
    return sum(w in head for w in words) / len(words)


PDFTOTEXT = "pdftotext"          # set from `fulltext.pdftotext` (a binary outside PATH, e.g. a user-level install)


def pdf_to_text(pdf: bytes) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.pdf"
        p.write_bytes(pdf)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")}
        try:
            r = subprocess.run([PDFTOTEXT or "pdftotext", "-layout", "-enc", "UTF-8", str(p), "-"], capture_output=True,
                               timeout=180, env=env)
        except FileNotFoundError:
            raise RuntimeError("pdftotext not found (install poppler)") from None
        return r.stdout.decode("utf-8", "replace")


def _strip_tags(t: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"[ \t]+", " ", html.unescape(t)).strip()


def elsevier_text(xml: str) -> str:
    """Article Retrieval API full-text XML -> title + section text ('' when only metadata/abstract came back)."""
    title = re.search(r"<dc:title>(.*?)</dc:title>", xml, re.S)
    body = re.search(r"<(?:ce:sections|body)[^>]*>(.*)</(?:ce:sections|body)>", xml, re.S)
    if not body:
        return ""
    t = re.sub(r"<ce:section-title[^>]*>(.*?)</ce:section-title>", r"\n\n## \1\n", body.group(1), flags=re.S)
    t = re.sub(r"</ce:para>|</ce:simple-para>", "\n", t)
    return ((_strip_tags(title.group(1)) + "\n\n") if title else "") + _strip_tags(t)


def jats_text(xml: str) -> str:
    title = re.search(r"<article-title[^>]*>(.*?)</article-title>", xml, re.S)
    body = re.search(r"<body[^>]*>(.*)</body>", xml, re.S)
    if not body:
        return ""
    t = re.sub(r"<title[^>]*>(.*?)</title>", r"\n\n## \1\n", body.group(1), flags=re.S)
    t = re.sub(r"</p>", "\n", t)
    return ((_strip_tags(title.group(1)) + "\n\n") if title else "") + _strip_tags(t)


_SECTION = re.compile(r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?(introduction|background|methods?|materials and methods|"
                      r"methodology|results?|discussion|conclusions?|experiments?|theory|model)\b", re.I)
_PAYWALL = re.compile(r"(purchase (this )?(article|pdf)|buy (this )?article|rent this article|get access|"
                      r"access through your institution|log ?in to (view|access)|subscribe to (view|access)|"
                      r"institutional (sign|log) ?in|check access|preview of subscription content)", re.I)


def html_fulltext(body: bytes) -> str | None:
    """Accept an HTML page as full text only if it has at least three distinct article section headings
    (h1-h4: introduction / methods / results / discussion / conclusions ...), shows no paywall prompt, and is not
    a bot-wall page.  The references section is cut before the length check (done by the caller)."""
    if challenge_page(body):
        return None
    doc = body.decode("utf-8", "replace")
    heads = [_strip_tags(h) for h in re.findall(r"<h[1-4][^>]*>(.*?)</h[1-4]>", doc, re.S | re.I)]
    kinds = {m.group(1).lower()[:5] for h in heads if (m := _SECTION.match(h))}
    if len(kinds) < 3:
        return None
    ref = re.search(r"<h[1-4][^>]*>\s*(?:<[^>]+>\s*)*(?:\d+\.?\s*)?(references|bibliography|literature cited)\b",
                    doc, re.I)
    if ref:
        doc = doc[: ref.start()]                    # body only: the length check must not count references
    txt = _strip_tags(doc)
    if _PAYWALL.search(txt):
        return None
    return txt


def safe_url(u: str) -> str:
    """URL without userinfo, query and fragment (they may carry credentials or signed tokens)."""
    try:
        p = urllib.parse.urlsplit(u or "")
    except ValueError:
        return ""
    netloc = p.hostname or ""
    if p.port:
        netloc += f":{p.port}"
    return urllib.parse.urlunsplit((p.scheme, netloc, p.path, "", "")) if p.scheme else (u or "").split("?")[0]


_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")
_TOKENISH = re.compile(r"\b[A-Za-z0-9_-]{8}([A-Za-z0-9_-]{24,})\b")


def redact(s: str, secrets: tuple = ()) -> str:
    """For logs, ledger notes and records: URLs lose query/fragment/userinfo, known secret values and long
    token-like strings are masked."""
    s = _URL_RE.sub(lambda m: safe_url(m.group(0)), s or "")
    for v in secrets:
        if v and len(v) >= 6:
            s = s.replace(v, "***")
    return _TOKENISH.sub(lambda m: m.group(0)[:8] + "***", s)


def paper_ident(c: dict) -> str:
    """Exact identity of a paper (queue key): "doi:<normalised DOI>", else "arxiv:<id>", else a title/URL hash."""
    if (d := norm_doi(c.get("doi"))):
        return "doi:" + d
    if c.get("arxiv"):
        return "arxiv:" + str(c["arxiv"]).lower().strip()
    ident = (c.get("title") or c.get("oa_url") or c.get("pdf_url") or "").lower().strip()
    return "h:" + hashlib.sha256(ident.encode()).hexdigest()[:16] if ident else ""


def paper_key(c: dict) -> str:
    """Cache file stem: doi_key(DOI) (the earlier pipeline's naming, so its cache is readable), else a key from
    the identity.  doi_key is lossy ("/" and "_" both map to "_"): cache reads verify the DOI in the provenance
    json, and a write that would collide with another DOI's file uses `<stem>-<sha8>` instead."""
    if (d := norm_doi(c.get("doi"))):
        return doi_key(d)
    ident = paper_ident(c)
    return doi_key(ident.replace(":", "_")) if ident else ""


def _alt_key(stem: str, ident: str) -> str:
    return f"{stem}-{hashlib.sha256(ident.encode()).hexdigest()[:8]}"


@contextlib.contextmanager
def _dir_lock(d: Path):
    """Exclusive lock for choosing + publishing a cache file name (writes are rare; reads need no lock because a
    file name is only ever published for one identity)."""
    import fcntl
    with open(d / ".write.lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_write(path: Path, data: bytes | str):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
    if isinstance(data, str):
        tmp.write_text(data)
    else:
        tmp.write_bytes(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------- HTTP
@dataclass
class Resp:
    status: int
    headers: dict
    body: bytes
    url: str
    error: str = ""
    truncated: bool = False


Transport = Callable[[str, dict, float, int], Resp]
CRED_HEADERS = ("x-els-apikey", "x-els-insttoken", "wiley-tdm-client-token", "authorization", "cookie")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


SYSTEM_CA_BUNDLES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                     "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", "/etc/ssl/ca-bundle.pem")


def ssl_context(ca_file: str | None = None):
    """Default verification; `ca_file` (fulltext.ca_file) or, when the interpreter's default CA file is missing
    (standalone Python builds on some Linux distributions), the first system bundle found."""
    import ssl
    if ca_file:
        return ssl.create_default_context(cafile=ca_file)
    dflt = ssl.get_default_verify_paths()
    if (dflt.cafile and os.path.exists(dflt.cafile)) or os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()
    for p in SYSTEM_CA_BUNDLES:
        if os.path.exists(p):
            return ssl.create_default_context(cafile=p)
    return ssl.create_default_context()


def make_transport(use_env_proxy: bool = False, ca_file: str | None = None) -> Transport:
    """One request, no automatic redirects (Fetcher follows them hop by hop).  Environment proxies are ignored
    unless use_env_proxy.  `timeout` is a total deadline for the whole request (socket ops are bounded by it
    too), so a slowly dripping response cannot keep a reservation alive."""
    handlers = [_NoRedirect(), urllib.request.HTTPSHandler(context=ssl_context(ca_file))] + \
        ([] if use_env_proxy else [urllib.request.ProxyHandler({})])
    opener = urllib.request.build_opener(*handlers)

    def transport(url: str, headers: dict, timeout: float, max_bytes: int) -> Resp:
        req = urllib.request.Request(url, headers=headers)
        deadline = time.monotonic() + timeout
        try:
            with opener.open(req, timeout=timeout) as r:
                chunks, n = [], 0
                while n <= max_bytes:
                    if time.monotonic() > deadline:
                        return Resp(0, {}, b"", url, f"total deadline {timeout:.0f}s exceeded")
                    # read1: at most one socket recv (each bounded by the socket timeout), so a trickling body
                    # cannot hold a read past the deadline by more than one timeout
                    b = r.read1(min(1 << 16, max_bytes + 1 - n))
                    if not b:
                        break
                    chunks.append(b)
                    n += len(b)
                body = b"".join(chunks)
                return Resp(r.status, dict(r.headers), body[:max_bytes], r.geturl(), truncated=len(body) > max_bytes)
        except urllib.error.HTTPError as e:
            body = b""
            with contextlib.suppress(Exception):
                while e.fp and len(body) < 20000 and time.monotonic() < deadline:
                    b = e.read1(20000 - len(body))
                    if not b:
                        break
                    body += b
            return Resp(e.code, dict(e.headers or {}), body, url)
        except Exception as e:  # noqa: BLE001
            return Resp(0, {}, b"", url, f"{type(e).__name__}: {str(e)[:200]}")
    return transport


class EgressRefused(RuntimeError):
    pass


def check_egress(ec: dict, transport: Transport) -> dict:
    """Public egress IP via the configured echo URLs; raises EgressRefused when outside allow_cidrs."""
    nets = [ipaddress.ip_network(c, strict=False) for c in ec.get("allow_cidrs") or []]
    if not nets:
        raise EgressRefused("egress_check.enabled but egress_check.allow_cidrs is empty")
    errs = []
    for u in ec.get("urls") or []:
        r = transport(u, {"User-Agent": "tbsgen-egress-check"}, float(ec.get("timeout", 15)), 4096)
        txt = r.body.decode("ascii", "replace") if r.status == 200 else ""
        # Cloudflare /cdn-cgi/trace ("ip=...") on a publisher's own domain shows the IP that publisher sees,
        # which is what matters when a proxy routes per domain; plain echo services return the bare IP.
        m = re.search(r"^ip=(\d{1,3}(?:\.\d{1,3}){3})\s*$", txt, re.M) or \
            re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", txt)
        if not m:
            errs.append(f"{safe_url(u)}: HTTP {r.status} {r.error}")
            continue
        ip = ipaddress.ip_address(m.group(1))
        if any(ip in n for n in nets):
            return dict(ok=True, ip=str(ip), via=safe_url(u))
        raise EgressRefused(f"egress IP {ip} (from {safe_url(u)}) is outside egress_check.allow_cidrs")
    raise EgressRefused("egress IP unknown: " + "; ".join(errs))


# ------------------------------------------------------------------------------------------- publishers
# One publisher can own several DOI prefixes and web hosts.  Browser downloads are booked under the earlier
# pipeline's key "chrome:<DOI prefix>"; all prefixes of one publisher share pacing, caps and cooldowns (aliases),
# and a cooldown on any of its web hosts (set by the HTTP channels) also stops the browser, and vice versa.
# The TDM API hosts (api.elsevier.com, api.wiley.com) are deliberately NOT in these groups: they are a separate
# service with per-key quotas, and the publishers ask text miners to use them instead of the website.  An API
# 429 means "slow down on the API", not "the campus IP is flagged on the website"; coupling them would let one
# throttle stop the other channel for 6-24 h.  A website block, likewise, must not stop the sanctioned API.
PUBLISHERS: dict[str, dict] = {
    "elsevier": dict(prefixes=["10.1016", "10.1006", "10.1053", "10.1067"],
                     hosts=["www.sciencedirect.com", "pdf.sciencedirectassets.com", "linkinghub.elsevier.com"]),
    "wiley": dict(prefixes=["10.1002", "10.1111", "10.1029", "10.1113"],
                  hosts=["onlinelibrary.wiley.com", "agupubs.onlinelibrary.wiley.com"]),
    "springernature": dict(prefixes=["10.1007", "10.1038", "10.1186", "10.1057"],
                           hosts=["link.springer.com", "www.nature.com"]),
    "tandf": dict(prefixes=["10.1080"], hosts=["www.tandfonline.com"]),
    "oup": dict(prefixes=["10.1093"], hosts=["academic.oup.com"]),
    "ieee": dict(prefixes=["10.1109"], hosts=["ieeexplore.ieee.org"]),
    "acs": dict(prefixes=["10.1021"], hosts=["pubs.acs.org"]),
    "iop": dict(prefixes=["10.1088"], hosts=["iopscience.iop.org"]),
    "sage": dict(prefixes=["10.1177"], hosts=["journals.sagepub.com"]),
    "informs": dict(prefixes=["10.1287"], hosts=["pubsonline.informs.org"]),
    "spie": dict(prefixes=["10.1117"], hosts=["www.spiedigitallibrary.org"]),
    "cambridge": dict(prefixes=["10.1017", "10.1154"], hosts=["www.cambridge.org"]),
    "aip": dict(prefixes=["10.1063", "10.1121"], hosts=["pubs.aip.org"]),
    "degruyter": dict(prefixes=["10.1515", "10.2478"], hosts=["www.degruyter.com", "www.degruyterbrill.com"]),
}


def publisher_of(doi: str) -> tuple[str, list[str], list[str]]:
    """(name, ledger keys "chrome:<prefix>" of all its prefixes, web hosts) for a DOI."""
    pre = doi.split("/", 1)[0] if "/" in doi else doi
    for name, p in PUBLISHERS.items():
        if pre in p["prefixes"]:
            return name, [f"chrome:{x}" for x in p["prefixes"]], list(p["hosts"])
    return pre, [f"chrome:{pre}"], []


def publisher_of_host(host: str) -> tuple[list[str], list[str]]:
    for p in PUBLISHERS.values():
        if host in p["hosts"]:
            return [f"chrome:{x}" for x in p["prefixes"]], [h for h in p["hosts"] if h != host]
    return [], []


# ------------------------------------------------------------------------------------------- fetcher
@dataclass
class Outcome:
    text: str | None
    rec: dict


@dataclass
class Fetcher:
    cfg: dict
    ledger: DownloadLedger | None = None
    transport: Transport | None = None
    browser_runner: Callable | None = None          # tests inject; default browser_ego.run_one
    secrets: dict = field(default_factory=dict)       # overrides env lookups (tests / in-memory injection)

    def __post_init__(self):
        ft = self.ft
        if self.transport is None:
            self.transport = make_transport(bool(ft.get("use_env_proxy")), ft.get("ca_file"))
        if self.ledger is None:
            self.ledger = DownloadLedger(ledger_path(self.cfg), ft.get("limits"), ft.get("global_limits"),
                                         ft.get("host_caps"))
        self._egress: dict | None = None
        self._egress_browser: dict | None = None
        self._egress_lock = threading.Lock()
        self._tl = threading.local()
        self._space_held = False            # set by drain_queue while it holds the TaskSpace lock
        self._space_fh = None
        # remote mode: results of the HTTP channels collected from the remote host, by paper identity
        self.remote_results: dict[str, dict] = {}
        global PDFTOTEXT
        if ft.get("pdftotext"):
            PDFTOTEXT = ft["pdftotext"]

    def remote_mode(self) -> bool:
        return bool((self.ft.get("remote") or {}).get("enabled"))

    def space_lock_path(self) -> Path:
        space = re.sub(r"[^A-Za-z0-9_.-]", "_", str((self.ft.get("browser") or {}).get("space", "tbsgen-s3-fulltext")))
        return Path(tempfile.gettempdir()) / f"tbsgen-ego-{space}.lock"

    def _try_space_lock(self) -> bool:
        """Inline browser use (browser.inline=true): take the TaskSpace lock for the rest of this process, or
        give up immediately if a `tbsgen fetch` (or another process) holds it."""
        import fcntl
        if self._space_fh is not None:
            return True
        fh = open(self.space_lock_path(), "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            return False
        self._space_fh = fh
        self._space_held = True
        return True

    @property
    def ft(self) -> dict:
        return self.cfg["fulltext"]

    # ---------------------------------------------------------------- helpers
    def secret(self, env_name: str | None) -> str | None:
        if not env_name:
            return None
        return self.secrets.get(env_name) or os.environ.get(env_name) or None

    def _secret_values(self) -> tuple:
        names = [(self.ft.get("elsevier") or {}).get("key_env"), (self.ft.get("elsevier") or {}).get("insttoken_env"),
                 (self.ft.get("wiley") or {}).get("token_env")]
        return tuple(v for n in names if (v := self.secret(n)))

    def _r(self, s: str) -> str:
        return redact(s, self._secret_values())

    def write_dir(self) -> Path | None:
        return write_cache_dir(self.cfg)

    def cached(self, c: dict) -> tuple[str, dict] | None:
        """Cached text for this paper.  A file whose provenance names another DOI (doi_key collision) is
        skipped; the collision-safe `<stem>-<sha8>` name is tried too.  Files without provenance (the earlier
        pipeline's cache always has one; hand-made caches may not) are accepted by name."""
        key, ident, doi = paper_key(c), paper_ident(c), norm_doi(c.get("doi"))
        if not key:
            return None
        dirs = [Path(d) for d in self.ft.get("cache_dirs") or []]
        if (w := self.write_dir()):
            dirs.append(w)
        for d in dirs:
            for stem in (key, _alt_key(key, ident)):
                p = d / f"{stem}.txt"
                if not p.exists():
                    continue
                prov: dict = {}
                with contextlib.suppress(OSError, ValueError):
                    j = json.loads((d / f"{stem}.json").read_text())
                    prov = j if isinstance(j, dict) else {}
                if doi and prov.get("doi") and norm_doi(prov["doi"]) != doi:
                    continue
                txt = p.read_text(errors="replace")
                if len(txt) >= self.ft["min_chars"]:
                    return txt, dict(source="cache", path=str(p), cached_channel=prov.get("channel"),
                                     cached_licence=prov.get("licence") or licence_of(prov.get("channel")))
        return None

    def _ua(self) -> str:
        m = self.ft.get("mailto")
        return f"tbsgen/0.2 (text mining for research{'; mailto:' + m if m else ''})"

    def _max_bytes(self) -> int:
        return int(float(self.ft["max_pdf_mb"]) * 1e6)

    def _egress_ok(self):
        ec = self.ft.get("egress_check") or {}
        if not ec.get("enabled"):
            return
        with self._egress_lock:
            if self._egress and time.time() - self._egress["ts"] < float(ec.get("ttl_s", 600)):
                if not self._egress["ok"]:
                    raise EgressRefused(self._egress["why"])
                return
            try:
                self._egress = dict(check_egress(ec, self.transport), ts=time.time())
            except EgressRefused as e:
                self._egress = dict(ok=False, ts=time.time(), why=str(e))
                raise

    def _egress_ok_browser(self):
        """The browser has its own network stack: check its egress through the browser itself (one checker at a
        time, on its own temporary Page; download workers wait for and share the result)."""
        ec = self.ft.get("egress_check") or {}
        if not ec.get("enabled"):
            return
        from . import browser_ego
        with self._egress_lock:
            eb = self._egress_browser
            if eb and time.time() - eb["ts"] < float(ec.get("ttl_s", 600)):
                if not eb["ok"]:
                    raise EgressRefused(eb["why"])
                return
            runner = self.browser_runner or browser_ego.run_one
            res = runner(dict(mode="egress", urls=ec.get("urls") or [], cfg=self.ft["browser"],
                              out_dir=str(self._browser_tmp())))

            def tr(url, headers, timeout, max_bytes):
                body = (res.get("bodies") or {}).get(url, "")
                return Resp(200 if body else 0, {}, body.encode(), url)
            try:
                self._egress_browser = dict(check_egress(ec, tr), ts=time.time())
            except EgressRefused as e:
                self._egress_browser = dict(ok=False, ts=time.time(), why=f"browser: {e}")
                raise EgressRefused(f"browser: {e}") from None

    def _validate(self, title: str, kind: str, body: bytes) -> tuple[str | None, str, float | None]:
        if len(body) > self._max_bytes():
            return None, f"too large ({len(body)} bytes)", None
        if kind == "pdf":
            if body[:5] != b"%PDF-":
                return None, "not a PDF" + (" (bot-wall page)" if challenge_page(body) else ""), None
            try:
                txt = pdf_to_text(body)
            except RuntimeError as e:
                return None, str(e), None
        elif kind == "jats":
            txt = jats_text(body.decode("utf-8", "replace"))
        elif kind == "elsevier_xml":
            txt = elsevier_text(body.decode("utf-8", "replace"))
        elif kind == "any":
            if body[:5] == b"%PDF-":
                return self._validate(title, "pdf", body)
            txt = html_fulltext(body) or ""
            if not txt:
                return None, "not a PDF or full-text article page", None
        else:
            raise ValueError(kind)
        if len(txt) < self.ft["min_chars"]:
            return None, f"short text {len(txt)}", None
        if title:
            tm = title_match(title, txt)
            if tm < self.ft["title_match_min"]:
                return None, f"title mismatch {tm:.2f}", tm
            return txt, "", tm
        return txt, "", None

    def _save(self, c: dict, channel: str, url: str, body: bytes, txt: str, tm: float | None, kind: str,
              **extra) -> dict:
        key, ident, doi = paper_key(c), paper_ident(c), norm_doi(c.get("doi"))
        prov = dict(doi=doi, ident=ident, channel=channel, licence=licence_of(channel), url=self._r(safe_url(url)),
                    kind=kind, bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(), text_chars=len(txt),
                    title_match=None if tm is None else round(tm, 3), fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    **extra)
        if (w := self.write_dir()) and key:
            w.mkdir(parents=True, exist_ok=True)
            with _dir_lock(w):
                return self._save_locked(w, key, ident, prov, body, txt, kind)
        return prov

    def save_imported(self, c: dict, prov: dict, txt: str, raw: bytes | None) -> dict:
        """Write a text fetched elsewhere (remote mode) to the write cache, with the same collision-safe naming."""
        key, ident = paper_key(c), paper_ident(c)
        prov = dict(prov, ident=ident)
        prov.setdefault("licence", licence_of(prov.get("channel")))
        if (w := self.write_dir()) and key:
            w.mkdir(parents=True, exist_ok=True)
            with _dir_lock(w):
                return self._save_locked(w, key, ident, prov, raw or b"", txt, prov.get("kind") or "bin")
        return prov

    def _save_locked(self, w: Path, key: str, ident: str, prov: dict, body: bytes, txt: str, kind: str) -> dict:
        stem = key
        with contextlib.suppress(OSError, ValueError):
            other = json.loads((w / f"{key}.json").read_text())
            if isinstance(other, dict) and other.get("ident", "doi:" + norm_doi(other.get("doi"))) != ident:
                stem = _alt_key(key, ident)          # another paper owns this file name
        if self.ft.get("keep_raw", True) and body:
            ext = {"pdf": "pdf", "jats": "xml", "elsevier_xml": "xml", "html": "html"}.get(kind, "bin")
            (w / "raw").mkdir(exist_ok=True)
            _atomic_write(w / "raw" / f"{stem}.{ext}", body)
        # provenance first, text last: a text file always has its (complete) provenance next to it
        _atomic_write(w / f"{stem}.json", json.dumps(prov, indent=1))
        _atomic_write(w / f"{stem}.txt", txt)
        prov["path"] = str(w / f"{stem}.txt")
        return prov

    def _related(self, host: str) -> tuple:
        keys, others = publisher_of_host(host)
        return tuple(keys + others)

    def _get(self, url: str, headers: dict, rec: dict) -> Resp:
        """GET with manual redirects: at most 5 hops, never https -> http, credential headers only sent to the
        original origin, and every hop's host must not be in cooldown."""
        origin = urllib.parse.urlsplit(url)[:2]
        hdr = dict(headers)
        for _ in range(6):
            r = self.transport(url, hdr, float(self.ft["timeout"]), self._max_bytes())
            if r.status not in (301, 302, 303, 307, 308):
                return r
            loc = {k.lower(): v for k, v in r.headers.items()}.get("location")
            if not loc:
                return r
            # some servers send raw spaces / control characters in Location: percent-encode them
            loc = urllib.parse.quote(loc.strip(), safe=":/?#[]@!$&'()*+,;=%~")
            nxt = urllib.parse.urljoin(url, loc)
            sp = urllib.parse.urlsplit(nxt)
            if urllib.parse.urlsplit(url).scheme == "https" and sp.scheme != "https":
                return Resp(0, {}, b"", url, "refused redirect https -> http")
            if sp.scheme not in ("http", "https"):
                return Resp(0, {}, b"", url, "refused redirect to non-http URL")
            cur_host = urllib.parse.urlsplit(url).hostname or ""
            if self.ledger and sp.hostname and sp.hostname != cur_host:
                hop = getattr(self._tl, "hop", None)
                # a refused redirect target is our own limit, not the paper's failure: booked as "limited"
                # (the queue keeps the paper due) and as a cancelled attempt (no breaker counts it)
                try:
                    if hop:
                        self.ledger.acquire_hop(hop[0], sp.hostname, hop[1], related=self._related(sp.hostname))
                    elif (msg := self.ledger.cooling(sp.hostname, *self._related(sp.hostname))):
                        rec.setdefault("limited", []).append(dict(
                            channel=hop[0] if hop else None, kind="cooldown", scope="host",
                            until=self.ledger.cooling_until(sp.hostname, *self._related(sp.hostname))))
                        return Resp(0, {}, b"", url, f"redirect target cooling: {msg}")
                except LimitReached as e:
                    rec.setdefault("limited", []).append(dict(channel=hop[0] if hop else None, kind=e.kind,
                                                              scope=e.scope, until=e.until))
                    return Resp(0, {}, b"", url, f"redirect target refused: {e}")
            if sp[:2] != origin:
                hdr = {k: v for k, v in hdr.items() if k.lower() not in CRED_HEADERS}
            rec.setdefault("redirects", []).append(safe_url(nxt))
            url = nxt
        return Resp(0, {}, b"", url, "too many redirects")

    @property
    def _cur(self) -> dict:
        return getattr(self._tl, "c", {})

    def _download(self, channel: str, key: str, doi: str, title: str, url: str, headers: dict, kind: str,
                  rec: dict, ledger_host: str | None = None) -> str | None:
        """One booked download.  Returns text on success; appends to rec['tried'/'errors']."""
        host = ledger_host or urllib.parse.urlsplit(url).hostname or ""
        if self.ledger is None:
            raise LimitReached("no download ledger configured (fulltext.ledger_path)", "cap")
        rid = self.ledger.acquire(channel, host, doi or key, related=self._related(host))
        self._tl.hop = (channel, doi or key)
        rec["tried"].append(f"{channel}:{safe_url(url)}")
        r = Resp(0, {}, b"", url, "not started")
        txt = None
        try:
            r = self._get(url, {"User-Agent": self._ua(), **headers}, rec)
            cool = block_signal(r.status, r.body)
            why = f"HTTP {r.status} {r.error}".strip()
            tm = None
            if r.truncated:
                why = f"too large (> {self._max_bytes()} bytes)"
            elif r.status == 200 and not cool:
                txt, why, tm = self._validate(title, kind, r.body)
            if cool:
                why = f"block signal HTTP {r.status}"
                for h in {host, urllib.parse.urlsplit(r.url).hostname or ""} - {""}:
                    self.ledger.block(h, self._r(f"{why} on {channel}"), seconds=cool, channel=channel)
            if txt is not None:
                # the reservation id travels with the text: a crash before record() can still be booked as ok
                prov = self._save(self._cur, channel, r.url, r.body, txt, tm, kind, reservation=rid)
                rec.update(source=channel, url=prov["url"], title_match=prov["title_match"], provenance=prov)
            else:
                if r.status == 200 and not cool and why.startswith("short text") and self._ne_applies(channel):
                    # metadata / abstract only: the subscription does not cover this paper
                    why = f"not_entitled: {why}"
                    self.ledger.mark_not_entitled(channel, doi or key, why)
                    rec.setdefault("not_entitled", []).append(channel)
                rec["errors"].append(self._r(f"{channel}: {why}"))
        finally:
            ok = rec.get("source") == channel
            note = "" if ok else (rec["errors"][-1] if rec["errors"] else self._r(r.error))
            if not ok and note.startswith(f"{channel}: not_entitled"):
                note = note[len(channel) + 2:]          # ledger note starts with "not_entitled" (breaker ignores it)
            elif not ok and r.status == 0 and r.error.startswith("redirect target "):
                note = "cancelled: " + note             # our own limit on the redirect target: no breaker count
            self.ledger.record(channel, host, doi or key, "ok" if ok else "fail", r.status, len(r.body),
                               note[:300], reservation=rid)
        return txt

    def _meta(self, url: str) -> dict | None:
        """Metadata lookup (Unpaywall / Europe PMC search): paced and capped, never counted as a download."""
        host = urllib.parse.urlsplit(url).hostname or ""
        if self.ledger is not None:
            self.ledger.acquire_meta(host)
        self._tl.hop = None
        r = self._get(url, {"User-Agent": self._ua(), "Accept": "application/json"}, {})
        if self.ledger is not None and (cool := block_signal(r.status, r.body)):
            self.ledger.block(host, f"block signal HTTP {r.status} on metadata", seconds=cool)
        if r.status != 200:
            return None
        try:
            return json.loads(r.body)
        except json.JSONDecodeError:
            return None

    def _ne_conf(self) -> dict:
        return self.ft.get("not_entitled") or {}

    def _ne_applies(self, channel: str) -> bool:
        ne = self._ne_conf()
        return bool(ne.get("enabled", True)) and channel in (ne.get("channels") or TDM_CHANNELS)

    def not_entitled(self, channel: str, doi: str) -> str | None:
        """Why `channel` must not ask for `doi` again (an earlier metadata-only answer), else None."""
        if not doi or self.ledger is None or not self._ne_applies(channel):
            return None
        return self.ledger.not_entitled(channel, doi, self._ne_conf().get("ttl_days"))

    @staticmethod
    def _limited(e: LimitReached, rec: dict, ch: str) -> None:
        """Book a ledger refusal: nothing was sent, so it is not a failure (the queue keeps the paper due)."""
        rec.setdefault("limited", []).append(dict(channel=ch, kind=e.kind, scope=e.scope, until=e.until))

    @classmethod
    def _host_scoped(cls, e: LimitReached, rec: dict, ch: str) -> None:
        """A host-level refusal skips that source only; channel-level ones stop the channel."""
        if e.scope != "host":
            raise e
        rec["skipped"].append(f"{ch}: {e}")
        cls._limited(e, rec, ch)

    # ---------------------------------------------------------------- channels
    def ch_oa(self, c: dict, key: str, doi: str, title: str, rec: dict) -> str | None:
        oa = self.ft.get("oa") or {}
        if oa.get("europepmc", True) and (doi or c.get("pmcid")):
            pmcid = c.get("pmcid")
            try:
                if not pmcid:
                    d = self._meta("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
                        {"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"}))
                    hits = ((d or {}).get("resultList") or {}).get("result") or []
                    if hits and hits[0].get("pmcid") and hits[0].get("inEPMC") == "Y" and \
                            hits[0].get("isOpenAccess") == "Y":
                        pmcid = hits[0]["pmcid"]
                if pmcid:
                    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                    if (t := self._download("oa", key, doi, title, url, {"Accept": "application/xml"}, "jats", rec)):
                        return t
            except LimitReached as e:
                self._host_scoped(e, rec, "oa")
        oap = c.get("openAccessPdf")
        urls = [u for u in (c.get("oa_url"), c.get("pdf_url"), oap.get("url") if isinstance(oap, dict) else oap)
                if isinstance(u, str) and u]
        if oa.get("unpaywall_email") and doi:
            try:
                u = self._meta(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?"
                               + urllib.parse.urlencode({"email": oa["unpaywall_email"]}))
                urls += [loc["url_for_pdf"] for loc in (u or {}).get("oa_locations") or [] if loc.get("url_for_pdf")]
            except LimitReached as e:
                self._host_scoped(e, rec, "oa")
        skip = set(oa.get("skip_hosts") or [])
        seen: list[str] = []
        for u in urls:
            u = re.sub(r"^http://", "https://", u)
            if u not in seen and urllib.parse.urlsplit(u).hostname not in skip:
                seen.append(u)
        seen.sort(key=lambda x: "arxiv.org" not in x)          # arXiv first: most tolerant host
        for u in seen[: int(oa.get("max_urls", 3))]:
            try:
                if (t := self._download("oa", key, doi, title, u, {"User-Agent": UA_BROWSERISH}, "any", rec)):
                    return t
            except LimitReached as e:
                self._host_scoped(e, rec, "oa")
        return None

    def ch_elsevier(self, c: dict, key: str, doi: str, title: str, rec: dict) -> str | None:
        e = self.ft["elsevier"]
        k = self.secret(e.get("key_env"))
        if not k:
            rec["skipped"].append(f"elsevier: {e.get('key_env')} not set")
            return None
        hdr = {"X-ELS-APIKey": k, "Accept": "text/xml"}
        if (tok := self.secret(e.get("insttoken_env"))):
            hdr["X-ELS-Insttoken"] = tok
        url = f"https://api.elsevier.com/content/article/doi/{urllib.parse.quote(doi, safe='/')}"
        return self._download("elsevier", key, doi, title, url, hdr, "elsevier_xml", rec)

    def ch_wiley(self, c: dict, key: str, doi: str, title: str, rec: dict) -> str | None:
        w = self.ft["wiley"]
        tok = self.secret(w.get("token_env"))
        if not tok:
            rec["skipped"].append(f"wiley: {w.get('token_env')} not set")
            return None
        url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/" + urllib.parse.quote(doi, safe="")
        return self._download("wiley", key, doi, title, url, {"Wiley-TDM-Client-Token": tok}, "pdf", rec,
                              ledger_host="api.wiley.com")

    def ch_inst(self, c: dict, key: str, doi: str, title: str, rec: dict) -> str | None:
        for r in self.ft["inst"].get("routes") or []:
            if doi.startswith(r["prefix"]):
                url = r["url"].format(doi=doi, suffix=doi.split("/", 1)[1] if "/" in doi else doi)
                try:
                    return self._download("inst", key, doi, title, url, {"User-Agent": UA_BROWSERISH}, "pdf", rec)
                except LimitReached as e:
                    self._host_scoped(e, rec, "inst")
                    return None
        return None

    def ch_browser(self, c: dict, key: str, doi: str, title: str, rec: dict, slot: int = 0) -> str | None:
        from . import browser_ego
        b = self.ft["browser"]
        _, keys, web_hosts = publisher_of(doi)
        lkey = "chrome:" + (doi.split("/", 1)[0] if "/" in doi else doi)
        aliases = tuple(k for k in keys if k != lkey)
        if self.ledger is None:
            raise LimitReached("no download ledger configured (fulltext.ledger_path)", "cap")
        if not self._space_held and not self._try_space_lock():
            raise LimitReached("browser TaskSpace is in use by another process (e.g. `tbsgen fetch`)", "cap", "host")
        rid = self.ledger.acquire("browser", lkey, doi, related=tuple(web_hosts), group=aliases)
        rec["tried"].append(f"browser:{lkey}")
        res: dict = {}
        ok = False
        try:
            runner = self.browser_runner or browser_ego.run_one
            res = runner(dict(doi=doi, title=title, pdf_url=browser_ego.pdf_url_hint(doi, c), slot=slot,
                              cfg=b, out_dir=str(self._browser_tmp()), max_bytes=self._max_bytes())) or {}
            p = Path(res["path"]) if res.get("path") else None
            size = p.stat().st_size if p and p.exists() else 0
            if size > self._max_bytes():
                rec["errors"].append(f"browser: too large ({size} bytes)")
                return None
            body = p.read_bytes() if size else b""
            http = 429 if res.get("http429") else res.get("http")
            cool = block_signal(http if isinstance(http, int) else None, body)
            if res.get("status") == "challenge" or res.get("blocked") or cool:
                secs = cool or float(b.get("challenge_cooldown_s", 24 * 3600))
                if http == 429:
                    secs = COOLDOWN_429_S
                why = self._r(f"HTTP {http}" if http == 429 else f"challenge: {res.get('title', '')[:60]}")
                for h in {lkey, *aliases, *web_hosts}:
                    self.ledger.block(h, f"browser {why}", seconds=secs, channel="browser" if h == lkey else None)
                rec["errors"].append(self._r(f"browser: {why}"))
                return None
            if not body:
                rec["errors"].append(self._r(f"browser: {res.get('status')} {res.get('error', '')[:160]}"))
                return None
            txt, why, tm = self._validate(title, "pdf" if body[:5] == b"%PDF-" else "any", body)
            if txt is None:
                rec["errors"].append(self._r(f"browser: {why}"))
                return None
            prov = self._save(c, "browser", res.get("final_url") or "", body, txt, tm,
                              "pdf" if body[:5] == b"%PDF-" else "html")
            rec.update(source="browser", url=prov["url"], title_match=prov["title_match"], provenance=prov)
            ok = True
            return txt
        finally:
            note = "" if ok else (rec["errors"][-1] if rec["errors"] else "")
            http = res.get("http")
            self.ledger.record("browser", lkey, doi, "ok" if ok else "fail", http if isinstance(http, int) else None,
                               int(res.get("bytes") or 0), note[:300], reservation=rid)
            for k in ("path", "out_file"):
                if res.get(k):
                    Path(res[k]).unlink(missing_ok=True)

    def _browser_tmp(self) -> Path:
        base = self.write_dir() or Path(tempfile.gettempdir())
        d = base / "_browser_tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d.resolve()

    # ---------------------------------------------------------------- cascade
    def channel_applies(self, ch: str, doi: str) -> bool:
        conf = self.ft.get(ch) or {}
        if ch != "cache" and not conf.get("enabled", ch == "oa"):
            return False
        if ch in ("elsevier", "wiley"):
            return bool(doi) and doi.startswith(tuple(conf.get("prefixes") or ()))
        if ch == "inst":
            return bool(doi) and any(doi.startswith(r["prefix"]) for r in conf.get("routes") or [])
        if ch == "browser":
            return bool(doi)
        return True

    def fetch(self, c: dict, *, browser: bool | None = None, channels: list[str] | None = None,
              slot: int = 0) -> Outcome:
        doi = norm_doi(c.get("doi"))
        key = paper_key(c)
        self._tl.c = c
        title = c.get("title") or ""
        rec: dict = dict(doi=doi, key=paper_ident(c), tried=[], errors=[], skipped=[], source=None)
        order = [ch for ch in (channels or self.ft.get("cascade") or CHANNELS) if ch in CHANNELS]
        skip_ch = set(c.get("skip_channels") or ())     # drain_queue: channels still backing off for this paper
        if browser is None:
            browser = bool((self.ft.get("browser") or {}).get("inline"))
        remote_used = False
        for ch in order:
            if ch == "cache":
                if (hit := self.cached(c)):
                    rec.update(hit[1])
                    return Outcome(hit[0], rec)
                continue
            if not self.channel_applies(ch, doi):
                continue
            if (why := self.not_entitled(ch, doi)):
                rec["skipped"].append(self._r(why))
                continue
            if ch in skip_ch:
                rec["skipped"].append(f"{ch}: backing off after a failed attempt")
                rec.setdefault("backing_off", []).append(ch)
                continue

            if ch == "browser" and not browser:
                rec["skipped"].append("browser: queued (run `tbsgen fetch`)")
                continue
            if ch != "browser" and self.remote_mode():
                # remote mode: the HTTP channels run on the remote host (`tbsgen fetch`); use its result if any
                r = self.remote_results.get(paper_ident(c))
                if r is None:
                    rec["skipped"].append(f"{ch}: remote (run `tbsgen fetch`)")
                    continue
                if remote_used:                 # one remote run covered all HTTP channels
                    continue
                remote_used = True
                rr = r.get("rec") or {}
                for k in ("tried", "errors", "skipped", "limited", "not_entitled"):
                    if rr.get(k):
                        rec.setdefault(k, []).extend(rr[k])
                if rr.get("egress_refused"):
                    rec["egress_refused"] = rr["egress_refused"]
                rec["remote_job"] = rr.get("remote_job")
                if r.get("ok") and r.get("text") is not None:
                    rec.update(source=rr.get("source"), url=rr.get("url"), title_match=rr.get("title_match"),
                               provenance=rr.get("provenance"))
                    return Outcome(r["text"], rec)
                continue
            if not self.ft.get("allow_network"):
                rec["skipped"].append(f"{ch}: allow_network=false")
                continue
            try:
                if ch != "browser":
                    self._egress_ok()
                else:
                    self._egress_ok_browser()
                fn = getattr(self, f"ch_{ch}")
                txt = fn(c, key, doi, title, rec, slot) if ch == "browser" else fn(c, key, doi, title, rec)
            except LimitReached as e:
                rec["skipped"].append(self._r(f"{ch}: {e}"))
                self._limited(e, rec, ch)
                continue
            except EgressRefused as e:
                rec["skipped"].append(self._r(f"{ch}: egress refused: {e}"))
                rec["egress_refused"] = self._r(str(e))
                continue
            if txt is not None:
                return Outcome(txt, rec)
        return Outcome(None, rec)


# ------------------------------------------------------------------------------------------- purge (TDM terms)
_EARLY_STAGES = ("S1", "S2", "S3", "S4", "S5")     # stages that still read items/<id>/fulltext.txt


def _text_channel(prov: dict) -> str | None:
    """Channel a text was obtained through (a cache hit names the channel that filled the cache)."""
    if not isinstance(prov, dict):
        return None
    ch = prov.get("channel") or prov.get("source")
    return prov.get("cached_channel") if ch == "cache" else ch


def _fetched_day(prov: dict, fallback: Path) -> str:
    d = str(prov.get("fetched_at") or "")[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    return time.strftime("%Y-%m-%d", time.localtime(fallback.stat().st_mtime)) if fallback.exists() else ""


def purge(cfg: dict, channels: list[str], *, before: str | None = None, dry_run: bool = False,
          runs: list[Path] | None = None) -> dict:
    """`tbsgen fulltext purge`: delete the full texts obtained through `channels` (TDM terms: delete the corpus at
    the end of the project) -- the text, the raw PDF/XML and the copies held by items of `runs` -- and keep the
    ledger rows and the provenance json (it holds no article text; it is marked `purged_at`).

    Only the write cache (`fulltext.write_cache_dir`) is purged.  Read-only caches (`fulltext.cache_dirs`) are
    never written: matching texts there are reported (`readonly_matches`) for the owner to delete.  Item copies
    of items still at S1..S5 (which read fulltext.txt) are not deleted unless the item is finished; they are
    reported as `items_active`."""
    from .state import read_json
    bad = [c for c in channels if c not in LICENCES]
    if not channels or bad:
        return dict(error=f"unknown channel(s) {bad or channels}; choose from {sorted(LICENCES)}")
    if before and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", before):
        return dict(error="--before must be YYYY-MM-DD")
    want = set(channels)
    out = dict(channels=sorted(want), before=before, dry_run=dry_run, texts=0, raw=0, items=0, items_active=[],
               readonly_matches=0, bytes=0)

    def match(prov: dict, path: Path) -> bool:
        return _text_channel(prov) in want and (not before or _fetched_day(prov, path) < before)

    w = write_cache_dir(cfg)
    if w and w.exists():
        with _dir_lock(w):
            for pj in sorted(w.glob("*.json")):
                prov = None
                with contextlib.suppress(OSError, ValueError):
                    prov = json.loads(pj.read_text())
                txt = pj.with_suffix(".txt")
                if not isinstance(prov, dict) or not match(prov, txt) or prov.get("purged_at"):
                    continue
                raws = sorted((w / "raw").glob(pj.stem + ".*")) if (w / "raw").exists() else []
                victims = [p for p in (txt, *raws) if p.exists()]
                out["texts"] += txt.exists()
                out["raw"] += len(raws)
                out["bytes"] += sum(p.stat().st_size for p in victims)
                if not dry_run:
                    for v in victims:
                        v.unlink()
                    prov.update(purged_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                    prov.pop("path", None)
                    _atomic_write(pj, json.dumps(prov, indent=1))
    for d in cfg["fulltext"].get("cache_dirs") or []:
        for pj in Path(d).glob("*.json") if Path(d).exists() else []:
            with contextlib.suppress(OSError, ValueError):
                prov = json.loads(pj.read_text())
                if isinstance(prov, dict) and match(prov, pj.with_suffix(".txt")) and pj.with_suffix(".txt").exists():
                    out["readonly_matches"] += 1
    for run in runs or []:
        items = Path(run) / "items"
        for idir in sorted(items.iterdir()) if items.exists() else []:
            ft = idir / "fulltext.txt"
            if not ft.exists():
                continue
            rec = read_json(idir / "S3_fulltext.json", {})
            if not match(rec, ft):
                continue
            st = read_json(idir / "state.json", {})
            if st.get("stage") in _EARLY_STAGES and st.get("status") not in ("dropped", "done"):
                out["items_active"].append(idir.name)
                continue
            out["items"] += 1
            out["bytes"] += ft.stat().st_size
            if not dry_run:
                ft.unlink()
                (idir / "fulltext_purged.json").write_text(json.dumps(dict(
                    channel=_text_channel(rec), purged_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))))
    return out


# ------------------------------------------------------------------------------------------- paths
def _run_root(cfg: dict) -> Path | None:
    r = (cfg.get("run") or {}).get("root")
    return Path(r) if r else None


def ledger_path(cfg: dict) -> Path:
    p = cfg["fulltext"].get("ledger_path")
    if p:
        return Path(p)
    if (r := _run_root(cfg)) is None:
        raise ValueError("fulltext.ledger_path or run.root must be set")
    return r / "fulltext_ledger.sqlite"


def write_cache_dir(cfg: dict) -> Path | None:
    d = cfg["fulltext"].get("write_cache_dir")
    if d:
        return Path(d)
    r = _run_root(cfg)
    return r / "fulltext_cache" if r else None


# ------------------------------------------------------------------------------------------- queue
# JSONL, one entry per paper key.  Every read-modify-write holds an exclusive flock on <queue>.lock, so S3 in
# one process and `tbsgen fetch` in another cannot lose each other's updates.
QUEUE_FIELDS = ("doi", "arxiv", "pmcid", "title", "oa_url", "pdf_url", "openAccessPdf")


def queue_path(cfg: dict) -> Path:
    q = cfg["fulltext"].get("queue_path")
    return Path(q) if q else Path(cfg["run"]["root"]) / "fulltext_queue.jsonl"


@contextlib.contextmanager
def queue_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _entry_key(e: dict) -> str:
    return e.get("ident") or paper_ident(e)


def read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = _entry_key(e)
        if k and k not in seen:
            seen.add(k)
            out.append(e)
    return out


def write_queue(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))


def queue_entry(c: dict, **extra) -> dict:
    e = {k: c.get(k) for k in QUEUE_FIELDS if c.get(k)}
    if e.get("doi"):
        e["doi"] = norm_doi(e["doi"])
    e["ident"] = paper_ident(c)
    e.update(extra)
    return e


def enqueue(path: Path, entry: dict) -> bool:
    """Idempotent (by paper identity).  For a paper already queued: new subscribers are added, retrieval fields
    that were missing or changed (a new OA link, a PMCID) are merged, and if anything changed the entry becomes
    due again (its backoff is reset)."""
    entry = dict(entry, ident=_entry_key(entry))
    entry.pop("key", None)
    if not entry["ident"]:
        return False
    with queue_lock(path):
        q = read_queue(path)
        for e in q:
            if _entry_key(e) == entry["ident"]:
                changed = False
                old_subs = [list(s) for s in _subs(e)]
                subs = old_subs + [list(s) for s in _subs(entry) if list(s) not in old_subs]
                if subs != old_subs:
                    e["subscribers"] = subs
                    changed = True
                for k in QUEUE_FIELDS:
                    if entry.get(k) and entry[k] != e.get(k):
                        e[k] = entry[k]
                        changed = True
                if changed:
                    e["next_try"] = 0
                    e["ver"] = int(e.get("ver") or 0) + 1      # a drain that read an older version keeps it due
                    write_queue(path, q)
                return False
        q.append(entry)
        write_queue(path, q)
        return True


def reconcile(store, path: Path) -> int:
    """Re-queue items parked with fulltext_unavailable whose queue write never happened (crash between parking
    and enqueueing, or a failed write).  Idempotent.  Returns the number of entries (re)added or updated."""
    from .state import read_json
    n = 0
    for it in store.items():
        st = it.state
        if st.get("stage") == "S3" and st.get("status") == "parked" and st.get("reason") == "fulltext_unavailable":
            q = read_json(it.p("S3_queue.json"), None)
            if q and (q.get("ident") or paper_ident(q)):
                e = {k: v for k, v in q.items() if k in (*QUEUE_FIELDS, "ident", "subscribers", "failed_channels",
                                                         "queued_at")}
                e.setdefault("subscribers", [[str(store.root), it.item_id]])
                n += bool(enqueue(path, e))
    return n


def _subs(e: dict) -> list[list[str]]:
    """[(run, item_id)] waiting for this paper."""
    return [list(s) for s in e.get("subscribers") or []]


# ------------------------------------------------------------------------------------------- queue drain
def match_only(e: dict, only: list[str]) -> bool:
    """`tbsgen fetch --only`: does queue entry `e` match one of the identifiers (DOI with or without "doi:" /
    https://doi.org/, arXiv id with or without "arxiv:", or the entry's ident)?"""
    ident = _entry_key(e)
    doi = norm_doi(e.get("doi"))
    arx = str(e.get("arxiv") or "").lower().strip()
    for x in only:
        x = x.strip()
        if not x:
            continue
        xl = x.lower()
        if xl == ident or (doi and norm_doi(xl) == doi) or (arx and xl.removeprefix("arxiv:") == arx):
            return True
    return False


def classify(out: Outcome) -> tuple[str, float | None, set[str], set[str]]:
    """Queue verdict for an attempt that produced no text -> (verdict, until, failed_channels, limited_channels).

    "limited": some channel was refused without sending anything (ledger cap / cooldown / breaker, egress
    refused); the paper is not charged a try and becomes due again at `until`, the earliest time a refusal lifts
    (None: unknown).  "failed": requests were made and nothing was refused, or no channel applies at all; the
    entry backs off.  failed_channels: channels that made a request and got no text (and were not also refused);
    they back off individually, so a paper waiting for a capped channel does not re-ask a failing one."""
    rec = out.rec
    limited = rec.get("limited") or []
    lim_ch = {x.get("channel") for x in limited if x.get("channel")}
    if rec.get("egress_refused"):
        lim_ch |= {t.split(":", 1)[0] for t in rec.get("skipped") or [] if "egress refused" in t}
    failed = {t.split(":", 1)[0] for t in rec.get("tried") or [] if not t.startswith("cache:")} - lim_ch
    if limited or rec.get("egress_refused"):
        ends = [float(x["until"]) for x in limited if x.get("until")]
        return "limited", (min(ends) if ends else None), failed, lim_ch
    if not rec.get("tried") and rec.get("backing_off"):
        return "limited", None, failed, lim_ch          # every applicable channel is still backing off
    return "failed", None, failed, lim_ch


def _requeue(e: dict, now: float, backoff: float, was_limited: bool, until: float | None, failed_ch: set[str],
             same_version: bool, backoff_only: bool = False) -> None:
    """Update a queue entry after an attempt without text (see drain_queue)."""
    if failed_ch:
        # every channel that really failed backs off on its own (doubling per channel)
        cb = dict(e.get("chan_backoff") or {})
        for ch in sorted(failed_ch):
            n_ch = int((cb.get(ch) or {}).get("tries") or 0) + 1
            cb[ch] = dict(tries=n_ch, next_try=now + backoff * min(8, 2 ** (n_ch - 1)))
        e["chan_backoff"] = cb
        e["failed_channels"] = sorted(set(e.get("failed_channels") or []) | failed_ch)
    if not same_version:                               # updated meanwhile (new link / subscriber): stay due
        e.update(last_try=now, next_try=0)
    elif was_limited:                                  # refused without a request: not a try, no backoff
        ends = [float(v["next_try"]) for v in (e.get("chan_backoff") or {}).values()
                if float(v.get("next_try") or 0) > now]
        if until is not None:
            nxt = until
        elif backoff_only and ends:                    # nothing refused: only waiting for backed-off channels
            nxt = min(ends)
        else:
            nxt = now + LIMIT_RECHECK_S
        # due again when the refusal lifts or a backed-off channel may retry, whichever comes first
        nxt = min([nxt, *ends])
        e.update(next_try=max(now, nxt), limited_until=nxt, last_limited=now)
    else:                                              # a real failure: back off
        n_try = int(e.get("tries") or 0) + 1
        e.update(tries=n_try, last_try=now, next_try=now + backoff * min(8, 2 ** (n_try - 1)))


def _chan_skip(e: dict, now: float) -> list[str]:
    """Channels of a queue entry that failed recently and are still backing off."""
    return sorted(ch for ch, v in (e.get("chan_backoff") or {}).items() if float(v.get("next_try") or 0) > now)


def drain_queue(fetcher: Fetcher, *, max_papers: int = 10, channels: list[str] | None = None,
                browser: bool = True, dry_run: bool = False, on_success=None, log=print,
                now: float | None = None, only: list[str] | None = None) -> dict:
    """`tbsgen fetch`: try queued papers again (cache first, then the cascade incl. the browser channel).

    Order: entries whose `next_try` has passed, least recently tried first.  Only a real failure (requests made,
    no text) backs off (`fulltext.queue_backoff_s`, doubling up to 8x), so permanently unreachable papers do not
    starve the rest.  A paper that was only refused by the ledger (caps, cooldowns, breakers) or by the egress
    check is not counted as a try: it becomes due again when the earliest refusal lifts (`classify`).  A paper
    the remote host did not process (job refused / not collected) stays due unchanged.
    `only`: process just these papers (DOIs / arXiv ids / idents), whether due or not.
    Parallelism: browser jobs of different publishers run in parallel (up to browser.concurrency, one Page per
    worker); papers of one publisher are serialized (the ledger also paces them 60-120 s apart).  An entry leaves
    the queue only after `on_success(entry, outcome)` returned True for all of its subscribers (the CLI's
    re-activation); subscribers added meanwhile keep the entry.
    """
    import fcntl
    from collections import defaultdict

    from . import browser_ego
    now = time.time() if now is None else now
    qp = queue_path(fetcher.cfg)
    qp.parent.mkdir(parents=True, exist_ok=True)
    # one drain at a time per queue (and so per browser TaskSpace / Page set); others return immediately
    drain_fh = open(qp.with_name(qp.name + ".drain.lock"), "a+")
    try:
        fcntl.flock(drain_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        drain_fh.close()
        return dict(busy=True, note="another `tbsgen fetch` is draining this queue")
    try:
        return _drain_locked(fetcher, qp, now, max_papers, channels, browser, dry_run, on_success, log,
                             defaultdict, browser_ego, only)
    finally:
        fcntl.flock(drain_fh, fcntl.LOCK_UN)
        drain_fh.close()


def _drain_locked(fetcher, qp, now, max_papers, channels, browser, dry_run, on_success, log, defaultdict,
                  browser_ego, only=None) -> dict:
    with queue_lock(qp):
        queue = read_queue(qp)
    if only:
        due = [e for e in queue if match_only(e, only)]
    else:
        due = sorted((e for e in queue if float(e.get("next_try") or 0) <= now),
                     key=lambda e: (max(float(e.get("last_try") or 0), float(e.get("last_limited") or 0)),
                                    queue.index(e)))
    todo = due[:max_papers] if max_papers else due
    vers = {_entry_key(e): int(e.get("ver") or 0) for e in todo}
    # per-channel backoff travels with the work copy (also to the remote worker); never written back as such
    todo = [dict(e, skip_channels=sk) if (sk := _chan_skip(e, now)) and not only else e for e in todo]
    stats = dict(queued=len(queue), due=len(due), considered=len(todo), ok=0, failed=0, limited=0, deferred=0,
                 by_channel={}, errors=[])
    if only:
        stats["only"] = list(only)
        stats["only_unmatched"] = [x for x in only if x.strip() and not any(match_only(e, [x]) for e in queue)]
    if dry_run:
        stats["plan"] = [dict(ident=_entry_key(e), doi=e.get("doi"), publisher=publisher_of(e.get("doi") or "")[0],
                              channels=[ch for ch in (channels or fetcher.ft["cascade"])
                                        if ch == "cache" or fetcher.channel_applies(ch, e.get("doi") or "")])
                         for e in todo]
        return stats
    bconf = fetcher.ft.get("browser") or {}
    # browser.batch_only: the browser runs only in egress-gated batches (`tbsgen fetch --browser`)
    use_browser = browser and bool(bconf.get("enabled")) and not bconf.get("batch_only")
    backoff = float(fetcher.ft.get("queue_backoff_s", 6 * 3600))
    lock = threading.Lock()
    results: dict[str, tuple[bool, list]] = {}     # key -> (fetched, acknowledged subscribers)
    deferred: set[str] = set()                     # not tried (remote job not collected yet): stays due
    limited: dict[str, float | None] = {}          # refused by the ledger / egress: due again at `until`
    chan_failed: dict[str, set[str]] = {}          # channels that made a request and got nothing
    backoff_only: set[str] = set()                 # nothing refused, every applicable channel still backing off
    if fetcher.remote_mode():
        # the HTTP channels of every due paper run as one job on the remote host; results feed the cascade
        from . import remote
        want = [ch for ch in (channels or fetcher.ft.get("cascade") or CHANNELS) if ch in remote.HTTP_CHANNELS]
        rjob = remote.run_job(fetcher, [e for e in todo if not fetcher.cached(e)], want or None,
                              log=lambda m: log(m)) if want else dict(results={}, jobs=[])
        fetcher.remote_results.update(rjob.get("results") or {})
        stats["remote"] = {k: v for k, v in rjob.items() if k != "results"}
        for j in stats["remote"].get("jobs") or []:
            j.pop("results", None)
        # a job the remote side refused (code fingerprint, bad spec) or that could not be collected: report it
        # loudly (the CLI exits non-zero); its papers were not tried and stay due
        errs = [f"{j.get('job')}: {j['error']}" for j in stats["remote"].get("jobs") or [] if j.get("error")]
        if rjob.get("refused"):
            errs.insert(0, f"not submitted: {rjob['refused']}")
        if errs:
            stats["remote_error"] = "; ".join(errs)
            # nothing was tried for the papers of a refused round: leave them untouched (no try, no backoff, no
            # browser attempt that would charge them a try); only results already collected are delivered
            keep = [e for e in todo if paper_ident(e) in fetcher.remote_results or fetcher.cached(e)]
            stats["not_tried"] = len(todo) - len(keep)
            todo = keep

    def one(e: dict, slot: int):
        k = _entry_key(e)
        t0 = time.time()
        acked: list = []
        try:
            out = fetcher.fetch(e, browser=use_browser, channels=channels, slot=slot)
            if out.text is not None and on_success:
                for s in _subs(e):
                    try:
                        if on_success(e, out, s):
                            acked.append(s)
                    except Exception as ex:  # noqa: BLE001
                        with lock:
                            stats["errors"].append(dict(key=k, error=fetcher._r(f"reactivate {s[1]}: {ex}")))
            elif out.text is not None:
                acked = _subs(e)
        except Exception as ex:  # noqa: BLE001
            out = Outcome(None, dict(errors=[fetcher._r(f"{type(ex).__name__}: {ex}")], skipped=[], source=None))
        dt = round(time.time() - t0, 1)
        with lock:
            src = out.rec.get("source")
            results[k] = (out.text is not None, acked)
            if out.text is not None:
                stats["ok"] += 1
                stats["by_channel"][src] = stats["by_channel"].get(src, 0) + 1
            elif not out.rec.get("tried") and any(" remote (run `tbsgen fetch`)" in x
                                                  for x in out.rec.get("skipped") or []):
                deferred.add(k)
                stats["deferred"] += 1
            else:
                verdict, until, failed_ch, _ = classify(out)
                chan_failed[k] = failed_ch
                if verdict == "limited":
                    limited[k] = until
                    if not out.rec.get("limited") and not out.rec.get("egress_refused"):
                        backoff_only.add(k)
                    stats["limited"] += 1
                else:
                    stats["failed"] += 1
                stats["errors"].append(dict(key=k, errors=out.rec.get("errors", [])[-3:],
                                            skipped=out.rec.get("skipped", [])[-3:]))
            log(json.dumps(dict(key=k, ok=out.text is not None, source=src, seconds=dt,
                                errors=out.rec.get("errors", [])[-2:], skipped=out.rec.get("skipped", [])[-2:]),
                           ensure_ascii=False))

    space_fh = None
    if use_browser and not fetcher._space_held:
        # one user of a browser TaskSpace at a time, across processes (fetch, inline S3): they share its Pages
        import fcntl
        space_fh = open(fetcher.space_lock_path(), "a+")
        try:
            fcntl.flock(space_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fetcher._space_held = True
        except BlockingIOError:
            space_fh.close()
            space_fh = None
            use_browser = False
            stats["note"] = "browser TaskSpace is in use by another process; browser channel skipped"
    try:
        _run_entries(fetcher, todo, use_browser, bconf, one, defaultdict, browser_ego)
    finally:
        if space_fh:
            import fcntl
            fetcher._space_held = False
            fcntl.flock(space_fh, fcntl.LOCK_UN)
            space_fh.close()
    with queue_lock(qp):
        q = read_queue(qp)
        keep = []
        for e in q:
            k = _entry_key(e)
            if k in results and k not in deferred:
                fetched, acked = results[k]
                rest = [s for s in _subs(e) if s not in acked]
                if fetched and not rest:
                    continue                                   # all subscribers re-activated: done
                if fetched:
                    e["subscribers"] = rest                    # new / unacknowledged subscribers stay
                    e["next_try"] = 0
                    e.pop("chan_backoff", None)
                else:
                    _requeue(e, now, backoff, k in limited, limited.get(k), chan_failed.get(k) or set(),
                             same_version=int(e.get("ver") or 0) == vers.get(k, 0), backoff_only=k in backoff_only)
            keep.append(e)
        write_queue(qp, keep)
        stats["remaining"] = len(keep)
    return stats


def _run_entries(fetcher, todo, use_browser, bconf, one, defaultdict, browser_ego):
    if use_browser and int(bconf.get("concurrency", 1)) > 1:
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in todo:
            groups[publisher_of(e.get("doi") or "")[0]].append(e)
        n = max(1, min(int(bconf.get("concurrency", 1)), len(groups)))
        if fetcher.browser_runner is None:
            browser_ego.ensure_pages(bconf, n)
        pending = list(groups.values())
        plock = threading.Lock()

        def worker(slot: int):
            while True:
                with plock:
                    if not pending:
                        return
                    grp = pending.pop(0)
                for e in grp:
                    one(e, slot)
        ths = [threading.Thread(target=worker, args=(k,)) for k in range(n)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    else:
        for e in todo:
            one(e, 0)
