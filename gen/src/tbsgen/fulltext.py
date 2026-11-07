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
A channel is skipped when it is disabled, its secret is missing, the DOI does not match its prefixes, or the
ledger refuses (cap, cooldown, breaker).  Nothing here bypasses access control: every channel only fetches what
the user's own subscriptions / API terms allow, and bot-wall pages are never clicked through.

Every download is booked in dlledger.DownloadLedger.  Successful texts go to the write cache as
<doi_key>.txt plus <doi_key>.json (provenance: channel, url, time, bytes, sha256, title match); raw bytes go to
<write_cache_dir>/raw/ when `keep_raw` is true.  Read-only caches are never written.

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


def pdf_to_text(pdf: bytes) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.pdf"
        p.write_bytes(pdf)
        r = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(p), "-"], capture_output=True, timeout=180)
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
    txt = _strip_tags(doc)
    if _PAYWALL.search(txt):
        return None
    cut = re.search(r"\n\s*(references|bibliography)\s*\n", txt, re.I)
    return txt[: cut.start()] if cut else txt


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


def paper_key(c: dict) -> str:
    """Stable key of a paper for the cache and the queue: DOI, else arXiv id, else a hash of title/URL."""
    if (d := norm_doi(c.get("doi"))):
        return doi_key(d)
    if c.get("arxiv"):
        return doi_key("arxiv_" + str(c["arxiv"]))
    ident = (c.get("title") or c.get("oa_url") or c.get("pdf_url") or "").lower().strip()
    return "untitled_" + hashlib.sha256(ident.encode()).hexdigest()[:16] if ident else ""


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


def make_transport(use_env_proxy: bool = False) -> Transport:
    """One request, no automatic redirects (Fetcher follows them hop by hop).  Environment proxies are ignored
    unless use_env_proxy."""
    handlers = [_NoRedirect()] + ([] if use_env_proxy else [urllib.request.ProxyHandler({})])
    opener = urllib.request.build_opener(*handlers)

    def transport(url: str, headers: dict, timeout: float, max_bytes: int) -> Resp:
        req = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as r:
                body = r.read(max_bytes + 1)
                return Resp(r.status, dict(r.headers), body[:max_bytes], r.geturl(), truncated=len(body) > max_bytes)
        except urllib.error.HTTPError as e:
            return Resp(e.code, dict(e.headers or {}), e.read(20000) if e.fp else b"", url)
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
        m = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", r.body.decode("ascii", "replace")) if r.status == 200 else None
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
            self.transport = make_transport(bool(ft.get("use_env_proxy")))
        if self.ledger is None:
            self.ledger = DownloadLedger(ledger_path(self.cfg), ft.get("limits"), ft.get("global_limits"),
                                         ft.get("host_caps"))
        self._egress: dict | None = None
        self._egress_browser: dict | None = None
        self._egress_lock = threading.Lock()

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

    def cached(self, key: str) -> tuple[str, dict] | None:
        if not key:
            return None
        dirs = [Path(d) for d in self.ft.get("cache_dirs") or []]
        if (w := self.write_dir()):
            dirs.append(w)
        for d in dirs:
            p = d / f"{key}.txt"
            if p.exists():
                txt = p.read_text(errors="replace")
                if len(txt) >= self.ft["min_chars"]:
                    prov = {}
                    with contextlib.suppress(OSError, ValueError):
                        prov = json.loads((d / f"{key}.json").read_text())
                    return txt, dict(source="cache", path=str(p),
                                     cached_channel=prov.get("channel") if isinstance(prov, dict) else None)
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
            txt = pdf_to_text(body)
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

    def _save(self, key: str, doi: str, channel: str, url: str, body: bytes, txt: str, tm: float | None,
              kind: str) -> dict:
        prov = dict(doi=doi, key=key, channel=channel, url=safe_url(url), kind=kind, bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(), text_chars=len(txt),
                    title_match=None if tm is None else round(tm, 3), fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if (w := self.write_dir()) and key:
            w.mkdir(parents=True, exist_ok=True)
            if self.ft.get("keep_raw", True):
                ext = {"pdf": "pdf", "jats": "xml", "elsevier_xml": "xml", "html": "html"}.get(kind, "bin")
                (w / "raw").mkdir(exist_ok=True)
                _atomic_write(w / "raw" / f"{key}.{ext}", body)
            # provenance first, text last: a text file always has its (complete) provenance next to it
            _atomic_write(w / f"{key}.json", json.dumps(prov, indent=1))
            _atomic_write(w / f"{key}.txt", txt)
            prov["path"] = str(w / f"{key}.txt")
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
            nxt = urllib.parse.urljoin(url, loc)
            sp = urllib.parse.urlsplit(nxt)
            if urllib.parse.urlsplit(url).scheme == "https" and sp.scheme != "https":
                return Resp(0, {}, b"", url, "refused redirect https -> http")
            if sp.scheme not in ("http", "https"):
                return Resp(0, {}, b"", url, "refused redirect to non-http URL")
            if self.ledger and (msg := self.ledger.cooling(sp.hostname or "", *self._related(sp.hostname or ""))):
                return Resp(0, {}, b"", url, f"redirect target cooling: {msg}")
            if sp[:2] != origin:
                hdr = {k: v for k, v in hdr.items() if k.lower() not in CRED_HEADERS}
            rec.setdefault("redirects", []).append(safe_url(nxt))
            url = nxt
        return Resp(0, {}, b"", url, "too many redirects")

    def _download(self, channel: str, key: str, doi: str, title: str, url: str, headers: dict, kind: str,
                  rec: dict, ledger_host: str | None = None) -> str | None:
        """One booked download.  Returns text on success; appends to rec['tried'/'errors']."""
        host = ledger_host or urllib.parse.urlsplit(url).hostname or ""
        if self.ledger is None:
            raise LimitReached("no download ledger configured (fulltext.ledger_path)", "cap")
        rid = self.ledger.acquire(channel, host, doi or key, related=self._related(host))
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
                    self.ledger.block(h, f"{why} on {channel}", seconds=cool, channel=channel)
            if txt is not None:
                prov = self._save(key, doi, channel, r.url, r.body, txt, tm, kind)
                rec.update(source=channel, url=prov["url"], title_match=prov["title_match"], provenance=prov)
            else:
                rec["errors"].append(self._r(f"{channel}: {why}"))
        finally:
            ok = rec.get("source") == channel
            note = "" if ok else (rec["errors"][-1] if rec["errors"] else self._r(r.error))
            self.ledger.record(channel, host, doi or key, "ok" if ok else "fail", r.status, len(r.body),
                               note[:300], reservation=rid)
        return txt

    def _meta(self, url: str) -> dict | None:
        """Metadata lookup (Unpaywall / Europe PMC search): paced and capped, never counted as a download."""
        host = urllib.parse.urlsplit(url).hostname or ""
        if self.ledger is not None:
            self.ledger.acquire_meta(host)
        r = self._get(url, {"User-Agent": self._ua(), "Accept": "application/json"}, {})
        if self.ledger is not None and (cool := block_signal(r.status, r.body)):
            self.ledger.block(host, f"block signal HTTP {r.status} on metadata", seconds=cool)
        if r.status != 200:
            return None
        try:
            return json.loads(r.body)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _host_scoped(e: LimitReached, rec: dict, ch: str) -> None:
        """A host-level refusal skips that source only; channel-level ones stop the channel."""
        if e.scope != "host":
            raise e
        rec["skipped"].append(f"{ch}: {e}")

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
        rid = self.ledger.acquire("browser", lkey, doi, related=aliases + tuple(web_hosts))
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
            http = res.get("http")
            cool = block_signal(http if isinstance(http, int) else None, body)
            if res.get("status") == "challenge" or cool:
                secs = cool or float(b.get("challenge_cooldown_s", 24 * 3600))
                if http == 429:
                    secs = COOLDOWN_429_S
                why = f"HTTP {http}" if http == 429 else f"challenge: {res.get('title', '')[:60]}"
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
            prov = self._save(key, doi, "browser", res.get("final_url") or "", body, txt, tm,
                              "pdf" if body[:5] == b"%PDF-" else "html")
            rec.update(source="browser", url=prov["url"], title_match=prov["title_match"], provenance=prov)
            ok = True
            return txt
        finally:
            note = "" if ok else (rec["errors"][-1] if rec["errors"] else "")
            http = res.get("http")
            self.ledger.record("browser", lkey, doi, "ok" if ok else "fail", http if isinstance(http, int) else None,
                               int(res.get("bytes") or 0), note[:300], reservation=rid)
            if res.get("path"):
                Path(res["path"]).unlink(missing_ok=True)

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
        title = c.get("title") or ""
        rec: dict = dict(doi=doi, key=key, tried=[], errors=[], skipped=[], source=None)
        order = [ch for ch in (channels or self.ft.get("cascade") or CHANNELS) if ch in CHANNELS]
        if browser is None:
            browser = bool((self.ft.get("browser") or {}).get("inline"))
        for ch in order:
            if ch == "cache":
                if (hit := self.cached(key)):
                    rec.update(hit[1])
                    return Outcome(hit[0], rec)
                continue
            if not self.channel_applies(ch, doi):
                continue
            if ch == "browser" and not browser:
                rec["skipped"].append("browser: queued (run `tbsgen fetch`)")
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
                continue
            except EgressRefused as e:
                rec["skipped"].append(f"{ch}: egress refused: {e}")
                rec["egress_refused"] = str(e)
                continue
            if txt is not None:
                return Outcome(txt, rec)
        return Outcome(None, rec)


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
    return e.get("key") or paper_key(e)


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
    e["key"] = paper_key(c)
    e.update(extra)
    return e


def enqueue(path: Path, entry: dict) -> bool:
    """Idempotent (by paper key); a later entry for the same paper adds its item_ids / runs."""
    entry = dict(entry, key=_entry_key(entry))
    if not entry["key"]:
        return False
    with queue_lock(path):
        q = read_queue(path)
        for e in q:
            if _entry_key(e) == entry["key"]:
                subs = {tuple(s) for s in _subs(e)} | {tuple(s) for s in _subs(entry)}
                if subs != {tuple(s) for s in _subs(e)}:
                    e["subscribers"] = sorted([list(s) for s in subs])
                    e.pop("item_ids", None)
                    e["next_try"] = 0
                    write_queue(path, q)
                return False
        if "item_ids" in entry:
            entry["subscribers"] = _subs(entry)
            entry.pop("item_ids")
        q.append(entry)
        write_queue(path, q)
        return True


def _subs(e: dict) -> list[list[str]]:
    """[(run, item_id)] waiting for this paper (older entries: item_ids + run)."""
    if e.get("subscribers"):
        return [list(s) for s in e["subscribers"]]
    return [[e.get("run") or "", i] for i in e.get("item_ids") or []]


# ------------------------------------------------------------------------------------------- queue drain
def drain_queue(fetcher: Fetcher, *, max_papers: int = 10, channels: list[str] | None = None,
                browser: bool = True, dry_run: bool = False, on_success=None, log=print,
                now: float | None = None) -> dict:
    """`tbsgen fetch`: try queued papers again (cache first, then the cascade incl. the browser channel).

    Order: entries whose `next_try` has passed, least recently tried first; a failed entry backs off
    (`fulltext.queue_backoff_s`, doubling up to 8x), so permanently unreachable papers do not starve the rest.
    Parallelism: browser jobs of different publishers run in parallel (up to browser.concurrency, one Page per
    worker); papers of one publisher are serialized (the ledger also paces them 60-120 s apart).  An entry leaves
    the queue only after `on_success(entry, outcome)` returned True for all of its subscribers (the CLI's
    re-activation); subscribers added meanwhile keep the entry.
    """
    from collections import defaultdict

    from . import browser_ego
    now = time.time() if now is None else now
    qp = queue_path(fetcher.cfg)
    with queue_lock(qp):
        queue = read_queue(qp)
    due = sorted((e for e in queue if float(e.get("next_try") or 0) <= now),
                 key=lambda e: (float(e.get("last_try") or 0), queue.index(e)))
    todo = due[:max_papers] if max_papers else due
    stats = dict(queued=len(queue), due=len(due), considered=len(todo), ok=0, failed=0, by_channel={}, errors=[])
    if dry_run:
        stats["plan"] = [dict(key=_entry_key(e), doi=e.get("doi"), publisher=publisher_of(e.get("doi") or "")[0],
                              channels=[ch for ch in (channels or fetcher.ft["cascade"])
                                        if ch == "cache" or fetcher.channel_applies(ch, e.get("doi") or "")])
                         for e in todo]
        return stats
    bconf = fetcher.ft.get("browser") or {}
    use_browser = browser and bool(bconf.get("enabled"))
    backoff = float(fetcher.ft.get("queue_backoff_s", 6 * 3600))
    lock = threading.Lock()
    results: dict[str, tuple[bool, list]] = {}     # key -> (fetched, acknowledged subscribers)

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
            else:
                stats["failed"] += 1
                stats["errors"].append(dict(key=k, errors=out.rec.get("errors", [])[-3:],
                                            skipped=out.rec.get("skipped", [])[-3:]))
            log(json.dumps(dict(key=k, ok=out.text is not None, source=src, seconds=dt,
                                errors=out.rec.get("errors", [])[-2:], skipped=out.rec.get("skipped", [])[-2:]),
                           ensure_ascii=False))

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
    with queue_lock(qp):
        q = read_queue(qp)
        keep = []
        for e in q:
            k = _entry_key(e)
            if k in results:
                fetched, acked = results[k]
                rest = [s for s in _subs(e) if s not in acked]
                if fetched and not rest:
                    continue                                   # all subscribers re-activated: done
                if fetched:
                    e["subscribers"] = rest                    # new / unacknowledged subscribers stay
                    e.pop("item_ids", None)
                    e["next_try"] = 0
                else:
                    n_try = int(e.get("tries") or 0) + 1
                    e.update(tries=n_try, last_try=now, next_try=now + backoff * min(8, 2 ** (n_try - 1)))
            keep.append(e)
        write_queue(qp, keep)
        stats["remaining"] = len(keep)
    return stats
