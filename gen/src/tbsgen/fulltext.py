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
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .dlledger import DownloadLedger, LimitReached, block_signal, challenge_page

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


_SECTION_WORDS = re.compile(r"\b(introduction|methods?|materials and methods|results|discussion|conclusions?|"
                            r"references|acknowledg(e)?ments?)\b", re.I)


def html_fulltext(body: bytes) -> str | None:
    """Accept an HTML page as full text only if it reads like an article (>= 4 distinct section words) and is
    not a bot-wall page; a landing page with abstract + references is rejected by the length check later."""
    if challenge_page(body):
        return None
    txt = _strip_tags(body.decode("utf-8", "replace"))
    found = {m.group(1).lower()[:6] for m in _SECTION_WORDS.finditer(txt)}
    return txt if len(found) >= 4 else None


# ------------------------------------------------------------------------------------------- HTTP
@dataclass
class Resp:
    status: int
    headers: dict
    body: bytes
    url: str
    error: str = ""


Transport = Callable[[str, dict, float, int], Resp]


def make_transport(use_env_proxy: bool = False) -> Transport:
    handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
    opener = urllib.request.build_opener(*handlers)

    def transport(url: str, headers: dict, timeout: float, max_bytes: int) -> Resp:
        req = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as r:
                return Resp(r.status, dict(r.headers), r.read(max_bytes + 1), r.geturl())
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
            errs.append(f"{u}: HTTP {r.status} {r.error}")
            continue
        ip = ipaddress.ip_address(m.group(1))
        if any(ip in n for n in nets):
            return dict(ok=True, ip=str(ip), via=u)
        raise EgressRefused(f"egress IP {ip} (from {u}) is outside egress_check.allow_cidrs")
    raise EgressRefused("egress IP unknown: " + "; ".join(errs))


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
    _egress: dict | None = None

    def __post_init__(self):
        ft = self.ft
        if self.transport is None:
            self.transport = make_transport(bool(ft.get("use_env_proxy")))
        if self.ledger is None:
            self.ledger = DownloadLedger(ledger_path(self.cfg), ft.get("limits"), ft.get("global_limits"),
                                         ft.get("host_caps"))

    @property
    def ft(self) -> dict:
        return self.cfg["fulltext"]

    # ---------------------------------------------------------------- helpers
    def secret(self, env_name: str | None) -> str | None:
        if not env_name:
            return None
        return self.secrets.get(env_name) or os.environ.get(env_name) or None

    def write_dir(self) -> Path | None:
        return write_cache_dir(self.cfg)

    def cached(self, doi: str) -> tuple[str, dict] | None:
        dirs = [Path(d) for d in self.ft.get("cache_dirs") or []]
        if (w := self.write_dir()):
            dirs.append(w)
        for d in dirs:
            p = d / f"{doi_key(doi)}.txt"
            if p.exists():
                txt = p.read_text(errors="replace")
                if len(txt) >= self.ft["min_chars"]:
                    meta = d / f"{doi_key(doi)}.json"
                    prov = json.loads(meta.read_text()) if meta.exists() else {}
                    return txt, dict(source="cache", path=str(p), cached_channel=prov.get("channel"))
        return None

    def _ua(self) -> str:
        m = self.ft.get("mailto")
        return f"tbsgen/0.2 (text mining for research{'; mailto:' + m if m else ''})"

    def _egress_ok(self):
        ec = self.ft.get("egress_check") or {}
        if not ec.get("enabled"):
            return
        ttl = float(ec.get("ttl_s", 600))
        if self._egress and time.time() - self._egress["ts"] < ttl:
            if not self._egress["ok"]:
                raise EgressRefused(self._egress["why"])
            return
        try:
            r = check_egress(ec, self.transport)
            self._egress = dict(r, ts=time.time())
        except EgressRefused as e:
            self._egress = dict(ok=False, ts=time.time(), why=str(e))
            raise

    def _validate(self, title: str, kind: str, body: bytes) -> tuple[str | None, str, float | None]:
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

    def _save(self, doi: str, channel: str, url: str, body: bytes, txt: str, tm: float | None, kind: str) -> dict:
        prov = dict(doi=doi, channel=channel, url=url, kind=kind, bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(), text_chars=len(txt),
                    title_match=None if tm is None else round(tm, 3), fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if (w := self.write_dir()):
            w.mkdir(parents=True, exist_ok=True)
            k = doi_key(doi)
            if self.ft.get("keep_raw", True):
                ext = {"pdf": "pdf", "jats": "xml", "elsevier_xml": "xml"}.get(kind, "bin")
                (w / "raw").mkdir(exist_ok=True)
                (w / "raw" / f"{k}.{ext}").write_bytes(body)
            tmp = w / f".{k}.txt.tmp{os.getpid()}"
            tmp.write_text(txt)
            os.replace(tmp, w / f"{k}.txt")
            (w / f"{k}.json").write_text(json.dumps(prov, indent=1))
            prov["path"] = str(w / f"{k}.txt")
        return prov

    def _download(self, channel: str, doi: str, title: str, url: str, headers: dict, kind: str, rec: dict,
                  ledger_host: str | None = None) -> str | None:
        """One booked download.  Returns text on success; appends to rec['tried'/'errors']."""
        host = ledger_host or urllib.parse.urlparse(url).netloc
        if self.ledger is None:
            raise LimitReached("no download ledger configured (fulltext.ledger_path)", "cap")
        rid = self.ledger.acquire(channel, host, doi)
        rec["tried"].append(f"{channel}:{_safe_url(url)}")
        r = Resp(0, {}, b"", url, "not started")
        try:
            r = self.transport(url, {"User-Agent": self._ua(), **headers}, float(self.ft["timeout"]),
                               int(self.ft["max_pdf_mb"] * 1e6))
            cool = block_signal(r.status, r.body)
            txt, why, tm = (None, f"HTTP {r.status} {r.error}".strip(), None)
            if r.status == 200 and not cool:
                txt, why, tm = self._validate(title, kind, r.body)
            if cool:
                why = f"block signal HTTP {r.status}"
                for h in {host, urllib.parse.urlparse(r.url).netloc} - {""}:
                    self.ledger.block(h, f"{why} on {channel}", seconds=cool, channel=channel)
            if txt is not None:
                prov = self._save(doi, channel, _safe_url(r.url), r.body, txt, tm, kind)
                rec.update(source=channel, url=_safe_url(r.url), title_match=prov["title_match"], provenance=prov)
            else:
                rec["errors"].append(f"{channel}: {why}")
        finally:
            ok = rec.get("source") == channel
            self.ledger.record(channel, host, doi, "ok" if ok else "fail", r.status, len(r.body),
                               "" if ok else (rec["errors"][-1] if rec["errors"] else r.error)[:300], reservation=rid)
        return txt

    def _meta(self, url: str) -> dict | None:
        """Metadata lookup (Unpaywall / Europe PMC search): paced and capped, never counted as a download."""
        host = urllib.parse.urlparse(url).netloc
        if self.ledger is not None:
            self.ledger.acquire_meta(host)
        r = self.transport(url, {"User-Agent": self._ua(), "Accept": "application/json"}, 30.0, 8 * 2**20)
        if self.ledger is not None and (cool := block_signal(r.status, r.body)):
            self.ledger.block(host, f"block signal HTTP {r.status} on metadata", seconds=cool)
        if r.status != 200:
            return None
        try:
            return json.loads(r.body)
        except json.JSONDecodeError:
            return None

    # ---------------------------------------------------------------- channels
    def ch_oa(self, c: dict, doi: str, title: str, rec: dict) -> str | None:
        oa = self.ft.get("oa") or {}
        if oa.get("europepmc", True) and doi:
            pmcid = c.get("pmcid")
            if not pmcid:
                d = self._meta("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
                    {"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"}))
                hits = ((d or {}).get("resultList") or {}).get("result") or []
                if hits and hits[0].get("pmcid") and hits[0].get("inEPMC") == "Y" and hits[0].get("isOpenAccess") == "Y":
                    pmcid = hits[0]["pmcid"]
            if pmcid:
                url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                if (t := self._download("oa", doi, title, url, {"Accept": "application/xml"}, "jats", rec)):
                    return t
        urls = [u for u in (c.get("oa_url"), c.get("pdf_url"),
                            (c.get("openAccessPdf") or {}).get("url") if isinstance(c.get("openAccessPdf"), dict)
                            else None) if u]
        if oa.get("unpaywall_email") and doi:
            u = self._meta(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?"
                           + urllib.parse.urlencode({"email": oa["unpaywall_email"]}))
            urls += [loc["url_for_pdf"] for loc in (u or {}).get("oa_locations") or [] if loc.get("url_for_pdf")]
        skip = set(oa.get("skip_hosts") or [])
        seen = []
        for u in urls:
            u = u.replace("http://", "https://", 1)
            if u not in seen and urllib.parse.urlparse(u).netloc not in skip:
                seen.append(u)
        seen.sort(key=lambda x: "arxiv.org" not in x)          # arXiv first: most tolerant host
        for u in seen[: int(oa.get("max_urls", 3))]:
            host = urllib.parse.urlparse(u).netloc
            if self.ledger and self.ledger.cooling(host):
                rec["skipped"].append(f"oa: {host} cooling")
                continue
            if (t := self._download("oa", doi, title, u, {"User-Agent": UA_BROWSERISH}, "any", rec)):
                return t
        return None

    def ch_elsevier(self, c: dict, doi: str, title: str, rec: dict) -> str | None:
        e = self.ft["elsevier"]
        key = self.secret(e.get("key_env"))
        if not key:
            rec["skipped"].append(f"elsevier: {e.get('key_env')} not set")
            return None
        hdr = {"X-ELS-APIKey": key, "Accept": "text/xml"}
        if (tok := self.secret(e.get("insttoken_env"))):
            hdr["X-ELS-Insttoken"] = tok
        url = f"https://api.elsevier.com/content/article/doi/{urllib.parse.quote(doi, safe='/')}"
        return self._download("elsevier", doi, title, url, hdr, "elsevier_xml", rec)

    def ch_wiley(self, c: dict, doi: str, title: str, rec: dict) -> str | None:
        w = self.ft["wiley"]
        tok = self.secret(w.get("token_env"))
        if not tok:
            rec["skipped"].append(f"wiley: {w.get('token_env')} not set")
            return None
        url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/" + urllib.parse.quote(doi, safe="")
        return self._download("wiley", doi, title, url, {"Wiley-TDM-Client-Token": tok}, "pdf", rec,
                              ledger_host="api.wiley.com")

    def ch_inst(self, c: dict, doi: str, title: str, rec: dict) -> str | None:
        for r in self.ft["inst"].get("routes") or []:
            if doi.startswith(r["prefix"]):
                url = r["url"].format(doi=doi, suffix=doi.split("/", 1)[1] if "/" in doi else doi)
                if self.ledger and self.ledger.cooling(urllib.parse.urlparse(url).netloc):
                    rec["skipped"].append(f"inst: {urllib.parse.urlparse(url).netloc} cooling")
                    return None
                return self._download("inst", doi, title, url, {"User-Agent": UA_BROWSERISH}, "pdf", rec)
        return None

    def ch_browser(self, c: dict, doi: str, title: str, rec: dict, slot: int = 0) -> str | None:
        from . import browser_ego
        b = self.ft["browser"]
        key = browser_ego.publisher_key(doi)
        if self.ledger is None:
            raise LimitReached("no download ledger configured (fulltext.ledger_path)", "cap")
        rid = self.ledger.acquire("browser", key, doi)
        rec["tried"].append(f"browser:{key}")
        res: dict = {}
        try:
            runner = self.browser_runner or browser_ego.run_one
            res = runner(dict(doi=doi, title=title, pdf_url=browser_ego.pdf_url_hint(doi, c), slot=slot,
                              cfg=b, out_dir=str(self._browser_tmp())))
            body = Path(res["path"]).read_bytes() if res.get("path") and Path(res["path"]).exists() else b""
            if res.get("status") == "challenge":
                self.ledger.block(key, f"challenge did not clear in browser: {res.get('title', '')[:80]}",
                                  seconds=float(b.get("challenge_cooldown_s", 24 * 3600)), channel="browser")
                rec["errors"].append(f"browser: challenge ({res.get('title', '')[:60]})")
                return None
            if not body:
                rec["errors"].append(f"browser: {res.get('status')} {res.get('error', '')[:160]}")
                return None
            txt, why, tm = self._validate(title, "pdf" if body[:5] == b"%PDF-" else "any", body)
            if txt is None:
                rec["errors"].append(f"browser: {why} ({res.get('content_type', '')})")
                return None
            prov = self._save(doi, "browser", _safe_url(res.get("final_url") or ""), body, txt, tm,
                              "pdf" if body[:5] == b"%PDF-" else "html")
            rec.update(source="browser", url=prov["url"], title_match=prov["title_match"], provenance=prov)
            return txt
        finally:
            ok = rec.get("source") == "browser"
            self.ledger.record("browser", key, doi, "ok" if ok else "fail", res.get("http"),
                               int(res.get("bytes") or 0), "" if ok else (rec["errors"][-1] if rec["errors"] else ""),
                               reservation=rid)
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
        title = c.get("title") or ""
        rec: dict = dict(doi=doi, tried=[], errors=[], skipped=[], source=None)
        order = [ch for ch in (channels or self.ft.get("cascade") or CHANNELS) if ch in CHANNELS]
        if browser is None:
            browser = bool((self.ft.get("browser") or {}).get("inline"))
        for ch in order:
            if ch == "cache":
                if doi and (hit := self.cached(doi)):
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
                txt = fn(c, doi, title, rec, slot) if ch == "browser" else fn(c, doi, title, rec)
            except LimitReached as e:
                rec["skipped"].append(f"{ch}: {e}")
                continue
            except EgressRefused as e:
                rec["skipped"].append(f"{ch}: egress refused: {e}")
                rec["egress_refused"] = str(e)
                continue
            if txt is not None:
                return Outcome(txt, rec)
        return Outcome(None, rec)

    def _egress_ok_browser(self):
        """The browser has its own network stack: check its egress through the browser itself."""
        ec = self.ft.get("egress_check") or {}
        if not ec.get("enabled"):
            return
        if getattr(self, "_egress_browser", None) and time.time() - self._egress_browser["ts"] < float(ec.get("ttl_s", 600)):
            if not self._egress_browser["ok"]:
                raise EgressRefused(self._egress_browser["why"])
            return
        from . import browser_ego
        runner = self.browser_runner or browser_ego.run_one
        res = runner(dict(mode="egress", urls=ec.get("urls") or [], cfg=self.ft["browser"], slot=0,
                          out_dir=str(self._browser_tmp())))

        def tr(url, headers, timeout, max_bytes):
            body = (res.get("bodies") or {}).get(url, "")
            return Resp(200 if body else 0, {}, body.encode(), url)
        try:
            r = check_egress(ec, tr)
            self._egress_browser = dict(r, ts=time.time())
        except EgressRefused as e:
            self._egress_browser = dict(ok=False, ts=time.time(), why=f"browser: {e}")
            raise EgressRefused(f"browser: {e}") from None


def _safe_url(u: str) -> str:
    """Drop query parameters that may carry credentials or signed tokens before logging / saving."""
    p = urllib.parse.urlparse(u or "")
    return urllib.parse.urlunparse(p._replace(query="", fragment="")) if p.query else (u or "")


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
def queue_path(cfg: dict) -> Path:
    q = cfg["fulltext"].get("queue_path")
    return Path(q) if q else Path(cfg["run"]["root"]) / "fulltext_queue.jsonl"


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
        d = norm_doi(e.get("doi"))
        if d and d not in seen:
            seen.add(d)
            out.append(e)
    return out


def write_queue(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    os.replace(tmp, path)


def enqueue(path: Path, entry: dict) -> bool:
    """Idempotent append (by DOI); later entries for the same DOI update item_ids."""
    q = read_queue(path)
    d = norm_doi(entry.get("doi"))
    for e in q:
        if norm_doi(e.get("doi")) == d:
            ids = set(e.get("item_ids") or []) | set(entry.get("item_ids") or [])
            if ids != set(e.get("item_ids") or []):
                e["item_ids"] = sorted(ids)
                write_queue(path, q)
            return False
    q.append(entry)
    write_queue(path, q)
    return True


# ------------------------------------------------------------------------------------------- queue drain
def drain_queue(fetcher: Fetcher, *, max_papers: int = 10, channels: list[str] | None = None,
                browser: bool = True, dry_run: bool = False, on_success=None, log=print) -> dict:
    """`tbsgen fetch`: try queued papers again (cache first, then the cascade incl. the browser channel).

    Parallelism: browser jobs of different publishers run in parallel (up to browser.concurrency, one Page per
    worker); papers of one publisher are serialized (the ledger also paces them 60-120 s apart).  Non-browser
    channels run sequentially.  Successful entries leave the queue and `on_success(entry, outcome)` is called
    (the CLI re-activates the items at S3).  A refused channel (cap / cooldown / breaker) leaves the entry.
    """
    import threading
    from collections import defaultdict

    from . import browser_ego
    cfg = fetcher.cfg
    qp = queue_path(cfg)
    queue = read_queue(qp)
    todo = queue[:max_papers] if max_papers else list(queue)
    stats = dict(queued=len(queue), considered=len(todo), ok=0, failed=0, by_channel={}, errors=[])
    if dry_run:
        stats["plan"] = [dict(doi=e["doi"], publisher=browser_ego.publisher_key(norm_doi(e["doi"])),
                              channels=[ch for ch in (channels or fetcher.ft["cascade"])
                                        if ch == "cache" or fetcher.channel_applies(ch, norm_doi(e["doi"]))])
                         for e in todo]
        return stats
    bconf = fetcher.ft.get("browser") or {}
    use_browser = browser and bool(bconf.get("enabled"))
    lock = threading.Lock()
    done: set[str] = set()

    def one(e: dict, slot: int):
        t0 = time.time()
        out = fetcher.fetch(e, browser=use_browser, channels=channels, slot=slot)
        dt = round(time.time() - t0, 1)
        with lock:
            src = out.rec.get("source")
            if out.text is not None:
                stats["ok"] += 1
                stats["by_channel"][src] = stats["by_channel"].get(src, 0) + 1
                done.add(norm_doi(e["doi"]))
            else:
                stats["failed"] += 1
                stats["errors"].append(dict(doi=e["doi"], errors=out.rec.get("errors")[-3:],
                                            skipped=out.rec.get("skipped")[-3:]))
            log(json.dumps(dict(doi=e["doi"], ok=out.text is not None, source=src, seconds=dt,
                                errors=out.rec.get("errors")[-2:], skipped=out.rec.get("skipped")[-2:]),
                           ensure_ascii=False))
        if out.text is not None and on_success:
            on_success(e, out)

    if use_browser and int(bconf.get("concurrency", 1)) > 1:
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in todo:
            groups[browser_ego.publisher_key(norm_doi(e["doi"]))].append(e)
        n = max(1, min(int(bconf.get("concurrency", 1)), len(groups)))
        browser_ego.ensure_pages(bconf, n) if fetcher.browser_runner is None else None
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
    if done:
        with lock:
            rest = [e for e in read_queue(qp) if norm_doi(e.get("doi")) not in done]
            write_queue(qp, rest)
    stats["remaining"] = len(read_queue(qp))
    return stats
