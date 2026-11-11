"""Code signals for papers: which candidates come with the authors' own code (code-grounded task generation).

Three sources, all optional and merged into one record per paper (`code_signals`) plus extra `code_links`:

* OpenAlex full-text search (`fulltext.search:github.com|zenodo.org|gitlab`): (a) a recall channel that runs the
  target queries with that filter (ranked list `openalex.code`), (b) a batch flag for the fused head (one call per
  50 DOIs).  Measured on 224 papers of two batches: recall 84 % of the papers with own code (precision 81 %).
* Papers with Code frozen archive (HF `pwc-archive/links-between-paper-and-code`, 300k paper<->repo links): an
  offline join by arXiv id or normalised title.  Point `code.pwc_path` at a TSV(.gz) with the columns
  `arxiv, title_key, repo_url, is_official, mentioned_in_paper` (see README: how to build it from the parquet).
* GitHub repository search for READMEs that cite the paper's DOI / arXiv id (`"<doi>" in:readme`): high precision
  (33/33 own repos in the probe).  Search API limit: 30 req/min with a token, 10 req/min without; paced by the
  ledger (`sources.github.min_interval_s`) and capped per target (`code.github_max_papers`).

Merged signal (OpenAlex ∪ metadata links ∪ GitHub README ∪ PwC official): recall 92 %, precision 82 % on the
same 224 papers.  Nothing here downloads full text.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import subprocess
import urllib.parse
from pathlib import Path

from .ledger import HTTPFailure
from .model import Paper, RankedList, Target
from .util import norm_doi, title_key

CODE_HOST_RE = re.compile(r"(github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|zenodo\.org|sourceforge\.net|"
                          r"cran\.r-project\.org|pypi\.org|bioconductor\.org|codeocean\.com)", re.I)


def has_meta_code(p: dict | Paper) -> bool:
    links = p.get("code_links") if isinstance(p, dict) else p.code_links
    return any(CODE_HOST_RE.search(str(u)) for u in links or [])


# ---------------------------------------------------------------------------------------------- OpenAlex
def openalex_fulltext_filter(cfg: dict) -> str:
    return "fulltext.search:" + "|".join(cfg["code"].get("fulltext_terms") or ["github.com", "zenodo.org", "gitlab"])


def openalex_code_search(oa, q: str, target: Target) -> RankedList:
    """Target query restricted to works whose full text mentions a code host (and year >= code.min_year)."""
    cfg = oa.cfg
    f = oa.scfg.get("filter", "")
    if target.openalex_filter:
        f = (f + "," if f else "") + target.openalex_filter
    f = (f + "," if f else "") + openalex_fulltext_filter(cfg)
    if cfg["code"].get("min_year"):
        f += f",publication_year:>{int(cfg['code']['min_year']) - 1}"
    from .sources import OA_SELECT, _oa_work
    params = {"search": q, "per_page": oa.scfg["per_page"], "select": OA_SELECT, "filter": f}
    r = oa._get("code_search", params, target.id, "search")
    works = r.json().get("results", []) if r.status == 200 else []
    return RankedList("openalex.code", q, papers=[_oa_work(w) for w in works])


def openalex_fulltext_flags(oa, dois: list[str], target_id: str = "") -> set[str]:
    """DOIs (normalised) among `dois` whose OpenAlex full text mentions a code host.  One call per 50 DOIs."""
    out: set[str] = set()
    dois = [d for d in dict.fromkeys(norm_doi(x) for x in dois) if d]
    flt = openalex_fulltext_filter(oa.cfg)
    for i in range(0, len(dois), 50):
        chunk = dois[i:i + 50]
        r = oa._get("code_flags", {"filter": "doi:" + "|".join(chunk) + "," + flt, "per_page": 50, "select": "doi"},
                    target_id, "list")
        if r.status == 200:
            out |= {d for w in r.json().get("results", []) if (d := norm_doi(w.get("doi")))}
    return out


# ---------------------------------------------------------------------------------------------- PwC archive
class PwcIndex:
    """Offline Papers-with-Code join.  Rows: arxiv, title_key, repo_url, is_official, mentioned_in_paper."""

    def __init__(self, path: str | os.PathLike | None):
        self.by_arxiv: dict[str, list[dict]] = {}
        self.by_title: dict[str, list[dict]] = {}
        self.path = str(path) if path else None
        if not path or not Path(path).exists():
            return
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                rec = dict(repo=row["repo_url"], official=row.get("is_official") in ("1", "True", "true"),
                           mentioned=row.get("mentioned_in_paper") in ("1", "True", "true"))
                a = re.sub(r"v\d+$", "", (row.get("arxiv") or "").strip().lower())
                if a:
                    self.by_arxiv.setdefault(a, []).append(rec)
                tk = (row.get("title_key") or "").strip()
                if len(tk) >= 25:
                    self.by_title.setdefault(tk, []).append(rec)

    def __bool__(self):
        return bool(self.by_arxiv or self.by_title)

    def lookup(self, arxiv: str | None, title: str | None) -> list[dict]:
        hits = list(self.by_arxiv.get(re.sub(r"v\d+$", "", (arxiv or "").lower()), [])) if arxiv else []
        if not hits and title:
            hits = list(self.by_title.get(title_key(title), []))
        seen, out = set(), []
        for h in sorted(hits, key=lambda h: (not h["official"], not h["mentioned"])):
            if h["repo"].lower() not in seen:
                seen.add(h["repo"].lower())
                out.append(h)
        return out


_PWC_CACHE: dict[str, PwcIndex] = {}


def pwc_index(path: str | None) -> PwcIndex:
    if not path:
        return PwcIndex(None)
    if path not in _PWC_CACHE:
        _PWC_CACHE[path] = PwcIndex(path)
    return _PWC_CACHE[path]


# ---------------------------------------------------------------------------------------------- GitHub search
def _gh_cli_transport(timeout: float = 60):
    """GitHub REST through the `gh` CLI (it authenticates itself; this process never sees a token)."""
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("https_proxy", "http_proxy", "all_proxy")}

    def call(path: str, params: dict) -> tuple[int, dict]:
        args = ["gh", "api", "-X", "GET", path, "-i"] + [x for k, v in params.items() for x in ("-f", f"{k}={v}")]
        try:
            r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 0, {"error": type(e).__name__}
        head, _, body = r.stdout.partition("\r\n\r\n") if "\r\n\r\n" in r.stdout else r.stdout.partition("\n\n")
        m = re.match(r"HTTP/[\d.]+ (\d+)", head)
        st = int(m.group(1)) if m else (200 if r.returncode == 0 else 0)
        try:
            return st, json.loads(body) if body.strip() else {}
        except ValueError:
            return st, {"raw": body[:200]}
    return call


def github_readme_search(http, cfg: dict, p: dict, target_id: str = "", gh=None) -> list[str]:
    """Repos whose README cites the paper's DOI or arXiv id (`"<id>" in:readme`).  With `code.github_client="gh"`
    the gh CLI does the call (its own auth, 30 req/min); otherwise anonymous REST through the ledger (10 req/min)."""
    terms = []
    doi = norm_doi(p.get("doi"))
    if doi and not doi.startswith("10.48550"):
        terms.append(doi)
    if p.get("arxiv"):
        terms.append(re.sub(r"v\d+$", "", str(p["arxiv"])))
    out: list[str] = []
    scfg = cfg["sources"]["github"]
    led = http.ledger
    for t in terms[:2]:
        q = f'"{t}" in:readme'
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": q, "per_page": 5})
        if gh is not None:
            # same cache / budget / pacing / cooldown / offline rule as Http.request; the call itself through gh
            key = led.cache_key("GET", url, None)
            hit = led.cache_get(key, cfg.get("cache_ttl_days", 30))
            if hit:
                led.book("github", "readme_search", url, hit[0], 0.0, cached=True, target=target_id)
                st, d = hit[0], json.loads(hit[1] or b"{}")
            else:
                if getattr(http, "offline", False) or led.cooling("github") > 0:
                    break
                led.check_budget("github", scfg)
                led.wait_turn("github", scfg.get("min_interval_s", 2.2))
                st, d = gh("search/repositories", {"q": q, "per_page": "5"})
                led.book("github", "readme_search", url, st, 0.0, target=target_id)
                if st == 200:
                    led.cache_put(key, "github", url, st, json.dumps(d).encode())
                elif st in (403, 429):
                    led.set_cooldown("github", scfg.get("cooldown_429_s", 600), f"http {st}")
                    break
        else:
            # anonymous search API: 10 requests / minute -> at least 6.5 s apart; a rate-limit 403 cools the source
            led.wait_turn("github", max(6.5, float(scfg.get("min_interval_s", 2.2))))
            # Http.request retries would send without this spacing: the github source never retries internally
            http.cfg["sources"]["github"] = dict(scfg, retries=0)
            try:
                r = http.request("github", "readme_search", url, target=target_id,
                                 headers={"Accept": "application/vnd.github+json"}, ok_statuses=(200, 422))
                st, d = r.status, (r.json() if r.status == 200 else {})
            except HTTPFailure as e:
                if e.status == 403:
                    led.set_cooldown("github", scfg.get("cooldown_429_s", 600), "http 403 (rate limit)")
                break
        if st == 200:
            out += [it["full_name"] for it in d.get("items", []) if it.get("full_name")]
    return sorted(set(out))


# ---------------------------------------------------------------------------------------------- merge
def annotate(papers: list[dict], *, oa_flags: set[str], pwc: PwcIndex, gh_hits: dict[str, list[str]]) -> None:
    """Attach `code_signals` to fused paper rows (in place) and add the found repos to `code_links`."""
    for p in papers:
        key = p.get("doi") or p.get("arxiv") or p.get("title", "")[:80]
        pw = pwc.lookup(p.get("arxiv"), p.get("title")) if pwc else []
        in_code_channel = bool((p.get("ranks") or {}).get("openalex.code"))
        sig: dict = dict(meta=has_meta_code(p),
                         openalex_fulltext=bool(norm_doi(p.get("doi")) in oa_flags) or in_code_channel,
                         pwc=[h["repo"] for h in pw if h["official"]][:5],                 # archive: is_official
                         pwc_mentioned=[h["repo"] for h in pw if h["mentioned"] and not h["official"]][:5],
                         pwc_unofficial=[h["repo"] for h in pw if not (h["official"] or h["mentioned"])][:3],
                         gh_readme=[f"https://github.com/{r}" for r in gh_hits.get(key, [])][:5])
        sig["any"] = bool(sig["meta"] or sig["openalex_fulltext"] or sig["pwc"] or sig["pwc_mentioned"]
                          or sig["gh_readme"])
        p["code_signals"] = sig
        # candidates only: ownership is decided downstream (tbsgen S3b) from the paper's own text
        extra = list(sig["pwc"]) + list(sig["pwc_mentioned"]) + list(sig["gh_readme"])
        if extra:
            p["code_links"] = sorted(set(p.get("code_links") or []) | set(extra))


def year_bias(papers: list[dict], cfg: dict) -> None:
    """Soft preference for recent papers (code more often public and runnable): rows older than code.min_year go
    behind the others (order kept within each group); fused_rank is renumbered."""
    y0 = cfg["code"].get("min_year")
    if not y0 or not cfg["code"].get("prefer_recent", True):
        return
    new = [p for p in papers if not p.get("year") or int(p["year"]) >= int(y0)]
    old = [p for p in papers if p.get("year") and int(p["year"]) < int(y0)]
    for p in old:
        p["year_bias"] = f"before {y0}: ranked after newer papers"
    papers[:] = new + old
    for r, p in enumerate(papers, 1):
        p["fused_rank"] = r
