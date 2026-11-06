"""Source adapters. Each returns RankedList objects (never raises for a single failed call; failures are recorded
in `self.errors`). Budget/cooldown exhaustion raises through so the orchestrator can mark the source stopped."""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

from .ledger import BudgetExceeded, Http, HTTPFailure, SourceCooling
from .model import Paper, RankedList, Resource, Target, informative_words
from .util import (arxiv_from_text, classify_url, clean, doi_from_text, inv2text, norm_doi, strip_site_suffix)

STOP_EXC = (BudgetExceeded, SourceCooling)


class Source:
    name = "base"

    def __init__(self, cfg: dict, http: Http):
        self.cfg = cfg
        self.scfg = cfg["sources"][self.name]
        self.http = http
        self.errors: list[str] = []
        self.key = os.environ.get(self.scfg.get("key_env") or "", "") or None

    def available(self) -> tuple[bool, str]:
        if not self.scfg.get("enabled", True):
            return False, "disabled in config"
        if self.scfg.get("require_key") and not self.key:
            return False, f"skipped: env {self.scfg.get('key_env')} not set"
        return True, ""

    def _safe(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except STOP_EXC:
            raise
        except HTTPFailure as e:
            self.errors.append(f"{fn.__name__}: {e}"[:300])
        except (ValueError, KeyError, ET.ParseError) as e:
            self.errors.append(f"{fn.__name__}: parse {e!r}"[:300])
        return None


# ====================================================================== OpenAlex
OA_SELECT = ("id,doi,title,publication_year,type,language,cited_by_count,abstract_inverted_index,open_access,"
             "best_oa_location,primary_location,referenced_works,ids,is_retracted")


def _oa_work(w: dict) -> Paper:
    pl = w.get("primary_location") or {}
    src = (pl.get("source") or {})
    best = w.get("best_oa_location") or {}
    ids = w.get("ids") or {}
    p = Paper(title=clean(w.get("title") or w.get("display_name") or ""), doi=w.get("doi"),
              openalex=(w.get("id") or "").rsplit("/", 1)[-1] or None,
              pmid=(ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
              abstract=inv2text(w.get("abstract_inverted_index")), year=w.get("publication_year"),
              venue=src.get("display_name") or "", type=w.get("type") or "", language=w.get("language") or "",
              oa_url=best.get("pdf_url") or best.get("landing_page_url") or (w.get("open_access") or {}).get("oa_url"),
              is_oa=(w.get("open_access") or {}).get("is_oa"), cited_by=w.get("cited_by_count"),
              url=pl.get("landing_page_url"),
              referenced_works=[r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])], origin="openalex")
    if not p.arxiv:
        p.arxiv = arxiv_from_text(pl.get("landing_page_url") or "") or arxiv_from_text(best.get("landing_page_url") or "")
    return p.add_clues()


class OpenAlex(Source):
    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def _url(self, params: dict) -> str:
        params = {**params, "mailto": self.cfg["mailto"]}
        if self.key:
            params["api_key"] = self.key   # stripped from cache keys / ledger / logs by strip_secret_params
        return self.BASE + "?" + urllib.parse.urlencode(params)

    def _cost(self, kind):
        return self.scfg["cost_usd"].get(kind, 0.001)

    def _get(self, kind, params, target, cost_kind):
        return self.http.request(self.name, kind, self._url(params), cost_usd=self._cost(cost_kind), target=target)

    def search(self, q: str, target: Target) -> RankedList:
        f = self.scfg.get("filter", "")
        if target.openalex_filter:
            f = (f + "," if f else "") + target.openalex_filter
        params = {"search": q, "per_page": self.scfg["per_page"], "select": OA_SELECT}
        if f:
            params["filter"] = f
        r = self._get("search", params, target.id, "search")
        works = r.json().get("results", []) if r.status == 200 else []
        return RankedList("openalex.search", q, papers=[_oa_work(w) for w in works])

    def semantic(self, text: str, target: Target) -> RankedList:
        r = self._get("semantic", {"search.semantic": text[:1500], "per_page": 25, "select": OA_SELECT}, target.id,
                      "semantic")
        return RankedList("openalex.semantic", text[:80],
                          papers=[_oa_work(w) for w in (r.json().get("results", []) if r.status == 200 else [])])

    def by_dois(self, dois: list[str], target_id: str = "") -> list[Paper]:
        out: list[Paper] = []
        dois = [d for d in dict.fromkeys(norm_doi(x) for x in dois) if d]
        for i in range(0, len(dois), 50):
            chunk = dois[i:i + 50]
            r = self._get("list", {"filter": "doi:" + "|".join(chunk), "per_page": 50, "select": OA_SELECT},
                          target_id, "list")
            if r.status == 200:
                out += [_oa_work(w) for w in r.json().get("results", [])]
        return out

    def by_ids(self, ids: list[str], target_id: str = "") -> list[Paper]:
        out: list[Paper] = []
        ids = list(dict.fromkeys(ids))
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            r = self._get("list", {"filter": "openalex:" + "|".join(chunk), "per_page": 50, "select": OA_SELECT},
                          target_id, "list")
            if r.status == 200:
                out += [_oa_work(w) for w in r.json().get("results", [])]
        return out

    def snowball(self, seeds: list[Paper], target: Target) -> RankedList:
        """Shared referenced_works of the seeds, ranked by (#seeds citing it, citations)."""
        from collections import Counter
        cnt: Counter = Counter()
        for s in seeds:
            cnt.update(set(s.referenced_works))
        need = self.scfg.get("snowball_min_shared", 2) if len(seeds) >= 2 else 1
        top = [w for w, c in cnt.most_common() if c >= need][: self.scfg.get("snowball_max", 50)]
        if not top:
            return RankedList("openalex.snowball", "seeds:" + ",".join(s.openalex or "" for s in seeds))
        papers = self.by_ids(top, target.id)
        papers.sort(key=lambda p: (-cnt[p.openalex], -(p.cited_by or 0)))
        return RankedList("openalex.snowball", "seeds:" + ",".join(s.openalex or "" for s in seeds), papers=papers)


# ====================================================================== Semantic Scholar
S2_FIELDS = ("paperId,externalIds,title,abstract,year,venue,citationCount,openAccessPdf,isOpenAccess,"
             "publicationTypes,url")


def _s2_paper(d: dict) -> Paper:
    ext = d.get("externalIds") or {}
    oap = d.get("openAccessPdf") or {}
    oa_url = oap.get("url") or None
    p = Paper(title=clean(d.get("title") or ""), doi=ext.get("DOI"), arxiv=ext.get("ArXiv"), s2=d.get("paperId"),
              pmid=str(ext["PubMed"]) if ext.get("PubMed") else None, abstract=clean(d.get("abstract") or ""),
              year=d.get("year"), venue=d.get("venue") or "", type=",".join(d.get("publicationTypes") or []),
              oa_url=oa_url, is_oa=bool(oa_url) or d.get("isOpenAccess"), cited_by=d.get("citationCount"),
              url=d.get("url"), origin="s2")
    return p.add_clues()


class SemanticScholar(Source):
    name = "s2"

    def _h(self):
        return {"x-api-key": self.key} if self.key else {}

    def search(self, q: str, target: Target) -> RankedList:
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
            {"query": q, "limit": self.scfg["limit"], "fields": S2_FIELDS})
        r = self.http.request(self.name, "search", url, headers=self._h(), target=target.id)
        return RankedList("s2.search", q, papers=[_s2_paper(d) for d in (r.json().get("data") or [])]
                          if r.status == 200 else [])

    def recommend(self, seeds: list[Paper], target: Target) -> RankedList:
        ids = []
        for s in seeds:
            if s.s2:
                ids.append(s.s2)
            elif s.doi:
                ids.append("DOI:" + s.doi)
            elif s.arxiv:
                ids.append("ARXIV:" + s.arxiv)
        ids = ids[:10]
        if not ids:
            return RankedList("s2.recommend", "")
        url = "https://api.semanticscholar.org/recommendations/v1/papers?" + urllib.parse.urlencode(
            {"fields": S2_FIELDS, "limit": self.scfg["recommend_limit"]})
        if len(ids) == 1:
            url = (f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{urllib.parse.quote(ids[0])}?"
                   + urllib.parse.urlencode({"fields": S2_FIELDS, "limit": self.scfg["recommend_limit"]}))
            r = self.http.request(self.name, "recommend", url, headers=self._h(), target=target.id)
        else:
            r = self.http.request(self.name, "recommend", url, method="POST", headers=self._h(),
                                  body={"positivePaperIds": ids}, target=target.id)
        data = r.json().get("recommendedPapers") or [] if r.status == 200 else []
        return RankedList("s2.recommend", "seeds:" + ",".join(ids), papers=[_s2_paper(d) for d in data])

    def batch(self, ids: list[str], target_id: str = "") -> list[Paper]:
        """POST /paper/batch: resolve DOI:/ARXIV: ids to S2 records (OA PDF links, S2 ids)."""
        out: list[Paper] = []
        for i in range(0, len(ids), 100):
            chunk = ids[i:i + 100]
            url = "https://api.semanticscholar.org/graph/v1/paper/batch?" + urllib.parse.urlencode({"fields": S2_FIELDS})
            r = self.http.request(self.name, "batch", url, method="POST", headers=self._h(), body={"ids": chunk},
                                  target=target_id)
            if r.status == 200:
                out += [_s2_paper(d) for d in r.json() if d]
        return out


# ====================================================================== Exa
EXA_CHANNEL = {
    # channel -> (query template, extra request fields)
    "code": ("{topic} open-source implementation code repository", {"includeDomains": ["github.com", "gitlab.com"]}),
    "data": ("{topic} public benchmark dataset download", {}),
    "docs": ("{topic} software documentation tutorial", {}),
    "paper": ("{desc}", {"category": "publication"}),   # was "research paper"; both return publication entities
}


class Exa(Source):
    """Exa search.  backend="pool" (default) goes through a local search-pool service (POST {pool_url}/v1/search,
    engine=exa, no Tavily fallback) that holds the Exa keys; backend="direct" calls api.exa.ai with key_env."""
    name = "exa"

    def __init__(self, cfg: dict, http: Http):
        super().__init__(cfg, http)
        self.backend = self.scfg.get("backend", "pool")
        if self.backend not in ("pool", "direct"):
            raise ValueError(f"sources.exa.backend must be 'pool' or 'direct', got {self.backend!r}")
        self.pool_url = (os.environ.get("TBS_SEARCH_POOL_URL") or self.scfg.get("pool_url") or "").rstrip("/")
        n = self.scfg.get("num_results")
        if self.backend == "pool" and not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 20):
            raise ValueError(f"sources.exa.num_results must be an integer 1-20 with backend='pool', got {n!r}")

    def available(self) -> tuple[bool, str]:
        if self.backend == "pool":
            if not self.scfg.get("enabled", True):
                return False, "disabled in config"
            return (True, "") if self.pool_url else (False, "skipped: sources.exa.pool_url not set")
        return super().available()

    def _search(self, kind: str, query: str, extra: dict, target: Target) -> dict:
        if self.backend == "pool":
            return self._search_pool(kind, query, extra, target)
        body = {"query": query, "type": self.scfg.get("type", "auto"), "numResults": self.scfg["num_results"],
                "contents": {"highlights": {"maxCharacters": self.scfg.get("highlights_chars", 400)}}, **extra}
        est = self.scfg.get("est_cost_usd", 0.012)

        def cost(st, hd, b):
            try:
                return float(json.loads(b).get("costDollars", {}).get("total", est)), 0.0
            except Exception:  # noqa: BLE001
                return est, 0.0
        r = self.http.request(self.name, kind, "https://api.exa.ai/search", method="POST",
                              headers={"x-api-key": self.key or ""}, body=body, cost_usd=est, target=target.id,
                              cost_from_response=cost)
        return r.json() if r.status == 200 else {}

    def _search_pool(self, kind: str, query: str, extra: dict, target: Target) -> dict:
        """Same Exa request shape as the direct path, sent through the pool.  The pool returns each Exa row
        unchanged under `raw` (entities included), so downstream parsing is identical."""
        opts = {"type": self.scfg.get("type", "auto"), "highlights_max_characters": self.scfg.get("highlights_chars", 400),
                **extra}
        body = {"query": query[:1500], "limit": self.scfg["num_results"], "engine": "exa",
                "include_contents": False, "exa_options": opts}
        est = self.scfg.get("est_cost_usd", 0.012)

        def cost(st, hd, b):
            try:
                return float((json.loads(b).get("usage") or {}).get("cost_usd", est)), 0.0
            except Exception:  # noqa: BLE001
                return est, 0.0
        def validate(b: bytes):
            # checked before caching: a wrong-provider or malformed 200 must not be cached as an empty result
            try:
                d = json.loads(b)
            except ValueError:
                return "pool response is not JSON"
            if not isinstance(d, dict) or not isinstance(d.get("results"), list):
                return "pool response has no results list"
            if d.get("provider") != "exa":   # engine=exa never falls back, but never mix providers silently
                return f"pool returned provider={d.get('provider')!r}"
            return None
        r = self.http.request(self.name, kind, f"{self.pool_url}/v1/search", method="POST", body=body,
                              cost_usd=est, target=target.id, cost_from_response=cost, validate=validate)
        if r.status != 200:    # Http passes 404 through; for the pool endpoint that is a misconfiguration
            raise HTTPFailure(r.status, f"pool endpoint {self.pool_url}/v1/search returned HTTP {r.status}")
        return {"results": [row.get("raw") or row for row in r.json()["results"]]}

    def channel(self, ch: str, target: Target, query: str | None = None) -> RankedList:
        tmpl, extra = EXA_CHANNEL[ch]
        q = query or tmpl.format(topic=target.topic_phrase(), desc=(target.description or " ".join(target.queries[:2]))[:600])
        d = self._search(ch, q, extra, target)
        return _web_list(f"exa.{ch}", q, d.get("results") or [], text_key="highlights", origin="exa",
                         paper_channel=(ch == "paper"))


# ====================================================================== Tavily
TAVILY_CHANNEL = {
    "code": ("{topic} source code github", {"include_domains": ["github.com", "gitlab.com"]}),
    "data": ("{topic} public dataset download", {}),
    "docs": ("{topic} software documentation tutorial", {}),
    "paper": ("{q0} paper", {"include_domains": ["arxiv.org", "nature.com", "sciencedirect.com", "springer.com",
                                                  "wiley.com", "iop.org", "aps.org", "oup.com", "biorxiv.org",
                                                  "ncbi.nlm.nih.gov", "plos.org", "pnas.org", "aanda.org",
                                                  "tandfonline.com", "informs.org", "ametsoc.org", "doi.org",
                                                  "semanticscholar.org", "openreview.net", "ieee.org", "acm.org"]}),
}


class Tavily(Source):
    name = "tavily"

    def channel(self, ch: str, target: Target, query: str | None = None) -> RankedList:
        tmpl, extra = TAVILY_CHANNEL[ch]
        q = query or tmpl.format(topic=target.topic_phrase(), q0=target.queries[0] if target.queries else target.id)
        depth = self.scfg.get("search_depth", "basic")
        credits = 2.0 if depth == "advanced" else 1.0
        body = {"query": q[:400], "search_depth": depth, "max_results": self.scfg["max_results"],
                "include_usage": True, **extra}

        def cost(st, hd, b):
            try:
                u = json.loads(b).get("usage") or {}
                return 0.0, float(u.get("credits", credits))
            except Exception:  # noqa: BLE001
                return 0.0, credits
        r = self.http.request(self.name, ch, "https://api.tavily.com/search", method="POST",
                              headers={"Authorization": f"Bearer {self.key or ''}"}, body=body, credits=credits,
                              target=target.id, cost_from_response=cost)
        d = r.json() if r.status == 200 else {}
        return _web_list(f"tavily.{ch}", q, d.get("results") or [], text_key="content", origin="tavily",
                         paper_channel=(ch == "paper"))


def _web_list(name, q, results, text_key, origin, paper_channel) -> RankedList:
    rl = RankedList(name, q)
    for it in results:
        url = it.get("url") or ""
        txt = it.get(text_key)
        if isinstance(txt, list):
            txt = " … ".join(txt)
        txt = clean(txt or it.get("text") or it.get("summary") or "")
        kind = classify_url(url)
        ent = None
        for e in it.get("entities") or []:       # Exa research-paper entities carry DOI/abstract when available
            if isinstance(e, dict) and isinstance(e.get("properties"), dict):
                ent = e["properties"]
        if kind == "paper" or (paper_channel and kind == "web"):
            doi = norm_doi((ent or {}).get("doi")) or doi_from_text(url)
            ax = arxiv_from_text(url)
            title = strip_site_suffix(it.get("title") or "")
            if doi or ax or paper_channel:
                e = ent or {}
                p = Paper(title=clean(e.get("title")) or title, doi=doi, arxiv=ax,
                          abstract=re.sub(r"^\s*abstract\s*", "", clean(e.get("abstract") or ""), flags=re.I),
                          year=e.get("year") or _year(it.get("publishedDate") or it.get("published_date")),
                          type=e.get("type") or "", language=e.get("language") or "",
                          cited_by=e.get("citationCount"), url=url, origin=origin)
                p.add_clues(txt)
                rl.papers.append(p)
                continue
        rl.resources.append(Resource(url=url, kind=kind if kind != "paper" else "paper", title=clean(it.get("title")),
                                     snippet=txt[:400], published=it.get("publishedDate") or it.get("published_date"),
                                     origin=origin))
    return rl


def _year(s):
    m = re.match(r"(\d{4})", s or "")
    return int(m.group(1)) if m else None


# ====================================================================== Crossref
class Crossref(Source):
    name = "crossref"
    SELECT = "DOI,title,abstract,container-title,publisher,issued,is-referenced-by-count,type,link,relation"

    def _item(self, it: dict) -> Paper:
        rel = it.get("relation") or {}
        rel_urls = [x.get("id", "") for v in rel.values() for x in (v if isinstance(v, list) else [])]
        p = Paper(title=clean(" ".join(it.get("title") or [])), doi=it.get("DOI"),
                  abstract=re.sub(r"^\s*(abstract|summary)\s*", "", clean(it.get("abstract") or ""), flags=re.I),
                  year=((it.get("issued") or {}).get("date-parts") or [[None]])[0][0],
                  venue=" ".join(it.get("container-title") or [])[:200], type=it.get("type") or "",
                  language=it.get("language") or "", cited_by=it.get("is-referenced-by-count"), origin="crossref")
        return p.add_clues(" ".join(rel_urls))

    def search(self, q: str, target: Target, kind: str = "search") -> RankedList:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
            "query.bibliographic": q, "rows": self.scfg["rows"], "mailto": self.cfg["mailto"],
            "filter": "type:journal-article,type:proceedings-article,type:posted-content,from-pub-date:1985",
            "select": self.SELECT})
        r = self.http.request(self.name, kind, url, target=target.id)
        items = r.json()["message"]["items"] if r.status == 200 else []
        return RankedList("crossref.search", q, papers=[self._item(i) for i in items])

    def by_dois(self, dois: list[str], target_id: str = "") -> list[Paper]:
        out = []
        dois = [d for d in dict.fromkeys(norm_doi(x) for x in dois) if d]
        for i in range(0, len(dois), 40):
            chunk = dois[i:i + 40]
            url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
                "filter": ",".join("doi:" + d for d in chunk), "rows": len(chunk), "mailto": self.cfg["mailto"],
                "select": self.SELECT})
            r = self.http.request(self.name, "lookup", url, target=target_id)
            if r.status == 200:
                out += [self._item(i) for i in r.json()["message"]["items"]]
        return out


# ====================================================================== arXiv
class Arxiv(Source):
    name = "arxiv"
    NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

    def search(self, q: str, target: Target, cats: list[str]) -> RankedList:
        # Terms grouped as all:(w1 w2 ...) = relevance-ranked OR inside the routed categories. A strict
        # all:w1 AND all:w2 ... form returned 0 hits for 13/20 eval queries (2026-11-05), so it is not used.
        words = informative_words(q, 8) or [q]
        terms = "all:%28" + "+".join(urllib.parse.quote(w) for w in words) + "%29"
        cat = "+OR+".join(f"cat:{c}*" if c not in ("quant-ph",) else f"cat:{c}" for c in cats)
        sq = f"%28{cat}%29+AND+{terms}" if cat else terms
        url = (f"https://export.arxiv.org/api/query?search_query={sq}&max_results={self.scfg['max_results']}"
               f"&sortBy=relevance")
        r = self.http.request(self.name, "search", url, headers={"Accept": "application/atom+xml"}, target=target.id)
        papers = []
        if r.status == 200:
            for e in ET.fromstring(r.body).findall("a:entry", self.NS):
                aid = re.sub(r"v\d+$", "", (e.findtext("a:id", "", self.NS) or "").rsplit("/abs/", 1)[-1])
                ab = " ".join((e.findtext("a:summary", "", self.NS) or "").split())
                cm = e.findtext("x:comment", "", self.NS) or ""
                pub = e.findtext("a:published", "", self.NS) or ""
                jr = e.findtext("x:journal_ref", "", self.NS) or ""
                p = Paper(title=" ".join((e.findtext("a:title", "", self.NS) or "").split()), arxiv=aid,
                          doi=e.findtext("x:doi", "", self.NS) or None, abstract=ab,
                          year=int(pub[:4]) if pub[:4].isdigit() else None, venue=jr or "arXiv", type="preprint",
                          oa_url=f"https://arxiv.org/pdf/{aid}", is_oa=True, url=f"https://arxiv.org/abs/{aid}",
                          origin="arxiv")
                papers.append(p.add_clues(cm))
        return RankedList("arxiv.search", sq, papers=papers)


# ====================================================================== Europe PMC
class EuropePMC(Source):
    name = "europepmc"

    def search(self, q: str, target: Target) -> RankedList:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": q + " AND NOT PUB_TYPE:\"review\"", "format": "json", "resultType": "core",
             "pageSize": self.scfg["page_size"]})
        r = self.http.request(self.name, "search", url, target=target.id)
        papers = []
        if r.status == 200:
            for it in ((r.json().get("resultList") or {}).get("result") or []):
                urls = ((it.get("fullTextUrlList") or {}).get("fullTextUrl") or [])
                oa = [u["url"] for u in urls if u.get("availabilityCode") in ("OA", "F")]
                p = Paper(title=clean(it.get("title")), doi=it.get("doi"), pmid=it.get("pmid"),
                          abstract=clean(it.get("abstractText")), year=int(it["pubYear"]) if it.get("pubYear") else None,
                          venue=((it.get("journalInfo") or {}).get("journal") or {}).get("title", ""),
                          type=",".join(((it.get("pubTypeList") or {}).get("pubType") or [])),
                          language=it.get("language") or "", oa_url=oa[0] if oa else None,
                          is_oa=it.get("isOpenAccess") == "Y" or bool(oa), cited_by=it.get("citedByCount"),
                          has_data_flag=it.get("hasData") == "Y", origin="europepmc")
                papers.append(p.add_clues())
        return RankedList("europepmc.search", q, papers=papers)


ALL = {c.name: c for c in (OpenAlex, SemanticScholar, Exa, Tavily, Crossref, Arxiv, EuropePMC)}
