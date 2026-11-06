"""Merge/dedupe candidates across sources, rule pre-filter, and reciprocal-rank fusion (RRF). Pure functions."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .model import Paper, RankedList, Resource
from .util import latin_ratio, resource_key

REVIEW_TITLE = re.compile(r"\b(review|survey|overview|tutorial|perspective|primer|state[- ]of[- ]the[- ]art|"
                          r"meta-analysis|systematic review|lecture notes|a guide to|introduction to|"
                          r"recent advances|advances in|progress in|challenges and opportunities)\b", re.I)
REVIEW_ABS = re.compile(r"^(in )?this (review|survey|overview|tutorial|perspective)|we review\b|we survey\b|"
                        r"this (article|paper|chapter) reviews", re.I)
BAD_TYPES = {"review", "editorial", "erratum", "letter", "retraction", "paratext", "dataset", "peer-review",
             "book", "reference-entry", "other", "component", "grant", "standard", "report-component"}
EN_CODES = {"", "en", "eng", "english"}


def source_of(list_name: str) -> str:
    return list_name.split(".", 1)[0]


# ------------------------------------------------------------------ merge
def _merge_into(a: Paper, b: Paper):
    for f in ("doi", "arxiv", "openalex", "s2", "pmid", "venue", "type", "language", "url", "year"):
        if not getattr(a, f) and getattr(b, f):
            setattr(a, f, getattr(b, f))
    if len(b.abstract or "") > len(a.abstract or ""):
        a.abstract = b.abstract
    if len(b.title or "") > 0 and (not a.title or (a.origin in ("exa", "tavily") and b.origin not in ("exa", "tavily"))):
        a.title = b.title
    if b.oa_url and (not a.oa_url or (a.oa_url and not a.oa_url.lower().endswith(".pdf") and b.oa_url.lower().endswith(".pdf"))):
        a.oa_url = b.oa_url
    a.is_oa = bool(a.is_oa) or bool(b.is_oa) or bool(a.oa_url)
    if b.cited_by is not None:
        a.cited_by = max(a.cited_by or 0, b.cited_by)
    if b.referenced_works and not a.referenced_works:
        a.referenced_works = b.referenced_works
    a.code_links = sorted(set(a.code_links) | set(b.code_links))
    a.data_links = sorted(set(a.data_links) | set(b.data_links))
    a.code_words |= b.code_words
    a.data_words |= b.data_words
    a.has_data_flag |= b.has_data_flag
    # Language: an explicit non-English code from any source wins only if no source says English
    if b.language and a.language and b.language.lower() in EN_CODES:
        a.language = b.language


class Pool:
    """Union of papers keyed by any shared identifier (doi / arxiv / openalex / s2 / normalised title)."""

    def __init__(self):
        self.papers: list[Paper] = []
        self.index: dict[str, int] = {}
        self.parent: list[int] = []
        self.hits: list[dict[str, list[int]]] = []   # per paper: list_name -> ranks (1-based)

    def _find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def add(self, p: Paper, list_name: str | None = None, rank: int | None = None) -> int:
        ks = p.keys()
        ids = {self._find(self.index[k]) for k in ks if k in self.index}
        if not ids:
            i = len(self.papers)
            self.papers.append(Paper(**{**p.__dict__}))
            self.parent.append(i)
            self.hits.append(defaultdict(list))
        else:
            i = min(ids)
            for j in ids - {i}:                       # merge other clusters into i
                self.parent[j] = i
                _merge_into(self.papers[i], self.papers[j])
                for ln, rs in self.hits[j].items():
                    self.hits[i][ln] += rs
                self.hits[j] = defaultdict(list)
            _merge_into(self.papers[i], p)
        for k in self.papers[i].keys():
            self.index[k] = i
        if list_name is not None and rank is not None:
            self.hits[i][list_name].append(rank)
        return i

    def add_list(self, rl: RankedList):
        for r, p in enumerate(rl.papers, 1):
            if p.title or p.doi or p.arxiv:
                self.add(p, rl.name, r)

    def roots(self) -> list[int]:
        return [i for i in range(len(self.papers)) if self._find(i) == i]


# ------------------------------------------------------------------ filters
def load_leak(path: str | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    d = json.loads(Path(path).read_text())
    return {x.lower() for v in d.values() for x in v}


def prefilter_reason(p: Paper, cfg: dict, leak: set[str], exclude: set[str]) -> str | None:
    pf = cfg["prefilter"]
    if pf.get("drop_leak", True):
        ids = {p.doi or "", "arxiv:" + (p.arxiv or "-")}
        if ids & (leak | exclude):
            return "leak:tbs-task-source"
    t = (p.type or "").lower()
    types = {x.strip() for x in re.split(r"[,;]", t) if x.strip()}
    if pf.get("drop_review", True):
        if types & {"review", "systematic-review", "review-article", "review article", "systematic review"} or \
                "review" in types or REVIEW_TITLE.search(p.title or "") or REVIEW_ABS.search((p.abstract or "")[:200]):
            return "review/overview"
    if types & (BAD_TYPES - {"review"}):
        return f"type:{t}"
    if pf.get("drop_non_english", True):
        lang = (p.language or "").lower()
        if lang not in EN_CODES:
            return f"non-english:{lang}"
        if latin_ratio((p.title or "") + " " + (p.abstract or "")[:500]) < 0.85:
            return "non-english:script"
    if p.year and pf.get("min_year") and p.year < pf["min_year"]:
        return f"old:{p.year}"
    if pf.get("drop_no_abstract", True) and len(p.abstract or "") < pf.get("min_abstract_chars", 200):
        return "no-abstract"
    if not p.title:
        return "no-title"
    return None


# ------------------------------------------------------------------ fusion
def rrf_scores(pool: Pool, cfg: dict, allowed_sources: set[str] | None = None) -> dict[int, dict]:
    """RRF over *source-kinds*: within one source-kind (e.g. openalex.search with 4 queries) a paper contributes
    its best rank fully and every further hit at 25%, so many queries of one source cannot swamp the others."""
    fc = cfg["fusion"]
    k, W = fc["rrf_k"], fc["weights"]
    out = {}
    for i in pool.roots():
        hits = pool.hits[i]
        score, ranks, srcs = 0.0, {}, set()
        for ln, rs in hits.items():
            if allowed_sources is not None and source_of(ln) not in allowed_sources:
                continue
            rs = sorted(rs)
            w = W.get(ln, 0.5)
            score += w * (1.0 / (k + rs[0]) + 0.25 * sum(1.0 / (k + r) for r in rs[1:]))
            ranks[ln] = rs[0]
            srcs.add(source_of(ln))
        if not ranks:
            continue
        score += fc.get("multi_source_bonus", 0.0) * max(0, len(srcs) - 1)
        out[i] = {"score": score, "ranks": ranks, "sources": sorted(srcs)}
    return out


def fuse(pool: Pool, cfg: dict, leak: set[str], exclude: set[str] | frozenset = frozenset(),
         allowed_sources: set[str] | None = None, top_n: int | None = None) -> tuple[list[dict], list[dict]]:
    sc = rrf_scores(pool, cfg, allowed_sources)
    # Missing abstract is a soft signal (methodlit.py: "A missing abstract is not a reason to exclude"): such papers
    # stay in the ranking with score x no_abstract_factor and abstract_missing=True; the hard drop only applies
    # when prefilter.drop_no_abstract is True.
    fac = cfg["prefilter"].get("no_abstract_factor", 0.5)
    minab = cfg["prefilter"].get("min_abstract_chars", 200)
    for i in sc:
        if len(pool.papers[i].abstract or "") < minab:
            sc[i]["score"] *= fac
    order = sorted(sc, key=lambda i: (-sc[i]["score"], -(pool.papers[i].cited_by or 0)))
    kept, dropped = [], []
    for i in order:
        p = pool.papers[i]
        reason = prefilter_reason(p, cfg, leak, set(exclude))
        row = {**p.to_dict(), "rrf": round(sc[i]["score"], 6), "ranks": sc[i]["ranks"], "sources": sc[i]["sources"],
               "n_sources": len(sc[i]["sources"]), "abstract_missing": len(p.abstract or "") < minab}
        if reason:
            row["drop_reason"] = reason
            dropped.append(row)
        else:
            kept.append(row)
    n = top_n if top_n is not None else cfg["fusion"]["top_n"]
    for r, row in enumerate(kept, 1):
        row["fused_rank"] = r
    return kept[:n], dropped


# ------------------------------------------------------------------ resources
def fuse_resources(lists: list[RankedList], papers: list[dict], k: int = 60,
                   leak_urls: tuple[str, ...] = ()) -> list[dict]:
    agg: dict[str, dict] = {}
    for rl in lists:
        for r, res in enumerate(rl.resources, 1):
            if res.kind == "paper" or not res.url or any(x in res.url.lower() for x in leak_urls):
                continue
            key = resource_key(res.url)
            a = agg.setdefault(key, {"key": key, "url": res.url, "kind": res.kind, "title": res.title,
                                     "snippet": res.snippet, "found_by": {}, "score": 0.0, "from_papers": []})
            if rl.name not in a["found_by"] or r < a["found_by"][rl.name]:
                a["found_by"][rl.name] = r
            a["score"] += 1.0 / (k + r)
            if res.kind == "code" and a["kind"] != "code":
                a["kind"] = "code"
    for p in papers:                        # code/data links mentioned in paper metadata
        for url, kind in [(u, "code") for u in p.get("code_links", [])] + [(u, "data") for u in p.get("data_links", [])]:
            url = url if url.startswith("http") else "https://" + url
            key = resource_key(url)
            a = agg.setdefault(key, {"key": key, "url": url, "kind": kind, "title": "", "snippet": "", "found_by": {},
                                     "score": 0.0, "from_papers": []})
            a["from_papers"].append(p.get("doi") or p.get("arxiv") or p.get("title", "")[:60])
            a["score"] += 0.5 / (k + p.get("fused_rank", 50))
    return sorted(agg.values(), key=lambda a: -a["score"])
