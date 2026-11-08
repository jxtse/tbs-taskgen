"""Orchestrator: target -> per-source ranked lists -> seeds -> snowball/recommend -> enrichment -> fusion.

    from tbsretr.pipeline import Retriever
    r = Retriever(load_config())
    out = r.retrieve(Target(id=..., queries=[...], description=..., seed_dois=[...]))

No LLM calls. Every network call goes through ledger.Http (pacing, budget, cache, 429 cooldown, sqlite accounting).
"""
from __future__ import annotations

import time
import uuid
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from .fusion import Pool, fuse, fuse_resources, load_leak, prefilter_reason, rrf_scores
from .ledger import BudgetExceeded, Http, Ledger, SourceCooling
from .model import Paper, RankedList, Target, informative_words
from .sources import ALL, STOP_EXC

PAPER_SOURCES = ("openalex", "s2", "crossref", "arxiv", "europepmc", "exa")
NEW4 = {"openalex", "s2", "exa"}
OLD3 = {"crossref", "arxiv", "europepmc"}


class Retriever:
    def __init__(self, cfg: dict, run: str | None = None, transport=None, offline: bool = False,
                 sources: set[str] | None = None):
        self.cfg = cfg
        self.run = run or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.ledger = Ledger(cfg["state_dir"], run=self.run)
        self.http = Http(cfg, self.ledger, transport=transport, offline=offline)
        self.src = {n: cls(cfg, self.http) for n, cls in ALL.items() if sources is None or n in sources}
        self.leak = load_leak(cfg.get("leak_refs"))
        self.stopped: dict[str, str] = {}
        self._enriched: list[Paper] = []

    # ---------------------------------------------------------------- helpers
    def status(self) -> dict:
        out = {}
        for n, s in self.src.items():
            ok, why = s.available()
            out[n] = {"available": ok, "note": why}
        return out

    def _run(self, name: str, fn, *a, **kw):
        """Call a source method unless that source is stopped/unavailable; record stop reasons."""
        s = self.src.get(name)
        if s is None or name in self.stopped:
            return None
        ok, why = s.available()
        if not ok:
            self.stopped[name] = why
            return None
        try:
            return s._safe(fn, *a, **kw)
        except STOP_EXC as e:
            self.stopped[name] = f"stopped: {e}"
            return None

    def crossref_queries(self, t: Target) -> list[str]:
        """2–3 precise queries: tool / author / model names plus a few context words."""
        n = self.cfg["sources"]["crossref"]["max_queries"]
        ex = t.exact()
        ctx = " ".join(informative_words(t.queries[0] if t.queries else t.topic_phrase(), 3))
        qs = []
        for term in ex:
            host = next((q for q in t.queries if term in q), None)
            words = [w for w in informative_words(host or ctx, 5) if w not in term and term not in w][:3]
            qs.append((term + " " + " ".join(words)).strip())
            if len(qs) >= n:
                break
        for q in t.queries:
            if len(qs) >= n:
                break
            if q not in qs:
                qs.append(q)
        return qs[:n]

    # ---------------------------------------------------------------- stage 1
    def _stage1_source(self, name: str, t: Target) -> list[RankedList]:
        s: Any = self.src.get(name)
        if s is None:
            return []
        sc = self.cfg["sources"][name]
        out: list[Any] = []
        if name == "openalex":
            for q in t.queries[: sc["max_queries"]]:
                out.append(self._run(name, s.search, q, t))
            if sc.get("semantic") and t.description:
                out.append(self._run(name, s.semantic, t.description, t))
        elif name == "s2":
            for q in t.queries[: sc["max_queries"]]:
                out.append(self._run(name, s.search, q, t))
        elif name == "crossref":
            for q in self.crossref_queries(t):
                out.append(self._run(name, s.search, q, t))
        elif name == "arxiv":
            cats = t.routed_arxiv()
            if cats:
                for q in t.queries[: sc["max_queries"]]:
                    out.append(self._run(name, s.search, q, t, cats))
        elif name == "europepmc":
            if t.routed_biomed():
                for q in t.queries[: sc["max_queries"]]:
                    out.append(self._run(name, s.search, q, t))
        elif name == "exa":
            for ch in sc.get("channels", []):
                if ch == "tool":
                    ex = t.exact()
                    if not ex:
                        continue
                    q = " ".join(ex[:3]) + " software package source code"
                    out.append(self._run(name, s.channel, "code", t, q))
                    if out[-1] is not None:
                        out[-1].name = f"{name}.tool"
                else:
                    out.append(self._run(name, s.channel, ch, t))
        return [x for x in out if x is not None]

    # ---------------------------------------------------------------- main
    def retrieve(self, t: Target, seed_mode: str = "auto", stage1_lists: list[RankedList] | None = None) -> dict:
        """seed_mode: 'auto' (target seeds if any, else pseudo-seeds) | 'pseudo' | 'given' | 'none'."""
        self.stopped = {}
        t0 = time.time()
        timings: dict[str, float] = {}
        lists: list[RankedList] = []
        if stage1_lists is None:
            names = [n for n in PAPER_SOURCES if n in self.src]
            if self.cfg.get("parallel_sources", True):
                with ThreadPoolExecutor(max_workers=len(names)) as ex:
                    futs = {n: ex.submit(self._timed, n, t) for n in names}
                    for n, f in futs.items():
                        ls, dt = f.result()
                        timings[n] = dt
                        lists += ls
            else:
                for n in names:
                    ls, dt = self._timed(n, t)
                    timings[n] = dt
                    lists += ls
        else:
            lists = list(stage1_lists)
        timings["stage1_wall"] = round(time.time() - t0, 2)

        # ---- seeds
        t1 = time.time()
        seeds, seed_origin = self._seeds(t, lists, seed_mode)
        stage2: list[RankedList] = []
        if seeds:
            if "openalex" in self.src and self.cfg["sources"]["openalex"].get("snowball", True):
                oa_seeds = [s for s in seeds if s.referenced_works]
                if oa_seeds:
                    r = self._run("openalex", self.src["openalex"].snowball, oa_seeds, t)
                    if r is not None:
                        stage2.append(r)
            s2c = self.cfg["sources"].get("s2", {})
            if "s2" in self.src and s2c.get("recommend", True) and \
                    (seed_origin == "given" or s2c.get("recommend_pseudo", False)):
                r = self._run("s2", self.src["s2"].recommend, seeds, t)
                if r is not None:
                    stage2.append(r)
        timings["stage2"] = round(time.time() - t1, 2)
        lists += stage2

        pool = Pool()
        for rl in lists:
            pool.add_list(rl)

        # ---- enrichment (abstracts / OA / metadata) for the head of the unfiltered ranking
        t2 = time.time()
        enrich = self._enrich(pool, t)
        timings["enrich"] = round(time.time() - t2, 2)

        exclude = {d.lower() for d in t.exclude_dois}
        papers, dropped = fuse(pool, self.cfg, self.leak, exclude)
        resources = fuse_resources(lists, papers, leak_urls=tuple(self.cfg.get("leak_url_substrings", ())))
        timings["total"] = round(time.time() - t0, 2)
        drop_counts: dict[str, int] = {}
        for d in dropped:
            k = d["drop_reason"].split(":")[0]
            drop_counts[k] = drop_counts.get(k, 0) + 1
        return {
            "target": t.id, "run": self.run, "seed_mode": seed_mode, "seed_origin": seed_origin,
            "seeds": [s.doi or s.arxiv or s.openalex for s in seeds],
            "routing": {"arxiv": t.routed_arxiv(), "europepmc": t.routed_biomed(), "crossref_queries": self.crossref_queries(t)},
            "papers": papers, "n_pool": len(pool.roots()), "dropped": drop_counts, "resources": resources,
            "sources": {n: {"lists": sum(1 for rl in lists if rl.name.startswith(n + ".")),
                            "returned": sum(len(rl.papers) + len(rl.resources) for rl in lists if rl.name.startswith(n + ".")),
                            "status": self.stopped.get(n, "ok"),
                            "errors": self.src[n].errors[-5:] if n in self.src else []}
                        for n in PAPER_SOURCES if n in self.src},
            "enrich": enrich, "timings": timings,
            "_lists": lists, "_pool": pool, "_enriched": list(self._enriched),   # in-memory only; dropped by to_json()
        }

    def _timed(self, n, t):
        a = time.time()
        ls = self._stage1_source(n, t)
        return ls, round(time.time() - a, 2)

    def _seeds(self, t: Target, lists: list[RankedList], mode: str) -> tuple[list[Paper], str]:
        k = self.cfg["pseudo_seeds"]["k"]
        if mode == "none":
            return [], "none"
        if t.seed_dois and mode in ("auto", "given"):
            ps = []
            if "openalex" in self.src:
                ps = self._run("openalex", self.src["openalex"].by_dois, t.seed_dois, t.id) or []
            got = {p.doi for p in ps}
            ps += [Paper(title="", doi=d) for d in t.seed_dois if d.lower() not in got]
            return ps[:10], "given"
        if mode == "given":
            return [], "given:empty"
        # pseudo-seeds: top-k pre-filtered papers from OpenAlex keyword lists only
        pool = Pool()
        for rl in lists:
            if rl.name == "openalex.search":
                pool.add_list(rl)
        sc = rrf_scores(pool, self.cfg)
        order = sorted(sc, key=lambda i: -sc[i]["score"])
        seeds = []
        for i in order:
            p = pool.papers[i]
            if prefilter_reason(p, self.cfg, self.leak, {d.lower() for d in t.exclude_dois}) is None:
                seeds.append(p)
            if len(seeds) >= k:
                break
        return seeds, "pseudo:openalex-top"

    def _enrich(self, pool: Pool, t: Target) -> dict:
        n = self.cfg["sources"]["openalex"].get("backfill_top", 120)
        sc = rrf_scores(pool, self.cfg)
        head = sorted(sc, key=lambda i: -sc[i]["score"])[:n]
        info = {"openalex_lookup": 0, "crossref_lookup": 0, "s2_batch": 0}
        self._enriched: list[Paper] = []
        # OpenAlex: abstracts / OA / type / language for papers not yet seen by OpenAlex
        need = [pool.papers[i].doi for i in head if pool.papers[i].doi and not pool.papers[i].openalex]
        if need and "openalex" in self.src:
            got = self._run("openalex", self.src["openalex"].by_dois, need, t.id) or []
            for p in got:
                pool.add(p)
                self._enriched.append(p)
            info["openalex_lookup"] = len(got)
        # Crossref: publisher/venue/type metadata for DOIs still missing venue or type
        if "crossref" in self.src and self.cfg["sources"]["crossref"].get("metadata_lookup", True):
            need = [pool.papers[i].doi for i in head if pool.papers[i].doi and (not pool.papers[i].venue or not pool.papers[i].type)]
            if need:
                got = self._run("crossref", self.src["crossref"].by_dois, need[:80], t.id) or []
                for p in got:
                    pool.add(p)
                    self._enriched.append(p)
                info["crossref_lookup"] = len(got)
        # S2: openAccessPdf + S2 ids for the head (1 batch call)
        if "s2" in self.src and self.cfg["sources"]["s2"].get("batch_lookup", True):
            ids = []
            for i in head[:100]:
                p = pool.papers[i]
                if p.s2:
                    continue
                if p.doi:
                    ids.append("DOI:" + p.doi)
                elif p.arxiv:
                    ids.append("ARXIV:" + p.arxiv)
            if ids:
                got = self._run("s2", self.src["s2"].batch, ids, t.id) or []
                for p in got:
                    pool.add(p)
                    self._enriched.append(p)
                info["s2_batch"] = len(got)
        return info


def to_json(out: dict) -> dict:
    return {k: v for k, v in out.items() if not k.startswith("_")}


__all__ = ["Retriever", "Target", "to_json", "NEW4", "OLD3", "BudgetExceeded", "SourceCooling", "fuse", "Pool"]
