"""Extra analysis: (1) can the new 4 sources cover what the old sources find?  Uses the *independently judged*
core papers from search_api/review (top-10 of Crossref / arXiv / Europe PMC with the same queries, judged before
this layer existed) and the 35 frozen selected papers as reference sets, and checks at which depth the new arms
contain them.  (2) per-target table for REPORT.md.
"""
import json
from collections import defaultdict
from pathlib import Path

from analyze import ARMS, HEAD, all_keys, label_of, load_labels, rebuild
from tbsretr.config import load_config
from tbsretr.fusion import load_leak

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / "search_api"


def main():
    cfg = load_config()
    leak = load_leak(cfg["leak_refs"])
    lab = load_labels()
    old_lab = json.loads((OLD / "review/labels.json").read_text())
    ref = defaultdict(list)   # target -> old-source core papers (crossref/arxiv/europepmc top-10, label 2)
    for api in ("crossref", "arxiv", "europepmc"):
        for line in (OLD / f"review/top10_{api}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if old_lab.get(f"{api}|{r['target']}|{r['rank']}") == 2:
                k = r["key"]
                d = {"title": r["title"], "doi": k[4:] if k.startswith("doi:") else None,
                     "arxiv": k[3:] if k.startswith("ax:") else None, "api": api}
                ref[r["target"]].append(d)
    out = {}
    tot = defaultdict(lambda: defaultdict(int))
    for p in sorted((HERE / "raw").glob("*.json")):
        rec = json.loads(p.read_text())
        tid = rec["target"]["id"]
        # dedupe reference set
        seen, R = set(), []
        for d in ref[tid]:
            ks = frozenset(all_keys(d))
            if not (ks & seen):
                seen |= ks
                R.append(d)
        row: dict = {"old_core_ref": len(R)}
        for arm in ("new4", "new4-s2", "oa", "full7"):
            kept, dropped, pool, origins = rebuild(rec, ARMS[arm], cfg, leak)
            pool_keys = set()
            for i in pool.roots():
                pool_keys |= all_keys(pool.papers[i].__dict__)
            head = set().union(*(all_keys(x) for x in kept[:HEAD]))
            drop_keys = {k: d["drop_reason"] for d in dropped for k in all_keys(d)}
            in_pool = [d for d in R if all_keys(d) & pool_keys]
            in_head = [d for d in R if all_keys(d) & head]
            row[arm] = {"in_pool": len(in_pool), "in_head60": len(in_head),
                        "dropped_by_prefilter": sum(1 for d in in_pool if any(k in drop_keys for k in all_keys(d))),
                        "missing": [f"{d['api']}: {d['title'][:70]}" for d in R if not (all_keys(d) & pool_keys)]}
            tot[arm]["in_pool"] += len(in_pool)
            tot[arm]["in_head60"] += len(in_head)
            tot[arm]["dropped_by_prefilter"] += row[arm]["dropped_by_prefilter"]
        tot["ref"]["n"] += len(R)
        out[tid] = row
    (HERE / "coverage_old_core.json").write_text(json.dumps({"per_target": out, "total": tot}, ensure_ascii=False, indent=1))
    print(json.dumps(tot, indent=0))
    for tid, r in out.items():
        print(tid[:40], r["old_core_ref"], {a: (r[a]["in_pool"], r[a]["in_head60"]) for a in ("new4", "new4-s2", "oa", "full7")})
        for m in r["new4"]["missing"]:
            print("    new4 missing:", m)


if __name__ == "__main__":
    main()
