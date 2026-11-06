"""Offline analysis of eval/raw/*.json -> eval/metrics.json + eval/tables.md (+ eval/to_label.jsonl for any paper
or resource in the judged sets that has no label yet).

Relevance labels (subjective, single annotator = this agent reading title + abstract; scale identical to
search_api/review/labels.json: 0 off-topic, 1 on-topic but peripheral/application/review, 2 core method paper):
  * reused from search_api/review (labels.json + top10_*.jsonl key mapping) when the same paper was judged there;
  * new labels in eval/labels_new.json (same scale); resource labels in eval/labels_res.json
    (0 irrelevant, 1 related/generic, 2 directly usable implementation/dataset/docs for this target).

Arms (re-fused offline from the very same raw ranked lists, so differences are only due to the source set):
  full7   = openalex + s2 + exa + tavily + crossref + arxiv + europepmc
  new4    = openalex + s2 + exa + tavily
  full-s2 = full7 without s2          new4-s2 = new4 without s2
  old2    = crossref + arxiv (the sources of the current production pipeline, for reference)
  oa      = openalex only
"""
from __future__ import annotations

import json
import statistics
import sqlite3
from collections import defaultdict
from pathlib import Path

from tbsretr.config import load_config
from tbsretr.fusion import Pool, fuse, load_leak
from tbsretr.model import Paper, RankedList, Resource
from tbsretr.util import norm_doi, resource_key, title_key

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / "search_api"
ARMS = {
    "full7": {"openalex", "s2", "exa", "tavily", "crossref", "arxiv", "europepmc"},
    "new4": {"openalex", "s2", "exa", "tavily"},
    "full-s2": {"openalex", "exa", "tavily", "crossref", "arxiv", "europepmc"},
    "new4-s2": {"openalex", "exa", "tavily"},
    "old2": {"crossref", "arxiv"},
    "oa": {"openalex"},
    # leave-one-out of the old sources from full7, and the recommended config
    "full-crossref": {"openalex", "s2", "exa", "tavily", "arxiv", "europepmc"},
    "full-arxiv": {"openalex", "s2", "exa", "tavily", "crossref", "europepmc"},
    "full-epmc": {"openalex", "s2", "exa", "tavily", "crossref", "arxiv"},
    "new4+crossref": {"openalex", "s2", "exa", "tavily", "crossref"},
    "full-exa-tavily": {"openalex", "s2", "crossref", "arxiv", "europepmc"},
}
TOPK = 20
HEAD = 60


def pkey(d: dict) -> str:
    if norm_doi(d.get("doi")):
        return "doi:" + str(norm_doi(d.get("doi")))
    if d.get("arxiv"):
        return "arxiv:" + d["arxiv"]
    return "t:" + title_key(d.get("title"))


def all_keys(d: dict) -> set[str]:
    ks = set()
    doi = norm_doi(d.get("doi"))
    if doi:
        ks.add("doi:" + doi)
        if doi.startswith("10.48550/arxiv."):
            ks.add("arxiv:" + doi.split("arxiv.", 1)[1])
    if d.get("arxiv"):
        ks.add("arxiv:" + d["arxiv"].removeprefix("arxiv:"))
    tk = title_key(d.get("title"))
    if len(tk) >= 25:
        ks.add("t:" + tk)
    return ks


def load_labels():
    """target -> {key: label} from old review + new labels."""
    lab = defaultdict(dict)
    old = json.loads((OLD / "review/labels.json").read_text())
    for f in (OLD / "review").glob("top10_*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            v = old.get(f"{r['api']}|{r['target']}|{r['rank']}")
            if v is None:
                continue
            k0 = r["key"]
            if k0.startswith("ax:"):
                k0 = "arxiv:" + k0[3:]
            for k in {k0, "t:" + title_key(r.get("title"))}:
                if k != "t:":
                    lab[r["target"]][k] = v
    p = HERE / "labels_new.json"
    if p.exists():
        for tid, d in json.loads(p.read_text()).items():
            if tid.startswith("_"):
                continue
            for k, v in d.items():
                lab[tid][k] = v
    return lab


def label_of(lab_t: dict, d: dict):
    for k in all_keys(d):
        if k in lab_t:
            return lab_t[k]
    return None


def rebuild(rec: dict, allowed: set[str], cfg, leak, seeded: bool = False):
    pool = Pool()
    lists = rec["stage1"] + (rec["stage2_seeded"] if seeded else rec["stage2_pseudo"])
    for lst in lists:
        if lst["name"].split(".")[0] not in allowed:
            continue
        pool.add_list(RankedList(lst["name"], lst["query"], papers=[Paper(**p) for p in lst["papers"]]))
    for p in rec["enriched"]:
        # enrichment may only attach to papers this arm already retrieved (otherwise an OpenAlex lookup of a
        # DOI that Crossref found would leak Crossref's recall into the new-4 arm)
        pp = Paper(**p)
        if p.get("origin") in allowed and any(k in pool.index for k in pp.keys()):
            pool.add(pp)
    kept, dropped = fuse(pool, cfg, leak, top_n=10_000)
    origins = defaultdict(set)   # pool-level: which sources returned each paper anywhere (any rank)
    for lst in lists:
        for p in lst["papers"]:
            for k in all_keys(p):
                origins[k].add(lst["name"].split(".")[0])
    return kept, dropped, pool, origins


def recall(sel, items):
    keys = set()
    for it in items:
        keys |= all_keys(it)
    return sum(1 for s in sel if all_keys(s) & keys)


def main():
    cfg = load_config()
    leak = load_leak(cfg["leak_refs"])
    lab = load_labels()
    recs = [json.loads(p.read_text()) for p in sorted((HERE / "raw").glob("*.json"))]
    out = {"per_target": {}, "arms": {}, "resources": {}, "cost": {}, "latency": {}}
    to_label, to_label_res = [], []
    res_lab = json.loads((HERE / "labels_res.json").read_text()) if (HERE / "labels_res.json").exists() else {}

    agg = {a: defaultdict(float) for a in ARMS}
    extra = {"full7_vs_new4": [], "s2_extra": [], "full7_vs_fulls2": []}
    for rec in recs:
        tid = rec["target"]["id"]
        sel = rec["selected"]
        held = [s for s in sel if s["doi"] not in rec["seed_dois"]]
        pt = {}
        heads = {}
        for arm, allowed in ARMS.items():
            kept, dropped, pool, origins = rebuild(rec, allowed, cfg, leak)
            heads[arm] = kept[:HEAD]
            top = kept[:TOPK]
            labs = [label_of(lab[tid], p) for p in top]
            for p, l in zip(top, labs):
                if l is None:
                    to_label.append({"target": tid, "key": pkey(p), "title": p["title"], "year": p["year"],
                                     "venue": p["venue"], "abstract": (p["abstract"] or "")[:500]})
            m = {"top20_rel": sum(1 for l in labs if l is not None and l >= 1),
                 "top20_core": sum(1 for l in labs if l == 2),
                 "top20_unlabeled": sum(1 for l in labs if l is None),
                 "pool": len(pool.roots()), "kept": len(kept),
                 "sel_n": len(sel), "sel_in_pool": recall(sel, [pp.__dict__ for pp in (pool.papers[i] for i in pool.roots())]),
                 "sel_in_head60": recall(sel, kept[:HEAD]), "sel_in_top20": recall(sel, top),
                 "abs_missing_head60": sum(1 for p in kept[:HEAD] if not p["abstract"]) / max(1, len(kept[:HEAD])),
                 "oa_head60": sum(1 for p in kept[:HEAD] if p["oa_url"]) / max(1, len(kept[:HEAD]))}
            if rec.get("seeded") and arm in ("full7", "new4", "oa", "full-s2"):
                ks, _, _, _ = rebuild(rec, allowed, cfg, leak, seeded=True)
                m["seeded_heldout_n"] = len(held)
                m["seeded_heldout_in_head60"] = recall(held, ks[:HEAD])
                m["pseudo_heldout_in_head60"] = recall(held, kept[:HEAD])
            pt[arm] = m
            for k, v in m.items():
                agg[arm][k] += v
        # ---- what the extra sources add (head-60 = what the LLM screen would see)
        for a, b, name in (("full7", "new4", "full7_vs_new4"), ("full7", "full-s2", "s2_extra"),
                           ("new4", "new4-s2", "s2_extra_new4"), ("full7", "full-crossref", "crossref_extra"),
                           ("full7", "full-arxiv", "arxiv_extra"), ("full7", "full-epmc", "epmc_extra"),
                           ("full7", "full-exa-tavily", "exa_tavily_extra"), ("new4+crossref", "new4", "crossref_over_new4")):
            kb = set().union(*(all_keys(p) for p in heads[b])) if heads[b] else set()
            add = [p for p in heads[a] if not (all_keys(p) & kb)]
            for p in add:
                l = label_of(lab[tid], p)
                if l is None:
                    to_label.append({"target": tid, "key": pkey(p), "title": p["title"], "year": p["year"],
                                     "venue": p["venue"], "abstract": (p["abstract"] or "")[:500]})
            _, _, _, origins = rebuild(rec, ARMS["full7"], cfg, leak)
            oa_keys = {k for k, s in origins.items() if "openalex" in s}
            b_keys = {k for k, s in origins.items() if s & ARMS[b]}
            pt[name] = {
                "added_head60": len(add),
                "added_rel": sum(1 for p in add if (label_of(lab[tid], p) or 0) >= 1),
                "added_core": sum(1 for p in add if label_of(lab[tid], p) == 2),
                "added_core_not_in_openalex": sum(1 for p in add if label_of(lab[tid], p) == 2
                                                  and not (all_keys(p) & oa_keys)),
                # core papers that the reduced arm never retrieved at any rank (not just ranked lower)
                "added_core_not_in_b_pool": sum(1 for p in add if label_of(lab[tid], p) == 2
                                                and not (all_keys(p) & b_keys)),
                "added_rel_not_in_b_pool": sum(1 for p in add if (label_of(lab[tid], p) or 0) >= 1
                                               and not (all_keys(p) & b_keys)),
                "added_unlabeled": sum(1 for p in add if label_of(lab[tid], p) is None),
                "added_sel": recall(sel, add),
                "examples_core": [p["title"][:90] for p in add if label_of(lab[tid], p) == 2][:6],
            }
            extra.setdefault(name, []).append(pt[name])
        # ---- resources (Exa vs Tavily, per channel)
        rr = {}
        for src in ("exa", "tavily"):
            for ch in ("code", "data", "docs", "paper"):
                lst = next((l for l in rec["stage1"] if l["name"] == f"{src}.{ch}"), None)
                if not lst:
                    continue
                items = lst["resources"] + [{"url": p["url"], "kind": "paper", "title": p["title"]} for p in lst["papers"]]
                kinds = defaultdict(int)
                for it in items:
                    kinds[it["kind"]] += 1
                labs_ = []
                for it in lst["resources"]:
                    k = resource_key(it["url"])
                    lv = res_lab.get(tid, {}).get(k)
                    if lv is None and ch in ("code", "data", "docs"):
                        to_label_res.append({"target": tid, "key": k, "kind": it["kind"], "title": it["title"][:120],
                                             "snippet": it["snippet"][:200], "by": f"{src}.{ch}"})
                    labs_.append(lv)
                rr[f"{src}.{ch}"] = {"n": len(items), "kinds": dict(kinds),
                                     "repos": len({resource_key(i["url"]) for i in lst["resources"] if i["kind"] == "code"}),
                                     "datasets": len({resource_key(i["url"]) for i in lst["resources"] if i["kind"] == "data"}),
                                     "good": sum(1 for l in labs_ if l == 2), "ok": sum(1 for l in labs_ if l is not None and l >= 1),
                                     "unlabeled": sum(1 for l in labs_ if l is None),
                                     "paper_with_id": sum(1 for p in lst["papers"] if p.get("doi") or p.get("arxiv"))}
        ex_code = {resource_key(r["url"]) for l in rec["stage1"] if l["name"].startswith("exa.") for r in l["resources"] if r["kind"] == "code"}
        tv_code = {resource_key(r["url"]) for l in rec["stage1"] if l["name"].startswith("tavily.") for r in l["resources"] if r["kind"] == "code"}
        rr["code_overlap"] = {"exa": len(ex_code), "tavily": len(tv_code), "both": len(ex_code & tv_code)}
        good = lambda keys: {k for k in keys if res_lab.get(tid, {}).get(k) == 2}
        rr["good_code_union"] = {"exa": len(good(ex_code)), "tavily": len(good(tv_code)), "both": len(good(ex_code) & good(tv_code))}
        ex_all = {resource_key(r["url"]) for l in rec["stage1"] if l["name"].startswith("exa.") and l["name"] != "exa.paper" for r in l["resources"]}
        tv_all = {resource_key(r["url"]) for l in rec["stage1"] if l["name"].startswith("tavily.") and l["name"] != "tavily.paper" for r in l["resources"]}
        rr["good_any_union"] = {"exa": len(good(ex_all)), "tavily": len(good(tv_all)), "both": len(good(ex_all) & good(tv_all)),
                                "union": len(good(ex_all | tv_all))}
        rr["ok_any_union"] = {"exa": len({k for k in ex_all if (res_lab.get(tid, {}).get(k) or 0) >= 1}),
                              "tavily": len({k for k in tv_all if (res_lab.get(tid, {}).get(k) or 0) >= 1}),
                              "exa_n": len(ex_all), "tavily_n": len(tv_all)}
        # paper channel: labelled-relevant papers contributed by each web source in the full7 head-60
        kept7, _, _, _ = rebuild(rec, ARMS["full7"], cfg, leak)
        for src in ("exa", "tavily"):
            hp = [p for p in kept7[:HEAD] if any(r.startswith(src + ".") for r in p["ranks"])]
            only = [p for p in hp if all(r.startswith(src + ".") for r in p["ranks"])]
            rr[f"{src}_papers_head60"] = {"n": len(hp), "core": sum(1 for p in hp if label_of(lab[tid], p) == 2),
                                          "only_this_source": len(only),
                                          "only_core": sum(1 for p in only if label_of(lab[tid], p) == 2)}
        out["resources"][tid] = rr
        out["per_target"][tid] = pt
        out["latency"][tid] = {"wall_s": rec["wall_s"], **rec["pseudo"]["timings"]}

    # ---- cost from ledger (live calls only, eval runs)
    db = sqlite3.connect(Path(cfg["state_dir"]) / "ledger.sqlite")
    # all live calls of every eval10-* run (incl. the first single-target smoke run and the Europe PMC re-run
    # after its 503 outage) = what this evaluation really spent
    runs = [r[0] for r in db.execute("SELECT DISTINCT run FROM calls WHERE run LIKE 'eval10-%'")]
    q = ",".join("?" * len(runs))
    rows = db.execute(f"SELECT target, source, kind, COUNT(*), SUM(status=429), SUM(cost_usd), SUM(credits), "
                      f"AVG(latency_s) FROM calls WHERE cached=0 AND run IN ({q}) GROUP BY target, source, kind", runs).fetchall()
    cost = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "n429": 0, "usd": 0.0, "credits": 0.0}))
    lat = defaultdict(list)
    for tgt, src, kind, n, n429, usd, cr, al in rows:
        c = cost[tgt][src]
        c["calls"] += n
        c["n429"] += n429 or 0
        c["usd"] += usd or 0
        c["credits"] += cr or 0
    for src, l in db.execute(f"SELECT source, latency_s FROM calls WHERE cached=0 AND status=200 AND run IN ({q})", runs):
        lat[src].append(l)
    out["cost"] = {t: dict(v) for t, v in cost.items()}
    out["call_latency_median"] = {s: round(statistics.median(v), 2) for s, v in lat.items()}
    tot = defaultdict(lambda: {"calls": 0, "n429": 0, "usd": 0.0, "credits": 0.0})
    for t, v in cost.items():
        for s, c in v.items():
            for k in c:
                tot[s][k] += c[k]
    out["cost_total"] = dict(tot)
    out["arms"] = {a: dict(v) for a, v in agg.items()}
    out["extra_total"] = {name: {k: sum(x[k] for x in xs) for k in xs[0] if k != "examples_core"}
                          for name, xs in extra.items() if xs}
    (HERE / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # dedupe to_label
    seen = set()
    with open(HERE / "to_label.jsonl", "w") as f:
        for r in to_label:
            k = (r["target"], r["key"])
            if k not in seen:
                seen.add(k)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    seen = set()
    with open(HERE / "to_label_res.jsonl", "w") as f:
        for r in to_label_res:
            k = (r["target"], r["key"])
            if k not in seen:
                seen.add(k)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("papers to label:", sum(1 for _ in open(HERE / "to_label.jsonl")),
          "resources to label:", sum(1 for _ in open(HERE / "to_label_res.jsonl")))
    for a, v in out["arms"].items():
        print(a, {k: round(x, 2) for k, x in v.items()})
    print(json.dumps(out["extra_total"], indent=0))


if __name__ == "__main__":
    main()
