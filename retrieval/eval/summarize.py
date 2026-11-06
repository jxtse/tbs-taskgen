"""Print the numbers used in REPORT.md from metrics.json, coverage_old_core.json and raw/."""
import json, glob, statistics
from collections import defaultdict
m = json.load(open("metrics.json"))
A = m["arms"]
print("ARMS (sum over 10 targets)")
for a in ("full7", "new4", "full-s2", "new4-s2", "old2", "oa"):
    v = A[a]
    print(f"{a:8s} top20 rel {int(v['top20_rel'])}/200 core {int(v['top20_core'])}/200 | sel: head60 {int(v['sel_in_head60'])}/35 top20 {int(v['sel_in_top20'])} pool {int(v['sel_in_pool'])} | "
          f"pool {int(v['pool'])} kept {int(v['kept'])} | OA-link head60 {v['oa_head60']/10:.0%} | seeded heldout {v.get('seeded_heldout_in_head60')} pseudo {v.get('pseudo_heldout_in_head60')} /{v.get('seeded_heldout_n')}")
print(json.dumps(m["extra_total"], indent=0))
cov = json.load(open("coverage_old_core.json"))
print("old core coverage", json.dumps(cov["total"]))
# S2 openAccessPdf coverage + per-source stats from raw lists
st = defaultdict(lambda: defaultdict(int))
drops = defaultdict(int)
for f in glob.glob("raw/*.json"):
    r = json.load(open(f))
    for l in r["stage1"] + r["stage2_pseudo"] + r["stage2_seeded"]:
        s = l["name"]
        for p in l["papers"]:
            st[s]["n"] += 1
            st[s]["oa"] += bool(p["oa_url"])
            st[s]["abs"] += bool(p["abstract"])
            st[s]["doi"] += bool(p["doi"] or p["arxiv"])
            st[s]["code"] += bool(p["code_links"])
    for k, v in r["pseudo"]["dropped"].items():
        drops[k] += v
    for p in r["enriched"]:
        if p["origin"] == "s2":
            st["s2.batch"]["n"] += 1; st["s2.batch"]["oa"] += bool(p["oa_url"])
for s, v in sorted(st.items()):
    print(f"{s:20s} n={v['n']:5d} oa={v['oa']/v['n']:.0%} abstract={v['abs']/v['n']:.0%} id={v['doi']/v['n']:.0%} codelink={v['code']/v['n']:.1%}")
print("prefilter drops (pseudo, 10 targets):", dict(drops))
print("cost_total", json.dumps(m["cost_total"]))
print("median call latency", m["call_latency_median"])
lat = [v["total"] for v in m["latency"].values()]
print("retrieve() total s: median", statistics.median(lat), "max", max(lat))
for k in ("stage1_wall", "stage2", "enrich"):
    print(k, statistics.median(v[k] for v in m["latency"].values()))
