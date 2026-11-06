"""Quick look at one eval/raw record: python eval/peek.py <raw.json>"""
import json
import sys

r = json.load(open(sys.argv[1]))
for k in ("pseudo", "seeded"):
    o = r[k]
    if not o:
        continue
    print("==", k, o["seed_origin"], o["seeds"], o["dropped"], o["enrich"], o["timings"])
    for p in o["papers"][: int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
        print(f"{p['fused_rank']:3d} {p['rrf']:.4f} {','.join(p['sources']):35s} "
              f"{(p['doi'] or p['arxiv'] or '')[:30]:30s} {p['title'][:80]}")
print("selected", [s["doi"] for s in r["selected"]], "seeds", r["seed_dois"])
print("-- resources")
for x in r["pseudo"]["resources"][:25]:
    print(x["kind"], x["key"][:60], list(x["found_by"]), len(x["from_papers"]))
for lst in r["stage2_pseudo"] + r["stage2_seeded"]:
    print(lst["name"], lst["query"][:60], len(lst["papers"]))
