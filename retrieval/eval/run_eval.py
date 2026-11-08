"""Live 10-target evaluation run (network; needs API keys in env). Saves everything needed for offline analysis:

  eval/raw/<target>.json   : every ranked list (stage 1 + stage 2 pseudo + stage 2 seeded), enrichment records,
                             per-source status/errors/timings, pseudo/seeded outputs.
  uv run python eval/run_eval.py --selected <earlier pipeline>/literature/methodlit/selected.jsonl [--only id,...]

Historical: the recorded run (eval/raw, REPORT.md) included the Tavily source, which has since been removed from
the code; a rerun now covers the remaining sources only.

Hard call caps for this evaluation (decision 2026-11-05) are enforced through the ledger: daily_max_calls for each
source = (calls already booked today) + cap - (calls this eval already made), so a source stops itself when its
cap is reached. 429 retries count as calls.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from tbsretr.config import load_config
from tbsretr.ledger import Ledger
from tbsretr.model import Target
from tbsretr.pipeline import Retriever, to_json

HERE = Path(__file__).resolve().parent
CAPS = {"openalex": 300, "exa": 80, "crossref": 120, "arxiv": 60, "europepmc": 60, "s2": 150}


def selected_by_target(sel: Path):
    out = {}
    for line in sel.read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["target"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r.get("selected_at") or "")
    return out


def seeds_for(sel):
    """2 seeds if >=4 selected, else 1 (keeps >=2 held out where possible). DOIs only (arXiv-only skipped)."""
    k = 2 if len(sel) >= 4 else 1
    return [r["doi"] for r in sel if not r["doi"].startswith("arxiv:")][:k]


def dump_lists(lists):
    return [{"name": rl.name, "query": rl.query, "papers": [asdict(p) for p in rl.papers],
             "resources": [asdict(r) for r in rl.resources]} for rl in lists]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--selected", required=True, help="selected.jsonl of the earlier pipeline (not in this repo)")
    a = ap.parse_args()
    targets = [Target.from_dict(d) for d in json.loads((HERE / "targets10.json").read_text())]
    if a.only:
        targets = [t for t in targets if t.id in set(a.only.split(","))]
    sel = selected_by_target(Path(a.selected))
    cfg = load_config()
    led = Ledger(cfg["state_dir"])
    run = "eval10-" + time.strftime("%Y%m%dT%H%M%S")
    already = {n: led.usage(n)["calls"] for n in CAPS}
    prior_eval = {}
    import sqlite3
    db = sqlite3.connect(led.path)
    for n in CAPS:
        prior_eval[n] = db.execute("SELECT COUNT(*) FROM calls WHERE run LIKE 'eval10-%' AND source=? AND cached=0",
                                   (n,)).fetchone()[0]
    db.close()
    for n, cap in CAPS.items():
        cfg["sources"][n]["daily_max_calls"] = already[n] + max(0, cap - prior_eval[n])
    cfg["sources"]["exa"]["daily_budget_usd"] = 2.0
    print("caps this run:", {n: cfg["sources"][n]["daily_max_calls"] - already[n] for n in CAPS})
    (HERE / "raw").mkdir(exist_ok=True)
    rt = Retriever(cfg, run=run)
    for t in targets:
        t0 = time.time()
        out = rt.retrieve(t, seed_mode="pseudo")
        lists1 = [rl for rl in out["_lists"] if rl.name not in ("openalex.snowball", "s2.recommend")]
        stage2_pseudo = [rl for rl in out["_lists"] if rl.name in ("openalex.snowball", "s2.recommend")]
        seed_dois = seeds_for(sel.get(t.id, []))
        seeded = None
        if seed_dois:
            t2 = Target.from_dict({**t.__dict__, "seed_dois": seed_dois})
            seeded = rt.retrieve(t2, seed_mode="given", stage1_lists=lists1)
        rec = {
            "target": asdict(t), "run": run, "selected": sel.get(t.id, []), "seed_dois": seed_dois,
            "stage1": dump_lists(lists1), "stage2_pseudo": dump_lists(stage2_pseudo),
            "stage2_seeded": dump_lists([rl for rl in (seeded or {}).get("_lists", [])
                                         if rl.name in ("openalex.snowball", "s2.recommend")]),
            "enriched": [asdict(p) for p in out["_enriched"] + ((seeded or {}).get("_enriched") or [])],
            "pseudo": to_json(out), "seeded": to_json(seeded) if seeded else None,
            "wall_s": round(time.time() - t0, 1),
        }
        fn = HERE / "raw" / (t.id.replace(":", "__") + ".json")
        fn.write_text(json.dumps(rec, ensure_ascii=False))
        print(f"{t.id[:50]:50s} {rec['wall_s']:6.1f}s pool={out['n_pool']} top={len(out['papers'])} "
              f"res={len(out['resources'])} status={ {k: v['status'][:30] for k, v in out['sources'].items() if v['status'] != 'ok'} }",
              flush=True)
    for row in led.summary(run):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
