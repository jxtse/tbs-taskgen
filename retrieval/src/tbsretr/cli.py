"""CLI.

  uv run tbsretr status                       # which sources are usable (no key values printed)
  uv run tbsretr run --targets T.json --out out/ [--only id1,id2] [--sources openalex,exa] [--seed-mode auto]
  uv run tbsretr usage [--run RUN]                    # ledger summary (calls, cache hits, USD, credits, 429s)

T.json: list of Target dicts {id, queries[], description?, field?, name?, libs?, seed_dois?, exclude_dois?, ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .ledger import Ledger, utc_day
from .model import Target
from .pipeline import Retriever, to_json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tbsretr")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--targets", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--only", default="")
    r.add_argument("--sources", default="")
    r.add_argument("--seed-mode", default="auto", choices=["auto", "pseudo", "given", "none"])
    r.add_argument("--offline", action="store_true", help="cache only; never hit the network")
    u = sub.add_parser("usage")
    u.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "status":
        rt = Retriever(cfg, run="status")
        st = rt.status()
        for n, v in st.items():
            used = rt.ledger.usage(n)
            print(f"{n:10s} {'ok' if v['available'] else 'SKIP':5s} {v['note']:45s} today(UTC {utc_day()}): "
                  f"{used['calls']} calls, ${used['usd']:.4f}, {used['credits']:.0f} credits")
        return 0
    if a.cmd == "usage":
        for row in Ledger(cfg["state_dir"]).summary(a.run):
            print(json.dumps(row))
        return 0

    targets = [Target.from_dict(d) for d in json.loads(Path(a.targets).read_text())]
    if a.only:
        keep = set(a.only.split(","))
        targets = [t for t in targets if t.id in keep]
    srcs = set(a.sources.split(",")) if a.sources else None
    rt = Retriever(cfg, sources=srcs, offline=a.offline)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in targets:
        res = to_json(rt.retrieve(t, seed_mode=a.seed_mode))
        fn = out / (t.id.replace(":", "__").replace("/", "_") + ".json")
        fn.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"{t.id}: {len(res['papers'])} papers, {len(res['resources'])} resources, "
              f"{res['timings']['total']}s, sources={ {k: v['status'] for k, v in res['sources'].items()} }",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
