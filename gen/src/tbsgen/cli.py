"""tbsgen command line.

  tbsgen ingest  --run DIR --candidates FILE [--config C]      S1: create item dirs from retrieval output
  tbsgen run     --run DIR [--config C] [--max-steps N] [--only ID,..]
  tbsgen resume  --run DIR                                       re-activate budget/model-paused items
  tbsgen status  --run DIR
  tbsgen report  --run DIR [--day YYYY-MM-DD]
  tbsgen ledger  --run DIR                                       spend summary (all days)
  tbsgen breaker --run DIR --model M --reset
  tbsgen calib-ingest --run DIR --item ID --job-dir JOBDIR      attach Harbor results to the pending S9 run
  tbsgen ping    --run DIR --calls N                             real Opus connectivity check (needs budget set)
Real model calls need budget.daily_usd_total in the config (or --budget-usd on ping).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import calib
from .budget import BudgetStop
from .config import load_config
from .models import ModelError, OpusClient
from .pipeline import Pipeline
from .report import daily_report
from .state import read_json, write_json


def _pipeline(a, overrides=None) -> Pipeline:
    ov = {"run": {"root": a.run}}
    if overrides:
        ov.update(overrides)
    return Pipeline(load_config(a.config, ov))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tbsgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "run", "resume", "status", "report", "ledger", "breaker", "calib-ingest", "ping"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--config", default=None)
        if name == "ingest":
            s.add_argument("--candidates", required=True)
            s.add_argument("--top", type=int, default=0)
        if name == "run":
            s.add_argument("--max-steps", type=int, default=10000)
            s.add_argument("--only", default="")
        if name == "report":
            s.add_argument("--day", default=None)
        if name == "breaker":
            s.add_argument("--model", required=True)
            s.add_argument("--reset", action="store_true")
        if name == "calib-ingest":
            s.add_argument("--item", required=True)
            s.add_argument("--job-dir", required=True)
        if name == "ping":
            s.add_argument("--calls", type=int, default=1)
            s.add_argument("--budget-usd", type=float, required=True)
            s.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args(argv)
    if a.cmd == "ping":
        return _ping(a)
    P = _pipeline(a)
    if a.cmd == "ingest":
        from .intake import load_candidates
        cs = load_candidates(__import__("pathlib").Path(a.candidates))
        if a.top:
            cs = cs[: a.top]
        print(json.dumps(dict(ingested=len(P.ingest(cs))), ensure_ascii=False))
    elif a.cmd == "run":
        print(json.dumps(P.run(a.max_steps, [x for x in a.only.split(",") if x] or None), indent=1, ensure_ascii=False))
    elif a.cmd == "resume":
        print(json.dumps(dict(resumed=P.resume()), ensure_ascii=False))
    elif a.cmd == "status":
        print(json.dumps(P.status(), indent=1, ensure_ascii=False))
    elif a.cmd == "report":
        print(daily_report(P, a.day))
    elif a.cmd == "ledger":
        print(json.dumps(P.ledger.summary(None), indent=1))
    elif a.cmd == "breaker":
        if a.reset:
            P.ledger.reset_breaker(a.model)
        print(json.dumps(P.ledger.breaker(a.model)))
    elif a.cmd == "calib-ingest":
        it = P.store.item(a.item)
        runs = read_json(it.p("calib", "runs.json"), [])
        pend = [r for r in runs if r.get("outcome") is None]
        if not pend:
            sys.exit("no pending calibration run for this item")
        trials = calib.collect(__import__("pathlib").Path(a.job_dir))
        if not trials:
            sys.exit("no trial results in job dir")
        t = trials[0]
        pend[-1].update(outcome=t["outcome"], minutes=t["minutes"], trial=t)
        write_json(it.p("calib", "runs.json"), runs)
        it.state["stage"], it.state["status"] = "S9", "active"
        it.save()
        print(json.dumps(t, ensure_ascii=False))
    return 0


def _ping(a):
    """Minimal real Opus call(s) through the new client + ledger.  Budget must be given explicitly."""
    cfg = load_config(a.config, {"run": {"root": a.run},
                                 "budget": {"daily_usd_total": a.budget_usd,
                                            "daily_calls_per_model": {"claude-opus-5-5": a.calls},
                                            "est_usd_per_call": 0.05}})
    P = Pipeline(cfg)
    c = OpusClient(P.ledger, cfg)
    out = []
    for i in range(a.calls):
        try:
            mt = a.max_tokens if i == 0 else max(a.max_tokens, 9000)   # 2nd call exercises the SSE (stream) path
            r = c.chat("Reply with exactly one word.", "Say: pong", stage="PING", item="ping", max_tokens=mt)
            out.append(dict(ok=True, text=r.text.strip()[:40], model=r.model, usage=r.usage, usd=round(r.usd, 6),
                            latency_s=r.latency_s))
        except (ModelError, BudgetStop) as e:
            out.append(dict(ok=False, error=str(e)[:300]))
    print(json.dumps(dict(results=out, ledger=P.ledger.summary(None)), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
