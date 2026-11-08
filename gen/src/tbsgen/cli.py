"""tbsgen command line.

  tbsgen ingest  --run DIR --candidates FILE [--config C]      S1: create item dirs from retrieval output
  tbsgen run     --run DIR [--config C] [--max-steps N] [--only ID,..]
  tbsgen resume  --run DIR                                       re-activate budget/model-paused items
  tbsgen status  --run DIR
  tbsgen report  --run DIR [--day YYYY-MM-DD]
  tbsgen ledger  --run DIR                                       spend summary (all days)
  tbsgen breaker --run DIR --model M --reset
  tbsgen calib-ingest --run DIR --item ID --job-dir JOBDIR      attach Harbor results to the pending S9 run
  tbsgen ping    --run DIR --calls N                             real teacher-model connectivity check (needs budget)
  tbsgen harbor-local --run DIR [--item ID]                       local Harbor oracle+nop on packaged items
  tbsgen reset   --run DIR --item ID --stage S5 [--regen]        re-activate an item at a stage (after a pipeline fix);
                                                                 --regen also forces S6A/S6C to regenerate files
  tbsgen fetch   --run DIR [--max N] [--channels a,b] [--no-browser] [--dry-run] [--status] [--finish-browser]
                 [--only IDENT[,IDENT]]                          drain the S3 full-text queue (incl. the ego-browser
                                                                 channel; HTTP channels on fulltext.remote.host when
                                                                 remote mode is on); fetched items go back to S3.
                                                                 --only: just these queue entries (DOI, arXiv id or
                                                                 ident), due or not.  Exit 3: the remote host refused
                                                                 the job (stats.remote_error; papers stay due)
  tbsgen fetch   --run DIR --browser [--wait-for-egress] [--max-wait S]
                                                                 one egress-gated, time-boxed browser batch (script,
                                                                 then a Hermes CLI agent); --wait-for-egress polls the
                                                                 browser egress and starts when it is allowed
  tbsgen remote deploy --run DIR [--dry-run] [--force]          ship the local HEAD to fulltext.remote.code_dir (git
                                                                 bundle + scp, fetch + checkout --detach, uv sync) and
                                                                 check both code fingerprints
  tbsgen remote status --run DIR                                 both sides' HEAD / fingerprint, remote jobs
  tbsgen fulltext purge --run DIR --channel elsevier,wiley [--before YYYY-MM-DD] [--dry-run]
                                                                 delete full texts (+ raw PDF/XML, + item copies in
                                                                 this run) obtained via these channels; ledger rows
                                                                 and provenance records are kept (TDM terms)
Real model calls need budget.daily_usd_total in the config (or --budget-usd on ping).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import calib
from .budget import BudgetStop
from .config import load_config
from .models import ModelError, TeacherClient
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
    for name in ("ingest", "run", "resume", "status", "report", "ledger", "breaker", "calib-ingest", "ping", "reset",
                 "harbor-local", "fetch", "remote", "fulltext"):
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
            s.add_argument("--job-dir", default=None)   # default: the pending run's own job dir
        if name == "harbor-local":
            s.add_argument("--item", default=None)
        if name == "reset":
            s.add_argument("--item", required=True)
            s.add_argument("--stage", required=True)
            s.add_argument("--regen", action="store_true")
        if name == "remote":
            s.add_argument("action", choices=["deploy", "status"])
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--force", action="store_true", help="deploy even if a remote job is not collected")
        if name == "fulltext":
            s.add_argument("action", choices=["purge"])
            s.add_argument("--channel", required=True, help="comma-separated, e.g. elsevier,wiley")
            s.add_argument("--before", default=None, help="only texts fetched before this day (YYYY-MM-DD)")
            s.add_argument("--dry-run", action="store_true")
        if name == "fetch":
            s.add_argument("--only", default="", help="IDENT[,IDENT]: DOI, arXiv id or queue ident")
            s.add_argument("--max", type=int, default=10)
            s.add_argument("--channels", default="")
            s.add_argument("--no-browser", action="store_true")
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--status", action="store_true")
            s.add_argument("--finish-browser", action="store_true")
            s.add_argument("--browser", action="store_true", help="run one egress-gated browser batch")
            s.add_argument("--wait-for-egress", action="store_true")
            s.add_argument("--max-wait", type=float, default=None)
        if name == "ping":
            s.add_argument("--calls", type=int, default=1)
            s.add_argument("--budget-usd", type=float, required=True)
            s.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args(argv)
    if a.cmd == "ping":
        return _ping(a)
    P = _pipeline(a)
    if a.cmd == "fetch":
        return _fetch(P, a)
    if a.cmd == "remote":
        return _remote(P, a)
    if a.cmd == "fulltext":
        from . import fulltext
        out = fulltext.purge(P.cfg, [x.strip() for x in a.channel.split(",") if x.strip()], before=a.before,
                             dry_run=a.dry_run, runs=[Path(a.run)])
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 2 if out.get("error") else 0
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
    elif a.cmd == "harbor-local":
        from . import harbor_local
        ids = [a.item] if a.item else [i for i in sorted(p.name for p in (Path(a.run) / "items").iterdir())
                                       if (Path(a.run) / "items" / i / "pkg" / "task.toml").exists()]
        for i in ids:
            print(json.dumps(harbor_local.check(P.cfg, Path(a.run), P.store.item(i).dir)))
    elif a.cmd == "reset":
        it = P.store.item(a.item)
        it.state.setdefault("history", []).append(dict(stage=it.state["stage"], result="manual_reset", to=a.stage,
                                                      prev_status=it.state["status"], prev_reason=it.state.get("reason"),
                                                      ts=__import__("time").time()))
        it.state.update(stage=a.stage, status="active")
        it.state.pop("reason", None)
        if a.regen:
            it.state["regen_6a"] = it.state["regen_6c"] = True
        it.save()
        P.store.event("manual_reset", item=a.item, stage=a.stage, regen=a.regen)
        print(json.dumps(dict(item=a.item, stage=a.stage, regen=a.regen)))
    elif a.cmd == "calib-ingest":
        from .stages_b import ingest
        try:
            t = ingest(P, P.store.item(a.item), a.job_dir)
        except ValueError as e:
            sys.exit(str(e))
        print(json.dumps(t, ensure_ascii=False))
    return 0


def _fetch(P: Pipeline, a) -> int:
    from . import browser_ego, fulltext
    from .state import Store
    f = P.fulltext_fetcher()
    if a.status:
        print(json.dumps(dict(ledger=str(fulltext.ledger_path(P.cfg)), usage=f.ledger.usage(),
                              queue=len(fulltext.read_queue(fulltext.queue_path(P.cfg)))), indent=1))
        return 0
    if a.finish_browser:
        print(json.dumps(browser_ego.finish(P.cfg["fulltext"]["browser"])))
        return 0

    def reactivate(entry, out, sub) -> bool:
        """Deliver the fetched text to one waiting item and move it on to S4 (the same transition S3 makes), so
        items of other runs (other caches) get the text too.  True when the item no longer waits for it."""
        run, iid = sub
        run = Path(run or P.store.root)
        if not (run / "items" / iid / "state.json").exists():
            return True                                    # item gone: nothing to wait for
        it = Store(run).item(iid)
        st = it.state
        if st.get("stage") == "S3" and st.get("status") == "parked" and st.get("reason") == "fulltext_unavailable":
            write_json(it.p("S3_fulltext.json"), dict(out.rec, delivered_by="tbsgen fetch"))
            it.p("fulltext.txt").write_text(out.text)
            it.advance("S3", "S4", source=out.rec.get("source"), chars=len(out.text), via="tbsgen fetch")
        return True                                        # delivered now, or already past S3 / not waiting

    readded = fulltext.reconcile(P.store, fulltext.queue_path(P.cfg)) if not a.dry_run else 0
    if getattr(a, "browser", False) and (getattr(a, "only", "") or "").strip():
        print("tbsgen fetch: --only is not supported with --browser (a browser batch plans its own papers)",
              file=sys.stderr)
        return 2
    if getattr(a, "browser", False):
        from . import browser_agent
        stats = browser_agent.run_batch(f, wait_for_egress=a.wait_for_egress, max_wait_s=a.max_wait, dry_run=a.dry_run,
                                        on_success=reactivate, log=lambda s: print(s, file=sys.stderr, flush=True))
        stats["requeued_from_items"] = readded
        print(json.dumps(stats, indent=1, ensure_ascii=False, default=str))
        return 2 if stats.get("refused") else 0
    only = [x.strip() for x in (getattr(a, "only", "") or "").split(",") if x.strip()] or None
    stats = fulltext.drain_queue(f, max_papers=0 if only else a.max,
                                 channels=[x for x in a.channels.split(",") if x] or None,
                                 browser=not a.no_browser, dry_run=a.dry_run, on_success=reactivate,
                                 log=lambda s: print(s, file=sys.stderr), only=only)
    stats["requeued_from_items"] = readded
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    if stats.get("remote_error"):
        print(f"tbsgen fetch: remote job refused: {stats['remote_error']}", file=sys.stderr)
        return 3
    if only and stats.get("only_unmatched"):
        print(f"tbsgen fetch: not in the queue: {', '.join(stats['only_unmatched'])}", file=sys.stderr)
        return 4 if not stats.get("considered") else 0
    return 0


def _remote(P: Pipeline, a) -> int:
    from . import remote
    f = P.fulltext_fetcher()
    if a.action == "deploy":
        out = remote.deploy(f, dry_run=a.dry_run, force=a.force, log=lambda s: print(s, file=sys.stderr))
    else:
        out = remote.status(f)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    if a.action == "status":
        return 0 if out.get("same_code") else 2
    return 0 if out.get("ok") else 2


def _ping(a):
    """Minimal real teacher-model call(s) through the client + ledger.  Budget must be given explicitly."""
    model = load_config(a.config)["models"]["teacher"]["model"]
    cfg = load_config(a.config, {"run": {"root": a.run},
                                 "budget": {"daily_usd_total": a.budget_usd,
                                            "daily_calls_per_model": {model: a.calls},
                                            "est_usd_per_call": 0.05}})
    P = Pipeline(cfg)
    c = TeacherClient(P.ledger, cfg)
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
