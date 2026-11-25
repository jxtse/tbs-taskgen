"""tbsgen command line.

  tbsgen ingest  --run DIR --candidates FILE [--config C]      S1: create item dirs from retrieval output
  tbsgen run     --run DIR [--config C] [--max-steps N] [--only ID,..] [--until STAGE] [--stages LIST] [--loop SECONDS]
                                                                 --until S4: stop after that stage (no model stage is
                                                                 started, so no model:config pause without a teacher)
  tbsgen resume  --run DIR [--only ID,..] [--stage S9,..] [--reason budget:,model:]
                                                                 re-activate paused items (default reasons budget:/model:);
                                                                 --only / --stage / --reason narrow which ones
  tbsgen status  --run DIR
  tbsgen report  --run DIR [--day YYYY-MM-DD | --throughput [--days N]]
  tbsgen ledger  --run DIR                                       spend summary (all days)
  tbsgen breaker --run DIR --model M --reset
  tbsgen calib-ingest --run DIR --item ID --job-dir JOBDIR      attach Harbor results to the pending S9 run (or the
                                                                 pending S9R review run of an item at S9R)
  tbsgen calib resolve-remote --run DIR --item ID --job NAME
                             (--synced | --ended-without-result | --not-started) [--remote-task PATH] [--force-local-gone]
  tbsgen calib watch --run DIR [--only ID,..] [--once] [--interval S] [--max-hours H]
                                                                 poll submitted S9 / S9R jobs: ingest finished ones,
                                                                 step S9 (label, or launch the next run GLM ->
                                                                 teacher) and S9R (launch the review, route verdict)
  tbsgen review build --run DIR --item ID [--round K] [--out DIR] build the S9R review task of an item (no launch;
                                                                 for replays / manual runs)
  tbsgen calib prepare --run DIR --item ID [--who glm|teacher]   run calib.prepare_command for one item (retries a
                                                                 failed preparation)
  tbsgen ping    --run DIR --calls N                             real teacher-model connectivity check (needs budget)
  tbsgen harbor-local --run DIR [--item ID]                       local Harbor oracle+nop on packaged items
  tbsgen harden  --run DIR --item ID --reason TEXT                             code-grounded item whose student passed:
                                                                 start a hardening round (back to S5G; difficulty.py)
  tbsgen regate  --run DIR --item ID --reason TEXT [--recalibrate]             a released / reviewed item after a harness change
                                                                 (tests/_metrics.py, test.sh): withdraw it, re-measure
                                                                 and re-gate with locked thresholds, repackage, S9R
  tbsgen reset   --run DIR --item ID --stage S5 [--regen]        re-activate an item at a stage (after a pipeline fix);
                                                                 --regen also forces S6A/S6C to regenerate files
  tbsgen fetch   --run DIR [--max N] [--channels a,b] [--no-browser] [--dry-run] [--status] [--finish-browser] [--loop SECONDS]
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
import math
import signal
import sys
import threading
import time
from pathlib import Path

from . import calib
from .budget import BudgetStop
from .config import load_config
from .models import ModelError, TeacherClient
from .pipeline import Pipeline, parse_stages
from .report import daily_report, throughput
from .state import STAGES, read_json, write_json


def _pipeline(a, overrides=None) -> Pipeline:
    ov = {"run": {"root": a.run}}
    if overrides:
        ov.update(overrides)
    return Pipeline(load_config(a.config, ov), init_deadline=getattr(a, "fetch_deadline", None))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tbsgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "run", "resume", "status", "report", "ledger", "diversity", "breaker", "calib-ingest", "ping", "reset",
                 "harbor-local", "fetch", "remote", "fulltext", "calib", "taskimage", "requeue-image", "review", "regate", "harden", "variant"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--config", default=None)
        if name == "diversity":
            s.add_argument("--runs-root", type=Path)
            s.add_argument("--json", action="store_true")
        if name == "ingest":
            s.add_argument("--candidates", required=True)
            s.add_argument("--top", type=int, default=0)
        if name in ("run", "fetch"):
            s.add_argument("--loop", type=float, metavar="SECONDS")
        if name == "run":
            s.add_argument("--stages", type=parse_stages)
            s.add_argument("--max-steps", type=int, default=10000)
            s.add_argument("--only", default="")
            s.add_argument("--until", default=None, choices=STAGES,
                           help="last stage to run (e.g. S4): later stages are not started, nothing is paused")
        if name == "resume":
            s.add_argument("--only", default="", help="ID[,ID]: just these items")
            s.add_argument("--stage", default="", help="STAGE[,STAGE]: just items paused at these stages")
            s.add_argument("--reason", default="budget:,model:,parked:probe_unreviewed", help="reason prefixes (comma-separated)")
        if name == "taskimage":
            s.add_argument("action", choices=["gc"])
            s.add_argument("--dry-run", action="store_true")
        if name == "requeue-image":
            s.add_argument("--item", required=True)
        if name == "calib":
            s.add_argument("action", choices=["watch", "prepare", "launch", "resolve-remote"])
            s.add_argument("--only", default="", help="watch: ID[,ID]")
            s.add_argument("--once", action="store_true", help="watch: one pass, then exit")
            s.add_argument("--interval", type=float, default=None, help="watch: seconds (calib.watch_interval_s)")
            s.add_argument("--max-hours", type=float, default=None, help="watch: stop after this long")
            s.add_argument("--item", default=None, help="prepare / resolve-remote: item id")
            s.add_argument("--job", help="resolve-remote: remote job name")
            s.add_argument("--attempt", type=int, help="launch: current attempt number")
            resolution = s.add_mutually_exclusive_group()
            resolution.add_argument("--synced", action="store_true")
            resolution.add_argument("--ended-without-result", "--died", action="store_true",
                                    help="confirm the remote job ended and is fully synced; ingest any complete trials")
            s.add_argument("--remote-task", metavar="PATH", help="remote task path when the launcher marker was lost")
            resolution.add_argument("--not-started", action="store_true")
            s.add_argument("--force-local-gone", action="store_true",
                           help="confirm the local launcher is gone when tmux cannot determine its status")
            s.add_argument("--who", default="glm", choices=["glm", "teacher"])
        if name == "review":
            s.add_argument("action", choices=["build", "summary", "deliverable"])
            s.add_argument("--item", default=None, help="build / deliverable: item id")
            s.add_argument("--job", default=None, help="deliverable: calibration job name")
            s.add_argument("--src", default=None, help="deliverable: directory holding the run's /app/output")
            s.add_argument("--note", default="", help="deliverable: provenance note (how it was recovered)")
            s.add_argument("--sample", type=int, default=3, help="summary: pass items to sample for spot checks")
            s.add_argument("--seed", type=int, default=None, help="summary: sampling seed (default: random)")
            s.add_argument("--round", type=int, default=None)
            s.add_argument("--out", default=None)
        if name == "report":
            s.add_argument("--day", default=None)
            s.add_argument("--throughput", action="store_true")
            s.add_argument("--days", type=int)
        if name == "breaker":
            s.add_argument("--model", required=True)
            s.add_argument("--reset", action="store_true")
        if name == "regate":
            s.add_argument("--item", required=True)
            s.add_argument("--reason", required=True, help="why (e.g. 'grader template v2.5')")
            s.add_argument("--recalibrate", action="store_true", help="archive the calibration epoch, S9 from GLM")
        if name == "harden":
            s.add_argument("--item", required=True)
            s.add_argument("--reason", required=True, help="why (operator note recorded in the round)")
        if name == "variant":
            s.add_argument("--item", default=None, help="ID[,ID]; omit to re-open every item ended label:easy")
            s.add_argument("--reason", default="S9b re-open", help="operator note recorded in the round")
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
            s.add_argument("--max", type=int, default=None,
                           help="papers per run (default: fulltext.remote.batch_max in remote mode, else 10)")
            s.add_argument("--channels", default="")
            s.add_argument("--no-browser", action="store_true")
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--status", action="store_true")
            s.add_argument("--release-stale", metavar="JOB_ID", help="release a verified stale remote fetch job")
            s.add_argument("--legacy-worker-stopped", action="store_true",
                           help="confirm no legacy fetch worker process is running on the original host")
            s.add_argument("--finish-browser", action="store_true")
            s.add_argument("--browser", action="store_true", help="run one egress-gated browser batch")
            s.add_argument("--wait-for-egress", action="store_true")
            s.add_argument("--max-wait", type=float, default=None)
        if name == "ping":
            s.add_argument("--calls", type=int, default=1)
            s.add_argument("--budget-usd", type=float, required=True)
            s.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args(argv)
    if getattr(a, "loop", None) is not None and (not math.isfinite(a.loop) or a.loop <= 0):
        ap.error("--loop requires a positive interval")
    if a.cmd == "report" and a.days is not None and (not a.throughput or a.days <= 0):
        ap.error("--days requires --throughput and a positive day count")
    if a.cmd == "fetch" and a.max_wait is not None and (not math.isfinite(a.max_wait) or a.max_wait < 0):
        ap.error("--max-wait requires a finite non-negative interval")
    if a.cmd == "ping":
        return _ping(a)
    if a.cmd == "diversity":
        from . import diversity
        result = diversity.report(Path(a.run), a.runs_root)
        print(json.dumps(result, indent=2) if a.json else diversity.format_report(result))
        return 0
    if a.cmd == "taskimage":
        from . import taskimage
        cfg = load_config(a.config or Path(a.run) / "config.toml", {"run": {"root": a.run}})
        print(json.dumps(taskimage.gc(cfg, dry_run=a.dry_run), indent=2))
        return 0
    if a.cmd == "fetch":
        return _fetch_command(a, ap)
    P = _pipeline(a)
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
        def run_pass(stopping):
            P.stopped_models.clear()
            result = P.run(a.max_steps, _ids(a.only), until=a.until, stages=a.stages, stopping=stopping,
                           poll_calib=a.loop is not None or a.stages is not None)
            print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
            return 0, result["progress"]
        if a.loop is not None:
            return _worker_loop(P.store.root, a.loop, run_pass)
        run_pass(None)
    elif a.cmd == "requeue-image":
        if not P.requeue_image(a.item):
            ap.error(f"{a.item}: requeue requires an S6I infrastructure park or a matching legacy drop and receipt")
        print(json.dumps(dict(requeued=a.item)))
    elif a.cmd == "resume":
        reasons = tuple(x.strip() for x in a.reason.split(",") if x.strip())
        if not reasons:
            sys.exit("tbsgen resume: --reason needs at least one prefix")
        print(json.dumps(dict(resumed=P.resume(reasons, only=_ids(a.only), stages=_ids(a.stage))),
                         ensure_ascii=False))
    elif a.cmd == "calib":
        return _calib(P, a)
    elif a.cmd == "review":
        from . import review
        if a.action == "summary":
            seed = a.seed if a.seed is not None else int(time.time())
            print(json.dumps(dict(review.summary(P, a.sample, seed), seed=seed), indent=1, ensure_ascii=False,
                             default=str))
            return 0
        if not a.item:
            sys.exit(f"tbsgen review {a.action}: --item is required")
        if a.action == "deliverable":
            it = P.store.item(a.item)
            with it.operator_lock():
                jobs = {x.get("job_name") for x in read_json(it.p("calib", "runs.json"), []) or []}
                if not a.job or not a.src or a.job not in jobs or not Path(a.src).is_dir():
                    sys.exit("tbsgen review deliverable: --job must name a run of the current calibration and --src "
                             "an existing directory")
                d = review.save_deliverable(it.dir, a.job, Path(a.src), a.note)
            P.store.event("review_deliverable_saved", item=a.item, job=a.job, note=a.note[:200], operator=True)
            print(json.dumps(dict(item=a.item, job=a.job, saved=str(d))))
            return 0
        it = P.store.item(a.item)
        with it.operator_lock():
            rnd = a.round or int(it.state.get("review_round") or 1)
            out = review.build_task(P.cfg, it.dir, it.state, rnd, Path(a.out) if a.out else None)
        print(json.dumps(dict(item=a.item, round=rnd, task=str(out))))
    elif a.cmd == "status":
        print(json.dumps(P.status(), indent=1, ensure_ascii=False))
    elif a.cmd == "report":
        print(json.dumps(throughput(P, a.days), indent=1) if a.throughput else daily_report(P, a.day))
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
        with it.operator_lock():
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
    elif a.cmd == "regate":
        from .stages_b import regate
        try:
            out = regate(P, P.store.item(a.item), a.reason, recalibrate=a.recalibrate)
        except ValueError as e:
            sys.exit(str(e))
        print(json.dumps(out, ensure_ascii=False))
    elif a.cmd == "harden":
        from .stages_b import harden_cmd
        try:
            out = harden_cmd(P, P.store.item(a.item), a.reason)
        except ValueError as e:
            sys.exit(str(e))
        print(json.dumps(out, ensure_ascii=False))
    elif a.cmd == "variant":
        from .stages_b import variant_cmd
        ids = _ids(a.item) or [i.item_id for i in P.store.items()
                               if i.state.get("status") == "done" and
                               str(i.state.get("reason") or "").startswith("label:easy")]
        out, failed = [], []
        for iid in ids:
            try:
                out.append(variant_cmd(P, P.store.item(iid), a.reason))
            except ValueError as e:                 # report every item, never stop the batch
                failed.append(dict(item=iid, error=str(e)))
        print(json.dumps(dict(reopened=out, failed=failed), ensure_ascii=False))
    elif a.cmd == "calib-ingest":
        from .stages_b import ingest
        try:
            t = ingest(P, P.store.item(a.item), a.job_dir)
        except ValueError as e:
            sys.exit(str(e))
        print(json.dumps(t, ensure_ascii=False))
    return 0


def _fetch_command(a, ap):
    from . import remote
    if a.legacy_worker_stopped and a.release_stale is None:
        ap.error("--legacy-worker-stopped requires --release-stale")
    if a.release_stale is not None and any((a.loop, a.dry_run, a.browser, a.status, a.finish_browser,
                                            a.only, a.channels)):
        ap.error("--release-stale is a separate fetch operation")
    if a.loop is not None and (a.status or a.dry_run or a.finish_browser or a.browser):
        ap.error("--loop is for draining the fetch queue")

    pipeline = None

    def fetch_pass(stopping=None):
        nonlocal pipeline
        a.fetch_deadline = time.monotonic() + a.max_wait if a.max_wait is not None else None
        if pipeline is None:
            try:
                pipeline = _pipeline(a)
            except TimeoutError as exc:
                print(f"tbsgen fetch: {exc}", file=sys.stderr)
                return (0, False) if stopping else 3
            if a.release_stale is not None:
                deadline = a.fetch_deadline
                try:
                    result = remote.release_stale(pipeline.fulltext_fetcher(deadline=deadline), a.release_stale,
                                                  deadline=deadline, legacy_worker_stopped=a.legacy_worker_stopped)
                except TimeoutError as exc:
                    result = dict(error=str(exc))
                print(json.dumps(result))
                return 3 if result.get("error") else 0
            for receipt in remote.outstanding_jobs(pipeline.cfg):
                if remote.legacy_receipt(receipt):
                    print(f"legacy fetch receipt {receipt['job']} ({receipt['stage']}) in the owning queue. "
                          f"After confirming the legacy worker stopped, use tbsgen fetch --release-stale "
                          f"{receipt['job']} --legacy-worker-stopped with the original --run and --config.",
                          file=sys.stderr)
        if stopping and stopping():
            return 0, False
        return _fetch(pipeline, a, looping=stopping is not None)

    if a.loop is not None:
        return _worker_loop(Path(a.run), a.loop, fetch_pass)
    return fetch_pass()


def _worker_loop(root, interval, run_pass):
    stopped = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    stop_file = root / "driver.stop"

    def stopping():
        return stopped.is_set() or stop_file.exists()

    try:
        while not stopping():
            code, progress = run_pass(stopping)
            if code:
                return code
            if not progress:
                until = time.monotonic() + interval
                while not stopping() and (left := until - time.monotonic()) > 0:
                    stopped.wait(min(left, 1.0))
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous)


def _ids(v: str) -> list[str] | None:
    return [x.strip() for x in (v or "").split(",") if x.strip()] or None


def _calib(P: Pipeline, a) -> int:
    import time
    from .stages_b import resolve_remote, watch_tick
    if a.action == "launch":
        from .budget import BudgetStop
        from .stages_b import manual_launch
        if (not (a.item and a.job and a.attempt) or a.synced or a.not_started or a.ended_without_result
                or a.remote_task or a.force_local_gone):
            sys.exit("calib launch requires --item ID --job NAME --attempt N")
        try:
            return manual_launch(P, P.store.item(a.item), a.job, a.attempt)
        except (BudgetStop, ValueError, OSError) as e:
            sys.exit(str(e))
    if a.attempt is not None:
        sys.exit("--attempt requires calib launch")
    if a.action == "resolve-remote":
        if not (a.item and a.job and (a.synced or a.ended_without_result or a.not_started)):
            sys.exit("resolve-remote requires --item ID --job NAME and one of --synced, --ended-without-result or --not-started")
        if a.not_started and a.remote_task:
            sys.exit("--remote-task requires --synced or --ended-without-result")
        if a.force_local_gone and not a.not_started:
            sys.exit("--force-local-gone requires --not-started")
        try:
            result = resolve_remote(P, P.store.item(a.item), a.job, synced=a.synced,
                                    ended_without_result=a.ended_without_result, remote_task=a.remote_task,
                                    force_local_gone=a.force_local_gone)
        except ValueError as e:
            sys.exit(str(e))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if a.job or a.synced or a.not_started or a.ended_without_result or a.remote_task or a.force_local_gone:
        sys.exit("remote resolution options require calib resolve-remote")
    if a.action == "prepare":
        if not a.item:
            sys.exit("tbsgen calib prepare: --item is required")
        if not P.cfg["calib"].get("prepare_command") or not calib.prepared_dir(P.cfg, a.who):
            sys.exit("tbsgen calib prepare: set calib.prepare_command and calib.prepared_tasks_dir")
        it = P.store.item(a.item)
        with it.operator_lock():
            if not (it.p("pkg") / "task.toml").exists():
                sys.exit(f"tbsgen calib prepare: {a.item} has no packaged task (pkg/task.toml)")
            res = calib.ensure_prepared(P.cfg, it.dir, a.who, force=True)
        P.store.event("calib_prepared", item=a.item, who=a.who, ok=res.get("ok"), sec=res.get("sec"), manual=True)
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res.get("ok") else 1
    only = _ids(a.only)
    interval = a.interval if a.interval is not None else float(P.cfg["calib"].get("watch_interval_s") or 120)
    t_end = time.time() + a.max_hours * 3600 if a.max_hours else None
    while True:
        out = watch_tick(P, only)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        running = [i for i, v in out["open"].items() if v["status"] == "waiting" and _running(P, i)]
        if a.once or not out["open"] or (t_end and time.time() >= t_end):
            return 0
        if not running and not P.cfg["calib"].get("allow_submit"):
            print(json.dumps(dict(stop="nothing running and calib.allow_submit=false; S9 items wait for a human",
                                  open=out["open"]), ensure_ascii=False), flush=True)
            return 0
        P.stopped_models.clear()
        time.sleep(interval)


def _running(P: Pipeline, item_id: str) -> bool:
    from .stages_b import job_stage
    it = P.store.item(item_id)
    stage = job_stage(it)
    if stage == "S5G":                                   # a hardening round's authoring job
        return (read_json(it.p("S5G", "run.json"), {}) or {}).get("state") == "launched"
    runs = read_json(it.p("review" if stage == "S9R" else "calib", "runs.json"), []) or []
    return any(calib.pending_launcher(r) for r in runs)


def fetch_max_default(cfg: dict, given: int | None) -> int:
    """`tbsgen fetch --max`: the given value, else the remote worker's per-job cap (fulltext.remote.batch_max)
    when remote mode is on — one fetch then fills one remote batch — else 10."""
    if given is not None:
        return int(given)
    rc = (cfg.get("fulltext") or {}).get("remote") or {}
    return int(rc.get("batch_max", 50)) if rc.get("enabled") else 10


def _fetch(P: Pipeline, a, *, looping=False):
    from . import browser_ego, fulltext
    from .state import Store
    max_wait = getattr(a, "max_wait", None)
    deadline = vars(a).pop("fetch_deadline", None)  # later loop passes get a new budget
    if deadline is None and max_wait is not None:
        deadline = time.monotonic() + max_wait
    try:
        f = P.fulltext_fetcher(deadline=deadline)
    except TimeoutError as exc:
        print(f"tbsgen fetch: {exc}", file=sys.stderr)
        return (0, False) if looping else 3
    if a.status:
        print(json.dumps(dict(ledger=str(fulltext.ledger_path(P.cfg)), usage=f.ledger.usage(),
                              queue=len(fulltext.read_queue(fulltext.queue_path(P.cfg)))), indent=1))
        return 0
    if a.finish_browser:
        print(json.dumps(browser_ego.finish(P.cfg["fulltext"]["browser"])))
        return 0

    delivery_deadline = deadline if not getattr(a, "browser", False) else None

    def reactivate(entry, out, sub) -> bool:
        """Deliver the fetched text to one waiting item and move it on to S4 (the same transition S3 makes), so
        items of other runs (other caches) get the text too.  True when the item no longer waits for it."""
        run, iid = sub
        run = Path(run or P.store.root)
        if not (run / "items" / iid / "state.json").exists():
            return True                                    # item gone: nothing to wait for
        it = Store(run).item(iid)
        if delivery_deadline is not None and time.monotonic() >= delivery_deadline:
            raise TimeoutError("subscriber delivery deadline reached")
        with it.operator_lock(deadline=delivery_deadline):
            st = it.state
            if st.get("stage") == "S3" and st.get("status") == "parked" and st.get("reason") == "fulltext_unavailable":
                write_json(it.p("S3_fulltext.json"), dict(out.rec, delivered_by="tbsgen fetch"))
                it.p("fulltext.txt").write_text(out.text)
                # the next stage belongs to the subscriber's run (its own config), not to this fetch process:
                # S3 records it in S3_queue.json when it parks; an item parked before code-grounding goes to S4
                nxt = (read_json(it.p("S3_queue.json"), {}) or {}).get("next_stage") or "S4"
                it.advance("S3", nxt, source=out.rec.get("source"), chars=len(out.text), via="tbsgen fetch")
            return True                                        # delivered now, or already past S3 / not waiting

    try:
        readded = fulltext.reconcile(P.store, fulltext.queue_path(P.cfg), deadline=delivery_deadline) if not a.dry_run else 0
    except TimeoutError as exc:
        print(f"tbsgen fetch: {exc}", file=sys.stderr)
        return (0, False) if looping else 3
    if getattr(a, "browser", False) and (getattr(a, "only", "") or "").strip():
        print("tbsgen fetch: --only is not supported with --browser (a browser batch plans its own papers)",
              file=sys.stderr)
        return 2
    if getattr(a, "browser", False):
        from . import browser_agent
        remaining = max(0, deadline - time.monotonic()) if deadline is not None else None
        stats = browser_agent.run_batch(f, wait_for_egress=a.wait_for_egress, max_wait_s=remaining, dry_run=a.dry_run,
                                        on_success=reactivate, log=lambda s: print(s, file=sys.stderr, flush=True))
        stats["requeued_from_items"] = readded
        print(json.dumps(stats, indent=1, ensure_ascii=False, default=str))
        return 2 if stats.get("refused") else 0
    only = [x.strip() for x in (getattr(a, "only", "") or "").split(",") if x.strip()] or None
    try:
        stats = fulltext.drain_queue(f, max_papers=0 if only else fetch_max_default(P.cfg, a.max),
                                     channels=[x for x in a.channels.split(",") if x] or None,
                                     browser=not a.no_browser, dry_run=a.dry_run, on_success=reactivate,
                                     log=lambda s: print(s, file=sys.stderr), only=only, poll_once=looping,
                                     max_wait_s=max(0, delivery_deadline - time.monotonic()) if delivery_deadline is not None else None)
    except TimeoutError as exc:
        print(f"tbsgen fetch: {exc}", file=sys.stderr)
        return (0, False) if looping else 3
    stats["requeued_from_items"] = readded
    print(json.dumps(stats, indent=1, ensure_ascii=False), flush=True)
    if looping:
        return 0, bool(stats.get("progress") or readded)
    if stats.get("remote_error"):
        print(f"tbsgen fetch: remote job refused: {stats['remote_error']}", file=sys.stderr)
        return 3
    if stats.get("undelivered") or stats.get("busy") or stats.get("deferred") or stats.get("remote", {}).get("busy"):
        print("tbsgen fetch: papers remain unprocessed or awaiting delivery", file=sys.stderr)
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
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out.get("same_code" if a.action == "status" else "ok") else 2


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
