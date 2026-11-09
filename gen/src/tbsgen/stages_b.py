"""Stage handlers S6C, S7, S8, PKG, S9, S10."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from . import calib, gates, package, repair
from .models import parse_files
from .prompts import METRICS_API, S6C_TESTS, S8_REPAIR, SYSTEM
from .stages_a import _write_files, chat_files, save_raw
from .state import read_json, write_json


def s6c(P, it):
    spec = read_json(it.p("_spec", "spec.json"))
    th = read_json(it.p("thresholds.json"))
    meas = read_json(it.p("S6A_measure.json"))
    task = it.p("task")
    if not (task / "tests" / "test_outputs.py").exists() or it.state.get("regen_6c"):
        thr = {k: {"threshold": v["threshold"], "direction": v["direction"]} for k, v in th["thresholds"].items()}
        g = spec["depgraph"]
        user = S6C_TESTS.format(spec=json.dumps(spec, indent=1),
                                measure=json.dumps({k: meas[k] for k in ("ref", "mid", "naive")}),
                                thresholds=json.dumps(thr), mid_drop=g.get("mid_drop"), naive_drop=g.get("naive_drop"),
                                hours=P.cfg["package"]["agent_hours"])
        need = {"tests/test_outputs.py", "instruction.md"}
        files, log = chat_files(P, it, "tests", "S6C", user, need, max_tokens=P.cfg["gen"].get("s6c_max_tokens"),
                                need_prefix=("baselines/exploit_",))
        write_json(it.p("S6C_calls.json"), log)
        if not need <= set(files) or not any(f.startswith("baselines/exploit_") for f in files):
            write_json(it.p("S6C_error.json"), dict(got=sorted(files), calls=log))
            return it.finish("dropped", "s6c_missing_files")
        bad = _write_files(task, files, ("tests/test_outputs.py", "instruction.md", "baselines/exploit_"))
        it.state.pop("regen_6c", None)
        it.state["s6c_rejected_paths"] = bad
        it.save()
    it.advance("S6C", "S7")


def s7(P, it):
    spec = read_json(it.p("_spec", "spec.json"))
    th = read_json(it.p("thresholds.json"))["thresholds"]
    n = it.bump("S7")
    rep = gates.run_gates(P.cfg, it.dir, spec, th, n)
    write_json(it.p("gates", f"attempt-{n}.json"), rep)
    if rep["ok"]:
        return it.advance("S7", "PKG", attempt=n)
    if rep["blocking"]:
        it.state["repair_from"] = "gates:" + ",".join(rep["blocking"])
        it.save()
        return it.goto("S8", it.state["repair_from"])
    # non-blocking only (thresholds_stated): programmatic fix = append a criteria table to the instruction
    if "thresholds_stated" in rep["nonblocking"]:
        ins = it.p("task", "instruction.md")
        rows = "\n".join(f"- `{k}` {v['direction']} {v['threshold']}" for k, v in th.items())
        ins.write_text(ins.read_text().rstrip() + "\n\n## Pass criteria (all must hold)\n" + rows + "\n")
        if n < 1 + P.cfg["gates"]["max_repair_rounds"]:
            return it.goto("S7", "programmatic_fix:thresholds_stated")
    it.finish("dropped", "nonblocking_unfixable:" + ",".join(rep["nonblocking"]))


def s8(P, it):
    rounds = it.state.get("repair_rounds", 0)
    if rounds >= P.cfg["gates"]["max_repair_rounds"]:
        return it.finish("dropped", f"repair_limit:{it.state.get('repair_from')}")
    task = it.p("task")
    spec = read_json(it.p("_spec", "spec.json"))
    last = sorted(it.p("gates").glob("attempt-*.json")) if it.p("gates").exists() else []
    gate_log = read_json(last[-1]) if last and str(it.state.get("repair_from", "")).startswith("gates:") else \
        read_json(it.p("S6_failure.json"))
    slim = json.dumps(_slim(gate_log), default=str)[:20000]
    user = S8_REPAIR.format(gates=slim, depgraph=json.dumps(spec["depgraph"]), files=repair.files_for_repair(task, gate_log),
                            metrics_api=METRICS_API)
    r = P.client("repair").chat(SYSTEM, user, stage="S8", item=it.item_id,
                                max_tokens=P.cfg["gen"].get("s6a_max_tokens"))
    save_raw(it, "S8", r)
    patch = parse_files(r.text, truncated=r.stop_reason == "max_tokens")
    rv = repair.review(task, patch, spec["depgraph"])
    d = it.p("repair", f"round-{rounds + 1}")
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "review.json", rv)
    for rel, body in patch.items():
        (d / "patch" / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / "patch" / rel).write_text(body)
    it.state["repair_rounds"] = rounds + 1
    it.save()
    if not rv["ok"] or not patch:
        return it.goto("S8", "repair_rejected:" + ";".join(rv["reasons"] or ["empty patch"]))
    repair.apply(task, patch)
    back = "S6A_run" if not str(it.state.get("repair_from", "")).startswith("gates:") else "S7"
    it.goto(back, f"repaired round {rounds + 1}")


def _slim(g):
    if not isinstance(g, dict):
        return g
    out = {k: g[k] for k in ("gates", "blocking", "nonblocking", "why") if k in g}
    det = g.get("details") or g.get("info") or {}
    if isinstance(det, dict):
        out["details"] = {k: det[k] for k in ("determinism", "unstated", "isolation", "runs", "truth_isolation")
                          if k in det}
        if "tdm_verbatim" in out.get("blocking", []):
            out["details"]["tdm_verbatim"] = det.get("tdm_verbatim")
            out["fix_hint"] = ("instruction.md / data files copy a passage of the paper's licensed full text verbatim "
                               "(over the TDM quote limit): paraphrase it in your own words or remove it.")
        if "truth_isolation" in out.get("blocking", []):
            out["fix_hint"] = ("tests/measure.py must execute /app/output/solution_cli.py ONLY via "
                               "`from _metrics import run_solution, hidden_sets` (one call per hidden set dir); "
                               "never subprocess it directly: run_solution drops privileges and locks /tests/truth.")
        if "readymade" in out.get("blocking", []):
            out["details"]["readymade"] = det.get("readymade")
            out["fix_hint"] = ("the task depends on the paper's own published implementation (see details.readymade): "
                               "remove that dependency / import / vendored code and implement the method in the "
                               "reference solution, so the agent has to do the work the task tests.")
        if "leak" in det:
            out["details"]["leak"] = dict(classes=det["leak"].get("classes"), findings=det["leak"].get("findings", [])[:8])
        if "seeds" in det:
            out["details"]["seeds"] = {s: {w: {k: v.get(k) for k in ("passed", "failed", "solve_rc", "solve_err")}
                                           for w, v in sd.items() if isinstance(v, dict)}
                                       for s, sd in det["seeds"].items()}
    return out


def pkg(P, it):
    spec = read_json(it.p("_spec", "spec.json"))
    last = sorted(it.p("gates").glob("attempt-*.json"))[-1]
    gate = read_json(last)
    gen0 = it.p("runs", f"s7-{gate['attempt']}", "detA")
    from . import prerelease
    cand = read_json(it.p("S1_candidate.json"), {})
    prov = dict(read_json(it.p("_spec", "provenance.json"), {}), item_id=it.item_id,
                companion_code=prerelease.signals(cand, spec)["repos"],     # for calibration / human review
                network=P.cfg["package"].get("network"),
                spec_sha=_sha_file(it.p("_spec", "spec.json")), gates=gate["gates"],
                thresholds=read_json(it.p("thresholds.json"))["thresholds"],
                model_cost=P.item_cost(it.item_id), packaged=time.time())
    out = package.package(P.cfg, it.dir, gen0, spec, prov)
    v = package.validate_pkg(out)
    # the packaged image (Dockerfile written here, not by the model) gets the same ready-made check
    rm = prerelease.readymade_check(out, cand, spec)
    if not rm["ok"]:
        v = dict(v, ok=False, errors=list(v.get("errors") or []) + [f"readymade: {rm['findings'][:3]}"])
    write_json(it.p("PKG_check.json"), v)
    if not v["ok"]:
        return it.finish("parked", "package_invalid")
    it.advance("PKG", "S9")


def _sha_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


@contextlib.contextmanager
def item_lock(it):
    """Per-item S9 lock (flock on calib/.lock): `run`, `calib watch` and `calib-ingest` may run at the same
    time; S9 steps and ingests of one item are serialised so an intent is never launched twice and a result is
    never attached twice.  Not re-entrant: never nest."""
    import fcntl
    p = it.p("calib", ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _cap_reached(P) -> float | None:
    cap = P.cfg["budget"].get("run_usd_total")
    if cap is not None and calib.actual_spend(P.ledger) >= float(cap):
        return float(cap)
    return None


def _save_run(it, rec):
    runs = read_json(it.p("calib", "runs.json"), [])
    for i, r in enumerate(runs):
        if r.get("job_name") == rec["job_name"]:
            runs[i] = rec
            break
    else:
        runs.append(rec)
    write_json(it.p("calib", "runs.json"), runs)


def s9(P, it):
    """Schedule the next calibration run, or label.  Results are attached by `tbsgen calib-ingest`.
    Run records are persisted before every side effect (intent -> reserved -> launched); resume reconciles."""
    with item_lock(it):
        it.load()                       # another process may have moved the item while this one waited
        if it.state.get("stage") != "S9" or it.state.get("status") != "active":
            return None
        return _s9(P, it)


def _s9(P, it):
    c = P.cfg["calib"]
    runs = read_json(it.p("calib", "runs.json"), [])
    for rec in runs:                                         # reconcile interrupted submissions
        if rec.get("outcome") is None and rec.get("state") in ("intent", "reserved"):
            if not c.get("allow_submit"):
                return it.finish("waiting", f"calib {rec['job_name']} in state {rec['state']}; submit disabled")
            cap = None if calib.job_started(rec) else _cap_reached(P)
            if cap is not None:
                # same gate as a new run: settled spend reached the run total -> do not start it; a held
                # reservation is released and the record goes back to intent (reserved again on resume)
                if rec["state"] == "reserved" and rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched",
                                    note="not launched: run_usd_total reached")
                    rec.update(state="intent", ledger_call=None)
                    _save_run(it, rec)
                return it.finish("paused", f"budget:run_total_actual ({calib.actual_spend(P.ledger):.2f} >= {cap})",
                                 paused_stage="S9")
            if rec["state"] == "intent" and not calib.job_started(rec) and c.get("prepare_command") and \
                    calib.prepared_dir(P.cfg, rec["who"]):
                prep = calib.ensure_prepared(P.cfg, it.dir, rec["who"])    # an intent written before the hook ran
                if not prep.get("ok"):
                    return it.finish("waiting", f"calib blocked: prepare_command failed "
                                                f"({(prep.get('error') or '')[:120]}); fix and run "
                                                "`tbsgen calib prepare --item ID`")
            rec = calib.launch(P.cfg, rec, P.ledger, lambda r: _save_run(it, r))
            if rec.get("blocked") and rec["state"] == "intent":
                return it.finish("waiting", f"calib blocked: {rec['blocked']}")
            P.store.event("calib_reconciled", item=it.item_id, job=rec["job_name"], state=rec["state"])
        elif rec.get("outcome") is None and rec.get("state") == "launched":
            why = calib.launcher_failed(rec, c)
            if why:
                if rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=why[:500])
                rec.update(state="launch_failed", outcome="infra", result_type="infra", error=why)
                _save_run(it, rec)
            elif calib.launcher_died(rec, c):
                _settle_dead_job(P, it, rec)
    runs = read_json(it.p("calib", "runs.json"), [])
    pending = [r for r in runs if r.get("outcome") is None]
    if pending:
        return it.finish("waiting", f"awaiting {pending[-1]['who']} run {pending[-1]['job_name']}")
    act, what = calib.next_action(runs, P.cfg)
    if act == "label":
        it.state["calib_label"] = what
        it.save()
        if what.startswith("parked"):
            return it.finish("parked", what)
        if what in c["keep_labels"]:
            return it.advance("S9", "S10", label=what)
        return it.finish("done", f"label:{what} (not kept; S9b variants off)")
    n = sum(1 for r in runs if r["who"] == what) + 1
    if c.get("prepare_command") and calib.prepared_dir(P.cfg, what):
        # environment-preparation hook: build the run-ready copy of pkg/ before the job config points at it
        prep = calib.ensure_prepared(P.cfg, it.dir, what)
        if not prep.get("reused"):
            P.store.event("calib_prepared", item=it.item_id, who=what, ok=prep.get("ok"), sec=prep.get("sec"),
                          error=(prep.get("error") or "")[:300])
        if not prep.get("ok"):
            return it.finish("waiting", f"calib blocked: prepare_command failed ({(prep.get('error') or '')[:120]}); "
                                        "fix and run `tbsgen calib prepare --item ID`")
    rec = calib.prepare(P.cfg, it.dir, what, n)
    if rec.get("blocked"):
        return it.finish("waiting", f"calib blocked: {rec['blocked']}")
    if not c.get("allow_submit"):
        _save_run(it, rec)
        P.store.event("calib_submit_prepared", item=it.item_id, who=what, config=rec["config"])
        return it.finish("waiting", f"prepared {what} run {n}; submit disabled (calib.allow_submit=false)")
    cap = P.cfg["budget"].get("run_usd_total")
    if cap is not None and calib.actual_spend(P.ledger) >= float(cap):
        # reservations only gate admission; a single Harbor run has no hard cost ceiling, so once settled
        # actual spend reaches the cap no further run is started (review #6)
        return it.finish("paused", f"budget:run_total_actual ({calib.actual_spend(P.ledger):.2f} >= {cap})")
    _save_run(it, rec)                                       # intent persisted before reserve/launch (review #3)
    rec = calib.launch(P.cfg, rec, P.ledger, lambda r: _save_run(it, r))
    P.store.event("calib_submitted" if rec.get("submitted") else "calib_launch_failed", item=it.item_id, who=what,
                  config=rec["config"], state=rec["state"])
    if rec.get("submitted"):
        return it.finish("waiting", f"submitted {what} run {n}: {rec['job_name']}")
    return it.goto("S9", f"launch failed: {rec.get('error', '')[:120]}")


def ingest(P, it, job_dir: str | None = None) -> dict:
    """Attach a finished Harbor job to its pending run.  The job must be that run's own job (job name, task path,
    agent model all match) and each run is consumed once; repairs a crash between saving runs.json and the item
    state (review #3/#4)."""
    with item_lock(it):
        it.load()
        return _ingest(P, it, job_dir)


def _ingest(P, it, job_dir: str | None = None) -> dict:
    runs = read_json(it.p("calib", "runs.json"), [])
    pend = [r for r in runs if r.get("outcome") is None and r.get("state", "launched") == "launched"]
    if not pend:
        if runs and runs[-1].get("outcome") is not None and it.state.get("status") == "waiting":
            it.state["stage"], it.state["status"] = "S9", "active"     # recover the item state
            it.save()
            return dict(recovered=True, job_name=runs[-1]["job_name"])
        raise ValueError("no pending calibration run for this item")
    rec = pend[0]
    jd = Path(job_dir or rec["job_dir"])
    jr = read_json(jd / "result.json", {})
    if not jr.get("finished_at"):           # Harbor writes result.json at job start; finished_at marks the end
        raise ValueError(f"job not finished yet: {jd}")
    conf = read_json(jd / "config.json", {})
    agents = conf.get("agents") or [{}]
    tasks = conf.get("tasks") or [{}]
    ident = dict(job_name=conf.get("job_name"), model=agents[0].get("model_name"), task=tasks[0].get("path"))
    want = dict(job_name=rec["job_name"], model=rec.get("model", ident["model"]), task=rec.get("task", ident["task"]))
    if ident != want:
        raise ValueError(f"job identity mismatch: job {ident} vs pending run {want}")
    used = {r.get("trial", {}).get("job_dir") for r in runs if r.get("outcome") is not None}
    if str(jd.resolve()) in used:
        raise ValueError(f"job dir already consumed: {jd}")
    trials = calib.collect(jd, rec.get("model"), P.cfg["models"].get("prices", {}))
    if not trials:
        raise ValueError("no trial results in job dir")
    t = dict(trials[0], job_dir=str(jd.resolve()))
    usd, kind, usage = calib.job_cost(trials)
    rec.update(outcome=t["outcome"], result_type=t["result_type"], minutes=t["minutes"], trial=t,
               cost_usd=usd, cost_kind=kind, n_trials=len(trials))
    if rec.get("ledger_call"):
        # unknown cost: the reservation stands in (cost_kind "reservation"), but actual_spend() does not count it
        P.ledger.settle(rec["ledger_call"], usd, kind=kind, usage=usage,
                        note=f"S9 cost {kind}, outcome={t['outcome']}, result_type={t['result_type']}")
    write_json(it.p("calib", "runs.json"), runs)
    it.state["stage"], it.state["status"] = "S9", "active"
    it.save()
    return t


def s10(P, it):
    rel = P.store.root / "release"
    rel.mkdir(exist_ok=True)
    with (rel / "manifest.jsonl").open("a") as f:
        runs = read_json(it.p("calib", "runs.json"), [])
        summ = [x for x in calib.run_summary(runs) if x["outcome"] is not None]
        f.write(json.dumps(dict(item_id=it.item_id, pkg=str(it.p("pkg")), label=it.state.get("calib_label"),
                                calib_runs=summ, result_types=[x["result_type"] for x in summ],
                                ts=time.time())) + "\n")
    it.advance("S10", None, status="done")


def _settle_dead_job(P, it, rec: dict) -> None:
    """The Harbor job of a launched run ended without finished_at (crash / kill): an infra run.  Its cost is what
    the trials written so far show (tokens), else unknown (reservation kept, not counted as actual spend)."""
    jd = Path(rec["job_dir"])
    try:
        trials = calib.collect(jd, rec.get("model"), P.cfg["models"].get("prices", {}))
    except (OSError, ValueError):
        trials = []
    usd, kind, usage = calib.job_cost(trials) if trials else (None, "reservation", None)
    if rec.get("ledger_call"):
        P.ledger.settle(rec["ledger_call"], usd, status="error", kind=kind, usage=usage,
                        note="harbor job ended without finished_at")
    exf = Path(rec.get("log", "") + ".exit")
    ex = exf.read_text().strip() if exf.exists() else "none (session gone)"
    rec.update(state="died", outcome="infra", result_type="infra", cost_usd=usd, cost_kind=kind,
               error=f"launcher exit {ex}; harbor job has no finished_at")
    _save_run(it, rec)
    P.store.event("calib_job_died", item=it.item_id, job=rec["job_name"], cost_kind=kind)


def rewake(cfg: dict, it) -> bool:
    """Under the item lock: re-read the item and, if still reschedulable, make it active (recorded in history)."""
    with item_lock(it):
        it.load()
        if not reschedulable(cfg, it):
            return False
        it.state["history"].append(dict(stage="S9", result="rewake", prev_reason=it.state.get("reason"),
                                        ts=time.time()))
        it.state["status"] = "active"
        it.save()
        return True


def reschedulable(cfg: dict, it) -> bool:
    """An S9 item that is waiting although no job of it is running: prepared-only intents (allow_submit was
    off), a blocked prepare / launch, or a reserved run.  With calib.allow_submit=true `run` steps it again."""
    if it.state.get("stage") != "S9" or it.state.get("status") != "waiting" or not cfg["calib"].get("allow_submit"):
        return False
    runs = read_json(it.p("calib", "runs.json"), [])
    return not any(r.get("outcome") is None and r.get("state", "launched") == "launched" for r in runs)


def watch_tick(P, only: list[str] | None = None, max_steps: int = 50) -> dict:
    """One pass of `tbsgen calib watch`: attach every finished Harbor job of an S9-waiting item (calib-ingest),
    flag launchers that died without a job, then step the S9 items (label, or launch the next run: GLM ->
    teacher).  Returns what it did and the S9 items still open."""
    did = []
    for it in P.store.items():
        if only and it.item_id not in only:
            continue
        if it.state.get("stage") != "S9" or it.state.get("status") != "waiting":
            continue
        runs = read_json(it.p("calib", "runs.json"), [])
        pend = [r for r in runs if r.get("outcome") is None and r.get("state", "launched") == "launched"]
        if not pend:
            continue
        rec = pend[0]
        jr = read_json(Path(rec["job_dir"]) / "result.json", {}) or {}
        if jr.get("finished_at"):
            try:
                t = ingest(P, it)
                did.append(dict(item=it.item_id, action="ingest", job=rec["job_name"], outcome=t.get("outcome"),
                                result_type=t.get("result_type")))
            except ValueError as e:
                did.append(dict(item=it.item_id, action="ingest_error", job=rec["job_name"], error=str(e)[:300]))
        else:
            c = P.cfg["calib"]
            with item_lock(it):
                it.load()                   # re-check under the lock: another watcher / run may have moved it
                runs = read_json(it.p("calib", "runs.json"), [])
                cur = next((r for r in runs if r.get("job_name") == rec["job_name"]), {})
                if it.state.get("stage") != "S9" or it.state.get("status") != "waiting" or \
                        cur.get("outcome") is not None or cur.get("state") != "launched":
                    continue
                kind = "launcher_failed" if calib.launcher_failed(cur, c) else \
                    "job_died" if calib.launcher_died(cur, c) else None
                if kind is None:
                    continue
                it.state["stage"], it.state["status"] = "S9", "active"   # s9() reconciles the dead launch
                it.save()
            did.append(dict(item=it.item_id, action=kind, job=rec["job_name"]))
    s9 = [i.item_id for i in P.store.items() if i.state.get("stage") == "S9" and (not only or i.item_id in only)]
    if s9:
        P.run(max_steps=max_steps, only=s9)
    open_ = {}
    for i in P.store.items():
        if (not only or i.item_id in only) and i.state.get("stage") == "S9" and \
                i.state.get("status") in ("waiting", "active"):
            open_[i.item_id] = dict(status=i.state["status"], reason=i.state.get("reason"))
    done = {i.item_id: dict(stage=i.state["stage"], status=i.state["status"], label=i.state.get("calib_label"))
            for i in P.store.items() if i.item_id in s9 and i.item_id not in open_}
    return dict(actions=did, open=open_, settled=done, ts=time.time())
