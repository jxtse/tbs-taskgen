"""Stage handlers S6C, S7, S8, PKG, S9, S10."""
from __future__ import annotations

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
        if "truth_isolation" in out.get("blocking", []):
            out["fix_hint"] = ("tests/measure.py must execute /app/output/solution_cli.py ONLY via "
                               "`from _metrics import run_solution, hidden_sets` (one call per hidden set dir); "
                               "never subprocess it directly: run_solution drops privileges and locks /tests/truth.")
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
    prov = dict(read_json(it.p("_spec", "provenance.json"), {}), item_id=it.item_id,
                spec_sha=_sha_file(it.p("_spec", "spec.json")), gates=gate["gates"],
                thresholds=read_json(it.p("thresholds.json"))["thresholds"],
                model_cost=P.item_cost(it.item_id), packaged=time.time())
    out = package.package(P.cfg, it.dir, gen0, spec, prov)
    v = package.validate_pkg(out)
    write_json(it.p("PKG_check.json"), v)
    if not v["ok"]:
        return it.finish("parked", "package_invalid")
    it.advance("PKG", "S9")


def _sha_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


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
    c = P.cfg["calib"]
    runs = read_json(it.p("calib", "runs.json"), [])
    for rec in runs:                                         # reconcile interrupted submissions
        if rec.get("outcome") is None and rec.get("state") in ("intent", "reserved"):
            if not c.get("allow_submit"):
                return it.finish("waiting", f"calib {rec['job_name']} in state {rec['state']}; submit disabled")
            rec = calib.launch(P.cfg, rec, P.ledger, lambda r: _save_run(it, r))
            P.store.event("calib_reconciled", item=it.item_id, job=rec["job_name"], state=rec["state"])
        elif rec.get("outcome") is None and rec.get("state") == "launched":
            why = calib.launcher_failed(rec)
            if why:
                if rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], 0.0, status="error", note=why[:500])
                rec.update(state="launch_failed", outcome="infra", error=why)
                _save_run(it, rec)
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
    trials = calib.collect(jd)
    if not trials:
        raise ValueError("no trial results in job dir")
    t = dict(trials[0], job_dir=str(jd.resolve()))
    rec.update(outcome=t["outcome"], minutes=t["minutes"], trial=t)
    if rec.get("ledger_call"):
        P.ledger.settle(rec["ledger_call"], t.get("cost_usd"), note=f"harbor cost_usd, outcome={t['outcome']}")
    write_json(it.p("calib", "runs.json"), runs)
    it.state["stage"], it.state["status"] = "S9", "active"
    it.save()
    return t


def s10(P, it):
    rel = P.store.root / "release"
    rel.mkdir(exist_ok=True)
    with (rel / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(dict(item_id=it.item_id, pkg=str(it.p("pkg")), label=it.state.get("calib_label"),
                                ts=time.time())) + "\n")
    it.advance("S10", None, status="done")
