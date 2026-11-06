"""Stage handlers S6C, S7, S8, PKG, S9, S10."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import calib, gates, package, repair
from .models import parse_files
from .prompts import S6C_TESTS, S8_REPAIR, SYSTEM
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
    user = S8_REPAIR.format(gates=slim, depgraph=json.dumps(spec["depgraph"]), files=repair.files_for_repair(task, gate_log))
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


def s9(P, it):
    """Schedule the next calibration run, or label.  Results are attached by `tbsgen calib-ingest`."""
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
        if what in P.cfg["calib"]["keep_labels"]:
            return it.advance("S9", "S10", label=what)
        return it.finish("done", f"label:{what} (not kept; S9b variants off)")
    n = sum(1 for r in runs if r["who"] == what) + 1
    rec = calib.submit(P.cfg, it.dir, what, n)
    rec["outcome"] = None
    runs.append(rec)
    write_json(it.p("calib", "runs.json"), runs)
    P.store.event("calib_submit_prepared", item=it.item_id, who=what, config=rec["config"])
    it.finish("waiting", f"prepared {what} run {n}; submit disabled (calib.allow_submit=false)")


def s10(P, it):
    rel = P.store.root / "release"
    rel.mkdir(exist_ok=True)
    with (rel / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(dict(item_id=it.item_id, pkg=str(it.p("pkg")), label=it.state.get("calib_label"),
                                ts=time.time())) + "\n")
    it.advance("S10", None, status="done")
