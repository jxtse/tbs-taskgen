"""Stage handlers S6C, S7, S8, PKG, S9, S9R, S10."""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path

import shutil

from . import calib, difficulty, gates, package, repair, review
from .models import parse_files
from .prompts import METRICS_API, S6C_TESTS, S8_EXPLOIT, S8_REPAIR, S8_REVIEW, SYSTEM
from .stages_a import _write_files, chat_files, save_raw
from .state import gate_attempts, latest_gate_report, passed_gate_attempt, passed_gate_report, read_json, write_json


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
    if it.p("_spec", "author_code.json").exists():
        from . import codeagent
        why = codeagent.env_ok(P.cfg)
        if why:
            return it.finish("paused", f"code:config:{why}", paused_stage="S7")
    spec = read_json(it.p("_spec", "spec.json"))
    th = read_json(it.p("thresholds.json"))["thresholds"]
    hround = int(it.state.get("harden_round", 0)) if it.state.get("harden_active") else 0
    if (it.p("_spec", difficulty.DESIGN_FILE).exists() or hround) and not it.state.get("thresholds_locked"):
        # (a review revision / regate never changes the science, so the design is not re-checked there)
        mode = difficulty.dcfg(P.cfg).get("design_gate", "block")
        if mode != "off" or hround:
            try:
                prof = difficulty.load_profile(P.cfg)
            except (OSError, ValueError):
                prof = None
            # a hardening round needs the measured levers for its preferred-lever check even with design_gate="off"
            chk = difficulty.check_lever_separation(it.dir, spec, th, prof) \
                if it.p("_spec", difficulty.DESIGN_FILE).exists() and (mode != "off" or hround) \
                else dict(ok=True, problems=[])
            hp = difficulty.check_hardening(it.dir, hround) if hround else []
            if hround and it.p("_spec", difficulty.DESIGN_FILE).exists():
                hp = hp + difficulty.preferred_problem(chk, prof, hround)
            if mode == "off":
                chk = dict(chk, ok=True, problems=[], off_problems=chk["problems"])   # off: only hp blocks
            chk = dict(chk, ok=chk["ok"] and not hp, problems=list(chk["problems"]) + hp, hardening=hp)
            write_json(it.p("gates", "difficulty.json"), dict(chk, mode=mode, ts=time.time()))
            if not chk["ok"] and (mode == "block" or hp):
                P.store.event("difficulty_design_failed", item=it.item_id, problems=chk["problems"][:5],
                              harden_round=it.state.get("harden_round", 0))
                write_json(it.p("S6_failure.json"), dict(why="difficulty_design", info=chk))
                from .stages_a import _agent_reauthor
                if not it.state.get("difficulty_reauthored") and _agent_reauthor(P, it, "difficulty_design"):
                    if it.state.get("stage") == "S5G":
                        it.state["difficulty_reauthored"] = True
                        it.save()
                    return None
                return it.finish("dropped", "difficulty_design_unproven")
    attempts = gate_attempts(it.p("gates"))
    if attempts:
        counters = it.state.setdefault("attempts", {})
        counters["S7"] = max(counters.get("S7", 0), int(attempts[-1].stem[8:]))
    n = it.bump("S7")
    rep = gates.run_gates(P.cfg, it.dir, spec, th, n)
    write_json(it.p("gates", f"attempt-{n}.json"), rep)
    if rep["ok"]:
        it.state["gates_attempt"] = n
        return it.advance("S7", "PKG", attempt=n)
    if rep["blocking"]:
        it.state["repair_from"] = "gates:" + ",".join(rep["blocking"])
        it.save()
        return it.goto("S8", it.state["repair_from"])
    # non-blocking only (thresholds_stated): programmatic fix = append a criteria table to the instruction
    if "thresholds_stated" in rep["nonblocking"]:
        ins = it.p("task", "instruction.md")
        if hashlib.sha256(ins.read_bytes()).hexdigest() == it.state.get("autofix_done_for"):
            return it.finish("dropped", "nonblocking_unfixable:" + ",".join(rep["nonblocking"]))
        rows = "\n".join(f"- `{k}` {v['direction']} {v['threshold']}" for k, v in th.items())
        fixed = (ins.read_text().rstrip() + "\n\n## Pass criteria (all must hold)\n" + rows + "\n").encode()
        it.state["autofix_done_for"] = hashlib.sha256(fixed).hexdigest()
        it.save()  # a crash before replacement leaves the original statement eligible for the same fix
        tmp = ins.with_suffix(".md.tmp")
        tmp.write_bytes(fixed)
        tmp.replace(ins)
        return it.goto("S7", "programmatic_fix:thresholds_stated")
    it.finish("dropped", "nonblocking_unfixable:" + ",".join(rep["nonblocking"]))


def s8(P, it):
    if str(it.state.get("repair_from", "")).startswith("review:") or it.state.get("review_revision_pending"):
        return _s8_review(P, it)
    reason = str(it.state.get("repair_from", ""))
    gate_log = latest_gate_report(it.p("gates"))
    if it.state.get("repair_from") == "gates:exploit_valid":
        return _s8_exploits(P, it)
    if it.state.get("thresholds_locked"):
        # a gate failed after a review revision or a regate: the ordinary S8 may change the reference / baselines
        # and is not bound by the revision limit, so it must not run — the review revision is retried (feedback)
        # while rounds remain, else the item is dropped (review.allow_human restores the old park)
        why = f"after the review revision: {it.state.get('repair_from')}"
        oc = (gate_log.get("details") or {}).get("output_contract")
        if oc and not oc.get("ok"):
            # compact per-file difference first, so the budget never cuts the columns that matter
            diff = {f: dict(missing_in_statement=sorted(set(m["reference"]) - set(m["stated"])),
                            not_in_reference=sorted(set(m["stated"]) - set(m["reference"])), line=m.get("line"))
                    for f, m in (oc.get("mismatch") or {}).items()}
            why += " output_contract mismatch (statement vs reference header): " + json.dumps(diff)[:1700]
        return _revision_retry(P, it, why)
    rounds = it.state.get("repair_rounds", 0)
    if rounds >= P.cfg["gates"]["max_repair_rounds"]:
        return it.finish("dropped", f"repair_limit:{it.state.get('repair_from')}")
    task = it.p("task")
    spec = read_json(it.p("_spec", "spec.json"))
    if not gate_log or not reason.startswith("gates:"):
        gate_log = read_json(it.p("S6_failure.json"))
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


_POST_PROBE_STAGES = ("S8", "PKG", "S9", "S9R", "S10")


def legacy_report_id(it):
    if it.state.get("stage") not in _POST_PROBE_STAGES or it.state.get("status") in ("done", "dropped"):
        return None
    attempt = passed_gate_attempt(it.state)
    if attempt is not None and it.state["stage"] != "S8":
        report = it.p("gates", f"attempt-{attempt}.json")
    else:
        reports = gate_attempts(it.p("gates"))
        if not reports:
            return None
        report = reports[-1]
    if gates.current_probe_report(read_json(report)) and (attempt is not None or it.state["stage"] == "S8"):
        return None
    # An attempt number can be reused by another hardening round.
    report_id = f"{report.stem}:{_sha_file(report) if report.exists() else 'missing'}"
    if not any(r.get("probe_report_id") == report_id for r in it.state.get("regates", [])):
        return report_id
    return None


def migrate_probe_report(P, it):
    if not legacy_report_id(it):
        return False
    with item_lock(it):
        it.load()
        report_id = legacy_report_id(it)
        if not report_id:
            return True  # another worker already moved this item
        stage = it.state["stage"]
        if stage == "S8" and it.state.get("review_revision_pending"):
            _finish_revision(P, it, it.state["review_revision_pending"])
        if stage in ("S9", "S9R"):
            records = read_json(it.p("review" if stage == "S9R" else "calib", "runs.json"), [])
            save = _save_review if stage == "S9R" else _save_run
            for rec in records:
                if stage == "S9R" and _received_drop(it, rec):
                    _route_review(P, it, rec, rec["n"])
                    return True
                if rec.get("outcome") is None:
                    if not (rec.get("state") == "launched" or rec.get("launch_attempted") or calib.job_started(rec)):
                        _cancel_probe_job(P, it, rec, save)
                        continue
                    # Adopt a manual submission through the same ledger transaction as normal recovery.
                    if rec.get("state") == "intent" and calib.job_started(rec):
                        cfg = review.review_cfg(P.cfg) if stage == "S9R" else P.cfg
                        calib.launch(cfg, rec, P.ledger, lambda r: save(it, r))
                    else:
                        rec["state"] = "launched"
                        save(it, rec)
                    it.state["probe_migration_job_stage"] = stage
        _regate(P, it, "probe_policy_migration", False, report_id)
        return True


def _cancel_probe_job(P, it, rec, save, kind="probe_migration_cancelled"):
    if rec.get("ledger_call"):
        P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind=kind)
    rec.update(state="cancelled_unstarted", outcome="infra", result_type="infra", obsolete_probe_job=True,
               cost_usd=0.0, cost_kind=kind)
    save(it, rec)


def _unstarted(rec):
    return (rec.get("outcome") is None and rec.get("state") in ("intent", "reserved")
            and rec.get("job_dir") and not rec.get("launch_attempted") and not calib.job_started(rec))


def finish_job_stage(P, it, status, reason, **info):
    """Close unstarted job reservations before persisting a parked or dropped item."""
    stage = it.state["stage"]
    if stage in ("S4c", "S5G"):
        path = it.p(stage, "run.json")
        agent = read_json(path, {})
        if _unstarted(agent):
            call = agent.get("ledger_call")
            if call is None and agent.get("reservation_key"):
                call = P.ledger.reserved_call(agent["reservation_key"])
            if call is not None:
                P.ledger.settle(call, 0.0, status="error", kind="cancelled_unstarted")
            agent.update(state="cancelled_unstarted", not_started=True, cost_usd=0.0, cost_kind="cancelled_unstarted")
            write_json(path, agent)
    for directory, save in (("review", _save_review), ("calib", _save_run)):
        for rec in read_json(it.p(directory, "runs.json"), []):
            if _unstarted(rec):
                _cancel_probe_job(P, it, rec, save, kind="cancelled_unstarted")
    return it.finish(status, reason, **info)


def _received_drop(it, rec):
    if (rec.get("outcome") not in ("done", "invalid") or rec.get("n") != it.state.get("review_round")
            or rec.get("calib_epoch", 0) != it.state.get("calib_epoch", 0)):
        return False
    v = _bound_verdict(it, rec, rec["n"])
    if v and (v.get("verdict") == "drop" or v["_normalized"].get("verdict") == "drop"):
        return (rec.get("pkg_sha") == calib.pkg_sha(it.p("pkg"))
                and _job_judgment(v, rec)["verdict"] == "drop")
    return False


def regate(P, it, reason: str, recalibrate: bool = False) -> dict:
    """Re-gate a packaged item after a harness change (the shared verifier template tests/_metrics.py / test.sh, the
    S7 gates): the statement, scoring script, generator, reference and thresholds stay as they are.
    Withdraws it from the release manifest, locks the thresholds (S6B never re-derives them; the reference must
    clear them under the new measurement), records the data hash (PKG checks the data / hidden inputs / truth did
    not change), and restarts at S6A_run -> S6B (locked) -> S6C (files kept) -> S7 -> PKG.  The calibration epoch
    and its runs are kept by default (they ran on the same statement, data and thresholds; S9 relabels from them;
    the operator re-scores the kept deliverables with the new verifier as evidence) -- recalibrate=True archives
    them and S9 starts again from the student.  The S9R review is bound to the package hash, so the new package is
    reviewed again before S10."""
    with item_lock(it):
        it.load()
        return _regate(P, it, reason, recalibrate)


def _regate(P, it, reason, recalibrate, report_id=None):
    st = it.state
    if not report_id and st.get("stage") not in ("S9", "S9R", "S10") and st.get("status") not in ("done", "parked"):
        raise ValueError(f"regate: item at {st.get('stage')}/{st.get('status')} is not packaged")
    packaged = it.p("pkg").is_dir()
    if (not report_id and not packaged) or not it.p("thresholds.json").exists():
        raise ValueError("regate: no pkg/ or thresholds.json")
    if not report_id and any(r.get("outcome") is None for r in read_json(it.p("calib", "runs.json"), []) or []):
        raise ValueError("regate: a calibration run is still pending")
    old_sha = calib.pkg_sha(it.p("pkg")) if packaged else None
    # withdraw first (idempotent): the package must be out of the manifest before the item can run again
    # and overwrite pkg/; a crash after this line leaves a withdrawn, still finished item -> regate again
    withdrawn = withdraw(P, it.item_id, f"regate: {reason}")
    rec = dict(ts=time.time(), reason=reason, from_stage=st.get("stage"), from_status=st.get("status"),
               pkg_sha=old_sha, calib_epoch=st.get("calib_epoch", 0), label=st.get("calib_label"))
    if report_id:
        rec["probe_report_id"] = report_id
        st.pop("repair_from", None)
        st.pop("regen_6c", None)
        st["gates_attempt"] = None
    st.setdefault("regates", []).append(rec)
    if not st.get("thresholds_locked") or not st.get("revision_data_hash"):
        st["revision_data_hash"] = review.pkg_data_hash(it.p("pkg")) if packaged else None
    st["thresholds_locked"] = True
    st["post_regate"] = True
    if packaged and (recalibrate or not (P.cfg.get("review") or {}).get("rescore_scoring_only", True)):
        # re-scoring switched off: the kept calibration cannot be checked against the new grader -> redo it
        if st.get("probe_migration_job_stage"):
            st["rescore_pending"] = dict(recalibrate=True)
        else:
            rec["archived_epoch"] = _archive_calibration(it)
        st.pop("calib_label", None)
    elif packaged and not st.get("rescore_pending"):
        # the shared verifier template changed: the kept calibration stands until S9 re-scores the saved
        # deliverables with the new grader (review.rescore_item), as the reference re-measurement does
        st["rescore_pending"] = dict(reason="regate", files=["tests/_metrics.py", "tests/test.sh"], ts=time.time())
    st.setdefault("history", []).append(dict(stage=st.get("stage"), result="regate", to="S6A_run", reason=reason,
                                             ts=time.time()))
    st.update(stage="S6A_run", status="waiting" if st.get("probe_migration_job_stage") else "active")
    st.pop("reason", None)
    it.save()
    P.store.event("regate", item=it.item_id, reason=reason, pkg_sha=old_sha)
    return dict(item=it.item_id, stage="S6A_run", pkg_sha=old_sha, withdrawn=withdrawn)


def _s8_exploits(P, it):
    """S8 when the only blocking gate is exploit_valid: rewrite the broken exploit baselines and nothing else
    (gates.exploit_patch_problems).  Exploits are not shipped and do not touch the reference, the tests, the
    statement or the thresholds, so this repair is allowed with locked thresholds (a released item re-gated after a
    harness change) and has its own round budget (gates.max_exploit_repair_rounds)."""
    n = int(it.state.get("exploit_repair_rounds", 0))
    locked = bool(it.state.get("thresholds_locked"))
    if n >= int(P.cfg["gates"].get("max_exploit_repair_rounds", 2)):
        if it.state.get("post_revision"):
            # after a review revision the review's own revision budget decides (feedback, else drop)
            return _revision_retry(P, it, f"exploit_valid still failing after {n} exploit repair rounds")
        if locked and (P.cfg.get("review") or {}).get("allow_human"):
            return it.finish("parked", f"exploit_repair_limit:{n}")
        return it.finish("dropped", f"exploit_repair_limit:{n}")
    task = it.p("task")
    gate_log = latest_gate_report(it.p("gates"))
    exploits = sorted(str(p.relative_to(task)) for p in (task / "baselines").glob("exploit_*.py"))
    files = "\n".join(f"=== FILE: {rel} ===\n{(task / rel).read_text()}" for rel in exploits)
    ctx = "\n".join(f"=== FILE: {rel} ===\n{(task / rel).read_text()[:30000]}"
                    for rel in ("instruction.md", "tests/measure.py", "solution/solve.py") if (task / rel).exists())
    user = S8_EXPLOIT.format(gates=json.dumps(_slim(gate_log), default=str)[:20000], metrics_api=METRICS_API,
                             context=ctx, files=files)
    r = P.client("repair").chat(SYSTEM, user, stage="S8", item=it.item_id,
                                max_tokens=P.cfg["gen"].get("s6a_max_tokens"))
    save_raw(it, "S8", r)
    patch = parse_files(r.text, truncated=r.stop_reason == "max_tokens")
    bad = gates.exploit_patch_problems(task, patch) or ([] if patch else ["empty patch"])
    d = it.p("repair", f"exploit-{n + 1}")
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "review.json", dict(ok=not bad, reasons=bad, files=sorted(patch), method="exploit-only"))
    it.state["exploit_repair_rounds"] = n + 1
    it.save()
    if bad:
        return it.goto("S8", "exploit_repair_rejected:" + ";".join(bad)[:300])
    for rel, body in patch.items():
        (d / "patch" / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / "patch" / rel).write_text(body)
        (d / "before" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task / rel, d / "before" / rel)
    repair.apply(task, patch)
    it.goto("S7", f"exploit repair round {n + 1}")


def _slim(g):
    if not isinstance(g, dict):
        return g
    out = {k: g[k] for k in ("gates", "blocking", "nonblocking", "why") if k in g}
    det = g.get("details") or g.get("info") or {}
    if isinstance(det, dict):
        out["details"] = {k: det[k] for k in ("determinism", "unstated", "isolation", "runs", "truth_isolation",
                                              "statement_clean", "threshold_format", "hidden_parity", "hidden_shift",
                                              "assumptions_stated", "oracle_margin")
                          if k in det}
        if "exploit_valid" in out.get("blocking", []):
            out["fix_hint"] = ("an exploit baseline did not really run (details.seeds.<seed>.exploit_*.valid.why): fix "
                               "baselines/exploit_*.py so it completes under the verifier's exact invocation (PROG: "
                               "the verifier calls `python /app/output/solution_cli.py <input_dir> <output_dir>` once "
                               "per hidden set, and a set may hold a single case) and writes every deliverable, "
                               "still taking the shortcut.  A crashing exploit tests nothing.")
        if "tdm_verbatim" in out.get("blocking", []):
            out["details"]["tdm_verbatim"] = det.get("tdm_verbatim")
            out["fix_hint"] = ("instruction.md / data files copy a passage of the paper's licensed full text verbatim "
                               "(over the TDM quote limit): paraphrase it in your own words or remove it.")
        if "truth_isolation" in out.get("blocking", []):
            out["fix_hint"] = ("tests/measure.py must execute /app/output/solution_cli.py ONLY via "
                               "`from _metrics import run_solution, hidden_sets` (one call per hidden set dir); "
                               "never subprocess it directly: run_solution drops privileges and locks /tests/truth.")
        if "output_contract" in out.get("blocking", []):
            out["details"]["output_contract"] = det.get("output_contract")
            out["fix_hint"] = ("instruction.md states a CSV header that differs from what the reference writes and "
                               "tests/measure.py reads (details.output_contract: file, stated columns, reference "
                               "columns): make the statement, the reference and the scorer agree on one format.")
        if "author_leak" in out.get("blocking", []):
            out["details"]["author_leak"] = det.get("author_leak")
            out["fix_hint"] = ("instruction.md or a public data file names the authors' repository / package / "
                               "function (details.author_leak): describe the method scientifically instead.")
        if "readymade" in out.get("blocking", []):
            out["details"]["readymade"] = det.get("readymade")
            out["fix_hint"] = ("the task depends on the paper's own published implementation (see details.readymade): "
                               "remove that dependency / import / vendored code and implement the method in the "
                               "reference solution, so the agent has to do the work the task tests.")
        if "leak" in det:
            out["details"]["leak"] = dict(classes=det["leak"].get("classes"), findings=det["leak"].get("findings", [])[:8])
        if "seeds" in det:
            out["details"]["seeds"] = {s: {w: {k: v.get(k) for k in ("passed", "failed", "solve_rc", "solve_err", "valid")
                                               if k in v}
                                           for w, v in sd.items() if isinstance(v, dict)}
                                       for s, sd in det["seeds"].items()}
    return out


def pkg(P, it):
    spec = read_json(it.p("_spec", "spec.json"))
    attempt = passed_gate_attempt(it.state)
    gate = passed_gate_report(it.p("gates"), it.state)
    if attempt is None or gate.get("attempt") != attempt:
        raise ValueError("PKG needs the report of the passing S7 attempt")
    gen0 = it.p("runs", f"s7-{attempt}", "detA")
    from . import prerelease
    cand = read_json(it.p("S1_candidate.json"), {})
    prov = dict(read_json(it.p("_spec", "provenance.json"), {}), item_id=it.item_id,
                companion_code=prerelease.signals(cand, spec)["repos"],     # for calibration / human review
                network=P.cfg["package"].get("network"),
                spec_sha=_sha_file(it.p("_spec", "spec.json")), gates=gate["gates"],
                thresholds=read_json(it.p("thresholds.json"))["thresholds"],
                model_cost=P.item_cost(it.item_id), packaged=time.time())
    import hashlib
    if it.state.get("rescore_pending") and not it.state.get("pkg_statement_sha"):
        # persisted before pkg/ is overwritten, so a crash between overwrite and archive still compares correctly
        try:
            it.state["pkg_statement_sha"] = hashlib.sha256(it.p("pkg", "instruction.md").read_bytes()).hexdigest()
            it.save()
        except OSError:
            pass
    old_sha = it.state.get("pkg_statement_sha")
    if "legacy_release_sha" not in it.state:
        previous = _live_release(P, it.item_id)
        if previous and "pkg_sha" not in previous and previous.get("pkg"):
            previous_pkg = Path(previous["pkg"])
            it.state["legacy_release_sha"] = calib.pkg_sha(previous_pkg) if previous_pkg.is_dir() else None
            it.save()  # Keep the original hash even if PKG is interrupted and retried.
    out = package.package(P.cfg, it.dir, gen0, spec, prov)
    v = package.validate_pkg(out, P.cfg)
    new_sha = hashlib.sha256((out / "instruction.md").read_bytes()).hexdigest()
    if it.state.get("rescore_pending") and old_sha and new_sha != old_sha:
        # a scoring-only revision / regate would re-score the kept calibration, but the packaged statement changed
        # (e.g. a note PKG adds, a criteria table S7 appended): those runs read another statement -> recalibrate
        epoch = _archive_calibration(it)
        it.state.pop("rescore_pending", None)
        P.store.event("calib_archived", item=it.item_id, epoch=epoch, reason="packaged statement changed")
    it.state.pop("pkg_statement_sha", None)
    # the packaged image (Dockerfile written here, not by the model) gets the same ready-made check
    spec_a = prerelease.with_author_code(it.dir, spec)
    rm = prerelease.readymade_check(out, cand, spec_a)
    if (it.p("_spec", "author_code.json")).exists():
        al = prerelease.author_name_leaks(out, spec_a)
        if not al["ok"]:
            rm = dict(rm, ok=False, findings=list(rm.get("findings") or []) + [dict(kind="author_name", **f)
                                                                               for f in al["findings"]])
    if not rm["ok"]:
        v = dict(v, ok=False, errors=list(v.get("errors") or []) + [f"readymade: {rm['findings'][:3]}"])
    if it.p("_spec", difficulty.DESIGN_FILE).exists() or difficulty.dcfg(P.cfg).get("profile_path"):
        try:
            prof = difficulty.load_profile(P.cfg)
        except (OSError, ValueError):
            prof = None
        tl = difficulty.targeting_leaks([out / "instruction.md", out / "environment", out / "task.toml"], prof,
                                        read_json(it.p("_spec", difficulty.DESIGN_FILE), None))
        if tl:
            v = dict(v, ok=False, errors=list(v.get("errors") or []) + [f"targeting_leak: {tl[:3]}"])
    write_json(it.p("PKG_check.json"), v)
    if not v["ok"]:
        raw = v.get("errors")
        errs = [str(x) for x in (raw if isinstance(raw, (list, tuple)) else []) if x]
        if it.state.get("thresholds_locked"):
            # the package the review revision produced is broken: retry the revision (feedback) / drop
            return _revision_retry(P, it, "package_invalid: " + "; ".join(errs)[:200])
        return it.finish("parked", "package_invalid")
    if it.state.get("thresholds_locked"):
        why = _revision_pkg_problem(it, gate)
        if why:
            return _revision_retry(P, it, why)
    it.state.pop("thresholds_locked", None)     # a review revision's lock ends with the new package
    it.state.pop("post_revision", None)
    it.state.pop("post_regate", None)
    if it.state.pop("harden_active", None):    # the hardened version is built: from here it is an ordinary item
        k = int(it.state.get("harden_round", 0))
        kind = difficulty.round_kind(it)
        it.state.pop("harden_kind", None)
        for key in ("harden_base", "harden_built"):
            it.state.pop(key, None)
        difficulty.write_round(it, k, packaged=time.time(), pkg_sha=calib.pkg_sha(it.p("pkg")))
        P.store.event("harden_packaged", item=it.item_id, round=k, rkind=kind)
    it.advance("PKG", "S9")


def _revision_pkg_problem(it, gate: dict) -> str | None:
    """After a review revision the new package must carry the reviewed package's data / hidden inputs / truth
    (README notes aside), and the verifier must have applied exactly the locked thresholds at run time, in every
    arm's metrics.jsonl of the passing S7 attempt (an arm that stopped before any check may log none)."""
    want = it.state.get("revision_data_hash")
    if want and review.pkg_data_hash(it.p("pkg")) != want:
        return "the revision changed packaged data / hidden inputs / truth; only README format notes may change"
    locked = review.locked_signature(read_json(it.p("thresholds.json"))["thresholds"])
    locked_checks = set(locked)
    logs = sorted(it.p("runs", f"s7-{gate['attempt']}").glob("logs-*/metrics.jsonl"))
    if not any(m.parent.name.startswith("logs-ref") for m in logs):
        return "no reference verifier log of the S7 attempt to check the thresholds applied at run time"
    for m in logs:                        # every arm (ref, mid, naive, exploits, nop): what check() applied
        got = review.runtime_thresholds(m)
        if got is None:
            return f"cannot read runtime thresholds ({m.parent.name})"
        reference = m.parent.name.startswith("logs-ref")
        if (reference and got != locked) or not set(got) <= locked_checks:
            return f"the verifier applied thresholds {got} != locked {locked} ({m.parent.name})"
    return None


def _sha_file(p: Path) -> str:
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


def _rescore_pending(P, it):
    """S9 entry with a pending re-score (a scoring-only review revision, or a shared-verifier regate): re-run the
    packaged verifier on every decisive run's saved deliverable (calib/deliverables/<job>/; a run that produced no
    deliverable at all keeps its fail outcome — the S7 nop gate proves the new grader fails an empty output).  All
    outcomes unchanged -> the calibration stands (S9 relabels from the kept runs).  Any outcome changed, or a
    decisive run without a saved deliverable -> archive the epoch and recalibrate from GLM.  A sandbox failure
    waits (infrastructure, not an attempt)."""
    pend = it.state.get("rescore_pending") or {}
    runs = read_json(it.p("calib", "runs.json"), []) or []
    if pend.get("recalibrate"):
        _archive_calibration(it)
        it.state.pop("rescore_pending", None)
        it.save()
        return None
    decisive = [r for r in runs if r.get("outcome") in ("pass", "fail")]
    if decisive:
        try:
            res = review.rescore_item(P.cfg, it.dir, runs)
        except Exception as e:                       # sandbox / docker down: not an attempt
            it.finish("waiting", f"rescore blocked: {str(e)[:200]}")
            return "stop"
        write_json(it.p("calib", "rescore.json"), dict(pend, keep=res["keep"], checked=res["checked"],
                                                       missing=res["missing"], changed=res["changed"],
                                                       details=res["details"], ts=time.time()))
        P.store.event("review_rescore", item=it.item_id, reason=pend.get("reason"), keep=res["keep"],
                      checked=res["checked"], missing=res["missing"], changed=res["changed"])
        it.state.pop("rescore_pending", None)
        if not res["keep"]:
            _archive_calibration(it)
            it.state.pop("calib_label", None)
            it.goto("S9", f"rescore: {len(res['missing'])} deliverable(s) missing, "
                          f"{len(res['changed'])} outcome(s) changed under the new grader; recalibrating")
            return "stop"
    else:
        it.state.pop("rescore_pending", None)
    return None


def _cap_reached(P) -> float | None:
    cap = P.cfg["budget"].get("run_usd_total")
    if cap is not None and calib.actual_spend(P.ledger) >= float(cap):
        return float(cap)
    return None


def job_stage(it):
    """The stage owning the outstanding job while a probe migration drains it."""
    return it.state.get("probe_migration_job_stage") or it.state.get("stage")


def running_jobs(P) -> list[str]:
    """Harbor jobs holding a calib.max_concurrent_jobs slot: records of the item's current stage (S9 -> calibration
    runs, S9R -> review runs; the stage's watcher is what settles them) without an outcome that are launched,
    reserved (about to launch / recovering), or whose job already exists."""
    out = []
    for it in P.store.items():
        st = job_stage(it)
        rf = ("calib", "runs.json") if st == "S9" else ("review", "runs.json") if st == "S9R" else None
        if rf is None:
            continue
        for r in read_json(it.p(*rf), []) or []:
            if r.get("outcome") is None and (r.get("state") in ("launched", "reserved") or calib.job_started(r)):
                out.append(r.get("job_name") or "?")
    return out


def _slots_full(P, own: str | None = None) -> str | None:
    """calib.max_concurrent_jobs (None = no limit): a new S9 / S9R job waits while that many are running; the
    waiting item is stepped again by the next `run` / `calib watch` (reschedulable).  own: the job being
    admitted (not counted against itself)."""
    n = P.cfg["calib"].get("max_concurrent_jobs")
    if n is None:
        return None
    cur = [j for j in running_jobs(P) if j != own]
    if len(cur) >= int(n):
        return f"slots full: {len(cur)}/{int(n)} jobs running ({', '.join(cur[:3])})"
    return None


@contextlib.contextmanager
def slot_lock(P):
    """Run-wide admission lock (flock on <run>/.calib-slots.lock), held only from the slot check through the
    launch (never around preparation), so concurrent `run` / `calib watch` processes cannot both take the last
    slot.  Taken inside an item lock; nothing takes an item lock while holding it."""
    import fcntl
    if P.cfg["calib"].get("max_concurrent_jobs") is None:
        yield
        return
    p = Path(P.cfg["run"]["root"]) / ".calib-slots.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _mark_refused(P, it, rec: dict, stage: str) -> None:
    """A run whose launcher exited because the model relay refused its preflight (calib.relay_preflight_refused):
    tag it so it is no attempt and uses up no infra retry (calib.next_action / S9R count), and the stage waits
    calib.preflight_retry_s before the next launch."""
    if calib.relay_preflight_refused(rec):
        rec.update(preflight_refused=True, refused_ts=time.time(), cost_kind="relay_refused")
        if rec.get("ledger_call"):
            P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="relay_refused",
                            note="relay preflight refused: no Harbor job, no model call")
        P.store.event("relay_preflight_refused", item=it.item_id, stage=stage, job=rec.get("job_name"))


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
    if it.state.get("rescore_pending"):
        if _rescore_pending(P, it) is not None:
            return None                     # waiting (sandbox trouble) or sent back to S9 to recalibrate
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
            full = _slots_full(P, rec["job_name"]) if rec["state"] == "intent" and not calib.job_started(rec) else None
            if full:
                return it.finish("waiting", f"calib {rec['job_name']} queued: {full}")
            if rec["state"] == "intent" and not calib.job_started(rec) and c.get("prepare_command") and \
                    calib.prepared_dir(P.cfg, rec["who"]):
                prep = calib.ensure_prepared(P.cfg, it.dir, rec["who"])    # an intent written before the hook ran
                if not prep.get("ok"):
                    return it.finish("waiting", f"calib blocked: prepare_command failed "
                                                f"({(prep.get('error') or '')[:120]}); fix and run "
                                                "`tbsgen calib prepare --item ID`")
            with slot_lock(P):
                full = _slots_full(P, rec["job_name"]) if rec["state"] == "intent" and \
                    not calib.job_started(rec) else None
                if not full:
                    rec = calib.launch(P.cfg, rec, P.ledger, lambda r: _save_run(it, r))
            if full:
                return it.finish("waiting", f"calib {rec['job_name']} queued: {full}")
            if rec.get("blocked") and rec["state"] == "intent":
                return it.finish("waiting", f"calib blocked: {rec['blocked']}")
            P.store.event("calib_reconciled", item=it.item_id, job=rec["job_name"], state=rec["state"])
        elif rec.get("outcome") is None and rec.get("state") == "launched":
            why = calib.launcher_failed(rec, c)
            if why:
                if rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=why[:500])
                rec.update(state="launch_failed", outcome="infra", result_type="infra", cost_usd=0.0,
                           cost_kind="not_launched", error=why)
                _mark_refused(P, it, rec, "S9")
                _save_run(it, rec)
            elif calib.launcher_died(rec, c):
                _settle_dead_job(P, it, rec)
    runs = read_json(it.p("calib", "runs.json"), [])
    pending = [r for r in runs if r.get("outcome") is None]
    if pending:
        return it.finish("waiting", f"awaiting {pending[-1]['who']} run {pending[-1]['job_name']}")
    if it.state.get("harden_pending") or difficulty.should_harden(P.cfg, it, runs):
        # hardening loop: the student solved this version on its first attempt -> back to S5G (difficulty.py)
        return start_hardening(P, it)
    act, what = calib.next_action(runs, P.cfg)
    if act == "run":
        wait = calib.preflight_wait(P.cfg, runs[-1] if runs else None)
        if wait:                        # the relay refused the last run: wait, then launch the same role again
            return it.finish("waiting", wait)
    if act == "label":
        it.state["calib_label"] = what
        it.save()
        if what == "easy" and difficulty.variant_rounds_left(P.cfg, it):
            # S9b: the student passed both decisive runs -> re-author a harder variant of this task and re-calibrate
            # it instead of discarding the version (difficulty.start_hardening, kind="variant"; own depth + budget)
            return start_hardening(P, it, kind=difficulty.KIND_VARIANT)
        if not P.cfg.get("review", {}).get("enabled"):
            scopes = _probe_scopes(it)
            if scopes or what == "frontier":
                why = review_gate(P, it, scopes)
                if why and scopes:
                    return _park_unreviewed(P, it, scopes)
                if why:
                    return finish_job_stage(P, it, "parked", f"frontier:unreviewed ({why})")
                if what == "frontier" or what in c["keep_labels"]:
                    return it.advance("S9", "S10", label=what)
        if review.reviewable(P.cfg, what):
            # S9R: kept labels must pass the review before S10; parked:teacher_failed is reviewed as a diagnosis.
            # A fresh round number per version: review job names / verdict records never collide with an earlier
            # version's (a hardened / variant version is reviewed again under its own round).
            it.state["review_round"] = review.next_round(it.dir, it.state)
            return it.goto("S9R", f"label:{what}")
        if what.startswith("parked"):
            return finish_job_stage(P, it, "parked", what)
        if what in c["keep_labels"]:
            return it.advance("S9", "S10", label=what)
        if what == "easy" and difficulty.variant_enabled(P.cfg, it):
            return it.finish("done", "label:easy (S9b exhausted)")
        return it.finish("done", f"label:{what} (not kept; S9b variants off)")
    n = sum(1 for r in runs if r["who"] == what) + 1
    full = _slots_full(P) if c.get("allow_submit") else None
    if full:
        # checked before anything is prepared or persisted: the next step starts this run from scratch (same n)
        return it.finish("waiting", f"calib {what} run {n} queued: {full}")
    if c.get("prepare_command") and calib.prepared_dir(P.cfg, what):
        # environment-preparation hook: build the run-ready copy of pkg/ before the job config points at it
        prep = calib.ensure_prepared(P.cfg, it.dir, what)
        if not prep.get("reused"):
            P.store.event("calib_prepared", item=it.item_id, who=what, ok=prep.get("ok"), sec=prep.get("sec"),
                          error=(prep.get("error") or "")[:300])
        if not prep.get("ok"):
            return it.finish("waiting", f"calib blocked: prepare_command failed ({(prep.get('error') or '')[:120]}); "
                                        "fix and run `tbsgen calib prepare --item ID`")
    stale = calib.stale_prepared(P.cfg, it.dir, what, int(it.state.get("calib_epoch", 0)))
    if stale:
        return it.finish("waiting", f"calib blocked: {stale}")
    rec = calib.prepare(P.cfg, it.dir, what, n, epoch=int(it.state.get("calib_epoch", 0)))
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
    with slot_lock(P):
        full = _slots_full(P, rec["job_name"])
        if not full:
            _save_run(it, rec)                               # intent persisted before reserve/launch (review #3)
            rec = calib.launch(P.cfg, rec, P.ledger, lambda r: _save_run(it, r))
    if full:
        # nothing persisted: the next step prepares this run again (same n; the prepared copy is reused)
        return it.finish("waiting", f"calib {what} run {n} queued: {full}")
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
        if job_stage(it) == "S9R":
            return _ingest_review(P, it, job_dir)
        return _ingest(P, it, job_dir)


def _after_ingest(P, it, rec, stage):
    if it.state.pop("probe_migration_job_stage", None):
        if stage == "S9R" and _received_drop(it, rec):
            return _route_review(P, it, rec, rec["n"])
        stage = "S6A_run"
    it.state.update(stage=stage, status="active")
    it.save()


def _ingest(P, it, job_dir: str | None = None) -> dict:
    runs = read_json(it.p("calib", "runs.json"), [])
    pend = [r for r in runs if r.get("outcome") is None and r.get("state", "launched") == "launched"]
    if not pend:
        if runs and runs[-1].get("outcome") is not None and it.state.get("status") == "waiting":
            _after_ingest(P, it, runs[-1], "S9")
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
    if calib.task_matches(jd, ident["task"], want["task"]):
        ident["task"] = want["task"]                  # a remote launcher's synced copy of the run's own task
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
    _after_ingest(P, it, rec, "S9")
    return t


def _oracle_gate_ok(it) -> bool:
    """The last S7 gate report of the current version passed the oracle (reference) gate."""
    g = passed_gate_report(it.p("gates"), it.state).get("gates", {})
    return g.get("oracle") is True


def _probe_scopes(it):
    return review.suspected_scopes(passed_gate_report(it.p("gates"), it.state))


def _park_unreviewed(P, it, scopes):
    why = "parked:probe_unreviewed"
    return finish_job_stage(P, it, "parked", why, detail={"suspected_scopes": scopes})


def review_gate(P, it, scopes=None) -> str | None:
    """Why S10 must not release this item yet (None = may release): with review on and a reviewable label, the
    current package, calibration epoch and review round need a recorded pass.  Frontier tasks also need a confirmed reference."""
    scopes = _probe_scopes(it) if scopes is None else scopes
    frontier = it.state.get("calib_label") == "frontier"
    if not frontier and not scopes and not review.reviewable(P.cfg, it.state.get("calib_label")):
        return None
    v = read_json(it.p("review", "verdict.json"), None)
    if not v:
        return "no review verdict"
    norm = v.get("_normalized") or {}
    if norm.get("verdict") != "pass":
        return f"review verdict {norm.get('verdict')}"
    if scopes or v.get("probe_findings"):
        runs = read_json(it.p("review", "runs.json"), [])
        job = next((r for r in runs if r.get("job_name") == v.get("job")), {})
        if (job.get("outcome") != "done" or job.get("n") != v.get("round") or job.get("pkg_sha") != v.get("pkg_sha")
                or job.get("calib_epoch", 0) != v.get("calib_epoch", 0)):
            return "probe verdict of another review job"
        if _job_judgment(v, job)["verdict"] != "pass":
            return "invalid verdict for the submitted review contract"
        contract = review.CURRENT_CONTRACT
        if not P.cfg["review"]["enabled"] and review.job_contract(job) != contract:
            if review.job_contract(job) not in ("s9r-v4", "s9r-v5"):
                return "review contract with execution evidence required"
            contract = "s9r-v5"  # preserve previously releasable, bound legacy passes when review is offline
        scopes = set(scopes) | set(job.get("suspected_scopes", []))
        if review.finding_errors(v, scopes, contract, repros=job.get("repros")):
            return "probe findings require a new review"
    if it.state.get("calib_label") == "frontier":
        fp = review.frontier_problems(norm, _oracle_gate_ok(it))
        if fp:
            return "frontier not confirmed: " + "; ".join(fp)[:200]
    if v.get("round") != it.state.get("review_round") or v.get("calib_epoch", 0) != it.state.get("calib_epoch", 0):
        return "review verdict of another round / calibration epoch"
    if v.get("pkg_sha") != calib.pkg_sha(it.p("pkg")):
        return "package changed after the review"
    return None


def s10(P, it):
    from . import diversity
    from .stages_a import _agent_slot_lock
    # Serialize the inventory read and manifest append with other S10 and S5G admissions in this run.
    with _agent_slot_lock(P):
        # A version already released (manifest append done, crash before "done") keeps its release: the diversity
        # decision is taken once per item, before the first append, not re-taken against a later inventory.
        scopes = _probe_scopes(it)
        previous = _live_release(P, it.item_id)
        why = None if previous else diversity.admission(P, it, "S10")
        if why:
            withdraw(P, it.item_id, why)
            return finish_job_stage(P, it, "parked", why)
        return _release_reviewed(P, it, previous, scopes)


def _release_reviewed(P, it, previous, scopes):
    why = review_gate(P, it, scopes)
    if why and not P.cfg["review"]["enabled"]:
        if scopes:
            return _park_unreviewed(P, it, scopes)
        return finish_job_stage(P, it, "parked", f"frontier:unreviewed ({why})")
    if why:
        # a release whose current package / epoch / round has no confirming pass must not stay published while the
        # review runs again (no-op when the item was never released)
        withdraw(P, it.item_id, f"S10 review gate: {why}")
        it.state["review_round"] = review.next_round(it.dir, it.state)
        return it.goto("S9R", f"S10 review gate: {why}")
    sha = calib.pkg_sha(it.p("pkg"))
    epoch = it.state.get("calib_epoch", 0)
    runs = read_json(it.p("calib", "runs.json"), [])
    summ = [x for x in calib.run_summary(runs) if x["outcome"] is not None]
    rv = read_json(it.p("review", "verdict.json"), None)
    nv = (rv or {}).get("_normalized") or {}
    rev = None if rv is None else dict(
        verdict=nv.get("verdict"), round=it.state.get("review_round"), revisions=it.state.get("review_revisions", 0),
        minor=[dict(id=i.get("id"), category=i.get("category"), title=i.get("title"))
               for i in nv.get("issues") or [] if i.get("severity") == "minor"],
        label_trustworthy=(nv.get("label_assessment") or {}).get("trustworthy"))
    label = it.state.get("calib_label")
    extra = {}
    if label == "frontier":
        # a separate release channel: no model solved it (no passing trajectory); consumers keep it apart
        extra = dict(frontier=calib.frontier_evidence(runs, P.cfg),
                     reference_check=nv.get("reference_check") or {})
    record = dict(item_id=it.item_id, pkg=str(it.p("pkg")), label=label,
                  channel="frontier" if label == "frontier" else "main", **extra,
                  calib_runs=summ, result_types=[x["result_type"] for x in summ],
                  pkg_sha=sha, calib_epoch=epoch, review=rev, difficulty=_difficulty_summary(it))
    if previous and "pkg_sha" not in previous and previous.get("pkg"):
        previous_pkg = Path(previous["pkg"])
        if "legacy_release_sha" in it.state:
            previous = dict(previous, pkg_sha=it.state["legacy_release_sha"])
        elif previous_pkg.is_dir():
            previous = dict(previous, pkg_sha=calib.pkg_sha(previous_pkg))
    if previous and all(previous.get(k) == v for k, v in record.items()):
        return it.advance("S10", None, status="done")
    rel = P.store.root / "release"
    rel.mkdir(exist_ok=True)
    _append_manifest(rel / "manifest.jsonl", dict(record, ts=time.time()))
    it.state.pop("legacy_release_sha", None)
    it.advance("S10", None, status="done")


def _settle_dead_job(P, it, rec: dict, save=_save_run) -> None:
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
    save(it, rec)
    P.store.event("calib_job_died", item=it.item_id, job=rec["job_name"], cost_kind=kind)


def rewake(cfg: dict, it) -> bool:
    """Under the item lock: re-read the item and, if still reschedulable, make it active (recorded in history)."""
    with item_lock(it):
        it.load()
        if not reschedulable(cfg, it):
            return False
        it.state["history"].append(dict(stage=it.state["stage"], result="rewake", prev_reason=it.state.get("reason"),
                                        ts=time.time()))
        it.state["status"] = "active"
        it.save()
        return True


def reschedulable(cfg: dict, it) -> bool:
    """An S9 item that is waiting although no job of it is running: prepared-only intents (allow_submit was
    off), a blocked prepare / launch, or a reserved run.  With calib.allow_submit=true `run` steps it again."""
    if it.state.get("stage") not in ("S9", "S9R") or it.state.get("status") != "waiting" or \
            not cfg["calib"].get("allow_submit"):
        return False
    runs = read_json(it.p("calib", "runs.json") if it.state["stage"] == "S9" else it.p("review", "runs.json"), [])
    if runs and calib.preflight_wait(cfg, runs[-1]):
        return False        # the relay refused its last run: nothing to do until calib.preflight_retry_s has passed
    return not any(r.get("outcome") is None and r.get("state", "launched") == "launched" for r in runs)


def watch_tick(P, only: list[str] | None = None, max_steps: int = 50) -> dict:
    """One pass of `tbsgen calib watch`: attach every finished Harbor job of an S9-waiting item (calib-ingest),
    flag launchers that died without a job, then step the S9 items (label, or launch the next run: GLM ->
    teacher).  Returns what it did and the S9 items still open."""
    did = []
    for it in P.store.items():
        if only and it.item_id not in only:
            continue
        stage = job_stage(it)
        if stage not in ("S9", "S9R") or it.state.get("status") != "waiting":
            continue
        rf = ("calib" if stage == "S9" else "review", "runs.json")
        runs = read_json(it.p(*rf), [])
        pend = [r for r in runs if r.get("outcome") is None and r.get("state", "launched") == "launched"]
        if not pend:
            if it.state.get("probe_migration_job_stage"):
                ingest(P, it)  # recover a crash after saving runs.json
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
                runs = read_json(it.p(*rf), [])
                cur = next((r for r in runs if r.get("job_name") == rec["job_name"]), {})
                if job_stage(it) != rf_stage(rf) or it.state.get("status") != "waiting" or \
                        cur.get("outcome") is not None or cur.get("state") != "launched":
                    continue
                kind = "launcher_failed" if calib.launcher_failed(cur, c) else \
                    "job_died" if calib.launcher_died(cur, c) else None
                if kind is None:
                    continue
                if it.state.get("probe_migration_job_stage"):
                    save = _save_run if stage == "S9" else _save_review
                    job_dir = Path(cur["job_dir"])
                    if ((job_dir / "config.json").exists() or (job_dir / "result.json").exists()
                            or any(job_dir.glob("*/result.json"))):
                        _settle_dead_job(P, it, cur, save=save)
                    else:
                        _cancel_probe_job(P, it, cur, save)
                    _after_ingest(P, it, cur, stage)
                else:
                    it.state["status"] = "active"       # s9() / s9r() reconcile the dead launch
                    it.save()
            did.append(dict(item=it.item_id, action=kind, job=rec["job_name"]))
    s9 = [i.item_id for i in P.store.items() if (not only or i.item_id in only) and
          (i.state.get("stage") in ("S9", "S9R") or (_in_revision(i) and i.state.get("status") == "active") or
           _in_hardening(i))]
    if s9:
        P.run(max_steps=max_steps, only=s9)
    open_ = {}
    for i in P.store.items():
        if (not only or i.item_id in only) and (
                (i.state.get("status") in ("waiting", "active") and
                 (i.state.get("stage") in ("S9", "S9R") or _in_revision(i) or _in_hardening(i))) or
                harden_needs_step(i)):
            open_[i.item_id] = dict(stage=i.state["stage"], status=i.state["status"], reason=i.state.get("reason"))
    done = {i.item_id: dict(stage=i.state["stage"], status=i.state["status"], label=i.state.get("calib_label"))
            for i in P.store.items() if i.item_id in s9 and i.item_id not in open_}
    return dict(actions=did, open=open_, settled=done, ts=time.time())


def _in_hardening(it) -> bool:
    """A hardening round in progress (S5G..PKG of the new version) or an unfinished hardening transaction."""
    return bool(it.state.get("harden_active") or harden_needs_step(it))


def _in_revision(it) -> bool:
    return bool(it.state.get("review_revisions") or it.state.get("thresholds_locked") or
                it.state.get("review_revision_pending") or str(it.state.get("repair_from", "")).startswith("review:"))


def rf_stage(rf: tuple) -> str:
    return "S9" if rf[0] == "calib" else "S9R"


# ------------------------------------------------------------------ S9R review
def _save_review(it, rec):
    runs = read_json(it.p("review", "runs.json"), [])
    for i, r in enumerate(runs):
        if r.get("job_name") == rec["job_name"]:
            runs[i] = rec
            break
    else:
        runs.append(rec)
    write_json(it.p("review", "runs.json"), runs)


def s9r(P, it):
    """S9R: launch the review agent for this round, or route its verdict.  Results are attached by
    `tbsgen calib-ingest` / `tbsgen calib watch` (same job lifecycle as S9: intent -> reserved -> launched)."""
    with item_lock(it):
        it.load()
        if it.state.get("stage") != "S9R" or it.state.get("status") != "active":
            return None
        return _s9r(P, it)


def _s9r(P, it):
    rcfg = review.review_cfg(P.cfg)
    c = rcfg["calib"]
    rv = P.cfg.get("review") or {}
    rnd = int(it.state.get("review_round") or 1)
    save = lambda r: _save_review(it, r)  # noqa: E731
    runs = [r for r in read_json(it.p("review", "runs.json"), []) if r.get("n") == rnd]
    sha, epoch = calib.pkg_sha(it.p("pkg")), int(it.state.get("calib_epoch", 0))
    for rec in runs:
        if _received_drop(it, rec):
            return _route_review(P, it, rec, rnd)
    scopes = _probe_scopes(it)
    review_disabled = bool(scopes) and not rv.get("enabled")
    for rec in runs:                                     # reconcile interrupted submissions / dead launches
        if rec.get("outcome") is None and rec.get("state") in ("intent", "reserved"):
            started = calib.job_started(rec)
            if not started and not rec.get("launch_attempted"):
                if (review_disabled or review.job_contract(rec) != review.CURRENT_CONTRACT
                        or set(scopes) - set(rec.get("suspected_scopes", []))):
                    _cancel_probe_job(P, it, rec, _save_review, kind="cancelled_unstarted")
                    continue
            elif review_disabled:
                # A manual job needs adoption. An uncertain launcher must be drained without resubmitting it.
                if started:
                    calib.launch(rcfg, rec, P.ledger, save)
                else:
                    rec.update(state="launched", recovered=True)
                    save(rec)
                continue
            if not c.get("allow_submit"):
                return it.finish("waiting", f"review {rec['job_name']} in state {rec['state']}; submit disabled")
            why = review.route_problem(P.cfg) if not calib.job_started(rec) else None
            if why:
                return it.finish("waiting", f"review blocked: {why}")
            cap = None if calib.job_started(rec) else _cap_reached(P)
            if cap is not None:
                if rec["state"] == "reserved" and rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched",
                                    note="not launched: run_usd_total reached")
                    rec.update(state="intent", ledger_call=None)
                    save(rec)
                return it.finish("paused", f"budget:run_total_actual ({calib.actual_spend(P.ledger):.2f} >= {cap})",
                                 paused_stage="S9R")
            with slot_lock(P):
                full = _slots_full(P, rec["job_name"]) if rec["state"] == "intent" and \
                    not calib.job_started(rec) else None
                if not full:
                    rec = calib.launch(rcfg, rec, P.ledger, save)
            if full:
                return it.finish("waiting", f"review {rec['job_name']} queued: {full}")
            if rec.get("blocked") and rec["state"] == "intent":
                return it.finish("waiting", f"review blocked: {rec['blocked']}")
        elif rec.get("outcome") is None and rec.get("state") == "launched":
            why = calib.launcher_failed(rec, c)
            if why:
                if rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=why[:500])
                rec.update(state="launch_failed", outcome="infra", cost_usd=0.0, cost_kind="not_launched", error=why)
                _mark_refused(P, it, rec, "S9R")
                save(rec)
            elif calib.launcher_died(rec, c):
                try:
                    trials = calib.collect(Path(rec["job_dir"]), rec.get("model"), P.cfg["models"].get("prices", {}))
                except (OSError, ValueError):
                    trials = []
                usd, kind, usage = calib.job_cost(trials) if trials else (None, "reservation", None)
                rec.update(state="died", outcome="infra", cost_usd=usd, cost_kind=kind,
                           error="review job ended without finished_at")
                if rec.get("ledger_call"):
                    P.ledger.settle(rec["ledger_call"], usd, status="error", kind=kind, usage=usage,
                                    note="review job ended without finished_at")
                save(rec)
    all_rnd = [r for r in read_json(it.p("review", "runs.json"), []) if r.get("n") == rnd]
    pending = [r for r in all_rnd if r.get("outcome") is None]
    if pending:
        return it.finish("waiting", f"awaiting review run {pending[-1]['job_name']}")
    # a verdict counts only for the package and calibration epoch it was submitted with
    runs = [r for r in all_rnd if r.get("pkg_sha") == sha and int(r.get("calib_epoch") or 0) == epoch]
    for rec in runs:
        if rec.get("outcome") == "invalid" and review.job_contract(rec) != review.CURRENT_CONTRACT:
            v = _bound_verdict(it, rec, rnd)
            if v:
                norm = _job_judgment(v, rec)
                if norm["verdict"]:
                    rec.update(outcome="done", verdict=norm["verdict"])
                    if _needs_new_review(rec, dict(v, _normalized=norm), scopes):
                        rec.update(outcome="infra", obsolete_probe_review=True)
                    save(rec)
    done = [r for r in runs if r.get("outcome") == "done"]
    if done:
        return _route_review(P, it, done[-1], rnd)
    if scopes and not rv.get("enabled"):
        return _park_unreviewed(P, it, scopes)
    # a run the model relay refused at its preflight is no attempt and uses up no retry (calib.relay_preflight_refused)
    n_infra = sum(1 for r in runs if r.get("outcome") == "infra" and not r.get("preflight_refused")
                  and not r.get("obsolete_probe_review") and not r.get("obsolete_probe_job"))
    n_invalid = sum(1 for r in runs if r.get("outcome") == "invalid")
    if n_infra + n_invalid > int(rv.get("infra_retries", 2)):
        if n_invalid:
            # the reviewer kept breaking the verdict contract: no usable judgment -> drop (no operator exit unless
            # review.allow_human)
            why = f"r{rnd}: {n_infra} infra, {n_invalid} invalid verdicts"
            if rv.get("allow_human"):
                return finish_job_stage(P, it, "parked", f"review:human ({why})")
            return finish_job_stage(P, it, "dropped", f"review:invalid {why}")
        return finish_job_stage(P, it, "parked", f"review:infra ({n_infra} infra, {n_invalid} invalid verdicts)")
    wait = calib.preflight_wait(P.cfg, all_rnd[-1] if all_rnd else None)
    if wait:                            # the relay refused the last review run: wait, then launch the next attempt
        return it.finish("waiting", wait)
    full = _slots_full(P) if c.get("allow_submit") else None
    if full:
        # checked before the review task is built or anything persisted: the next step starts it from scratch
        return it.finish("waiting", f"review round {rnd} queued: {full}")
    task = review.build_task(P.cfg, it.dir, it.state, rnd)
    rec = review.prepare(P.cfg, it.dir, task, rnd, len(all_rnd) + 1, epoch=epoch)
    if rec.get("blocked"):
        return it.finish("waiting", f"review blocked: {rec['blocked']}")
    if not c.get("allow_submit"):
        save(rec)
        P.store.event("review_prepared", item=it.item_id, round=rnd, config=rec["config"])
        return it.finish("waiting", f"prepared review round {rnd}; submit disabled (calib.allow_submit=false)")
    cap = _cap_reached(P)
    if cap is not None:
        return it.finish("paused", f"budget:run_total_actual ({calib.actual_spend(P.ledger):.2f} >= {cap})",
                         paused_stage="S9R")
    with slot_lock(P):
        full = _slots_full(P, rec["job_name"])
        if not full:
            save(rec)
            rec = calib.launch(rcfg, rec, P.ledger, save)
    if full:
        return it.finish("waiting", f"review round {rnd} queued: {full}")
    P.store.event("review_submitted" if rec.get("submitted") else "review_launch_failed", item=it.item_id,
                  round=rnd, state=rec["state"])
    if rec.get("submitted"):
        return it.finish("waiting", f"submitted review round {rnd}: {rec['job_name']}")
    return it.goto("S9R", f"review launch failed: {rec.get('error', '')[:120]}")


def _ingest_review(P, it, job_dir: str | None = None) -> dict:
    runs = read_json(it.p("review", "runs.json"), [])
    pend = [r for r in runs if r.get("outcome") is None and r.get("state", "launched") == "launched"]
    if not pend:
        if runs and runs[-1].get("outcome") is not None and it.state.get("status") == "waiting":
            _after_ingest(P, it, runs[-1], "S9R")
            return dict(recovered=True, job_name=runs[-1]["job_name"])
        raise ValueError("no pending review run for this item")
    rec = pend[0]
    jd = Path(job_dir or rec["job_dir"])
    jr = read_json(jd / "result.json", {})
    if not jr.get("finished_at"):
        raise ValueError(f"job not finished yet: {jd}")
    conf = read_json(jd / "config.json", {})
    agents = conf.get("agents") or [{}]
    tasks = conf.get("tasks") or [{}]
    ident = dict(job_name=conf.get("job_name"), model=agents[0].get("model_name"), task=tasks[0].get("path"))
    want = dict(job_name=rec["job_name"], model=rec.get("model"), task=rec.get("task"))
    if calib.task_matches(jd, ident["task"], want["task"]):
        ident["task"] = want["task"]
    if ident != want:
        raise ValueError(f"job identity mismatch: job {ident} vs pending review run {want}")
    assigned = rec.get("suspected_scopes", [])
    res = review.classify(jd, rec.get("model"), P.cfg["models"].get("prices", {}),
                          allowed_verdicts=rec.get("allowed_verdicts"), allow_human=True,
                          suspected_scopes=assigned, contract=review.job_contract(rec))
    rec.update(outcome="invalid" if (res["outcome"] == "done" and (res["verdict"] or {}).get("verdict") is None)
               else res["outcome"],
               minutes=res["minutes"], cost_usd=res["cost_usd"], cost_kind=res["cost_kind"],
               trial=res["trial"], n_trials=res["n_trials"], usage=res["usage"], repros=res["repros"])
    if res["verdict"] is not None:
        d = it.p("review", f"r{rec['n']}")
        d.mkdir(parents=True, exist_ok=True)
        raw = res["verdict_raw"]
        write_json(d / "verdict.raw.json", raw)
        body = dict(raw if isinstance(raw, dict) else {"_raw": raw}, _normalized=res["verdict"],
                    verifier_reward=res["verifier_reward"],
                    pkg_sha=rec.get("pkg_sha"), calib_epoch=rec.get("calib_epoch", 0), job=rec["job_name"])
        jf = d / f"{rec['job_name']}.verdict.json"           # one immutable verdict per job
        if not jf.exists():
            write_json(jf, body)
        write_json(d / "verdict.json", body)                  # latest of the round (records for the next round)
        rec["verdict"] = res["verdict"]["verdict"]
        if rec["verdict"] not in (None, "drop") and _needs_new_review(rec, body, _probe_scopes(it)):
            rec.update(outcome="infra", obsolete_probe_review=True)
    if rec.get("ledger_call"):
        P.ledger.settle(rec["ledger_call"], res["cost_usd"], kind=res["cost_kind"], usage=res["usage"],
                        note=f"S9R review r{rec['n']} outcome={rec['outcome']} verdict={rec.get('verdict')}")
    _save_review(it, rec)
    _after_ingest(P, it, rec, "S9R")
    outcome = "infra" if rec.get("obsolete_probe_review") else res["outcome"]
    return dict(outcome=outcome, verdict=rec.get("verdict"), job_name=rec["job_name"])


def _append_manifest(path: Path, record: dict) -> None:
    with path.open("a+b") as f:
        prefix = b""
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                prefix = b"\n"
        f.write(prefix + json.dumps(record).encode() + b"\n")


def _live_release(P, item_id: str) -> dict | None:
    m = P.store.root / "release" / "manifest.jsonl"
    if not m.exists():
        return None
    live = None
    for line in m.read_text().splitlines():
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("item_id") == item_id:
            live = None if o.get("withdrawn") else o
    return live


def _released(P, item_id: str) -> bool:
    return _live_release(P, item_id) is not None


def withdraw(P, item_id: str, reason: str) -> bool:
    """Append a withdrawal line for a released item (the manifest is append-only; the last line per item wins)."""
    if not _released(P, item_id):
        return False
    _append_manifest(P.store.root / "release" / "manifest.jsonl",
                     dict(item_id=item_id, withdrawn=True, reason=reason, ts=time.time()))
    P.store.event("release_withdrawn", item=item_id, reason=reason)
    return True


def _revision_retry(P, it, why: str):
    """A review revision failed to clear a locked-threshold gate (S6B reference / S7 gate / PKG checks): retry the
    revision with the failure fed back while revision rounds remain; otherwise drop the item.  review.allow_human
    (off by default) restores the old operator park instead."""
    rv = P.cfg.get("review") or {}
    if rv.get("allow_human"):
        return finish_job_stage(P, it, "parked", f"review:human ({why})")
    n = int(it.state.get("review_revisions", 0))
    rnd = int(it.state.get("review_round") or 1)
    verdict = read_json(it.p("review", "verdict.json"), {}) or {}
    routed = verdict.get("round") == rnd and (verdict.get("_normalized") or {}).get("verdict") == "revise"
    if it.state.get("post_revision") and n < int(rv.get("max_revisions", 3)) and routed:
        it.state["revision_feedback"] = str(why)[:2000]
        it.state["repair_from"] = f"review:r{rnd}"
        return it.goto("S8", f"review revision retry: {why[:160]}")
    return finish_job_stage(P, it, "dropped", f"review:gate_failed ({why[:300]})")


def _bound_verdict(it, rec, rnd):
    v = read_json(it.p("review", f"r{rnd}", f"{rec.get('job_name')}.verdict.json"), {}) or {}
    if (v.get("job") == rec.get("job_name") and v.get("pkg_sha") == rec.get("pkg_sha")
            and v.get("calib_epoch", 0) == rec.get("calib_epoch", 0) and isinstance(v.get("_normalized"), dict)):
        return v
    return None


def _job_judgment(v, rec):
    reward = v.get("verifier_reward")
    if reward is None and v["_normalized"].get("verdict") is None:
        # Older records did not persist the verifier reward separately. Only reconsider a rejected
        # non-drop verdict if the trial that actually produced it passed its bundled verifier.
        trial = rec.get("trial") or {}
        matched = trial.get("trial") and trial["trial"] == trial.get("verdict_trial")
        reward = trial.get("reward") if matched and trial.get("reward") is not None else 0
    # For older jobs the bundled verifier already checked membership. A reward of one
    # (or its persisted human verdict) is evidence of opt-in under that job's contract.
    legacy_human = reward == 1 or v["_normalized"].get("verdict") == "human"
    return review.normalize(v, reward=reward, allow_human=legacy_human, allowed_verdicts=rec.get("allowed_verdicts"),
                            suspected_scopes=rec.get("suspected_scopes", []), contract=review.job_contract(rec),
                            repros=rec.get("repros"))


def _needs_new_review(rec, verdict, scopes):
    """A valid judgment whose job did not require the evidence now needed."""
    routed = verdict["_normalized"].get("verdict")
    if routed in (None, "drop"):
        return False
    assigned = rec.get("suspected_scopes", [])
    if review.job_contract(rec) == review.CURRENT_CONTRACT and not set(scopes) - set(assigned):
        return False
    return bool(review.finding_errors(verdict, set(assigned) | set(scopes), repros=rec.get("repros")))


def _route_review(P, it, rec: dict, rnd: int):
    """pass -> S10 (kept label) / stays parked (parked label); revise -> S8 review revision while rounds remain
    (review.max_revisions, default 3; after that drop); drop -> dropped.  An invalid verdict (malformed, blocking
    issue without a fix_kind, bound to another job / package / epoch) re-runs the review within the round, up to
    review.infra_retries invalid verdicts; after that the item is dropped (or parked behind review.allow_human)."""
    rv = P.cfg.get("review") or {}
    v = _bound_verdict(it, rec, rnd)
    if not v:
        return _review_invalid(P, it, rnd, rec, f"verdict of job {rec.get('job_name')} missing or bound to "
                                                 f"another job / package / epoch")
    norm = _job_judgment(v, rec)
    if norm["verdict"] not in (None, "drop") and _needs_new_review(rec, dict(v, _normalized=norm), _probe_scopes(it)):
        _save_review(it, dict(rec, outcome="infra", obsolete_probe_review=True))
        return it.goto("S9R", "review contract changed; re-running")
    verdict = norm["verdict"]
    if verdict is None:
        return _review_invalid(P, it, rnd, rec, "; ".join(norm.get("problems") or ["invalid verdict"])[:400])
    if rec.get("outcome") != "done" or rec.get("verdict") != verdict:
        _save_review(it, dict(rec, outcome="done", verdict=verdict))
    # the current verdict (S10's review gate checks round, calibration epoch and package hash)
    write_json(it.p("review", "verdict.json"), dict(v, _normalized=norm, round=rnd,
                                                    calib_epoch=int(rec.get("calib_epoch") or 0),
                                                    pkg_sha=rec.get("pkg_sha"), job=rec.get("job_name")))
    label = it.state.get("calib_label") or ""
    kept = label in P.cfg["calib"]["keep_labels"]
    fp = None
    if label == "frontier" and verdict == "pass":
        fp = review.frontier_problems(norm, _oracle_gate_ok(it))
        kept = not fp
    it.state["review_verdict"] = verdict
    P.store.event("review_verdict", item=it.item_id, round=rnd, verdict=verdict, raw=norm.get("raw_verdict"),
                  blocking=norm.get("blocking"), minor=norm.get("minor"), label=label,
                  problems=(norm.get("problems") or [])[:6])
    if verdict == "pass":
        if kept:
            return it.advance("S9R", "S10", review="pass", round=rnd)
        if fp:
            withdraw(P, it.item_id, f"review r{rnd}: frontier not confirmed")
            P.store.event("frontier_unconfirmed", item=it.item_id, round=rnd, why=fp[:8])
            return finish_job_stage(P, it, "parked", f"frontier:unconfirmed (review r{rnd}: pass; {'; '.join(fp)[:300]})")
        return finish_job_stage(P, it, "parked", f"{label} (review r{rnd}: pass)")
    withdraw(P, it.item_id, f"review r{rnd}: {verdict}")
    if verdict == "human":                  # only reachable behind review.allow_human (explicit opt-in)
        return finish_job_stage(P, it, "parked", f"review:human r{rnd} ({','.join(norm.get('blocking') or [])})")
    if verdict == "revise" and int(it.state.get("review_revisions", 0)) >= int(rv.get("max_revisions", 3)):
        return finish_job_stage(P, it, "dropped", f"review:drop r{rnd} (revision limit {rv.get('max_revisions', 3)}: "
                                    f"{','.join(norm.get('blocking') or [])})")
    if verdict == "revise":
        it.state["repair_from"] = f"review:r{rnd}"
        return it.goto("S8", f"review r{rnd}: revise ({','.join(norm.get('blocking') or [])})")
    return finish_job_stage(P, it, "dropped", f"review:drop r{rnd} ({','.join(norm.get('blocking') or [])})")


def _review_invalid(P, it, rnd: int, rec: dict, why: str) -> dict:
    """An invalid verdict is not a judgment about the task: re-run the review within the round.  After
    review.infra_retries invalid verdicts the item is dropped (review.allow_human parks it for an operator)."""
    rv = P.cfg.get("review") or {}
    if rec.get("outcome") == "done":
        # mark the run itself invalid, so S9R does not route the same verdict again (and the retry budget counts)
        rec = dict(rec, outcome="invalid", invalid_why=why[:300])
        _save_review(it, rec)
    n_bad = sum(1 for r in read_json(it.p("review", "runs.json"), [])
                if r.get("n") == rnd and r.get("outcome") == "invalid" and r.get("job_name") != rec.get("job_name"))
    if n_bad + 1 > int(rv.get("infra_retries", 2)):
        if rv.get("allow_human"):
            return finish_job_stage(P, it, "parked", f"review:human (r{rnd}: {n_bad + 1} invalid verdicts; last: {why[:200]})")
        return finish_job_stage(P, it, "dropped", f"review:invalid r{rnd} ({n_bad + 1} invalid verdicts; last: {why[:200]})")
    P.store.event("review_verdict_invalid", item=it.item_id, round=rnd, job=rec.get("job_name"), why=why[:400])
    return it.goto("S9R", f"review r{rnd} verdict invalid ({why[:160]}); re-running")


def _move_records(src: Path, dst: Path) -> None:
    """Move the calibration record files of src into dst; never overwrite (a clash keeps both, suffixed)."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name != ".lock":
            t = dst / p.name
            k = 1
            while t.exists():
                t = dst / f"{p.name}.{k}"
                k += 1
            shutil.move(str(p), str(t))


def _archive_calibration(it) -> int:
    """Close the current calibration epoch: its records move to calib/epoch-<k>/ (job dirs stay where they are);
    the next S9 starts from GLM with job names suffixed -e<k+1>."""
    e = int(it.state.get("calib_epoch", 0))
    d = it.p("calib", f"epoch-{e}")
    d.mkdir(parents=True, exist_ok=True)
    _move_records(it.p("calib"), d)
    it.state.setdefault("calib_history", []).append(dict(
        epoch=e, label=it.state.get("calib_label"), review_verdict=it.state.get("review_verdict"),
        review_round=it.state.get("review_round"), archived=str(d), ts=time.time()))
    it.state["calib_epoch"] = e + 1
    it.state.pop("calib_label", None)
    return e + 1


def _s8_review(P, it):
    """S8 for a review verdict "revise": one local edit of the statement / format notes (via the generator) /
    test and scoring scripts.  Thresholds and the reference are locked (review.revision_review; S6B keeps
    thresholds.json; the regenerated data must hash the same as before apart from README* notes).  A revision that
    changes only the grader (tests/test_outputs.py, tests/measure.py) is scoring-only: the calibration is kept and
    re-scored with the new grader at S9 (review.rescore_pending) instead of being archived; anything else archives
    it and goes back to S6A_run (re-measure) -> S6B (locked) -> S6C -> S7 -> PKG -> S9 (from GLM) -> S9R.
    Crash-safe: an accepted patch is saved before it is applied, and a pending revision is completed (idempotently)
    on resume without another model call."""
    with item_lock(it):
        it.load()
        if it.state.get("stage") != "S8" or it.state.get("status") != "active":
            return None
        return _s8_review_locked_(P, it)


def _s8_review_locked_(P, it):
    rv = P.cfg.get("review") or {}
    if it.state.get("review_revision_pending"):
        return _finish_revision(P, it, it.state["review_revision_pending"])
    n = int(it.state.get("review_revisions", 0))
    if n >= int(rv.get("max_revisions", 3)):
        if rv.get("allow_human"):
            return it.finish("parked", f"review:human (revision limit {n})")
        return it.finish("dropped", f"review:drop (revision limit {n}: {rv.get('max_revisions', 3)})")
    rnd = int(it.state.get("review_round") or 1)
    verdict = read_json(it.p("review", "verdict.json"), {}) or {}        # the routed verdict (bound in S9R)
    norm = verdict.get("_normalized") or {}
    # the package hash may differ from the reviewed one after a failed gate forced a rebuild (S6B/S7/PKG checks);
    # the round and the routed revise verdict still bind the revision, and PKG checks the data hash
    if verdict.get("round") != rnd or norm.get("verdict") != "revise":
        return it.finish("parked", f"review:error (no routed revise verdict for r{rnd} and the current package)")
    issues = [i for i in norm["issues"] if i["severity"] == "blocking"] + \
             [i for i in norm["issues"] if i["severity"] == "minor"]
    feedback = it.state.pop("revision_feedback", None)
    if feedback:
        issues = [dict(id="IF", category="c", severity="blocking", fix_kind="local", title="the last revision failed",
                       evidence=[feedback], fix="fix the failure it caused (see the S6B/S7/PKG records)")] + issues
    spec = read_json(it.p("_spec", "spec.json"))
    task = it.p("task")
    th = read_json(it.p("thresholds.json"))["thresholds"]
    user = S8_REVIEW.format(issues=json.dumps(issues, indent=1, ensure_ascii=False)[:30000],
                            thresholds=json.dumps({k: {"direction": v["direction"], "threshold": v["threshold"]}
                                                   for k, v in th.items()}),
                            files=review.files_for_revision(it.dir), allowed=", ".join(review.REVISION_PATHS)
                                                                     + ", baselines/exploit_*.py, "
                                                                     + review.GATING_PATH,
                            gating=json.dumps(bool(spec.get("resource_metric_gating"))))
    r = P.client("repair").chat(SYSTEM, user, stage="S8", item=it.item_id,
                                max_tokens=P.cfg["gen"].get("s6a_max_tokens"))
    save_raw(it, "S8", r)
    patch = parse_files(r.text, truncated=r.stop_reason == "max_tokens")
    chk = review.revision_review(task, patch, spec["depgraph"], th)
    d = it.p("repair", f"review-{n + 1}")
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "patch.json", patch)                  # never written as files before the paths are checked
    write_json(d / "review.json", dict(chk, review_round=rnd, issues=[i.get("id") for i in issues],
                                       gate_feedback=feedback))
    if not chk["ok"] or not patch:
        # a rejected patch spends its revision round: retry while rounds remain, else drop (allow_human parks)
        it.state["review_revisions"] = n + 1
        it.state["revision_feedback"] = "the last revision patch was rejected: " + \
            "; ".join(chk["reasons"] or ["empty patch"])[:400]
        it.state["repair_from"] = f"review:r{rnd}"
        it.save()
        if n + 1 >= int(rv.get("max_revisions", 3)):
            if rv.get("allow_human"):
                return it.finish("parked", "review:human (revision rejected: " +
                                 "; ".join(chk["reasons"] or ["empty patch"])[:300] + ")")
            return it.finish("dropped", "review:drop (revision rejected: " +
                             "; ".join(chk["reasons"] or ["empty patch"])[:300] + ")")
        return it.goto("S8", f"review r{rnd}: revision rejected, retrying")
    for rel, body in patch.items():                      # paths are in review.REVISION_PATHS here
        (d / "patch" / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / "patch" / rel).write_text(body)
        if (task / rel).exists():
            (d / "before" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(task / rel, d / "before" / rel)
    shutil.copy2(it.p("thresholds.json"), d / "thresholds.locked.json")
    # the reviewed package's data / hidden inputs / truth: the revised package must carry the same (PKG checks)
    # a retry after a failed revision (thresholds still locked) keeps the hash recorded before the first attempt:
    # pkg/ may already hold the rejected attempt's data, which must never become the new baseline
    data = (it.state.get("revision_data_hash") if it.state.get("thresholds_locked") else None) or \
        review.pkg_data_hash(it.p("pkg"))
    rescore = bool(rv.get("rescore_scoring_only", True)) and review.scoring_only(patch)
    pend = dict(n=n + 1, dir=str(d), round=rnd, epoch=int(it.state.get("calib_epoch", 0)) + 1, data_hash=data,
                archive=not rescore, files=sorted(patch))
    it.state["review_revision_pending"] = pend
    it.save()
    return _finish_revision(P, it, pend)


def _finish_revision(P, it, pend: dict):
    d = Path(pend["dir"])
    patch = read_json(d / "patch.json", {}) or {}
    gating = patch.pop(review.GATING_PATH, None)
    if gating is not None:                              # spec flag, never a task file (idempotent)
        spec = read_json(it.p("_spec", "spec.json"))
        spec["resource_metric_gating"] = gating.strip().lower() == "true"
        write_json(it.p("_spec", "spec.json"), spec)
    repair.apply(it.p("task"), patch)                   # idempotent (whole-file writes)
    if pend.get("archive", True):
        if int(it.state.get("calib_epoch", 0)) < pend["epoch"]:
            _archive_calibration(it)
        else:                                           # crashed after the epoch moved: finish moving the files
            _move_records(it.p("calib"), it.p("calib", f"epoch-{pend['epoch'] - 1}"))
    elif not it.state.get("rescore_pending"):
        # scoring-only revision: the statement, data, thresholds and reference are untouched, so the calibration
        # stands until S9 re-scores the saved deliverables with the new grader (review.rescore_item)
        it.state["rescore_pending"] = dict(reason="scoring-only revision", round=pend["round"], n=pend["n"],
                                           files=pend.get("files") or sorted(patch), ts=time.time())
    it.state["thresholds_locked"] = True
    it.state["review_revisions"] = pend["n"]
    it.state["revision_data_hash"] = pend.get("data_hash")
    it.state["post_revision"] = True
    it.state.pop("repair_from", None)
    it.state.pop("review_revision_pending", None)
    P.store.event("review_revision_applied", item=it.item_id, round=pend["round"], files=sorted(patch),
                  calib_epoch=it.state.get("calib_epoch", 0), scoring_only=not pend.get("archive", True))
    it.goto("S6A_run", f"review revision {pend['n']} (r{pend['round']})")      # one atomic save


# ------------------------------------------------------------------ hardening loop (difficulty.py)
def _difficulty_summary(it) -> dict | None:
    d = read_json(it.p("_spec", difficulty.DESIGN_FILE), None)
    prov = read_json(it.p("_spec", "provenance.json"), {}) or {}
    if not isinstance(d, dict) and not it.state.get("harden_round"):
        return None
    return dict(profile=prov.get("difficulty_profile"),
                weakness_ids=[lv.get("weakness_id") for lv in (d or {}).get("levers") or [] if isinstance(lv, dict)],
                harden_round=int(it.state.get("harden_round", 0)), harden_kind=it.state.get("harden_kind"),
                harden_reverted=it.state.get("harden_reverted"),
                variant_round=int(it.state.get("variant_round", 0) or 0),
                variant_exhausted=bool(it.state.get("variant_exhausted")))


def _first_student_pass(runs: list[dict], cfg: dict) -> dict | None:
    real = difficulty.decisive(runs, cfg) or []
    return next((r for r in real if r["who"] == "glm" and r["outcome"] == "pass"), None)


def _last_student_pass(runs: list[dict], cfg: dict) -> dict | None:
    """The last decisive student pass (an S9b variant round: the run that made the version easy)."""
    real = difficulty.decisive(runs, cfg) or []
    return next((r for r in reversed(real) if r["who"] == "glm" and r["outcome"] == "pass"), None)


def _s5g_base(it, k: int) -> dict:
    """S5G spend and job count before hardening round k (the round's own allowance starts from here)."""
    charges = dict(read_json(it.p("S5G", "spent.json"), {}) or {})
    rec = read_json(it.p("S5G", "run.json"), None) or {}
    if rec.get("job_name") and rec["job_name"] not in charges and rec.get("launch_attempted"):
        b = rec.get("budget") or {}
        usd = rec.get("cost_usd")
        charges[rec["job_name"]] = dict(usd=float(usd) if usd is not None else float(b.get("max_usd") or 0.0),
                                        sec=60.0 * float(rec["minutes"]) if rec.get("minutes") is not None else 0.0)
    from .stages_a import _jobs_used
    return dict(round=k, usd=sum(c["usd"] for c in charges.values()), sec=sum(c["sec"] for c in charges.values()),
                jobs=_jobs_used(it, "S5G"))


def start_hardening(P, it, operator: str | None = None, kind: str = "harden"):
    """Send the current version back to S5G for a hardening round (kind="harden": the student solved it on its
    first attempt; kind="variant": S9b, the student passed both decisive runs and the item was labelled easy).
    Crash-safe and idempotent: the round number, the kind, the trigger and the epoch are persisted
    (state.harden_pending) before anything moves; an item with a pending round is scheduled whatever its stage /
    status (pipeline.runnable) and finishes the same round without another model call.  Steps: feedback the author
    may see (harden/round-k/feedback.json: the student's own summary, minutes, per-metric aggregate margins),
    snapshot of the version (harden/round-k/prev), round record, release withdrawal, archive of this calibration
    epoch (never pooled with the next version), reset of the per-version build state, -> S5G (a new authoring job
    with the previous version and the feedback; its own allowance)."""
    st = it.state
    pend = st.get("harden_pending")
    if not pend:
        runs = read_json(it.p("calib", "runs.json"), []) or []
        # a variant round feeds the author the decisive run that solved it; a hardening round its first attempt
        trig = _last_student_pass(runs, P.cfg) if kind == difficulty.KIND_VARIANT else _first_student_pass(runs, P.cfg)
        if trig is None:
            raise ValueError("hardening needs a passing student run in the current calibration")
        k = int(st.get("harden_round", 0)) + 1
        th = (read_json(it.p("thresholds.json"), {}) or {}).get("thresholds") or {}
        rd = it.p("harden", f"round-{k}")
        rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / "feedback.json", difficulty.feedback(trig, th))
        pend = dict(round=k, kind=kind, trigger=trig["job_name"], epoch=int(st.get("calib_epoch", 0)),
                    operator=operator, from_stage=st.get("stage"), from_status=st.get("status"),
                    base=_s5g_base(it, k), ts=time.time())
        st["harden_pending"] = pend
        it.save()
    kind = str(pend.get("kind") or "harden")
    k = int(pend["round"])
    rd = it.p("harden", f"round-{k}")
    difficulty.snapshot_version(it, rd / "prev")
    rec = read_json(rd / "record.json", {}) or {}
    if "prev_task_sha" not in rec:                     # written once, before the epoch is archived
        runs = read_json(it.p("calib", "runs.json"), []) or []
        difficulty.write_round(
            it, k, round=k, kind=kind, trigger=pend["trigger"], operator=pend.get("operator"),
            prev_review_round=int(st.get("review_round") or 0),    # reviews above this round belong to the new build
            prev_task_sha=difficulty.task_sha(it.p("task")), prev_science_sha=difficulty.science_sha(it.p("task")),
            prev_pkg_sha=calib.pkg_sha(it.p("pkg")) if it.p("pkg").exists() else None,
            prev_gates_attempt=passed_gate_attempt(st),
            prev_label=st.get("calib_label"), archived_epoch=pend["epoch"],
            prev_runs=[{x: r.get(x) for x in ("job_name", "who", "outcome", "minutes", "cost_usd")} for r in runs],
            started=time.time())
    withdraw(P, it.item_id, f"hardening round {k}")     # no-op unless released
    if int(st.get("calib_epoch", 0)) == int(pend["epoch"]):
        _archive_calibration(it)                         # only while the epoch is still current (idempotent)
    for p in it.p("calib").glob("prepare*.json") if it.p("calib").exists() else []:
        p.unlink()                                       # the prepared sandbox copy is of the old package
    for d in ("gates", "runs"):                         # the previous version's copies are in round-k/prev
        if it.p(d).exists():
            shutil.rmtree(it.p(d))
    for f in difficulty.STALE_FILES:
        if it.p(f).is_file():
            it.p(f).unlink()
    for key in ("thresholds_locked", "repair_rounds", "back_to_spec", "ref_rewrite", "avoid_mid_drop", "regen_6a",
                "s6a_feedback", "repair_from", "calib_label", "rescore_pending", "post_revision",
                "post_regate", "difficulty_reauthored", "harden_built", "harden_pending"):
        st.pop(key, None)
    st.setdefault("attempts", {}).pop("S7", None)
    st["gates_attempt"] = None
    st["regen_6c"] = True
    st["harden_round"] = k
    st["harden_kind"] = kind
    st["harden_active"] = True
    st["harden_base"] = pend["base"]
    st["S5G_rerun"] = True
    if kind == difficulty.KIND_VARIANT:
        st["variant_round"] = int(st.get("variant_round", 0) or 0) + 1   # its own counter (state.harden_round is
        # the shared attempt index that names harden/round-k); budget tagging and the depth check use this one
    st.setdefault("history", []).append(dict(stage=st.get("stage"), result=kind, to="S5G", round=k,
                                             trigger=pend["trigger"], ts=time.time()))
    st.update(stage="S5G", status="active")
    st.pop("reason", None)
    it.save()
    P.store.event("harden_start", item=it.item_id, round=k, rkind=kind, trigger=pend["trigger"],
                  operator=pend.get("operator"))
    return None


HARDEN_REVERT_STAGES = {"S5", "S5G", "S6A", "S6A_run", "S6B", "S6M", "S6C", "S7", "S8", "PKG"}


def _review_of_version(it) -> list[dict]:
    """Review runs of the version a round is building (a round that reached PKG and failed): those of a review round
    above the one the replaced version had (the round record's prev_review_round).  The replaced version's own
    reviews, and any earlier version's, never block its revert.  Legacy round records (no prev_review_round) fall
    back to the package hash."""
    k = int(it.state.get("harden_round", 0) or 0)
    rec = (read_json(it.p("harden", f"round-{k}", "record.json"), {}) or {}) if k else {}
    prev = rec.get("prev_review_round")
    if prev is None:
        return review.runs_of_version(it.dir, it.state)
    runs = read_json(it.p("review", "runs.json"), []) or []
    return [r for r in runs if int(r.get("n") or 0) > int(prev)]


def harden_needs_step(it) -> bool:
    """An unfinished hardening / S9b variant transaction (start or revert) that must be completed whatever the
    item's status.  Same eligibility as harden_revert (a version that never reached S9: no review runs of it)."""
    st = it.state
    if st.get("harden_pending") or st.get("harden_revert_pending"):
        return True
    return bool(st.get("harden_active") and st.get("status") in ("dropped", "parked") and
                st.get("stage") in HARDEN_REVERT_STAGES and not _review_of_version(it))


def harden_step(P, it) -> None:
    """Complete a pending hardening start or revert (pipeline.step calls this before the stage handler)."""
    with item_lock(it):
        it.load()
        if it.state.get("harden_pending"):
            start_hardening(P, it)
        elif harden_needs_step(it):
            harden_revert(P, it, locked=True)


def harden_revert(P, it, locked: bool = False) -> bool:
    """A hardened version that could not be built (dropped / parked before S9, never reviewed): put the previous
    version back (files from harden/round-k/prev, calibration records from its archived epoch), stop hardening
    this item and continue its calibration under the original rule.  The failed version's files go to
    harden/round-k/failed/; its own calibration records (if any) stay archived.  A replayable transaction:
    state.harden_revert_pending fixes the round and both epochs before anything moves, every step is idempotent."""
    if not locked:
        with item_lock(it):
            it.load()
            return harden_revert(P, it, locked=True)
    st = it.state
    pend = st.get("harden_revert_pending")
    if not pend:
        if not st.get("harden_active") or st.get("status") not in ("dropped", "parked") or \
                st.get("stage") not in HARDEN_REVERT_STAGES:
            return False
        k = int(st.get("harden_round", 0))
        rd = it.p("harden", f"round-{k}")
        rec = read_json(rd / "record.json", {}) or {}
        if not (rd / "prev" / "DONE").exists() or rec.get("archived_epoch") is None or _review_of_version(it):
            return False
        pend = dict(round=k, kind=difficulty.round_kind(it), why=f"{st.get('stage')}:{st.get('reason')}",
                    prev_epoch=int(rec["archived_epoch"]), failed_epoch=int(st.get("calib_epoch", 0)), ts=time.time(),
                    infra=difficulty.revert_infra(it, st.get("stage"), st.get("reason")))
        st["harden_revert_pending"] = pend
        it.save()
    kind = str(pend.get("kind") or "harden")
    k = int(pend["round"])
    rd = it.p("harden", f"round-{k}")
    difficulty.restore_version(it, rd / "prev", rd / "failed")
    record = read_json(rd / "record.json", {})
    if "prev_gates_attempt" in record:
        st["gates_attempt"] = record["prev_gates_attempt"]
    else:
        st.pop("gates_attempt", None)  # An older snapshot uses its passing history entry.
    if not pend.get("failed_archived"):
        if int(st.get("calib_epoch", 0)) == pend["failed_epoch"]:
            _archive_calibration(it)                    # the failed version's own records (normally none)
        pend["failed_archived"] = True
        it.save()
    _move_records(it.p("calib", f"epoch-{pend['prev_epoch']}"), it.p("calib"))
    for p in it.p("calib").glob("prepare*.json"):
        p.unlink()                                       # a prepared sandbox copy may be of the hardened version
    for key in ("harden_active", "calib_label", "thresholds_locked", "repair_from", "rescore_pending", "harden_base",
                "harden_built", "regen_6c", "harden_revert_pending", "harden_kind"):
        st.pop(key, None)
    infra = pend.get("infra")
    if not infra:
        # a real round failed: no more rounds of this kind, the original rule decides (S9b: label easy, exhausted)
        st["variant_exhausted" if kind == difficulty.KIND_VARIANT else "harden_exhausted"] = True
    # an infra revert does not use up the depth; the attempts cap bounds the retries (rounds_left_kind)
    if not any(r.get("round") == k for r in st.get("harden_reverted") or []):
        st.setdefault("harden_reverted", []).append(dict(round=k, kind=kind, why=pend["why"], infra=infra,
                                                         ts=time.time()))
    st.setdefault("history", []).append(dict(stage=st.get("stage"), result="harden_revert", to="S9", round=k,
                                             reason=pend["why"], infra=infra, kind=kind, ts=time.time()))
    st.update(stage="S9", status="active")
    st.pop("reason", None)
    it.save()
    difficulty.write_round(it, k, reverted=dict(why=pend["why"], infra=infra, ts=time.time()))
    P.store.event("harden_reverted", item=it.item_id, round=k, rkind=kind, why=pend["why"], infra=infra,
                  rounds_used=difficulty.rounds_used_kind(it, kind),
                  attempts=difficulty.attempts_used(it, kind))
    return True


def harden_authored(P, it) -> None:
    """S5G finished a hardening round: record the new version's cost in the round record."""
    k = int(it.state.get("harden_round", 0))
    if not k or not it.state.get("harden_active"):
        return
    r = read_json(it.p("S5G_result.json"), {}) or {}
    rec = read_json(it.p("harden", f"round-{k}", "record.json"), {}) or {}
    jobs = list(rec.get("author_jobs") or [])
    if r.get("job") not in [j.get("job") for j in jobs]:
        jobs.append(dict(job=r.get("job"), cost_usd=r.get("cost_usd"), minutes=r.get("minutes")))
    it.state["harden_built"] = True
    difficulty.write_round(it, k, author_jobs=jobs, author_cost_usd=round(sum(float(j.get("cost_usd") or 0)
                                                                              for j in jobs), 4),
                           new_task_sha=difficulty.task_sha(it.p("task")), authored=time.time())
    P.store.event("harden_authored", item=it.item_id, round=k, rkind=difficulty.round_kind(it),
                  job=r.get("job"),
                  cost_usd=r.get("cost_usd"))


def harden_cmd(P, it, reason: str) -> dict:
    """`tbsgen harden --item ID`: start a hardening round for an item whose current calibration has a passing
    student run (e.g. one already labelled easy before the loop existed).  The calibration epoch is archived."""
    with item_lock(it):
        it.load()
        st = it.state
        if st.get("harden_pending"):                 # an interrupted start: finish the same round
            start_hardening(P, it)
            return dict(item=it.item_id, stage=it.state["stage"], round=it.state.get("harden_round"), resumed=True)
        if st.get("stage") not in ("S9", "S9R", "S10") or st.get("status") not in ("active", "done", "waiting",
                                                                                     "parked"):
            raise ValueError(f"harden: item at {st.get('stage')}/{st.get('status')}")
        if not difficulty.harden_enabled(P.cfg, it):
            raise ValueError("harden: not a code-grounded item, or difficulty.harden is off / excludes it")
        if not difficulty.rounds_left(P.cfg, it):
            raise ValueError("harden: no hardening rounds left")
        runs = read_json(it.p("calib", "runs.json"), []) or []
        if any(r.get("outcome") is None for r in runs):
            raise ValueError("harden: a calibration run is still pending")
        if st.get("rescore_pending") or st.get("thresholds_locked") or st.get("review_revision_pending"):
            raise ValueError("harden: finish the pending re-score / revision / regate first")
        rv = read_json(it.p("review", "runs.json"), []) or []
        if rv:
            # review job names follow the review round; a reviewed item would reuse them after hardening
            raise ValueError("harden: the item already has S9R review runs (not supported)")
        if _first_student_pass(runs, P.cfg) is None:
            raise ValueError("harden: no passing student run in the current calibration")
        start_hardening(P, it, operator=reason)
    return dict(item=it.item_id, stage=it.state["stage"], round=it.state.get("harden_round"))


def variant_cmd(P, it, reason: str) -> dict:
    """`tbsgen variant --item ID`: start an S9b variant round on an item the calibration labelled easy (the student
    passed every decisive run), e.g. a version finished `label:easy` before S9b existed.  Idempotent: an interrupted
    start is finished, a variant round already in progress is reported as such (state.harden_active + kind).  Works
    for an S9R-reviewed item: the new version gets a fresh review round (review.next_round at its label), so its
    review job names and verdicts never collide with the earlier version's."""
    with item_lock(it):
        it.load()
        st = it.state
        if st.get("harden_pending") or st.get("harden_active"):
            if not st.get("harden_pending") and difficulty.round_kind(it) != difficulty.KIND_VARIANT:
                raise ValueError("variant: a first-pass hardening round is in progress")
            if st.get("harden_pending"):
                start_hardening(P, it)               # finish an interrupted start (kind is in the pending record)
            if st.get("variant_reopened"):
                return dict(item=it.item_id, stage=it.state["stage"], round=it.state.get("harden_round"),
                            kind="variant", variant_round=it.state.get("variant_round"),
                            skipped="already re-opened for S9b")
            return dict(item=it.item_id, stage=it.state["stage"], round=it.state.get("harden_round"), kind="variant",
                        resumed=True, variant_round=it.state.get("variant_round"))
        if st.get("stage") not in ("S9", "S9R", "S10") or st.get("status") not in ("active", "done", "waiting",
                                                                                "parked"):
            raise ValueError(f"variant: item at {st.get('stage')}/{st.get('status')}")
        if not difficulty.variant_enabled(P.cfg, it):
            raise ValueError("variant: S9b off (calib.variant_depth), not code-grounded, or excluded")
        if not difficulty.variant_rounds_left(P.cfg, it):
            raise ValueError("variant: no S9b rounds left")
        if st.get("variant_reopened"):
            return dict(item=it.item_id, stage=st.get("stage"), round=it.state.get("harden_round"),
                        variant_round=it.state.get("variant_round"), skipped="already re-opened for S9b")
        runs = read_json(it.p("calib", "runs.json"), []) or []
        if any(r.get("outcome") is None for r in runs):
            raise ValueError("variant: a calibration run is still pending")
        if st.get("rescore_pending") or st.get("thresholds_locked") or st.get("review_revision_pending"):
            raise ValueError("variant: finish the pending re-score / revision / regate first")
        dec = difficulty.decisive(runs, P.cfg) or []
        act, what = calib.next_action(runs, P.cfg)
        if (act, what) != ("label", "easy"):
            raise ValueError(f"variant: the current calibration is not easy ({act}: {what}); "
                             "a version needs both decisive student passes (or a quick pass)")
        glm = [r for r in dec if r.get("who") == "glm"]
        if not glm or any(r.get("outcome") != "pass" for r in glm):
            raise ValueError("variant: the student did not pass every decisive run (not an easy version)")
        start_hardening(P, it, operator=reason, kind=difficulty.KIND_VARIANT)
        it.state["variant_reopened"] = True
        it.save()
        P.store.event("s9b_reopen", item=it.item_id, round=it.state.get("harden_round"),
                      variant_round=it.state.get("variant_round"), reason=reason)
    return dict(item=it.item_id, stage=it.state["stage"], round=it.state.get("harden_round"),
                variant_round=it.state.get("variant_round"), kind=difficulty.KIND_VARIANT)
