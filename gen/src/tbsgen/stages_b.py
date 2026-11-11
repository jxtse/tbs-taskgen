"""Stage handlers S6C, S7, S8, PKG, S9, S9R, S10."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import shutil

from . import calib, gates, package, repair, review
from .models import parse_files
from .prompts import METRICS_API, S6C_TESTS, S8_EXPLOIT, S8_REPAIR, S8_REVIEW, SYSTEM
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
    if str(it.state.get("repair_from", "")).startswith("review:") or it.state.get("review_revision_pending"):
        return _s8_review(P, it)
    if it.state.get("repair_from") == "gates:exploit_valid":
        return _s8_exploits(P, it)
    if it.state.get("thresholds_locked"):
        # a gate failed after a review revision or a regate: the ordinary S8 may change the reference / baselines
        # and is not bound by the revision limit, so it must not run — the review revision is retried (feedback)
        # while rounds remain, else the item is dropped (review.allow_human restores the old park)
        why = f"after the review revision: {it.state.get('repair_from')}"
        last = sorted(it.p("gates").glob("attempt-*.json"),
                      key=lambda q: int(q.stem.split("-")[-1]) if q.stem.split("-")[-1].isdigit() else -1) \
            if it.p("gates").exists() else []
        oc = ((read_json(last[-1], {}) or {}).get("details") or {}).get("output_contract") if last else None
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
        st = it.state
        if st.get("stage") not in ("S9", "S9R", "S10") and st.get("status") not in ("done", "parked"):
            raise ValueError(f"regate: item at {st.get('stage')}/{st.get('status')} is not packaged")
        if not it.p("pkg").exists() or not it.p("thresholds.json").exists():
            raise ValueError("regate: no pkg/ or thresholds.json")
        if any(r.get("outcome") is None for r in read_json(it.p("calib", "runs.json"), []) or []):
            raise ValueError("regate: a calibration run is still pending")
        old_sha = calib.pkg_sha(it.p("pkg"))
        # withdraw first (idempotent): the package must be out of the manifest before the item can run again
        # and overwrite pkg/; a crash after this line leaves a withdrawn, still finished item -> regate again
        withdrawn = withdraw(P, it.item_id, f"regate: {reason}")
        rec = dict(ts=time.time(), reason=reason, from_stage=st.get("stage"), from_status=st.get("status"),
                   pkg_sha=old_sha, calib_epoch=st.get("calib_epoch", 0), label=st.get("calib_label"))
        st.setdefault("regates", []).append(rec)
        st["thresholds_locked"] = True
        st["revision_data_hash"] = review.pkg_data_hash(it.p("pkg"))
        st["post_regate"] = True
        if recalibrate or not (P.cfg.get("review") or {}).get("rescore_scoring_only", True):
            # re-scoring switched off: the kept calibration cannot be checked against the new grader -> redo it
            rec["archived_epoch"] = _archive_calibration(it)
            st.pop("calib_label", None)
        elif not st.get("rescore_pending"):
            # the shared verifier template changed: the kept calibration stands until S9 re-scores the saved
            # deliverables with the new grader (review.rescore_item), as the reference re-measurement does
            st["rescore_pending"] = dict(reason="regate", files=["tests/_metrics.py", "tests/test.sh"], ts=time.time())
        st.setdefault("history", []).append(dict(stage=st.get("stage"), result="regate", to="S6A_run", reason=reason,
                                                 ts=time.time()))
        st.update(stage="S6A_run", status="active")
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
    last = sorted(it.p("gates").glob("attempt-*.json"))
    gate_log = read_json(last[-1]) if last else {}
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
        out["details"] = {k: det[k] for k in ("determinism", "unstated", "isolation", "runs", "truth_isolation")
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
    import hashlib
    if it.state.get("rescore_pending") and not it.state.get("pkg_statement_sha"):
        # persisted before pkg/ is overwritten, so a crash between overwrite and archive still compares correctly
        try:
            it.state["pkg_statement_sha"] = hashlib.sha256(it.p("pkg", "instruction.md").read_bytes()).hexdigest()
            it.save()
        except OSError:
            pass
    old_sha = it.state.get("pkg_statement_sha")
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
    rm = prerelease.readymade_check(out, cand, spec)
    if not rm["ok"]:
        v = dict(v, ok=False, errors=list(v.get("errors") or []) + [f"readymade: {rm['findings'][:3]}"])
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
    it.advance("PKG", "S9")


def _revision_pkg_problem(it, gate: dict) -> str | None:
    """After a review revision the new package must carry the reviewed package's data / hidden inputs / truth
    (README notes aside), and the verifier must have applied exactly the locked thresholds at run time, in every
    arm's metrics.jsonl of the passing S7 attempt (an arm that stopped before any check may log none)."""
    want = it.state.get("revision_data_hash")
    if want and review.pkg_data_hash(it.p("pkg")) != want:
        return "the revision changed packaged data / hidden inputs / truth; only README format notes may change"
    locked = review.locked_signature(read_json(it.p("thresholds.json"))["thresholds"])
    logs = sorted(it.p("runs", f"s7-{gate['attempt']}").glob("logs-*/metrics.jsonl"))
    if not any(m.parent.name.startswith("logs-ref") for m in logs):
        return "no reference verifier log of the S7 attempt to check the thresholds applied at run time"
    for m in logs:                        # every arm (ref, mid, naive, exploits, nop): what check() applied
        got = review.runtime_thresholds(m)
        if got and got != locked:
            return f"the verifier applied thresholds {got} != locked {locked} ({m.parent.name})"
    return None


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


def _rescore_pending(P, it):
    """S9 entry with a pending re-score (a scoring-only review revision, or a shared-verifier regate): re-run the
    packaged verifier on every decisive run's saved deliverable (calib/deliverables/<job>/; a run that produced no
    deliverable at all keeps its fail outcome — the S7 nop gate proves the new grader fails an empty output).  All
    outcomes unchanged -> the calibration stands (S9 relabels from the kept runs).  Any outcome changed, or a
    decisive run without a saved deliverable -> archive the epoch and recalibrate from GLM.  A sandbox failure
    waits (infrastructure, not an attempt)."""
    pend = it.state.get("rescore_pending") or {}
    runs = read_json(it.p("calib", "runs.json"), []) or []
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


def running_jobs(P) -> list[str]:
    """Harbor jobs holding a calib.max_concurrent_jobs slot: records of the item's current stage (S9 -> calibration
    runs, S9R -> review runs; the stage's watcher is what settles them) without an outcome that are launched,
    reserved (about to launch / recovering), or whose job already exists."""
    out = []
    for it in P.store.items():
        st = it.state.get("stage")
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
        if review.reviewable(P.cfg, what):
            # S9R: kept labels must pass the review before S10; parked:teacher_failed is reviewed as a diagnosis
            it.state["review_round"] = int(it.state.get("review_revisions", 0)) + 1
            return it.goto("S9R", f"label:{what}")
        if what.startswith("parked"):
            return it.finish("parked", what)
        if what in c["keep_labels"]:
            return it.advance("S9", "S10", label=what)
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
        if it.state.get("stage") == "S9R":
            return _ingest_review(P, it, job_dir)
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
    it.state["stage"], it.state["status"] = "S9", "active"
    it.save()
    return t


def review_gate(P, it) -> str | None:
    """Why S10 must not release this item yet (None = may release): with review on and a reviewable label, the
    current package, calibration epoch and review round need a recorded pass."""
    if not review.reviewable(P.cfg, it.state.get("calib_label")):
        return None
    v = read_json(it.p("review", "verdict.json"), None)
    if not v:
        return "no review verdict"
    norm = v.get("_normalized") or {}
    if norm.get("verdict") != "pass":
        return f"review verdict {norm.get('verdict')}"
    if v.get("round") != it.state.get("review_round") or v.get("calib_epoch", 0) != it.state.get("calib_epoch", 0):
        return "review verdict of another round / calibration epoch"
    if v.get("pkg_sha") != calib.pkg_sha(it.p("pkg")):
        return "package changed after the review"
    return None


def s10(P, it):
    why = review_gate(P, it)
    if why:
        it.state["review_round"] = int(it.state.get("review_revisions", 0)) + 1
        return it.goto("S9R", f"S10 review gate: {why}")
    rel = P.store.root / "release"
    rel.mkdir(exist_ok=True)
    with (rel / "manifest.jsonl").open("a") as f:
        runs = read_json(it.p("calib", "runs.json"), [])
        summ = [x for x in calib.run_summary(runs) if x["outcome"] is not None]
        rv = read_json(it.p("review", "verdict.json"), None)
        nv = (rv or {}).get("_normalized") or {}
        rev = None if rv is None else dict(
            verdict=nv.get("verdict"), round=it.state.get("review_round"), revisions=it.state.get("review_revisions", 0),
            minor=[dict(id=i.get("id"), category=i.get("category"), title=i.get("title"))
                   for i in nv.get("issues") or [] if i.get("severity") == "minor"],
            label_trustworthy=(nv.get("label_assessment") or {}).get("trustworthy"))
        f.write(json.dumps(dict(item_id=it.item_id, pkg=str(it.p("pkg")), label=it.state.get("calib_label"),
                                calib_runs=summ, result_types=[x["result_type"] for x in summ],
                                calib_epoch=it.state.get("calib_epoch", 0), review=rev,
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
    return not any(r.get("outcome") is None and r.get("state", "launched") == "launched" for r in runs)


def watch_tick(P, only: list[str] | None = None, max_steps: int = 50) -> dict:
    """One pass of `tbsgen calib watch`: attach every finished Harbor job of an S9-waiting item (calib-ingest),
    flag launchers that died without a job, then step the S9 items (label, or launch the next run: GLM ->
    teacher).  Returns what it did and the S9 items still open."""
    did = []
    for it in P.store.items():
        if only and it.item_id not in only:
            continue
        if it.state.get("stage") not in ("S9", "S9R") or it.state.get("status") != "waiting":
            continue
        rf = ("calib", "runs.json") if it.state["stage"] == "S9" else ("review", "runs.json")
        runs = read_json(it.p(*rf), [])
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
                runs = read_json(it.p(*rf), [])
                cur = next((r for r in runs if r.get("job_name") == rec["job_name"]), {})
                if it.state.get("stage") != rf_stage(rf) or it.state.get("status") != "waiting" or \
                        cur.get("outcome") is not None or cur.get("state") != "launched":
                    continue
                kind = "launcher_failed" if calib.launcher_failed(cur, c) else \
                    "job_died" if calib.launcher_died(cur, c) else None
                if kind is None:
                    continue
                it.state["status"] = "active"       # s9() / s9r() reconcile the dead launch
                it.save()
            did.append(dict(item=it.item_id, action=kind, job=rec["job_name"]))
    s9 = [i.item_id for i in P.store.items() if (not only or i.item_id in only) and
          (i.state.get("stage") in ("S9", "S9R") or (_in_revision(i) and i.state.get("status") == "active"))]
    if s9:
        P.run(max_steps=max_steps, only=s9)
    open_ = {}
    for i in P.store.items():
        if (not only or i.item_id in only) and i.state.get("status") in ("waiting", "active") and \
                (i.state.get("stage") in ("S9", "S9R") or _in_revision(i)):
            open_[i.item_id] = dict(stage=i.state["stage"], status=i.state["status"], reason=i.state.get("reason"))
    done = {i.item_id: dict(stage=i.state["stage"], status=i.state["status"], label=i.state.get("calib_label"))
            for i in P.store.items() if i.item_id in s9 and i.item_id not in open_}
    return dict(actions=did, open=open_, settled=done, ts=time.time())


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
    for rec in runs:                                     # reconcile interrupted submissions / dead launches
        if rec.get("outcome") is None and rec.get("state") in ("intent", "reserved"):
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
    sha, epoch = calib.pkg_sha(it.p("pkg")), int(it.state.get("calib_epoch", 0))
    # a verdict counts only for the package and calibration epoch it was submitted with
    runs = [r for r in all_rnd if r.get("pkg_sha") == sha and int(r.get("calib_epoch") or 0) == epoch]
    done = [r for r in runs if r.get("outcome") == "done"]
    if done:
        return _route_review(P, it, done[-1], rnd)
    n_infra = sum(1 for r in runs if r.get("outcome") == "infra")
    n_invalid = sum(1 for r in runs if r.get("outcome") == "invalid")
    if n_infra + n_invalid > int(rv.get("infra_retries", 2)):
        if n_invalid:
            # the reviewer kept breaking the verdict contract: no usable judgment -> drop (no operator exit unless
            # review.allow_human)
            why = f"r{rnd}: {n_infra} infra, {n_invalid} invalid verdicts"
            if rv.get("allow_human"):
                return it.finish("parked", f"review:human ({why})")
            return it.finish("dropped", f"review:invalid {why}")
        return it.finish("parked", f"review:infra ({n_infra} infra, {n_invalid} invalid verdicts)")
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
            it.state["status"] = "active"
            it.save()
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
    res = review.classify(jd, rec.get("model"), P.cfg["models"].get("prices", {}),
                          allow_human=bool((P.cfg.get("review") or {}).get("allow_human")))
    rec.update(outcome="invalid" if (res["outcome"] == "done" and (res["verdict"] or {}).get("verdict") is None)
               else res["outcome"],
               minutes=res["minutes"], cost_usd=res["cost_usd"], cost_kind=res["cost_kind"],
               trial=res["trial"], n_trials=res["n_trials"], usage=res["usage"])
    if res["verdict"] is not None:
        d = it.p("review", f"r{rec['n']}")
        d.mkdir(parents=True, exist_ok=True)
        raw = res["verdict_raw"]
        write_json(d / "verdict.raw.json", raw)
        body = dict(raw if isinstance(raw, dict) else {"_raw": raw}, _normalized=res["verdict"],
                    pkg_sha=rec.get("pkg_sha"), calib_epoch=rec.get("calib_epoch", 0), job=rec["job_name"])
        jf = d / f"{rec['job_name']}.verdict.json"           # one immutable verdict per job
        if not jf.exists():
            write_json(jf, body)
        write_json(d / "verdict.json", body)                  # latest of the round (records for the next round)
        rec["verdict"] = res["verdict"]["verdict"]
    if rec.get("ledger_call"):
        P.ledger.settle(rec["ledger_call"], res["cost_usd"], kind=res["cost_kind"], usage=res["usage"],
                        note=f"S9R review r{rec['n']} outcome={res['outcome']} verdict={rec.get('verdict')}")
    _save_review(it, rec)
    it.state["status"] = "active"
    it.save()
    return dict(outcome=res["outcome"], verdict=rec.get("verdict"), job_name=rec["job_name"])


def _released(P, item_id: str) -> bool:
    m = P.store.root / "release" / "manifest.jsonl"
    if not m.exists():
        return False
    live = False
    for line in m.read_text().splitlines():
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("item_id") == item_id:
            live = not o.get("withdrawn")
    return live


def withdraw(P, item_id: str, reason: str) -> bool:
    """Append a withdrawal line for a released item (the manifest is append-only; the last line per item wins)."""
    if not _released(P, item_id):
        return False
    with (P.store.root / "release" / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(dict(item_id=item_id, withdrawn=True, reason=reason, ts=time.time())) + "\n")
    P.store.event("release_withdrawn", item=item_id, reason=reason)
    return True


def _revision_retry(P, it, why: str):
    """A review revision failed to clear a locked-threshold gate (S6B reference / S7 gate / PKG checks): retry the
    revision with the failure fed back while revision rounds remain; otherwise drop the item.  review.allow_human
    (off by default) restores the old operator park instead."""
    rv = P.cfg.get("review") or {}
    if rv.get("allow_human"):
        return it.finish("parked", f"review:human ({why})")
    n = int(it.state.get("review_revisions", 0))
    rnd = int(it.state.get("review_round") or 1)
    verdict = read_json(it.p("review", "verdict.json"), {}) or {}
    routed = verdict.get("round") == rnd and (verdict.get("_normalized") or {}).get("verdict") == "revise"
    if it.state.get("post_revision") and n < int(rv.get("max_revisions", 3)) and routed:
        it.state["revision_feedback"] = str(why)[:2000]
        it.state["repair_from"] = f"review:r{rnd}"
        return it.goto("S8", f"review revision retry: {why[:160]}")
    return it.finish("dropped", f"review:gate_failed ({why[:300]})")


def _route_review(P, it, rec: dict, rnd: int):
    """pass -> S10 (kept label) / stays parked (parked label); revise -> S8 review revision while rounds remain
    (review.max_revisions, default 3; after that drop); drop -> dropped.  An invalid verdict (malformed, blocking
    issue without a fix_kind, bound to another job / package / epoch) re-runs the review within the round, up to
    review.infra_retries invalid verdicts; after that the item is dropped (or parked behind review.allow_human)."""
    rv = P.cfg.get("review") or {}
    v = read_json(it.p("review", f"r{rnd}", f"{rec.get('job_name')}.verdict.json"), {}) or {}
    if (v.get("job") != rec.get("job_name") or v.get("pkg_sha") != rec.get("pkg_sha") or
            int(v.get("calib_epoch") or 0) != int(rec.get("calib_epoch") or 0) or "_normalized" not in v):
        return _review_invalid(P, it, rnd, rec, f"verdict of job {rec.get('job_name')} missing or bound to "
                                                 f"another job / package / epoch")
    norm = v["_normalized"]
    verdict = norm["verdict"]
    if verdict is None:
        return _review_invalid(P, it, rnd, rec, "; ".join(norm.get("problems") or ["invalid verdict"])[:400])
    # the current verdict (S10's review gate checks round, calibration epoch and package hash)
    write_json(it.p("review", "verdict.json"), dict(v, _normalized=norm, round=rnd,
                                                    calib_epoch=int(rec.get("calib_epoch") or 0),
                                                    pkg_sha=rec.get("pkg_sha"), job=rec.get("job_name")))
    label = it.state.get("calib_label") or ""
    kept = label in P.cfg["calib"]["keep_labels"]
    it.state["review_verdict"] = verdict
    P.store.event("review_verdict", item=it.item_id, round=rnd, verdict=verdict, raw=norm.get("raw_verdict"),
                  blocking=norm.get("blocking"), minor=norm.get("minor"), label=label,
                  problems=(norm.get("problems") or [])[:6])
    if verdict == "pass":
        if kept:
            return it.advance("S9R", "S10", review="pass", round=rnd)
        return it.finish("parked", f"{label} (review r{rnd}: pass)")
    withdraw(P, it.item_id, f"review r{rnd}: {verdict}")
    if verdict == "human":                  # only reachable behind review.allow_human (explicit opt-in)
        return it.finish("parked", f"review:human r{rnd} ({','.join(norm.get('blocking') or [])})")
    if verdict == "revise" and int(it.state.get("review_revisions", 0)) >= int(rv.get("max_revisions", 3)):
        return it.finish("dropped", f"review:drop r{rnd} (revision limit {rv.get('max_revisions', 3)}: "
                                    f"{','.join(norm.get('blocking') or [])})")
    if verdict == "revise":
        it.state["repair_from"] = f"review:r{rnd}"
        return it.goto("S8", f"review r{rnd}: revise ({','.join(norm.get('blocking') or [])})")
    return it.finish("dropped", f"review:drop r{rnd} ({','.join(norm.get('blocking') or [])})")


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
            return it.finish("parked", f"review:human (r{rnd}: {n_bad + 1} invalid verdicts; last: {why[:200]})")
        return it.finish("dropped", f"review:invalid r{rnd} ({n_bad + 1} invalid verdicts; last: {why[:200]})")
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
