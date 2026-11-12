"""Difficulty targeting (difficulty.py): weakness-profile loading, S5G difficulty design, the S7 lever check, the
S9 hardening loop (trigger, feedback contents, versioning, idempotent resume, round cap, revert), S9R checklist.
No Docker, no network, no real jobs."""
import json
from pathlib import Path

import pytest

from tbsgen import calib, difficulty, review
from tbsgen.config import load_config
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json, write_json

from test_codeagent import SPEC, _agent_item

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "weakness_profile.example.md"
DESIGN = {"profile": "example-weakness-v0",
          "levers": [{"weakness_id": "EX02", "lever": "generic smoother cannot model the drift",
                      "paper_evidence": "Sec. 3.2", "code_evidence": "fastsmoother/core.py:40",
                      "why_naive_fails": "no drift", "why_mid_fails": "drift ignored",
                      "reference_handles": "joint estimation", "metrics": ["rmse"]}],
          "artificial_difficulty_check": "complete statement; thresholds from measurements"}


def _files(design=DESIGN, ins="Estimate the state."):
    f = {"_spec/spec.json": json.dumps(SPEC),
         "_spec/author_code.json": json.dumps({"package_names": ["fastsmoother"], "main_functions": ["smooth_rts"]}),
         "task/instruction.md": ins, "task/generator/generate.py": "print(1)",
         "task/solution/solve.py": "print(1)", "task/tests/measure.py": "print(1)",
         "task/baselines/mid.py": "print(1)", "task/baselines/naive.py": "print(1)", "report.md": "ok"}
    if design is not None:
        f["_spec/difficulty_design.json"] = json.dumps(design)
    return f


def _profiled(tmp_path, monkeypatch, files, stage="S5G", **diff):
    P, it, fl = _agent_item(tmp_path, monkeypatch, files, stage=stage)
    P.cfg["difficulty"] = dict(P.cfg["difficulty"], profile_path=str(EXAMPLE), **diff)
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    return P, it, fl


# ------------------------------------------------------------------ profile
def test_profile_loading_and_example(tmp_path):
    cfg = load_config(None, {"difficulty": {"profile_path": str(EXAMPLE)}})
    prof = difficulty.load_profile(cfg)
    assert prof["id"] == "example-weakness-v0" and prof["ids"] == ["EX01", "EX02", "EX03"] and len(prof["sha"]) == 16
    assert difficulty.load_profile(load_config(None, {})) is None            # off by default
    bad = tmp_path / "p.md"
    bad.write_text("# no entries\n")
    with pytest.raises(ValueError):
        difficulty.load_profile(load_config(None, {"difficulty": {"profile_path": str(bad)}}))


def test_validate_design():
    prof = dict(id="example-weakness-v0", ids=["EX01", "EX02", "EX03"])
    assert difficulty.validate_design(DESIGN, prof, SPEC) == []
    d = json.loads(json.dumps(DESIGN))
    d["levers"][0]["weakness_id"] = "T07"
    d["levers"][0]["metrics"] = ["nope"]
    errs = difficulty.validate_design(d, prof, SPEC)
    assert any("not in the profile" in e for e in errs) and any("not metrics of the spec" in e for e in errs)
    assert any("hardening" in e for e in difficulty.validate_design(DESIGN, prof, SPEC, harden_round=1))
    d = dict(DESIGN, levers=DESIGN["levers"] * 4)
    assert any("1-3" in e for e in difficulty.validate_design(d, prof, SPEC))


# ------------------------------------------------------------------ S5G
def test_s5g_with_profile_writes_design_and_provenance(tmp_path, monkeypatch):
    P, it, _ = _profiled(tmp_path, monkeypatch, _files())
    P.run(max_steps=1, until="S5G")
    work = it.p("S5G", "work")
    assert (work / "weakness_profile.md").read_text() == EXAMPLE.read_text()
    ins = it.p("S5G", "task", "instruction.md").read_text()
    assert "difficulty_design.json" in ins and "example-weakness-v0" in ins
    P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["stage"] == "S6A_run", st
    prov = read_json(it.p("_spec", "provenance.json"))
    assert prov["difficulty_profile"]["id"] == "example-weakness-v0" and prov["difficulty_levers"] == ["EX02"]
    assert read_json(it.p("_spec", "difficulty_design.json"))["levers"][0]["weakness_id"] == "EX02"


def test_s5g_invalid_design_reauthors_then_drops(tmp_path, monkeypatch):
    d = dict(DESIGN, levers=[dict(DESIGN["levers"][0], weakness_id="ZZ9")])
    P, it, fl = _profiled(tmp_path, monkeypatch, _files(design=d))
    for _ in range(6):
        P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"].startswith("s5g_jobs_exhausted:difficulty_design_invalid"), st
    assert len(fl.calls) == 2                                   # authoring_max_jobs
    assert read_json(it.p("S5G", "work", "previous_attempt.json"))["S6_failure.json"]["why"] == \
        "difficulty_design_invalid"


def test_s5g_without_profile_unchanged(tmp_path, monkeypatch):
    P, it, _ = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    P.run(max_steps=1, until="S5G")
    assert not (it.p("S5G", "work", "weakness_profile.md")).exists()
    P.run(max_steps=1, until="S5G")
    assert P.store.item(it.item_id).state["stage"] == "S6A_run"
    assert read_json(it.p("_spec", "provenance.json"))["difficulty_profile"] is None


# ------------------------------------------------------------------ S7 lever check
def _item_dir(tmp_path, ref=0.1, mid=0.5, naive=0.9, ins="Estimate the state."):
    d = tmp_path / "item"
    (d / "_spec").mkdir(parents=True)
    (d / "task").mkdir()
    (d / "task" / "instruction.md").write_text(ins)
    write_json(d / "_spec" / "difficulty_design.json", DESIGN)
    write_json(d / "S6A_measure.json", {w: {s: {"rmse": v} for s in ("0", "1")}
                                        for w, v in (("ref", ref), ("mid", mid), ("naive", naive))})
    return d


def test_lever_separation_check(tmp_path):
    th = {"rmse": {"threshold": 0.3, "direction": "<="}}
    assert difficulty.check_lever_separation(_item_dir(tmp_path / "a"), SPEC, th)["ok"]
    r = difficulty.check_lever_separation(_item_dir(tmp_path / "b", mid=0.2, naive=0.25), SPEC, th)
    assert not r["ok"] and "lever 0" in r["problems"][0]
    r = difficulty.check_lever_separation(_item_dir(tmp_path / "c", ins="This targets weakness profile EX02."),
                                          SPEC, th)
    assert not r["ok"] and any("reveals" in p for p in r["problems"])


# ------------------------------------------------------------------ hardening loop (S9)
def _calib_item(tmp_path, monkeypatch):
    """A code-grounded item at S9 with a packaged task and a passing first student run."""
    from test_calib_s9 import _env, _finish, _runs
    P, ids, counter, jobs = _env(tmp_path, n_items=1)
    P.cfg["difficulty"] = dict(P.cfg["difficulty"], profile_path=str(EXAMPLE))
    iid = ids[0]
    it = P.store.item(iid)
    write_json(it.p("_spec", "author_code.json"), {"package_names": ["x"]})
    write_json(it.p("_spec", "spec.json"), SPEC)
    write_json(it.p("_spec", "difficulty_design.json"), DESIGN)
    for rel in ("generator/generate.py", "solution/solve.py", "tests/measure.py", "baselines/mid.py",
                "baselines/naive.py", "instruction.md"):
        (it.p("task", rel)).parent.mkdir(parents=True, exist_ok=True)
        (it.p("task", rel)).write_text("v1 " + rel)
    write_json(it.p("thresholds.json"), {"thresholds": {"m": {"threshold": 0.5, "direction": ">="}}})
    write_json(it.p("S6A_measure.json"), {"ref": {}})
    P.run()
    job = _runs(P, iid)[0]["job_name"]
    jd = _finish(jobs, job, reward=1.0, cost=5.0)
    tr = jd / "trial__a"
    (tr / "verifier" / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in (
        {"name": "_solution_cli_run", "value": 12.0, "aux": True},
        {"name": "m", "value": 0.8, "threshold": 0.5, "direction": ">=", "passed": True},
        {"name": "m", "value": 0.1},                                           # duplicate: first wins
        {"name": "per_object_detail", "value": [1, 2, 3], "objects": {"a": "truth-label-x"}},
        {"name": "unknown_metric", "value": 0.4})) + "\n")
    (tr / "verifier" / "test-stdout.txt").write_text("SECRET-VERIFIER-OUTPUT per-object truth 0.123")
    (tr / "agent").mkdir(parents=True, exist_ok=True)
    (tr / "agent" / "claude-code.txt").write_text(json.dumps({"type": "result", "result": "I used method X."}) + "\n")
    return P, iid, counter, jobs


def test_first_pass_triggers_hardening_with_sanitised_feedback(tmp_path, monkeypatch):
    from test_calib_s9 import _runs
    from tbsgen.stages_b import watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    st = it.state
    # (watch_tick's run continues into S5G, which pauses here: the test config has no build host)
    assert st["stage"] == "S5G" and st["harden_round"] == 1 and st["harden_active"], st
    assert counter.read_text().count("x") == 1                     # no second student run was launched
    fb = read_json(it.p("harden", "round-1", "feedback.json"))
    assert fb["student_final_message"] == "I used method X." and fb["solution_runtime_sec"] == 12.0
    assert fb["metric_margins"] == [{"metric": "m", "value": 0.8, "threshold": 0.5, "direction": ">=",
                                     "relative_margin": 0.6}]
    blob = json.dumps(fb)
    assert "SECRET-VERIFIER-OUTPUT" not in blob and "truth" not in blob
    assert "per_object_detail" not in blob and "unknown_metric" not in blob
    rec = read_json(it.p("harden", "round-1", "record.json"))
    assert rec["archived_epoch"] == 0 and rec["prev_runs"][0]["outcome"] == "pass"
    assert (it.p("harden", "round-1", "prev", "DONE")).exists()
    assert (it.p("harden", "round-1", "prev", "task", "solution", "solve.py")).read_text() == "v1 solution/solve.py"
    assert st["calib_epoch"] == 1 and not it.p("calib", "runs.json").exists()
    assert (it.p("calib", "epoch-0", "runs.json")).exists()
    kinds = [e["kind"] for e in P.store.events()]
    assert "harden_start" in kinds
    # S5G inputs of the round: previous version + feedback, nothing from the verifier beyond the margins
    from tbsgen import codeagent
    work = tmp_path / "w"
    codeagent._stage_difficulty(P.cfg, it, work.mkdir() or work)
    assert (work / "previous_version" / "task" / "solution" / "solve.py").exists()
    assert read_json(work / "student_feedback.json") == fb
    staged = " ".join(p.read_text(errors="replace") for p in work.rglob("*") if p.is_file())
    assert "SECRET-VERIFIER-OUTPUT" not in staged


def test_hardening_start_is_idempotent_after_a_crash(tmp_path, monkeypatch):
    from tbsgen import stages_b
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    from tbsgen.stages_b import ingest
    it = P.store.item(iid)
    rec = read_json(it.p("calib", "runs.json"))[0]
    ingest(P, it, rec["job_dir"])
    real_archive = stages_b._archive_calibration
    calls = {"n": 0}

    def crash(it_):
        calls["n"] += 1
        raise RuntimeError("crash after the snapshot")
    monkeypatch.setattr(stages_b, "_archive_calibration", crash)
    it = P.store.item(iid)
    with pytest.raises(RuntimeError):
        stages_b.start_hardening(P, it)
    it = P.store.item(iid)
    assert it.state["harden_pending"]["round"] == 1 and it.state["stage"] == "S9"
    monkeypatch.setattr(stages_b, "_archive_calibration", real_archive)
    P.run(only=[iid], until="S9")
    it = P.store.item(iid)
    assert it.state["stage"] == "S5G" and it.state["harden_round"] == 1 and "harden_pending" not in it.state
    assert it.state["calib_epoch"] == 1 and len(list(it.p("calib").glob("epoch-*"))) == 1
    assert counter.read_text().count("x") == 1


def test_round_cap_then_original_rule(tmp_path, monkeypatch):
    from tbsgen.stages_b import watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    it = P.store.item(iid)
    it.state.update(harden_round=2)                              # rounds used up
    it.save()
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    assert it.state["stage"] == "S9" and counter.read_text().count("x") == 2      # second student run (old rule)


def test_harden_off_and_non_grounded(tmp_path, monkeypatch):
    from tbsgen.stages_b import watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    P.cfg["difficulty"]["harden"] = False
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["stage"] == "S9" and counter.read_text().count("x") == 2
    (tmp_path / "b").mkdir()
    P2, iid2, counter2, _ = _calib_item(tmp_path / "b", monkeypatch)
    P2.store.item(iid2).p("_spec", "author_code.json").unlink()   # a classic (non-code-grounded) item
    watch_tick(P2, only=[iid2])
    assert P2.store.item(iid2).state["stage"] == "S9" and counter2.read_text().count("x") == 2


def test_failed_hardened_version_reverts_to_previous(tmp_path, monkeypatch):
    from tbsgen.stages_b import harden_revert, watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    (it.p("task", "solution", "solve.py")).write_text("v2 broken")
    it.state.update(stage="S6B")
    it.finish("dropped", "not_separable")
    assert harden_revert(P, it)
    it = P.store.item(iid)
    st = it.state
    assert st["stage"] == "S9" and st["status"] == "active" and st["harden_exhausted"] and not st.get("harden_active")
    assert (it.p("task", "solution", "solve.py")).read_text() == "v1 solution/solve.py"
    assert (it.p("harden", "round-1", "failed", "task", "solution", "solve.py")).read_text() == "v2 broken"
    runs = read_json(it.p("calib", "runs.json"))
    assert len(runs) == 1 and runs[0]["outcome"] == "pass"          # the old version's calibration is back
    # continues under the original rule: a second student run on the old version, no new hardening
    P.run(only=[iid], until="S9")
    assert counter.read_text().count("x") == 2
    assert P.store.item(iid).state["harden_round"] == 1


def test_harden_cli_for_an_easy_item(tmp_path, monkeypatch):
    from tbsgen.stages_b import harden_cmd, watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    P.cfg["difficulty"]["harden"] = False
    watch_tick(P, only=[iid])                                    # second student run launched (loop off)
    from test_calib_s9 import _finish, _runs
    _finish(jobs, _runs(P, iid)[1]["job_name"], reward=1.0, cost=5.0)
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "done" and st["calib_label"] == "easy"
    P.cfg["difficulty"]["harden"] = True
    out = harden_cmd(P, P.store.item(iid), "operator: easy before the loop existed")
    st = P.store.item(iid).state
    assert out["round"] == 1 and st["stage"] == "S5G" and st["status"] == "active"
    rec = read_json(P.store.item(iid).p("harden", "round-1", "record.json"))
    assert rec["operator"].startswith("operator") and rec["prev_label"] == "easy"


def test_s9r_checklist_has_genuine_difficulty(tmp_path):
    assert "j_genuine_difficulty" in review.CHECKLIST
    item = tmp_path / "item"
    (item / "pkg" / "tests").mkdir(parents=True)
    (item / "_spec").mkdir()
    write_json(item / "_spec" / "difficulty_design.json", DESIGN)
    (item / "harden" / "round-1").mkdir(parents=True)
    write_json(item / "harden" / "round-1" / "record.json", {"round": 1})
    (item / "harden" / "round-1" / "feedback.json").write_text("{}")
    review.write_bundle(item, tmp_path / "b", {"history": []})
    assert (tmp_path / "b" / "records" / "difficulty_design.json").exists()
    assert (tmp_path / "b" / "records" / "hardening" / "round-1-record.json").exists()
    assert not list((tmp_path / "b").rglob("feedback.json"))
    assert "Artificial difficulty is blocking" in review.REVIEW_PROMPT


def test_next_action_unchanged():
    cfg = load_config(None, {})
    p = dict(who="glm", outcome="pass", minutes=30)
    f = dict(who="glm", outcome="fail", minutes=30)
    assert calib.next_action([p], cfg) == ("run", "glm")
    assert calib.next_action([p, f], cfg) == ("label", "unstable")
    assert calib.next_action([p, p], cfg) == ("label", "easy")
    assert calib.next_action([f, dict(who="teacher", outcome="pass")], cfg) == ("label", "hard")


def test_targeting_leak_in_public_files_and_package(tmp_path):
    prof = dict(id="example-weakness-v0", ids=["EX01", "EX02"])
    pub = tmp_path / "env" / "data"
    pub.mkdir(parents=True)
    (pub / "README.md").write_text("Data notes. (designed for EX02)")
    (pub / "x.csv").write_text("a,b\n1,2\n")
    assert difficulty.targeting_leaks([tmp_path / "env"], prof, None) == ["README.md: 'EX02'"]
    (pub / "README.md").write_text("Notes: example-weakness-v0")
    assert difficulty.targeting_leaks([tmp_path / "env"], prof, None)
    (pub / "README.md").write_text("Plain notes; EX020 is a sample id; FLEX02B too.")
    assert difficulty.targeting_leaks([tmp_path / "env"], prof, None) == []
    (pub / "rows.jsonl").write_text('{"note": "weakness profile EX02"}\n')            # any extension
    assert difficulty.targeting_leaks([tmp_path / "env"], prof, None)
    (pub / "rows.jsonl").write_bytes(b"x" * (3 << 20) + b" EX01 ")                     # large file, chunk edge
    assert difficulty.targeting_leaks([tmp_path / "env"], prof, None) == ["rows.jsonl: 'EX01'"]


def test_hardening_must_change_the_science(tmp_path):
    d = tmp_path / "item"
    for root in (d / "task", d / "harden" / "round-1" / "prev" / "task"):
        for rel in ("generator/generate.py", "solution/solve.py", "tests/measure.py", "baselines/mid.py"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("same " + rel)
    (d / "task" / "baselines" / "mid.py").write_text("weaker mid -> tighter thresholds")
    (d / "task" / "instruction.md").write_text("vaguer")
    assert difficulty.check_hardening(d, 1)                     # baseline / wording only: rejected
    (d / "task" / "generator" / "generate.py").write_text("new regime")
    assert difficulty.check_hardening(d, 1) == []
    assert "new regime" in difficulty.version_diff(d / "harden" / "round-1" / "prev" / "task", d / "task")


def test_revert_is_replayable_after_a_crash(tmp_path, monkeypatch):
    from tbsgen import stages_b
    from tbsgen.stages_b import watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    (it.p("task", "solution", "solve.py")).write_text("v2 broken")
    it.state.update(stage="S6B")
    it.finish("dropped", "not_separable")
    real = stages_b._move_records
    monkeypatch.setattr(stages_b, "_move_records", lambda a, b: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        stages_b.harden_revert(P, P.store.item(iid))
    assert P.store.item(iid).state["harden_revert_pending"]["prev_epoch"] == 0
    monkeypatch.setattr(stages_b, "_move_records", real)
    P.run(only=[iid], until="S9")                               # a dropped item with a pending revert is resumed
    it = P.store.item(iid)
    assert it.state["stage"] == "S9" and "harden_revert_pending" not in it.state
    assert (it.p("task", "solution", "solve.py")).read_text() == "v1 solution/solve.py"
    assert (it.p("harden", "round-1", "failed", "task", "solution", "solve.py")).read_text() == "v2 broken"
    runs = read_json(it.p("calib", "runs.json"))
    assert runs[0]["outcome"] == "pass" and counter.read_text().count("x") == 2


def test_harden_cmd_resumes_an_interrupted_start(tmp_path, monkeypatch):
    from tbsgen import stages_b
    from tbsgen.stages_b import harden_cmd, watch_tick
    from test_calib_s9 import _finish, _runs
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    P.cfg["difficulty"]["harden"] = False
    watch_tick(P, only=[iid])
    _finish(jobs, _runs(P, iid)[1]["job_name"], reward=1.0, cost=5.0)
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["status"] == "done"
    P.cfg["difficulty"]["harden"] = True
    real = stages_b._archive_calibration
    monkeypatch.setattr(stages_b, "_archive_calibration", lambda it_: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        harden_cmd(P, P.store.item(iid), "op")
    monkeypatch.setattr(stages_b, "_archive_calibration", real)
    st = P.store.item(iid).state
    assert st["status"] == "done" and st["harden_pending"]["round"] == 1
    P.run(only=[iid])                                           # the scheduler finishes the pending start
    st = P.store.item(iid).state
    assert st["stage"] == "S5G" and st["harden_round"] == 1 and "harden_pending" not in st


def test_round_allowance_and_rerun_survive_a_pause(tmp_path, monkeypatch):
    """S5G of a hardening round: the old version's finished job is never accepted, the rerun intent survives a
    pause, and the round's allowance is its own (not the unspent rest of earlier rounds)."""
    P, it, fl = _profiled(tmp_path, monkeypatch, _files(design=dict(DESIGN, hardening={
        "changes": "c", "why_previous_solution_fails_now": "w"})),
        harden_authoring_timeout_s=900, harden_authoring_max_usd=2.0)
    P.run(max_steps=2, until="S5G")                             # round-0 authoring job
    it = P.store.item(it.item_id)
    assert it.state["stage"] == "S6A_run" and len(fl.calls) == 1
    from tbsgen.stages_b import _s5g_base
    it.state.update(stage="S5G", status="active", harden_round=1, harden_active=True, S5G_rerun=True,
                    harden_base=_s5g_base(it, 1))
    it.save()
    (it.p("harden", "round-1", "prev", "task")).mkdir(parents=True)
    (it.p("harden", "round-1", "prev", "DONE")).write_text("x")
    monkeypatch.setattr("tbsgen.codeagent.env_ok", lambda cfg, agent=False: "paused for the test")
    P.run(max_steps=1, until="S5G")
    it = P.store.item(it.item_id)
    assert it.state["status"] == "paused" and read_json(it.p("S5G", "run.json"))["retry"] is True
    monkeypatch.setattr("tbsgen.codeagent.env_ok", lambda cfg, agent=False: None)
    P.resume(("code:",))
    P.run(max_steps=1, until="S5G")
    assert len(fl.calls) == 2                                   # a new job, not the old version's result
    conf = json.loads(Path(read_json(it.p("S5G", "run.json"))["config"]).read_text())
    assert conf["agents"][0]["kwargs"]["max_budget_usd"] == "2.00"
    assert read_json(it.p("S5G", "run.json"))["harden_round"] == 1


def test_watch_follows_a_hardening_round_and_harden_refuses_pending_rescore(tmp_path, monkeypatch):
    from tbsgen.stages_b import harden_cmd, watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    assert it.state["stage"] == "S5G" and it.state["harden_active"]
    it.state.update(status="waiting", reason="S5G:agent_running")    # as with a running authoring job
    it.save()
    polled = []
    monkeypatch.setattr("tbsgen.stages_a.s5g", lambda P_, it_: polled.append(it_.item_id))
    from tbsgen import pipeline
    monkeypatch.setitem(pipeline.HANDLERS, "S5G", lambda P_, it_: polled.append(it_.item_id))
    out = watch_tick(P, only=[iid])
    assert iid in out["open"] and iid not in out["settled"] and polled == [iid]   # the S5G job is polled
    P2, iid2, _, jobs2 = _calib_item(tmp_path / "b" if (tmp_path / "b").mkdir() is None else None, monkeypatch)
    it = P2.store.item(iid2)
    it.state["rescore_pending"] = dict(reason="regate")
    it.save()
    with pytest.raises(ValueError):
        harden_cmd(P2, P2.store.item(iid2), "op")
