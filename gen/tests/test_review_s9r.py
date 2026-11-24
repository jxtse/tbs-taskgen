"""S9R review stage: routing after S9 labels, review job lifecycle (prepare / launch / ingest), verdict contract
(normalize), the S8 review revision (thresholds locked, only allowed files, calibration archived and restarted
from GLM), the second-review rule, manifest notes and withdrawal.  No Docker, no network, no real jobs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tbsgen import gates, review
from tbsgen.config import load_config
from tbsgen.models import FakeClient
from tbsgen.pipeline import Pipeline
from tbsgen.stages_b import ingest, watch_tick
from tbsgen.state import read_json, write_json


# ------------------------------------------------------------------ fixtures
TEST_OUTPUTS = '''from _metrics import check
def test_a():
    check("err", 0.5, 1.0e-3, "<=")
def test_b():
    check("score", 0.5, 0.9, ">=")
'''


def _env(tmp_path, review_kw=None, **calib_kw):
    hr = tmp_path / "harbor"
    hr.mkdir(parents=True, exist_ok=True)
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    tmux = tmp_path / "fake-tmux"
    tmux.write_text("#!/bin/bash\nif [ \"$1\" = has-session ]; then exit 0; fi\nshift 4\n\"$@\"\n"
                    "for f in %s/items/*/calib/*.log.exit %s/items/*/review/*.log.exit; do [ -f \"$f\" ] || continue; "
                    "j=$(basename \"$f\" .log.exit); [ -f \"%s/$j/result.json\" ] && rm \"$f\"; done; true\n"
                    % (tmp_path / "run", tmp_path / "run", jobs))
    tmux.chmod(0o755)
    mk = ('J=$(python3 -c \'import json,sys;print(json.load(open(sys.argv[1]))["job_name"])\' $2); '
          'mkdir -p "%s/$J" && cp $2 "%s/$J/config.json" && echo "{}" > "%s/$J/result.json"' % (jobs, jobs, jobs))
    for name in ("launch.sh", "teach.sh"):
        (hr / name).write_text("#!/bin/bash\n%s\n" % mk)
        (hr / name).chmod(0o755)
    cal = dict(allow_submit=True, harbor_root=str(hr), jobs_dir=str(jobs), student_launcher="launch.sh",
               teacher_launcher="teach.sh", agent_allowed_hosts=["models.example.com"], tmux_bin=str(tmux),
               est_usd_per_run=30.0)
    cal.update(calib_kw)
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "calib": cal, "review": review_kw or {},
                             "budget": {"daily_usd_total": 1000.0, "run_usd_total": 500.0}})
    P = Pipeline(cfg, clients={})
    [iid] = P.ingest([dict(doi="10.1/s9r", title="t", abstract="algorithm estimation")])
    it = P.store.item(iid)
    pkg = it.p("pkg")
    for d in ("environment/data", "tests", "solution"):
        (pkg / d).mkdir(parents=True, exist_ok=True)
    (pkg / "task.toml").write_text('[environment]\nnetwork_mode = "public"\ncpus = 2\n[agent]\n'
                                   'network_mode = "allowlist"\nallowed_hosts = []\n')
    (pkg / "instruction.md").write_text("Write out.csv with column spec.\n")
    (pkg / "environment" / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\nCOPY data /app/data\n")
    (pkg / "environment" / "data" / "README.md").write_text("format notes\n")
    (pkg / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (pkg / "tests" / "test_outputs.py").write_text(TEST_OUTPUTS)
    (pkg / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    task = it.p("task")
    for d in ("tests", "solution", "generator", "baselines"):
        (task / d).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Write out.csv with column spec.\n")
    (task / "tests" / "test_outputs.py").write_text(TEST_OUTPUTS)
    (task / "tests" / "measure.py").write_text("COL = 'spec'\n")
    (task / "solution" / "solve.py").write_text("print('ref')\n")
    (task / "generator" / "generate.py").write_text("README = 'format notes'\n")
    (task / "baselines" / "mid.py").write_text("print('mid')\n")
    write_json(it.p("thresholds.json"), dict(thresholds={"err": dict(direction="<=", threshold=1.0e-3, ref=1e-4),
                                                         "score": dict(direction=">=", threshold=0.9, ref=0.95)}))
    write_json(it.p("_spec", "spec.json"), dict(depgraph={"nodes": []}, metrics=[]))
    it.state.update(stage="S9", status="active")
    it.save()
    return P, iid, jobs


def _runs(P, iid, sub="calib"):
    return read_json(P.store.item(iid).p(sub, "runs.json"), [])


def _receipt(scope, key, reward=1):
    return dict(repro_id=key, scope_id=scope, reward=reward, exit_code=0, command=review.repro_command(scope), output_sha256="a" * 64)


def _finish(jobs, job_name, reward=None, cost=5.0, verdict=None, name="trial__a", repros=()):
    if reward is None:
        reward = 1.0 if verdict is not None else 0.0     # a review job with a well-formed verdict has reward 1
    jd = jobs / job_name
    t = jd / name
    (t / "verifier").mkdir(parents=True, exist_ok=True)
    (t / "agent").mkdir(parents=True, exist_ok=True)
    (t / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": reward}}, "agent_result": {"cost_usd": cost},
        "agent_execution": {"started_at": "2026-11-09T01:00:00Z", "finished_at": "2026-11-09T01:30:00Z"}}))
    (t / "verifier" / "metrics.jsonl").write_text(json.dumps({"name": "m", "value": 1.0, "passed": reward >= 1}) + "\n")
    (t / "agent" / "claude-code.txt").write_text(
        json.dumps({"type": "assistant", "message": {"id": "m1", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls /app/data"}}]}}) + "\n" +
        json.dumps({"type": "result", "subtype": "success", "result": "done"}) + "\n")
    if repros:
        (t / "verifier/repro.jsonl").write_text("".join(json.dumps(row) + "\n" for row in repros))
    if verdict is not None:
        (t / "verifier" / "verdict.json").write_text(json.dumps(verdict))
    (jd / "result.json").write_text(json.dumps({"finished_at": "2026-11-09T01:31:00Z"}))


def _to_label(P, iid, jobs, seq=(0.0, 1.0)):
    """GLM fail, teacher pass -> hard (default seq)."""
    P.run(only=[iid])
    for k, rw in enumerate(seq):
        _finish(jobs, _runs(P, iid)[k]["job_name"], reward=rw)
        ingest(P, P.store.item(iid))
        P.run(only=[iid])


def V(verdict, issues=(), trust=True):
    return dict(verdict=verdict, issues=list(issues), checklist={},
                label_assessment=dict(label="hard", trustworthy=trust, reason="r"), summary="s")


BLOCK = dict(id="I1", category="a", severity="blocking", problem="statement", title="column name conflict",
             evidence=["instruction.md:1 says spectrum_index; tests/measure.py:1 reads spec"], fix="use spec")
MINOR = dict(id="I2", category="d", severity="minor", title="timing margin small", evidence=[], fix="none")


# ------------------------------------------------------------------ verdict contract
def test_normalize_contract():
    """pass / revise / drop only; the verdict is derived from the blocking issues' problem classes; a structural
    problem makes the verdict invalid (None: the review runs again), never human and never pass."""
    n = review.normalize(V("pass", [MINOR]))
    assert n["verdict"] == "pass" and n["minor"] == ["I2"] and not n["problems"]
    # blocking without evidence -> minor; a revise with no evidenced blocking issue is invalid (not a pass)
    n = review.normalize(V("revise", [dict(BLOCK, evidence=[])]))
    assert n["downgraded"] == ["I1"] and n["blocking"] == [] and n["verdict"] is None
    n = review.normalize(V("revise", [BLOCK]))
    assert n["verdict"] == "revise" and n["blocking"] == ["I1"] and n["issues"][0]["fix_kind"] == "local"
    # a pass with an evidenced blocking issue is routed by the issue (revise), the disagreement is recorded
    n = review.normalize(V("pass", [BLOCK]))
    assert n["verdict"] == "revise" and any("overridden" in p for p in n["problems"])
    assert review.normalize(V("drop", [MINOR]))["verdict"] is None
    assert review.normalize(V("ok"))["verdict"] is None
    assert review.normalize(V("human", [BLOCK]))["verdict"] is None              # human is not a verdict ...
    assert review.normalize(V("human", [BLOCK]), allow_human=True)["verdict"] == "human"   # ... unless opted in
    assert review.normalize(None)["verdict"] is None
    n = review.normalize(V("revise", [dict(BLOCK, severity="critical")]))
    assert n["verdict"] is None and n["issues"][0]["severity"] == "minor"
    # strict: a structural problem can never yield pass
    assert review.normalize(V("pass", [dict(MINOR, severity="critical")]))["verdict"] is None
    assert review.normalize(V("pass", ["not an object"]))["verdict"] is None
    assert review.normalize(V("pass"), reward=0.0)["verdict"] is None
    assert review.normalize(V("pass"), reward=1.0)["verdict"] == "pass"


def test_problem_class_decides_fixability():
    """Local problem classes -> revise, fundamental ones -> drop; the class overrides the reviewer's fix_kind, so
    a "revise" whose blocking issues are all fundamental is routed as drop."""
    for pc in ("statement", "format_notes", "grader_io", "exploit_baseline", "test_gap", "timing_gate"):
        assert review.normalize(V("revise", [dict(BLOCK, problem=pc)]))["verdict"] == "revise", pc
    for pc in ("reference_wrong", "truth_regen", "core_invalid", "calibration"):
        n = review.normalize(V("revise", [dict(BLOCK, problem=pc, fix_kind="local")]))
        assert n["verdict"] == "drop" and n["issues"][0]["fix_kind"] == "fundamental", pc
        assert any("contradicts" in p for p in n["problems"])
    mixed = [BLOCK, dict(BLOCK, id="I3", problem="reference_wrong")]
    assert review.normalize(V("revise", mixed))["verdict"] == "drop"
    assert review.normalize(V("drop", [BLOCK]))["verdict"] == "drop"
    # unknown class / no class at all -> invalid; a legacy verdict with fix_kind only is still accepted
    assert review.normalize(V("revise", [dict(BLOCK, problem="typo")]))["verdict"] is None
    nb = {k: v for k, v in BLOCK.items() if k != "problem"}
    assert review.normalize(V("revise", [nb]))["verdict"] is None
    assert review.normalize(V("revise", [dict(nb, fix_kind="local")]))["verdict"] == "revise"
    assert review.normalize(V("revise", [dict(nb, fix_kind="fundamental")]))["verdict"] == "drop"
    # the review task's validator demands the class too
    assert set(review.PROBLEM_CLASSES.values()) == {"local", "fundamental"}


def test_reviewable_labels():
    cfg = load_config()
    assert cfg["review"]["enabled"] is True and cfg["review"]["max_revisions"] == 3
    assert cfg["review"]["allow_human"] is False and cfg["review"]["rescore_scoring_only"] is True
    assert review.reviewable(cfg, "hard") and review.reviewable(cfg, "unstable")
    assert review.reviewable(cfg, "parked:teacher_failed") and review.reviewable(cfg, "frontier")
    assert not review.reviewable(cfg, "easy") and not review.reviewable(cfg, "parked:infra")
    assert not review.reviewable(load_config(None, {"review": {"enabled": False}}), "hard")


def test_review_cfg_falls_back_to_teacher_route():
    cfg = load_config(None, {"calib": {"teacher_environment": "env:T", "teacher_agent_allowed_hosts": ["10.0.0.1"],
                                       "teacher_harbor_root": "/t", "teacher_launcher": "t.sh",
                                       "teacher_agent": "a:T"}})
    c = review.review_cfg(cfg)["calib"]
    assert (c["teacher_environment"], c["teacher_agent_allowed_hosts"], c["teacher_harbor_root"],
            c["teacher_launcher"], c["teacher_agent"], c["teacher_model"]) == \
        ("env:T", ["10.0.0.1"], "/t", "t.sh", "a:T", "claude-opus-5-5")
    c = review.review_cfg(load_config(None, {"calib": {"teacher_launcher": "t.sh"},
                                             "review": {"launcher": "r.sh", "model": "m-r", "est_usd_per_run": 7}}))
    assert c["calib"]["teacher_launcher"] == "r.sh" and c["calib"]["teacher_model"] == "m-r"
    assert c["calib"]["est_usd_per_run"] == 7.0


# ------------------------------------------------------------------ routing
def test_hard_goes_through_review_before_release(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    it = P.store.item(iid)
    assert it.state["calib_label"] == "hard" and it.state["stage"] == "S9R" and it.state["status"] == "waiting"
    assert not (P.store.root / "release" / "manifest.jsonl").exists()
    [rr] = _runs(P, iid, "review")
    assert rr["who"] == "review" and rr["model"] == "claude-opus-5-5" and rr["job_name"].endswith("-review1")
    conf = json.loads(Path(rr["config"]).read_text())
    assert conf["agents"][0]["extra_allowed_hosts"] == ["models.example.com"]
    t = Path(rr["task"])
    assert (t / "environment" / "review" / "pkg" / "instruction.md").exists()
    assert (t / "environment" / "review" / "extra" / "baselines" / "mid.py").exists()
    assert (t / "environment" / "review" / "records" / "thresholds.json").exists()
    cal = t / "environment" / "review" / "calibration"
    assert len(json.loads((cal / "runs.json").read_text())) == 2
    assert "ls /app/data" in next(cal.glob("*glm1/trace_digest.md")).read_text()
    assert "COPY review /review" in (t / "environment" / "Dockerfile").read_text()
    assert "hard" in (t / "instruction.md").read_text()
    # ledger: the review run is reserved under stage S9R
    import sqlite3
    with sqlite3.connect(P.ledger.path) as db:
        assert db.execute("SELECT count(*) FROM calls WHERE stage='S9R'").fetchone()[0] == 1
    _finish(jobs, rr["job_name"], reward=1.0, cost=3.0, verdict=V("pass", [MINOR]))
    out = watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["stage"] == "S10" and st["status"] == "done", (st, out)
    m = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[-1])
    assert m["review"]["verdict"] == "pass" and m["review"]["minor"][0]["id"] == "I2"
    assert m["label"] == "hard" and m["calib_epoch"] == 0


def test_parked_teacher_failed_reviewed_but_never_released(tmp_path):
    """calib.frontier off: the old rule (teacher failures parked, reviewed as a diagnosis only)."""
    P, iid, jobs = _env(tmp_path)
    P.cfg["calib"]["frontier"] = False
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    it = P.store.item(iid)
    assert it.state["calib_label"] == "parked:teacher_failed" and it.state["stage"] == "S9R"
    [rr] = _runs(P, iid, "review")
    _finish(jobs, rr["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("parked:teacher_failed")
    assert not (P.store.root / "release" / "manifest.jsonl").exists()


FRONTIER_OK = dict(V("pass", [MINOR]), reference_check=dict(reward=1.0, notes="reproduced"),
                   checklist={k: dict(status="ok", note="n") for k in
                              ("a_consistency", "b_missing_conditions", "e_shortcuts_leaks", "j_genuine_difficulty")})


def _oracle_ok(P, iid, ok=True):
    it = P.store.item(iid)
    it.state["gates_attempt"] = 9
    it.save()
    write_json(it.p("gates", "attempt-9.json"), dict(probe_policy_version=gates.PROBE_POLICY_VERSION, ok=ok, gates=dict(oracle=ok)))


def test_frontier_released_on_its_own_channel_after_a_confirming_review(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))                     # student fail, teacher fail x2 (real fails)
    it = P.store.item(iid)
    assert it.state["calib_label"] == "frontier" and it.state["stage"] == "S9R"
    [rr] = _runs(P, iid, "review")
    assert "This is a `frontier` candidate" in (Path(rr["task"]) / "instruction.md").read_text()
    _oracle_ok(P, iid)
    _finish(jobs, rr["job_name"], verdict=FRONTIER_OK)
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["stage"] == "S10" and st["status"] == "done", st
    m = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[-1])
    assert m["label"] == "frontier" and m["channel"] == "frontier"
    assert m["frontier"]["teacher_decisive"] == 2 and m["frontier"]["teacher_timeouts"] == 0
    assert m["reference_check"]["reward"] == 1.0


@pytest.mark.parametrize("change", ["no_reference", "reference_fails", "check_issue", "untrusted", "oracle_gate",
                                    "bad_types"])
def test_frontier_not_confirmed_stays_parked(tmp_path, change):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    v = json.loads(json.dumps(FRONTIER_OK))
    if change == "no_reference":
        v.pop("reference_check")
    elif change == "reference_fails":
        v["reference_check"]["reward"] = 0.0
    elif change == "check_issue":
        v["checklist"]["b_missing_conditions"]["status"] = "na"
    elif change == "untrusted":
        v["label_assessment"]["trustworthy"] = None
    elif change == "bad_types":
        v.update(reference_check="unknown", checklist=["ok"], label_assessment="yes")
    _oracle_ok(P, iid, ok=change != "oracle_gate")
    [rr] = _runs(P, iid, "review")
    _finish(jobs, rr["job_name"], verdict=v)
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("frontier:unconfirmed"), st
    assert not (P.store.root / "release" / "manifest.jsonl").exists()


def test_frontier_needs_real_teacher_failures():
    from tbsgen import calib
    cfg = load_config()
    run = lambda who, out, rt=None, exc=None: dict(who=who, outcome=out, result_type=rt or out,
                                                   trial=dict(exception=exc))
    g = run("glm", "fail", "fail_delivered")
    t = run("teacher", "fail", "fail_delivered")
    assert calib.next_action([g, t, t], cfg) == ("label", "frontier")
    assert calib.next_action([g, t, run("teacher", "infra")], cfg) == ("run", "teacher")   # infra is no attempt
    assert calib.next_action([g, t, run("teacher", "fail", "timeout_with_output")], cfg) == \
        ("label", "parked:teacher_failed")
    assert calib.next_action([g, t, run("teacher", "fail", "fail_delivered", "AgentTimeoutError")], cfg) == \
        ("label", "parked:teacher_failed")
    cfg["calib"]["frontier"] = False
    assert calib.next_action([g, t, t], cfg) == ("label", "parked:teacher_failed")
    ev = calib.frontier_evidence([g, t, run("teacher", "infra"), t], cfg)
    assert (ev["teacher_runs"], ev["teacher_decisive"], ev["teacher_infra"], ev["teacher_timeouts"]) == (3, 2, 1, 0)


def test_frontier_not_confirmed_withdraws_an_earlier_release(tmp_path):
    """A released frontier task reviewed again (e.g. after a reset) whose new pass does not confirm it is withdrawn."""
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    _oracle_ok(P, iid)
    [rr] = _runs(P, iid, "review")
    _finish(jobs, rr["job_name"], verdict=FRONTIER_OK)
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["status"] == "done"
    it = P.store.item(iid)
    it.state.update(stage="S9R", status="active", review_round=2)
    it.save()
    P.run(only=[iid])
    rr2 = [r for r in _runs(P, iid, "review") if r.get("n") == 2][0]
    _finish(jobs, rr2["job_name"], verdict=dict(FRONTIER_OK, reference_check=dict(reward=0.0)))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("frontier:unconfirmed"), st
    last = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[-1])
    assert last["item_id"] == iid and last["withdrawn"]


def test_frontier_reaching_s10_with_review_disabled_is_not_released(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    it = P.store.item(iid)
    assert it.state["stage"] == "S9R"
    P.cfg["review"]["enabled"] = False                                  # switched off mid-review, then S10
    it.state.update(stage="S10", status="active")
    it.save()
    P.run(only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("frontier:unreviewed"), st
    assert not (P.store.root / "release" / "manifest.jsonl").exists()


def test_frontier_s10_without_a_confirming_verdict_goes_back_to_review(tmp_path):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    it = P.store.item(iid)
    assert stages_b.review_gate(P, it) == "no review verdict"
    _oracle_ok(P, iid, ok=False)
    write_json(it.p("review", "verdict.json"), dict(FRONTIER_OK, _normalized=review.normalize(FRONTIER_OK),
                                                    round=it.state.get("review_round"), calib_epoch=0,
                                                    pkg_sha=__import__("tbsgen.calib").calib.pkg_sha(it.p("pkg"))))
    assert stages_b.review_gate(P, it).startswith("frontier not confirmed")


def test_frontier_relabelled_with_review_disabled_keeps_confirmed_release(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    _oracle_ok(P, iid)
    [rr] = _runs(P, iid, "review")
    _finish(jobs, rr["job_name"], verdict=FRONTIER_OK)
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["status"] == "done"
    P.cfg["review"]["enabled"] = False
    it = P.store.item(iid)
    it.state.update(stage="S9", status="active")                       # e.g. reset --stage S9 (runs kept)
    it.save()
    P.run(only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "done"
    rows = (P.store.root / "release/manifest.jsonl").read_text().splitlines()
    assert len(rows) == 1 and not json.loads(rows[0]).get("withdrawn")


def test_frontier_s10_gate_sends_it_back_to_review(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    it = P.store.item(iid)
    it.state.update(stage="S10", status="active")                      # e.g. reset --stage S10 without a verdict
    it.save()
    from tbsgen import stages_b
    stages_b.s10(P, P.store.item(iid))
    st = P.store.item(iid).state
    assert st["stage"] == "S9R" and "review gate" in st["history"][-1].get("reason", st.get("reason", "")), st
    assert not (P.store.root / "release" / "manifest.jsonl").exists()


def test_released_frontier_with_a_changed_package_is_withdrawn_at_s10(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    _oracle_ok(P, iid)
    [rr] = _runs(P, iid, "review")
    _finish(jobs, rr["job_name"], verdict=FRONTIER_OK)
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    assert it.state["status"] == "done"
    (it.p("pkg") / "instruction.md").write_text("changed after the review")   # stale verdict
    it.state.update(stage="S10", status="active")                      # e.g. reset --stage S10
    it.save()
    from tbsgen import stages_b
    stages_b.s10(P, P.store.item(iid))
    assert P.store.item(iid).state["stage"] == "S9R"
    last = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[-1])
    assert last["item_id"] == iid and last["withdrawn"]


def test_frontier_review_disabled_is_never_released(tmp_path):
    P, iid, jobs = _env(tmp_path)
    P.cfg["review"]["enabled"] = False
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("frontier:unreviewed")


def test_easy_not_reviewed_and_review_disabled(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(1.0, 1.0))
    st = P.store.item(iid).state
    assert st["calib_label"] == "easy" and st["status"] in ("done", "dropped") and st["stage"] != "S9R" and not _runs(P, iid, "review")
    P, iid, jobs = _env(tmp_path / "off", review_kw={"enabled": False})
    _to_label(P, iid, jobs)
    assert P.store.item(iid).state["stage"] == "S10"


def test_review_infra_retries_then_park(tmp_path):
    P, iid, jobs = _env(tmp_path, review_kw={"infra_retries": 1})
    _to_label(P, iid, jobs)
    for k in range(2):
        rr = _runs(P, iid, "review")[-1]
        _finish(jobs, rr["job_name"])                 # no verdict file -> infra
        watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert [r["outcome"] for r in _runs(P, iid, "review")] == ["infra", "infra"]
    assert st["status"] == "parked" and st["reason"].startswith("review:infra (2 infra")
    assert _runs(P, iid, "review")[1]["job_name"].endswith("-review1-a2")


def test_drop_and_human(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("drop", [dict(BLOCK, problem="reference_wrong")]))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "dropped" and st["reason"].startswith("review:drop")
    # a revise whose only blocking issue lacks evidence is invalid: the review runs again within the round, and
    # after review.infra_retries invalid verdicts the item is dropped (no human exit)
    P, iid, jobs = _env(tmp_path / "h", review_kw={"infra_retries": 1})
    _to_label(P, iid, jobs)
    for k in range(2):
        rr = _runs(P, iid, "review")[-1]
        _finish(jobs, rr["job_name"], verdict=V("revise", [dict(BLOCK, evidence="")]))
        watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert [r["outcome"] for r in _runs(P, iid, "review")] == ["invalid", "invalid"]
    assert st["status"] == "dropped" and st["reason"].startswith("review:invalid r1: 0 infra, 2 invalid")
    # behind review.allow_human (explicit opt-in) the same exit parks for an operator
    P, iid, jobs = _env(tmp_path / "opt", review_kw={"infra_retries": 0, "allow_human": True})
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [dict(BLOCK, evidence="")]))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("review:human")


def test_submit_off_writes_review_intent(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    assert _runs(P, iid, "review")[0]["state"] == "launched"
    P2, iid2, jobs2 = _env(tmp_path / "off")
    P2.run(only=[iid2])
    _finish(jobs2, _runs(P2, iid2)[0]["job_name"], reward=0.0)
    ingest(P2, P2.store.item(iid2))
    P2.run(only=[iid2])
    _finish(jobs2, _runs(P2, iid2)[1]["job_name"], reward=1.0)
    ingest(P2, P2.store.item(iid2))
    P2.cfg["calib"]["allow_submit"] = False
    P2.run(only=[iid2])
    st = P2.store.item(iid2).state
    [rr] = _runs(P2, iid2, "review")
    assert st["stage"] == "S9R" and st["status"] == "waiting" and rr["state"] == "intent" and not rr["submitted"]
    P2.cfg["calib"]["allow_submit"] = True
    P2.run(only=[iid2])                                # re-woken: the intent is launched once
    assert _runs(P2, iid2, "review")[0]["state"] == "launched"


# ------------------------------------------------------------------ revision
def _revise_client(P, files: dict):
    body = "\n".join(f"=== FILE: {k} ===\n{v}" for k, v in files.items()) + "\n=== END ==="
    fake = FakeClient(P.ledger, P.cfg, lambda stage, system, user: body, name="fake-opus")
    P._clients["repair"] = fake
    return fake


def test_revision_applies_archives_calibration_and_locks_thresholds(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [BLOCK, MINOR]))
    _revise_client(P, {"instruction.md": "Write out.csv with column spec (one row per spectrum).\n"})
    import tbsgen.pipeline as pl
    seen = []
    orig = pl.HANDLERS["S6A_run"]
    pl.HANDLERS["S6A_run"] = lambda P_, it_: (seen.append(it_.state.get("thresholds_locked")),
                                               it_.finish("paused", "test:stop", paused_stage="S6A_run"))
    try:
        watch_tick(P, only=[iid])
    finally:
        pl.HANDLERS["S6A_run"] = orig
    it = P.store.item(iid)
    assert seen == [True]
    assert it.state["review_revisions"] == 1 and it.state["calib_epoch"] == 1 and "calib_label" not in it.state
    assert "spectrum" in it.p("task", "instruction.md").read_text()
    assert it.p("repair", "review-1", "before", "instruction.md").read_text().startswith("Write out.csv with column spec.")
    assert "review_revision_pending" not in it.state
    assert (it.p("repair", "review-1", "thresholds.locked.json")).exists()
    assert not it.p("calib", "runs.json").exists() and it.p("calib", "epoch-0", "runs.json").exists()
    assert it.state["calib_history"][0]["label"] == "hard"
    # a new S9 starts from GLM with an epoch suffix
    from tbsgen import calib
    assert calib.job_name_for(it.dir, "glm", 1, 1).endswith("-glm1-e1")


def test_revision_rejects_threshold_change_and_disallowed_paths(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [BLOCK]))
    _revise_client(P, {"tests/test_outputs.py": TEST_OUTPUTS.replace("1.0e-3", "1.0e-2")})
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    # a rejected patch spends its revision round; with the same (bad) patch every time all 3 rounds are spent
    assert st["status"] == "dropped" and "threshold" in st["reason"] and st["review_revisions"] == 3
    assert "1.0e-3" in P.store.item(iid).p("task", "tests", "test_outputs.py").read_text()   # not applied
    task = P.store.item(iid).p("task")
    th = read_json(P.store.item(iid).p("thresholds.json"))["thresholds"]
    rv = review.revision_review(task, {"solution/solve.py": "x"}, {"nodes": []}, th)
    assert not rv["ok"] and "not allowed" in rv["reasons"][0]
    # direction flip is a change too, even with the same literals
    flip = TEST_OUTPUTS.replace('"score", 0.5, 0.9, ">="', '"score", 0.5, 0.9, "<="')
    assert not review.revision_review(task, {"tests/test_outputs.py": flip}, {"nodes": []}, th)["ok"]
    # an indirection (named constant) cannot be verified -> rejected
    ind = "THR = 1.0e-2\n" + TEST_OUTPUTS.replace('"err", 0.5, 1.0e-3,', '"err", 0.5, THR,')
    assert not review.revision_review(task, {"tests/test_outputs.py": ind}, {"nodes": []}, th)["ok"]
    # same literal text, different value from the locked record -> rejected
    th2 = dict(th, err=dict(th["err"], threshold=5e-4))
    assert not review.revision_review(task, {"tests/test_outputs.py": TEST_OUTPUTS}, {"nodes": []}, th2)["ok"]
    assert review.revision_review(task, {"tests/test_outputs.py": TEST_OUTPUTS}, {"nodes": []}, th)["ok"]
    # a wrapper that rescales the threshold at run time is rejected
    wrap = "from _metrics import check as _c\ndef check(n, v, t, d):\n    _c(n, v, t * 10, d)\n" + \
        TEST_OUTPUTS.replace("from _metrics import check\n", "")
    assert not review.revision_review(task, {"tests/test_outputs.py": wrap}, {"nodes": []}, th)["ok"]
    assert review.rebinds_check("import _metrics\n_metrics.check = None\n")
    assert not review.rebinds_check(TEST_OUTPUTS)
    # parameter binding (codex r3): def test_x(check=relaxed)
    par = TEST_OUTPUTS + "\ndef relaxed(n, v, t, d):\n    pass\ndef test_z(check=relaxed):\n    check('err', 1, 1.0e-3, '<=')\n"
    assert review.rebinds_check(par)
    assert not review.revision_review(task, {"tests/test_outputs.py": par}, {"nodes": []}, th)["ok"]
    assert review.rebinds_check(TEST_OUTPUTS + "\nf = lambda check: check\n")
    assert review.rebinds_check(TEST_OUTPUTS + "\nglobals()['check'] = print\n")
    assert review.rebinds_check(TEST_OUTPUTS.replace("from _metrics import check", "from _metrics import check as c"))
    assert review.revision_review(task, {"tests/measure.py": "COL = 'spectrum_index'\n"}, {"nodes": []}, th)["ok"]


def test_revision_bad_path_writes_nothing(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [BLOCK]))
    _revise_client(P, {"../../thresholds.json": "{}", "instruction.md": "x\n"})
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    assert it.state["status"] == "dropped" and "not allowed" in it.state["reason"]
    assert read_json(it.p("thresholds.json"))["thresholds"]["err"]["threshold"] == 1.0e-3
    assert not it.p("repair", "review-1", "patch").exists() and it.p("repair", "review-1", "patch.json").exists()
    assert it.p("task", "instruction.md").read_text().startswith("Write out.csv")


def test_gate_failure_after_revision_retries_or_drops_never_ordinary_s8(tmp_path):
    """A gate failing after a review revision never runs the ordinary S8 (which may touch the reference): the
    review revision is retried with the failure fed back while rounds remain, else the item is dropped."""
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    write_json(it.p("review", "verdict.json"), dict(V("revise", [BLOCK]), round=1,
                                                    _normalized=review.normalize(V("revise", [BLOCK]))))
    it.state.update(stage="S8", status="active", thresholds_locked=True, review_revisions=1, review_round=1,
                    post_revision=True, repair_from="gates:oracle")
    it.save()
    stages_b.s8(P, it)
    st = P.store.item(iid).state
    assert st["stage"] == "S8" and st["status"] == "active" and st["repair_from"] == "review:r1"
    assert "gates:oracle" in st["revision_feedback"]
    it = P.store.item(iid)
    it.state.update(review_revisions=3, repair_from="gates:oracle")
    it.save()
    stages_b.s8(P, it)
    st = P.store.item(iid).state
    assert st["status"] == "dropped" and st["reason"].startswith("review:gate_failed")


def test_s10_gate_sends_unreviewed_hard_item_to_review(tmp_path):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="S10", status="active", calib_label="hard")
    it.save()
    stages_b.s10(P, it)
    st = P.store.item(iid).state
    assert st["stage"] == "S9R" and not (P.store.root / "release" / "manifest.jsonl").exists()


def test_stale_prepared_copy_blocks_recalibration(tmp_path):
    from tbsgen import calib
    P, iid, _ = _env(tmp_path, prepared_tasks_dir=str(tmp_path / "prep"))
    it = P.store.item(iid)
    assert calib.stale_prepared(P.cfg, it.dir, "glm", 0) is None
    assert "predate" in calib.stale_prepared(P.cfg, it.dir, "glm", 1)
    write_json(it.p("calib", "prepare.json"), dict(ok=True, pkg_sha=calib.pkg_sha(it.p("pkg"))))
    assert calib.stale_prepared(P.cfg, it.dir, "glm", 1) is None


def test_second_review_revise_revises_again_until_the_limit(tmp_path):
    """A second (or third) non-pass review is not a human exit: revise again while review.max_revisions allows,
    then drop."""
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="S9R", status="active", calib_label="hard", review_round=2, review_revisions=1)
    it.save()
    P.run(only=[iid])
    rr = _runs(P, iid, "review")[0]
    assert rr["job_name"].endswith("-review2")
    _finish(jobs, rr["job_name"], verdict=V("revise", [BLOCK]))
    import tbsgen.pipeline as pl
    orig = pl.HANDLERS["S8"]
    pl.HANDLERS["S8"] = lambda P_, it_: it_.finish("paused", "test:stop", paused_stage="S8")
    try:
        watch_tick(P, only=[iid])
    finally:
        pl.HANDLERS["S8"] = orig
    st = P.store.item(iid).state
    assert st["stage"] == "S8" and st["repair_from"] == "review:r2"
    # at the limit (3 revisions used) the same verdict drops the item
    P, iid, jobs = _env(tmp_path / "lim")
    it = P.store.item(iid)
    it.state.update(stage="S9R", status="active", calib_label="hard", review_round=4, review_revisions=3)
    it.save()
    P.run(only=[iid])
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [BLOCK]))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "dropped" and st["reason"].startswith("review:drop r4 (revision limit 3")


def test_locked_s6b_keeps_thresholds_and_parks_if_reference_fails(tmp_path):
    from tbsgen import stages_a
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="S6B", status="active", thresholds_locked=True)
    it.save()
    write_json(it.p("S6A_measure.json"), dict(ref={"0": {"err": 2e-4, "score": 0.95}}, mid={"0": {"err": 1.0, "score": 0.1}},
                                              naive={"0": {"err": 2.0, "score": 0.0}}, nop={}, runs={}))
    import tbsgen.stages_a as A
    orig = A.set_thresholds
    A.set_thresholds = lambda spec, meas, cfg: dict(thresholds={"err": dict(direction="<=", threshold=5e-3),
                                                                "score": dict(direction=">=", threshold=0.5)})
    try:
        stages_a.s6b(P, it)
        it = P.store.item(iid)
        assert it.state["stage"] == "S6C"
        th = read_json(it.p("thresholds.json"))["thresholds"]
        assert th["err"]["threshold"] == 1.0e-3 and th["score"]["threshold"] == 0.9     # unchanged
        rl = read_json(it.p("thresholds.relock.json"))
        assert rl["compare"]["err"]["fresh_would_be_looser"] is True
        # reference now misses a locked threshold -> not a review revision in flight: dropped
        write_json(it.p("S6A_measure.json"), dict(ref={"0": {"err": 5e-3, "score": 0.95}}, mid={}, naive={}, nop={},
                                                  runs={}))
        it.state.update(stage="S6B", status="active")
        it.save()
        stages_a.s6b(P, it)
        st = P.store.item(iid).state
        assert st["status"] == "dropped" and "locked threshold" in st["reason"]
    finally:
        A.set_thresholds = orig


def test_withdraw_released_item_on_non_pass(tmp_path):
    from tbsgen.stages_b import _released, withdraw
    P, iid, _ = _env(tmp_path)
    rel = P.store.root / "release"
    rel.mkdir()
    (rel / "manifest.jsonl").write_text(json.dumps(dict(item_id=iid, label="hard")) + "\n")
    assert _released(P, iid)
    assert withdraw(P, iid, "review r1: revise")
    assert not _released(P, iid) and not withdraw(P, iid, "again")


def test_remote_alias_task_identity(tmp_path):
    from tbsgen import calib
    jd = tmp_path / "job"
    jd.mkdir()
    assert not calib.task_matches(jd, "/remote/t", "/local/t")
    (jd / "tbsgen-remote.json").write_text(json.dumps(dict(remote_task="/remote/t", local_task="/local/t")))
    assert calib.task_matches(jd, "/remote/t", "/local/t")
    assert not calib.task_matches(jd, "/remote/other", "/local/t")


def test_check_signature_and_trace_digest(tmp_path):
    sig = review.check_calls(TEST_OUTPUTS)
    assert sig == sorted([("err", 1.0e-3, "<="), ("score", 0.9, ">=")])
    assert review.check_calls("check('a', v, X, '<=')") is None
    t = tmp_path / "trial"
    (t / "agent").mkdir(parents=True)
    (t / "agent" / "claude-code.txt").write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"id": "a", "content": [{"type": "text", "text": "Plan: read data"}]}},
        {"type": "assistant", "message": {"id": "a", "content": [{"type": "text", "text": "Plan: read data"}]}},
        {"type": "assistant", "message": {"id": "b", "content": [
            {"type": "tool_use", "id": "x", "name": "Write", "input": {"file_path": "/app/s.py", "content": "abc"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}},
        {"type": "result", "subtype": "success", "num_turns": 3, "result": "final words"}]))
    d = review.trace_digest(t)
    assert d.count("Plan: read data") == 1 and "Write: /app/s.py (3 chars)" in d and "final words" in d


def test_stale_pass_is_not_rebound_to_a_changed_package(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["stage"] == "S10"
    # package changes after the pass (e.g. by hand), item is put back before S10
    it = P.store.item(iid)
    it.p("pkg", "instruction.md").write_text("changed\n")
    it.state.update(stage="S10", status="active")
    it.save()
    P.run(only=[iid])
    it = P.store.item(iid)
    rr = _runs(P, iid, "review")
    assert it.state["stage"] == "S9R" and len(rr) == 2 and rr[1]["outcome"] is None      # a new review job
    assert rr[1]["pkg_sha"] != rr[0]["pkg_sha"]


def test_revision_pkg_checks_runtime_thresholds_and_data(tmp_path):
    from tbsgen.stages_b import _revision_pkg_problem
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    it.state["revision_data_hash"] = review.pkg_data_hash(it.p("pkg"))
    m = it.p("runs", "s7-1", "logs-ref0", "metrics.jsonl")
    m.parent.mkdir(parents=True)
    rows = [dict(name="err", value=1e-4, threshold=1.0e-3, direction="<="),
            dict(name="score", value=0.95, threshold=0.9, direction=">="),
            dict(name="_solution_cli_run", value=1.0, threshold=1800.0, direction="<=", aux=True)]
    m.write_text("\n".join(json.dumps(r) for r in rows))
    assert _revision_pkg_problem(it, dict(attempt=1)) is None
    rows[0]["threshold"] = 1.0e-2
    m.write_text("\n".join(json.dumps(r) for r in rows))
    assert "applied thresholds" in _revision_pkg_problem(it, dict(attempt=1))
    rows[0]["threshold"] = 1.0e-3
    m.write_text("\n".join(json.dumps(r) for r in rows))
    (it.p("pkg", "environment", "data", "x.npy")).write_text("new data")
    assert "changed packaged data" in _revision_pkg_problem(it, dict(attempt=1))
    (it.p("pkg", "environment", "data", "x.npy")).unlink()
    (it.p("pkg", "environment", "data", "README.md")).write_text("better notes")
    assert _revision_pkg_problem(it, dict(attempt=1)) is None


def test_non_object_verdict_is_invalid_and_rerun(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=42, reward=0.0)
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    rr = _runs(P, iid, "review")
    assert rr[0]["outcome"] == "invalid" and st["stage"] == "S9R" and len(rr) == 2     # a new review run


def test_verdict_of_one_job_cannot_be_routed_for_another(tmp_path):
    """codex r3: round-level verdict.json overwritten by another job's pass must not release this job's package."""
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    rr = _runs(P, iid, "review")
    _finish(jobs, rr[0]["job_name"], verdict=V("pass"))
    from tbsgen.stages_b import _route_review
    it = P.store.item(iid)
    # a forged / overwritten verdict bound to a different job
    other = dict(rr[0], job_name="tbsgen-other-job", outcome="done")
    write_json(it.p("review", "r1", "tbsgen-other-job.verdict.json"),
               dict(V("pass"), _normalized=review.normalize(V("pass")), pkg_sha="deadbeef", calib_epoch=0,
                    job="tbsgen-other-job"))
    _route_review(P, it, other, 1)
    st = P.store.item(iid).state
    assert st["stage"] == "S9R" and st["status"] == "active"                 # re-run, not routed
    assert any(r.get("outcome") == "invalid" and "bound to" in r.get("invalid_why", "") for r in _runs(P, iid, "review"))
    assert not (P.store.root / "release" / "manifest.jsonl").exists()


def test_regate_released_item_withdraws_locks_and_rereviews(tmp_path):
    """A harness change (verifier template): regate withdraws the released item, locks thresholds and data, goes
    back to S6A_run; calibration runs kept by default; the S10 review gate then demands a new review because the
    package hash changed."""
    from tbsgen.stages_b import _released, regate, review_gate
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    P.run(only=[iid])
    it = P.store.item(iid)
    assert it.state["status"] == "done" and _released(P, iid)
    out = regate(P, it, "grader template v2.5")
    it = P.store.item(iid)
    assert out["withdrawn"] and not _released(P, iid)
    assert it.state["stage"] == "S6A_run" and it.state["status"] == "active" and it.state["thresholds_locked"]
    assert it.state["revision_data_hash"] and it.state["regates"][0]["reason"] == "grader template v2.5"
    assert len(_runs(P, iid)) == 2 and it.state.get("calib_epoch", 0) == 0
    it.p("pkg", "tests", "_metrics.py").write_text("# new template\n")      # what the new PKG step changes
    it.state.update(stage="S10", status="active")
    assert review_gate(P, it) == "package changed after the review"


def test_regate_recalibrate_archives_epoch(tmp_path):
    from tbsgen.stages_b import regate
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    it = P.store.item(iid)
    it.state.update(stage="S10", status="done")
    it.save()
    regate(P, it, "x", recalibrate=True)
    it = P.store.item(iid)
    assert it.state["calib_epoch"] == 1 and _runs(P, iid) == [] and "calib_label" not in it.state


def test_regate_withdraws_before_the_item_can_run(tmp_path, monkeypatch):
    """Codex r3: if the withdrawal fails, the item must stay finished (not runnable) and regate can be retried."""
    import pytest
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    it = P.store.item(iid)
    it.state.update(stage="S10", status="done")
    it.save()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(stages_b, "withdraw", boom)
    with pytest.raises(OSError):
        stages_b.regate(P, it, "x")
    it = P.store.item(iid)
    assert it.state["status"] == "done" and not it.state.get("thresholds_locked")
    monkeypatch.undo()
    assert stages_b.regate(P, it, "x")["stage"] == "S6A_run"



def test_scoring_only_revision_rescores_instead_of_recalibrating(tmp_path, monkeypatch):
    """A revision that changes only the grader keeps the calibration: S9 re-scores the saved deliverables with the
    new grader; unchanged outcomes keep the label (S9R reviews again), a changed or missing one recalibrates."""
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"],
            verdict=V("revise", [dict(BLOCK, problem="grader_io", title="public mask read from /app/data")]))
    _revise_client(P, {"tests/measure.py": "COL = 'spec'  # mask from /tests/public_inputs\n"})
    import tbsgen.pipeline as pl
    orig = pl.HANDLERS["S6A_run"]
    pl.HANDLERS["S6A_run"] = lambda P_, it_: it_.finish("paused", "test:stop", paused_stage="S6A_run")
    try:
        watch_tick(P, only=[iid])
    finally:
        pl.HANDLERS["S6A_run"] = orig
    it = P.store.item(iid)
    assert it.state.get("calib_epoch", 0) == 0 and it.state["calib_label"] == "hard"      # not archived
    assert it.state["rescore_pending"]["files"] == ["tests/measure.py"] and len(_runs(P, iid)) == 2
    glm, teacher = [r["job_name"] for r in _runs(P, iid)]
    for j in (glm, teacher):
        d = tmp_path / "deliv" / j
        d.mkdir(parents=True)
        (d / "solution_cli.py").write_text("print(1)\n")
        review.save_deliverable(it.dir, j, d, note="replayed from trace")
    # unchanged outcomes: GLM still fails, teacher still passes -> keep
    monkeypatch.setattr(review, "rescore_run", lambda cfg, d, r: dict(ok=True, reward="1" if r["job_name"] == teacher
                                                                        else "0", passed=r["job_name"] == teacher))
    it.state.update(stage="S9", status="active")
    it.save()
    assert stages_b._rescore_pending(P, it) is None
    assert "rescore_pending" not in it.state and it.state.get("calib_epoch", 0) == 0
    assert read_json(it.p("calib", "rescore.json"))["keep"] is True
    # changed outcome (the new grader fails the teacher) -> archive and recalibrate from GLM
    it.state["rescore_pending"] = dict(reason="scoring-only revision")
    monkeypatch.setattr(review, "rescore_run", lambda cfg, d, r: dict(ok=True, reward="0", passed=False))
    stages_b._rescore_pending(P, it)
    assert it.state["calib_epoch"] == 1 and "calib_label" not in it.state and _runs(P, iid) == []
    # a sandbox that wrote no reward is not a re-score: counted as missing -> recalibrate, never a pass
    assert review.rescore_item(P.cfg, it.dir, [dict(job_name="nope", outcome="pass")])["missing"] == ["nope"]


def test_rescore_can_be_switched_off(tmp_path):
    P, iid, jobs = _env(tmp_path, review_kw={"rescore_scoring_only": False})
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [dict(BLOCK, problem="grader_io")]))
    _revise_client(P, {"tests/measure.py": "COL = 'spec'\n"})
    import tbsgen.pipeline as pl
    orig = pl.HANDLERS["S6A_run"]
    pl.HANDLERS["S6A_run"] = lambda P_, it_: it_.finish("paused", "test:stop", paused_stage="S6A_run")
    try:
        watch_tick(P, only=[iid])
    finally:
        pl.HANDLERS["S6A_run"] = orig
    it = P.store.item(iid)
    assert it.state["calib_epoch"] == 1 and "rescore_pending" not in it.state


def test_rescore_failure_stops_s9_and_retry_keeps_first_data_hash(tmp_path, monkeypatch):
    """Codex r1: a rescore that cannot run leaves the item waiting (S9 does not relabel); a revision retry keeps
    the data hash recorded before the first revision attempt."""
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    it = P.store.item(iid)
    it.state.update(stage="S9", status="active", rescore_pending=dict(reason="t"))
    it.save()
    def boom(*a, **k):
        raise RuntimeError("docker down")
    monkeypatch.setattr(review, "rescore_item", boom)
    stages_b.s9(P, it)
    st = P.store.item(iid).state
    assert st["stage"] == "S9" and st["status"] == "waiting" and st.get("rescore_pending")
    # retry with thresholds still locked: the recorded hash wins over the current pkg/
    it = P.store.item(iid)
    it.state.update(thresholds_locked=True, revision_data_hash="first-hash")
    it.save()
    assert ((it.state.get("revision_data_hash") if it.state.get("thresholds_locked") else None)
            or "pkg") == "first-hash"


def test_codex_r3_regate_without_rescore_recalibrates_and_exploit_limit_after_revision(tmp_path):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path, review_kw={"rescore_scoring_only": False})
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    stages_b.regate(P, P.store.item(iid), "template v3")
    st = P.store.item(iid).state
    assert st["calib_epoch"] == 1 and "calib_label" not in st and "rescore_pending" not in st
    # exploit repair budget spent after a review revision: the review revision budget decides (retry), not a drop
    P, iid, _ = _env(tmp_path / "ex")
    it = P.store.item(iid)
    write_json(it.p("review", "verdict.json"), dict(V("revise", [BLOCK]), round=1,
                                                    _normalized=review.normalize(V("revise", [BLOCK]))))
    it.state.update(stage="S8", status="active", thresholds_locked=True, post_revision=True, review_revisions=1,
                    review_round=1, exploit_repair_rounds=2, repair_from="gates:exploit_valid")
    it.save()
    stages_b.s8(P, it)
    st = P.store.item(iid).state
    assert st["status"] == "active" and st["repair_from"] == "review:r1", st


def test_review_relay_refusals_do_not_park(tmp_path):
    """S9R runs refused at the relay preflight are no attempts: review.infra_retries is untouched, the item is not
    parked review:infra, and the review launches again once the relay answers."""
    P, iid, jobs = _env(tmp_path, review_kw={"infra_retries": 1}, preflight_retry_s=0)
    _to_label(P, iid, jobs)
    teach = Path(P.cfg["calib"]["harbor_root"]) / "teach.sh"
    good = teach.read_text()
    rr = _runs(P, iid, "review")[-1]
    # the first review job is already launched by the good launcher: turn it into a refusal
    import shutil
    shutil.rmtree(jobs / rr["job_name"])
    Path(rr["log"]).write_text("relay preflight http=400\nJ: harbor exit 3\n")
    Path(rr["log"] + ".exit").write_text("3")
    teach.write_text("#!/bin/bash\necho 'relay preflight http=400'\necho 'J: harbor exit 3'\nexit 3\n")
    for _ in range(4):
        watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    runs = _runs(P, iid, "review")
    assert st["stage"] == "S9R" and st["status"] != "parked", st
    assert len(runs) >= 3 and all(r["preflight_refused"] for r in runs if r.get("outcome") == "infra")
    teach.write_text(good)
    watch_tick(P, only=[iid])
    P.run(only=[iid])
    last = _runs(P, iid, "review")[-1]
    assert last["state"] == "launched" and not last.get("preflight_refused") and \
        (jobs / last["job_name"] / "config.json").exists(), last


def test_suspected_scopes_bind_bundle_validator_and_ingest(tmp_path, caplog):
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    trial = dict(status="shortcut_suspected", transformed_files=[dict(path="side.csv", declared_deliverable=False)],
                 verifier=dict(out="accepted", err="", metrics=[]), reward="1")
    report = dict(scopes={"hidden:A": trial})
    it.state["gates_attempt"] = 10
    it.save()
    write_json(it.p("gates/attempt-9.json"), dict(details={"coverage": dict(scopes={"public": trial})}))
    write_json(it.p("gates/attempt-10.json"), dict(probe_policy_version=gates.PROBE_POLICY_VERSION,
                                                   details={"coverage": report, "hidden_garbage": report}))
    _to_label(P, iid, jobs)
    [rec] = _runs(P, iid, "review")
    task = Path(rec["task"])
    bundle = read_json(task / "environment/review/records/output_probes.json")
    first, second = "one_confident/hidden:A", "hidden_garbage/hidden:A"
    assert bundle["suspected_scopes"] == rec["suspected_scopes"] == [first, second]
    assert bundle["probes"]["one_confident"] == report
    prompt = (task / "instruction.md").read_text()
    assert all(s in prompt for s in (first, second, review.REVIEW_PROMPT_VERSION, "reproduce"))
    artifact = dict(classification="probe_artefact", observed_reward="1",
                    observation="The optional plotting branch was never executed.",
                    reason="Only the diagnostic file changed.", repro_id="first-run")
    other = dict(artifact, repro_id="second-run")
    verdict = dict(V("pass"), probe_findings={first: artifact, second: other})
    journal = tmp_path / "repro.jsonl"
    rows = [_receipt(first, "first-run"), _receipt(second, "second-run"), _receipt("extra/public", "extra-run")]
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def check(expected):
        src = tmp_path / "verdict.json"
        write_json(src, verdict)
        validation = subprocess.run([sys.executable, task / "tests/s9r_validate.py", src, tmp_path / "parsed.json",
                                     journal], capture_output=True, text=True)
        assert validation.returncode == (1 if expected is None else 0), validation.stdout
        assert review.normalize(verdict, suspected_scopes=[first, second],
                                repros=review.read_repros(journal))["verdict"] == expected

    assert "command" not in artifact
    assert bundle["repro_commands"] == {scope: review.repro_command(scope) for scope in (first, second)}
    for argv in (["bash", "/review/pkg/tests/test.sh"], rows[1]["command"], [*rows[0]["command"], "extra"]):
        journal.write_text("\n".join(json.dumps(row) for row in [dict(rows[0], command=argv), *rows[1:]]))
        check(None)
    journal.write_bytes(json.dumps(rows[0]).encode() + b'\n{"truncated"\n' +
                        json.dumps(rows[1]).encode() + b"\n\xff\n" + json.dumps(rows[2]).encode())
    check("pass")
    assert "Skipped 2 unparseable reproduction journal lines" in caplog.text
    del verdict["probe_findings"][second]
    check(None)
    verdict["probe_findings"][second] = other
    # A verdict file cannot supply its own receipts. With no trial log this attempt is invalid.
    _finish(jobs, rec["job_name"], verdict=dict(verdict, repros=review.read_repros(journal)))
    watch_tick(P, only=[iid])
    invalid, retry = _runs(P, iid, "review")
    assert invalid["outcome"] == "invalid" and retry["suspected_scopes"] == [first, second]
    for scope, key in ((first, "first-run"), ("extra/public", "extra-run")):
        gap = dict(artifact, repro_id=key, classification="grader_gap", issue_id="I1")
        verdict["probe_findings"][scope] = gap
        check(None)
        for problem, routed in (("test_gap", "revise"), ("core_invalid", "drop")):
            verdict["issues"] = [dict(BLOCK, problem=problem)]
            check(routed)
        for change in ({"repro_id": "absent"}, {"repro_id": "second-run"}, {"observed_reward": True},
                       {"observed_reward": 0.5}, {"observation": "short"}, {"issue_id": "absent"},
                       {"repro_id": "", "observation": "I have not attempted to execute the grader on this scope."}):
            verdict["probe_findings"][scope] = dict(gap, **change)
            check(None)
        verdict.update(issues=[], probe_findings={first: artifact, second: other})
    _finish(jobs, retry["job_name"], verdict=verdict, repros=rows)
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["status"] == "done"


def test_short_probe_threshold_log(tmp_path):
    from tbsgen.stages_b import _revision_pkg_problem
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    locked = read_json(it.p("thresholds.json"))["thresholds"]
    rows = [json.dumps(dict(name=k, **v)) for k, v in locked.items()]
    ref = it.p("runs/s7-1/logs-ref0/metrics.jsonl")
    probe = it.p("runs/s7-1/logs-exploit__one_confident__public0/metrics.jsonl")
    ref.parent.mkdir(parents=True)
    probe.parent.mkdir(parents=True)
    ref.write_text("\n".join(rows))
    probe.write_text(rows[0])
    assert _revision_pkg_problem(it, {"attempt": 1}) is None
    probe.write_text(json.dumps(dict(json.loads(rows[0]), direction=">=")))
    assert probe.parent.name in _revision_pkg_problem(it, {"attempt": 1})
    probe.write_text(rows[0])
    for partial in ("", rows[0]):
        ref.write_text(partial)
        assert ref.parent.name in _revision_pkg_problem(it, {"attempt": 1})


def test_pkg_and_bundle_use_historical_passing_attempt(tmp_path):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    for n in (9, 10):
        d = it.p(f"runs/s7-{n}/detA")
        write_json(d / "environment/data/seed.json", n)
        write_json(d / "tests/truth/sample.json", n)
        write_json(it.p(f"gates/attempt-{n}.json"), dict(probe_policy_version=gates.PROBE_POLICY_VERSION, attempt=n, gates={"oracle": True},
                   details={"coverage": dict(scopes={str(n): dict(status="shortcut_suspected")})}))
    it.advance("S7", "PKG", attempt=9)
    stages_b.pkg(P, it)
    assert read_json(it.p("pkg/environment/data/seed.json")) == 9
    bundle = review.write_bundle(it.dir, tmp_path / "bundle", it.state)
    assert bundle["suspected_scopes"] == ["one_confident/9"]
    assert [p.name for p in (tmp_path / "bundle/records/s7_gates").iterdir()] == ["attempt-9.json"]
    write_json(it.p("gates/attempt-9.json"), {"attempt": 9})
    review.write_bundle(it.dir, tmp_path / "bundle", it.state)
    assert not (tmp_path / "bundle/records/s7_gates").exists()


def _current_probe(it):
    it.state["gates_attempt"] = 1
    it.save()
    write_json(it.p("gates/attempt-1.json"), dict(probe_policy_version=gates.PROBE_POLICY_VERSION,
               details=dict(coverage=dict(scopes={"public": dict(status="shortcut_suspected")}))))
    return "one_confident/public"


def test_disabled_review_parks_and_resumes(tmp_path):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path, review_kw={"enabled": False})
    it = P.store.item(iid)
    scope = _current_probe(it)
    _to_label(P, iid, jobs)
    it.load()
    assert it.state["status"] == "parked" and it.state["reason"] == "parked:probe_unreviewed"
    assert it.state["history"][-1]["detail"]["suspected_scopes"] == [scope] and not stages_b._released(P, iid)
    assert not P.resume(only=[iid])
    it.state.update(stage="S10", status="active")
    stages_b.s10(P, it)
    assert it.state["reason"] == "parked:probe_unreviewed" and not stages_b._released(P, iid)
    P.cfg["review"]["enabled"] = True
    assert P.resume(only=[iid]) == [iid]
    P.run(only=[iid])
    [rec] = _runs(P, iid, "review")
    assert rec["suspected_scopes"] == [scope]
    finding = dict(classification="probe_artefact", command="python verify.py", observation="Only the notes changed.",
                   observed_reward="1", reason="The declared table is identical.", repro_id="run")
    _finish(jobs, rec["job_name"], verdict=dict(V("pass"), probe_findings={scope: finding}), repros=[_receipt(scope, "run")])
    watch_tick(P, only=[iid])
    assert P.store.item(iid).state["status"] == "done" and stages_b._released(P, iid)


@pytest.mark.parametrize("stage,status", [("S8", "active"), ("PKG", "active"), ("S9", "paused"),
                                          ("S9R", "waiting"), ("S10", "active")])
def test_legacy_report_regates_once_without_spend(tmp_path, stage, status):
    import shutil
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    if stage in ("S8", "PKG"):
        shutil.rmtree(it.p("pkg"))
    it.state.update(stage=stage, status=status, repair_rounds=2, review_revisions=1,
                    attempts={"S7": 4, "S5G": 2}, repair_from="gates:coverage")
    it.save()
    legacy = dict(details={"coverage": {"scopes": "unreadable by the current parser"}})
    write_json(it.p("gates/attempt-4.json"), legacy)
    assert review.suspected_scopes(legacy) == []
    before = P.item_cost(iid)
    if status == "paused":
        assert P.resume(only=[iid]) == [iid]
    else:
        P.run(max_steps=1, only=[iid])
    it.load()
    assert (it.state["stage"], it.state["status"]) == ("S6A_run", "active")
    [record] = it.state["regates"]
    assert record["probe_report_id"] and record["reason"] == "probe_policy_migration"
    assert it.state["thresholds_locked"] and "repair_from" not in it.state
    assert it.state["attempts"] == {"S7": 4, "S5G": 2}
    assert it.state["repair_rounds"] == 2 and it.state["review_revisions"] == 1
    assert P.item_cost(iid) == before and _runs(P, iid, "review") == []
    it.state.update(stage=stage, status="active")
    it.save()
    assert P.resume(only=[iid]) == []
    assert it.load().state["regates"] == [record]
    write_json(it.p("gates/attempt-5.json"), legacy)
    assert P.resume(only=[iid]) == [iid]
    assert it.load().state["regates"][-1]["probe_report_id"] != record["probe_report_id"]


@pytest.mark.parametrize("status,current", [("done", False), ("dropped", False), ("active", True)])
def test_terminal_and_current_reports_stay_put(tmp_path, status, current):
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="S10", status=status)
    it.save()
    manifest = P.store.root / "release/manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(dict(item_id=iid)) + "\n")
    _current_probe(it)
    if not current:
        write_json(it.p("gates/attempt-1.json"), dict(probe_policy_version="old"))
    before = it.p("state.json").read_bytes(), manifest.read_bytes()
    assert not P.resume(only=[iid])
    if status != "active":
        P.run(only=[iid])
    assert (it.p("state.json").read_bytes(), manifest.read_bytes()) == before


def _v2_job(it, rec):
    rec.pop("review_contract_version")
    rec["review_prompt_version"] = "s9r-v2"
    write_json(it.p("review/runs.json"), [rec])


@pytest.mark.parametrize("state", ["launched", "done", "invalid"])
def test_v2_valid_verdict_gets_one_free_review(tmp_path, state):
    P, iid, jobs = _env(tmp_path, review_kw={"infra_retries": 0})
    it = P.store.item(iid)
    scope = _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    _v2_job(it, rec)
    finding = dict(reproduction="Ran the packaged verifier", reason="Only diagnostic output changed.",
                   classification="probe_artefact")
    verdict = dict(V("pass"), probe_findings={scope: finding})
    if state != "launched":
        contract = "s9r-v2" if state == "done" else review.CURRENT_CONTRACT
        norm = review.normalize(verdict, suspected_scopes=[scope], contract=contract)
        write_json(it.p("review/r1", rec["job_name"] + ".verdict.json"),
                   dict(verdict, _normalized=norm, pkg_sha=rec["pkg_sha"], calib_epoch=0, job=rec["job_name"]))
        write_json(it.p("review/runs.json"), [dict(rec, outcome=state, verdict=norm["verdict"],
                   trial=dict(trial="trial", verdict_trial="trial", reward=1))])
        it.load().state["status"] = "active"
        it.save()
    else:
        _finish(jobs, rec["job_name"], verdict=verdict)
        assert ingest(P, it)["outcome"] == "infra"
    P.run(only=[iid])
    obsolete, fresh = _runs(P, iid, "review")
    assert obsolete["outcome"] == "infra" and obsolete["obsolete_probe_review"]
    assert fresh["review_contract_version"] == review.CURRENT_CONTRACT
    P.resume(only=[iid])
    P.run(only=[iid])
    assert len(_runs(P, iid, "review")) == 2
    _finish(jobs, fresh["job_name"], verdict=verdict)
    watch_tick(P, only=[iid])
    assert it.load().state["reason"].startswith("review:invalid") and len(_runs(P, iid, "review")) == 2


def test_invalid_v2_verdict_spends_invalid_budget(tmp_path):
    P, iid, jobs = _env(tmp_path, review_kw={"infra_retries": 0})
    it = P.store.item(iid)
    _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    _v2_job(it, rec)
    _finish(jobs, rec["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    assert it.load().state["reason"].startswith("review:invalid")
    assert _runs(P, iid, "review")[0]["outcome"] == "invalid"


@pytest.mark.parametrize("old", [True, False])
@pytest.mark.parametrize("legacy_report", [True, False])
def test_received_drop_survives_contract_and_regate(tmp_path, old, legacy_report):
    from tbsgen.stages_b import running_jobs
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    scope = _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    if old:
        _v2_job(it, rec)
    if legacy_report:
        write_json(it.p("gates/attempt-1.json"), {})
        P.resume(only=[iid])
        from tbsgen.cli import _running
        assert it.load().state["stage"] == "S6A_run" and running_jobs(P) == [rec["job_name"]]
        assert _running(P, iid)
    verdict = V("drop", [BLOCK])
    if not old:
        verdict["probe_findings"] = {scope: dict(classification="grader_gap", issue_id="I1", reason="Mismatch",
            command="pytest", observed_reward=1, repro_id="run", observation="The synthetic grader reported success.")}
    _finish(jobs, rec["job_name"], reward=0, verdict=verdict, repros=[_receipt(scope, "run")])
    watch_tick(P, only=[iid])
    assert it.load().state["status"] == "dropped" and it.state["reason"].startswith("review:drop")
    assert _runs(P, iid, "review")[0]["outcome"] == "done" and running_jobs(P) == []


@pytest.mark.parametrize("change", ["complete", "v4_complete", "finding", "package", "round", "epoch", "contract"])
def test_s10_review_disabled_checks_bound_findings(tmp_path, change):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    scope = _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    finding = dict(classification="probe_artefact", command="pytest", observed_reward="1", repro_id="run",
                   observation="5 passed, 1 skipped in 0.3s", reason="Only diagnostic output changed.")
    _finish(jobs, rec["job_name"], verdict=dict(V("pass"), probe_findings={scope: finding}), repros=[_receipt(scope, "run")])
    ingest(P, it)
    P.run(only=[iid], max_steps=1)
    assert it.load().state["stage"] == "S10"
    P.cfg["review"]["enabled"] = False
    if change == "finding":
        verdict = read_json(it.p("review/verdict.json"))
        verdict["probe_findings"] = {}
        write_json(it.p("review/verdict.json"), verdict)
    elif change == "package":
        it.p("pkg/instruction.md").write_text("New package statement.")
    elif change in ("round", "epoch"):
        it.state.update({"round": {"review_round": 2}, "epoch": {"calib_epoch": 1}}[change])
        it.save()
    elif change in ("contract", "v4_complete"):
        job, = _runs(P, iid, "review")
        version = "s9r-v4" if change == "v4_complete" else "s9r-v2"
        write_json(it.p("review/runs.json"), [dict(job, review_contract_version=version)])
    stages_b.s10(P, it)
    if change in ("complete", "v4_complete"):
        assert it.state["status"] == "done" and stages_b._released(P, iid)
        manifest = P.store.root / "release/manifest.jsonl"
        before = manifest.read_bytes()
        write_json(it.p("review/verdict.json"), {})
        it.state.update(stage="S10", status="active")
        stages_b.s10(P, it)
        assert manifest.read_bytes() == before and stages_b._released(P, iid)
    else:
        assert it.state["reason"] == "parked:probe_unreviewed" and not stages_b._released(P, iid)


@pytest.mark.parametrize("phrase", ["not run", "did not run", "didn't run", "was not executed", "could not run",
                                    "unable to run", "skipped running", "no run"])
@pytest.mark.parametrize("field", ["observation", "command"])
def test_legacy_v5_reproduction_rejects_explicit_negations(phrase, field):
    finding = dict(classification="probe_artefact", command="pytest", observed_reward=0.5,
                   observation="5 passed, 1 skipped in 0.3s", reason="Only diagnostic output changed.")
    verdict = dict(V("pass"), probe_findings={"scope": finding})
    assert not review.finding_errors(verdict, ["scope"], "s9r-v5")
    finding[field] = f"pytest ./checks.py ({phrase} in this environment)"
    assert review.finding_errors(verdict, ["scope"], "s9r-v5")


@pytest.mark.parametrize("started", [True, False])
def test_regate_settles_old_calibration_without_charging_a_retry(tmp_path, started):
    from tbsgen import calib
    P, iid, jobs = _env(tmp_path, allow_submit=started, infra_retries=0)
    P.run(only=[iid])
    rec, = _runs(P, iid)
    it = P.store.item(iid)
    write_json(it.p("gates/attempt-1.json"), {})
    assert P.resume(only=[iid]) == [iid]
    assert it.load().state["stage"] == "S6A_run"
    if started:
        _finish(jobs, rec["job_name"], reward=0.0)
        watch_tick(P, only=[iid], max_steps=0)
        assert it.load().state["status"] == "active" and _runs(P, iid)[0]["outcome"] == "fail"
    else:
        old, = _runs(P, iid)
        assert old["obsolete_probe_job"] and calib.next_action([old], P.cfg) == ("run", "glm")


def test_s7_allocates_above_disk_and_binds_passing_report(tmp_path, monkeypatch):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="PKG", attempts={"S7": 1})
    it.save()
    write_json(it.p("gates/attempt-3.json"), {"attempt": 3})
    assert not it.p("runs/s7-3/detA").exists()
    assert stages_b.migrate_probe_report(P, it)
    assert it.state["stage"] == "S6A_run"

    def run_gates(cfg, item, spec, thresholds, attempt):
        assert attempt == 4
        run = item / f"runs/s7-{attempt}"
        (run / "detA/tests/truth").mkdir(parents=True)
        write_json(run / "detA/environment/data/README.json", attempt)
        logs = run / "logs-ref0/metrics.jsonl"
        logs.parent.mkdir()
        logs.write_text("\n".join(json.dumps(dict(name=k, **v)) for k, v in thresholds.items()))
        return dict(attempt=attempt, ok=True, gates={"oracle": True}, probe_policy_version=gates.PROBE_POLICY_VERSION,
                    details=dict(coverage=dict(scopes={"public": dict(status="shortcut_suspected")})))

    monkeypatch.setattr(gates, "run_gates", run_gates)
    it.goto("S7", "remeasured")
    stages_b.s7(P, it)
    assert it.load().state["gates_attempt"] == it.state["attempts"]["S7"] == 4
    # A stray report cannot select another data tree, suppress review, or trigger a second migration.
    write_json(it.p("gates/attempt-5.json"), {"attempt": 5})
    assert not stages_b.legacy_report_id(it)
    stages_b.pkg(P, it)
    assert it.state["stage"] == "S9" and read_json(it.p("pkg/environment/data/README.json")) == 4
    assert stages_b._probe_scopes(it) == ["one_confident/public"]
    bundle = review.write_bundle(it.dir, tmp_path / "bundle", it.state)
    assert bundle["suspected_scopes"] == ["one_confident/public"]
    assert [p.name for p in (tmp_path / "bundle/records/s7_gates").iterdir()] == ["attempt-4.json"]


def test_migrated_gate_failure_retries_bound_revision(tmp_path, monkeypatch):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path, review_kw={"max_revisions": 3})
    it = P.store.item(iid)
    it.state.update(stage="S8", post_revision=True, review_round=2, review_revisions=2,
                    thresholds_locked=True, repair_from="gates:coverage", attempts={"S7": 1})
    it.save()
    verdict = dict(V("revise", [BLOCK]), _normalized={"verdict": "revise"}, round=2, job="review-two",
                   pkg_sha=stages_b.calib.pkg_sha(it.p("pkg")), calib_epoch=0)
    write_json(it.p("review/verdict.json"), verdict)
    write_json(it.p("gates/attempt-3.json"), {"attempt": 3})
    assert stages_b.migrate_probe_report(P, it)
    assert it.state["stage"] == "S6A_run" and it.state["post_revision"]
    it.goto("S7", "remeasured")
    monkeypatch.setattr(gates, "run_gates", lambda *a: dict(attempt=4, ok=False, blocking=["oracle"],
                        probe_policy_version=gates.PROBE_POLICY_VERSION))
    stages_b.s7(P, it)
    stages_b.s8(P, it)
    assert (it.state["stage"], it.state["status"], it.state["repair_from"]) == ("S8", "active", "review:r2")
    assert it.state["review_revisions"] == 2 and "gates:oracle" in it.state["revision_feedback"]
    assert read_json(it.p("review/verdict.json")) == verdict


@pytest.mark.parametrize("submitted_human", [True, False])
def test_submitted_verdict_set_controls_ingest_and_routing(tmp_path, submitted_human):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path, review_kw={"allow_human": submitted_human, "infra_retries": 0})
    _to_label(P, iid, jobs)
    it = P.store.item(iid)
    rec, = _runs(P, iid, "review")
    allowed = [*review.VERDICTS, *(["human"] if submitted_human else [])]
    assert rec["allowed_verdicts"] == allowed
    verdict = V("human")
    src = tmp_path / "verdict.json"
    write_json(src, verdict)
    result = subprocess.run([sys.executable, Path(rec["task"]) / "tests/s9r_validate.py", src, tmp_path / "parsed.json"],
                            capture_output=True)
    assert (result.returncode == 0) == submitted_human
    P.cfg["review"]["allow_human"] = not submitted_human
    _finish(jobs, rec["job_name"], verdict=verdict, reward=1)
    ingest(P, it)
    row, = _runs(P, iid, "review")
    if submitted_human:
        assert (row["outcome"], row["verdict"]) == ("done", "human")
        stages_b.s9r(P, it)
        assert it.state["status"] == "parked" and it.state["reason"].startswith("review:human r1")
        assert _runs(P, iid, "review")[0]["outcome"] == "done"
    else:
        assert row["outcome"] == "invalid"


@pytest.mark.parametrize("stage", ["S9", "S9R"])
@pytest.mark.parametrize("launched", [False, True])
def test_migration_cancels_unstarted_job_at_zero(tmp_path, stage, launched):
    from tbsgen import calib, stages_b
    P, iid, jobs = _env(tmp_path, allow_submit=False, infra_retries=0)
    it = P.store.item(iid)
    it.state.update(stage=stage, status="waiting", review_round=1)
    it.save()
    P.cfg["budget"].update(run_calls_per_model={"model": 1}, daily_calls_per_model={"model": 1})
    call = P.ledger.reserve("model", stage, iid, True, est_usd=5)
    log = it.p("old.log")
    rec = dict(who="teacher" if stage == "S9" else "review", job_name="old", n=1, ledger_call=call,
               job_dir=str(jobs / "old"), state="launched" if launched else "reserved", log=str(log), outcome=None)
    if launched:
        log.write_text("launcher exited before starting the job")
        Path(str(log) + ".exit").write_text("1")
    sub = "calib" if stage == "S9" else "review"
    write_json(it.p(sub, "runs.json"), [rec])
    write_json(it.p("gates/attempt-1.json"), {"attempt": 1})
    assert stages_b.migrate_probe_report(P, it)
    if launched:
        assert it.state["status"] == "waiting"
        watch_tick(P, only=[iid], max_steps=0)
    assert it.load().state["stage"] == "S6A_run" and it.state["status"] == "active"
    old, = _runs(P, iid, sub)
    assert old["obsolete_probe_job"] and old["cost_usd"] == 0 and old["cost_kind"] == "probe_migration_cancelled"
    assert calib.next_action([old], P.cfg) == ("run", "glm")
    with P.ledger._db() as db:
        assert db.execute("SELECT usd, cost_kind FROM calls WHERE id=?", (call,)).fetchone() == (
            0, "probe_migration_cancelled")
    assert P.item_cost(iid)["calls"] == P.ledger.summary()["calls_total"] == 0
    assert P.ledger.reserve("model", stage, iid, True, est_usd=5) != call


@pytest.mark.parametrize("stage", ["S9", "S9R"])
def test_cancel_recovers_committed_launch_key(tmp_path, stage):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path, allow_submit=False)
    it = P.store.item(iid)
    it.state.update(stage=stage, review_round=1, calib_label="hard")
    it.save()
    P.run(only=[iid])
    sub = "calib" if stage == "S9" else "review"
    rec, = _runs(P, iid, sub)
    assert rec["state"] == "intent" and not rec.get("ledger_call")
    model = rec["model"]
    key = json.dumps([iid, stage, rec["job_name"], rec.get("attempt", rec["n"])])
    # #44 has committed the reservation, but the item still has its earlier intent.
    call = P.ledger.reserve(model, stage, iid, True, est_usd=30, reservation_key=key)
    other = P.ledger.reserve(model, stage, iid, True, est_usd=4,
                             reservation_key=json.dumps([iid, stage, "other-job", 1]))
    P.cfg["budget"]["run_calls_per_model"] = {model: 2}
    if stage == "S9":
        write_json(it.p("gates/attempt-1.json"), {})
        P.resume(only=[iid])
    else:
        P.cfg["review"]["enabled"] = False
        P.run(only=[iid])
    cancelled, = _runs(P, iid, sub)
    assert cancelled["state"] == "cancelled_unstarted" and cancelled["ledger_call"] == call
    with P.ledger._db() as db:
        assert db.execute("SELECT usd FROM calls WHERE id=?", (call,)).fetchone() == (0,)
        assert db.execute("SELECT status FROM calls WHERE id=?", (other,)).fetchone() == ("reserved",)
    assert P.ledger.reserve(model, stage, "next", True, est_usd=1)
    assert not stages_b.running_jobs(P)


@pytest.mark.parametrize("state", ["fresh", "intent", "reserved"])
def test_disabled_review_without_scopes_releases_without_launch(tmp_path, state):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path, allow_submit=False)
    it = P.store.item(iid)
    it.state.update(stage="S9R", review_round=1, calib_label="hard")
    it.save()
    if state != "fresh":
        P.run(only=[iid])
        rec, = _runs(P, iid, "review")
        if state == "reserved":
            call = P.ledger.reserve(rec["model"], "S9R", iid, True, est_usd=30)
            write_json(it.p("review/runs.json"), [dict(rec, state=state, ledger_call=call)])
    P.cfg["calib"]["allow_submit"] = True
    P.cfg["review"]["enabled"] = False
    P.run(only=[iid])
    assert it.load().state["status"] == "done" and stages_b._released(P, iid)
    assert not list(jobs.iterdir()) and not stages_b.running_jobs(P)
    assert [r["state"] for r in _runs(P, iid, "review")] == ([] if state == "fresh" else ["cancelled_unstarted"])
    assert P.item_cost(iid) == dict(calls=0, usd=0, in_tok=0, out_tok=0)


@pytest.mark.parametrize("stage", ["S9R", "S10", "S9R_after_ingest"])
def test_review_off_accepts_received_v5_pass(tmp_path, stage):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    scope = _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    write_json(it.p("review/runs.json"), [dict(rec, review_contract_version="s9r-v5")])
    finding = dict(classification="probe_artefact", command="bash /tests/test.sh", observed_reward=1,
                   observation="Only the diagnostic output changed.", reason="The declared output matched.")
    P.cfg["review"]["enabled"] = stage == "S9R_after_ingest"
    _finish(jobs, rec["job_name"], verdict=dict(V("pass"), probe_findings={scope: finding}))
    assert ingest(P, it)["outcome"] == ("infra" if stage == "S9R_after_ingest" else "done")
    P.cfg["review"]["enabled"] = False
    if stage == "S10":
        v = read_json(it.p("review/r1/verdict.json"))
        write_json(it.p("review/verdict.json"), dict(v, round=1))
        it.goto("S10", "received pass")
    P.run(only=[iid])
    assert it.load().state["status"] == "done" and stages_b._released(P, iid)
    assert len(_runs(P, iid, "review")) == 1


def test_obsolete_calibration_excluded_from_manifest(tmp_path):
    from tbsgen import calib, stages_b
    P, iid, _ = _env(tmp_path, review_kw={"enabled": False})
    it = P.store.item(iid)
    runs = [dict(job_name="cancelled", who="teacher", outcome="infra", obsolete_probe_job=True),
            dict(job_name="completed", who="teacher", outcome="pass", result_type="pass")]
    write_json(it.p("calib/runs.json"), runs)
    it.state.update(stage="S10", calib_label="hard")
    stages_b.s10(P, it)
    record = stages_b._live_release(P, iid)
    assert record["calib_runs"] == calib.run_summary(runs[1:])
    assert record["result_types"] == ["pass"]


def test_legacy_manifest_hash_survives_in_place_pkg_retry(tmp_path, monkeypatch):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path, review_kw={"enabled": False})
    it = P.store.item(iid)
    it.state.update(stage="S10", calib_label="hard", gates_attempt=1)
    write_json(it.p("gates/attempt-1.json"), dict(attempt=1, gates={}, probe_policy_version=gates.PROBE_POLICY_VERSION))
    stages_b.s10(P, it)
    previous = stages_b._live_release(P, iid)
    old_sha = previous.pop("pkg_sha")
    manifest = P.store.root / "release/manifest.jsonl"
    manifest.write_text(json.dumps(previous) + "\n")
    gen = it.p("runs/s7-1/detA")
    write_json(gen / "environment/data/seed.json", 1)
    write_json(gen / "tests/truth/target.json", 0)
    it.goto("PKG", "rebuild")
    build = stages_b.package.package

    def interrupted(*args):
        build(*args)
        raise InterruptedError

    with monkeypatch.context() as m:
        m.setattr(stages_b.package, "package", interrupted)
        with pytest.raises(InterruptedError):
            stages_b.pkg(P, it)
    assert it.load().state["legacy_release_sha"] == old_sha
    assert stages_b.calib.pkg_sha(it.p("pkg")) != old_sha
    stages_b.pkg(P, it)
    it.goto("S10", "release")
    stages_b.s10(P, it)
    stages_b.s10(P, it)
    old, new = map(json.loads, manifest.read_text().splitlines())
    assert old == previous and new["pkg_sha"] == stages_b.calib.pkg_sha(it.p("pkg"))
    assert {k: v for k, v in new.items() if k not in ("pkg_sha", "ts")} == {k: v for k, v in old.items() if k != "ts"}


@pytest.mark.parametrize("contract,observation", [
    ("s9r-v4", "I never executed the grader for this scope."),
    ("s9r-v5", "Synthetic grader reported a passing score."),
])
def test_old_probe_contract_gets_free_review(tmp_path, contract, observation):
    P, iid, jobs = _env(tmp_path, review_kw={"infra_retries": 0})
    it = P.store.item(iid)
    scope = _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    rec.pop("allowed_verdicts")
    write_json(it.p("review/runs.json"), [dict(rec, review_contract_version=contract)])
    finding = dict(classification="probe_artefact", command="bash /tests/test.sh", observed_reward=1,
                   observation=observation, reason="Inspected the table.")
    _finish(jobs, rec["job_name"], verdict=dict(V("pass"), probe_findings={scope: finding}))
    watch_tick(P, only=[iid])
    old, fresh = _runs(P, iid, "review")
    assert old["obsolete_probe_review"] and old["outcome"] == "infra"
    assert fresh["review_contract_version"] == review.CURRENT_CONTRACT and fresh["outcome"] is None


def test_s7_autofix_once_per_statement(tmp_path, monkeypatch):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    monkeypatch.setattr(gates, "run_gates", lambda *args: dict(ok=False, blocking=[], nonblocking=["thresholds_stated"]))
    ins = it.p("task/instruction.md")
    for version in ("Original", "Updated"):
        ins.write_text(version)
        it.goto("S7", "review revision")
        stages_b.s7(P, it)
        fixed = ins.read_bytes()
        assert it.load().state["status"] == "active" and it.state["autofix_done_for"] == hashlib.sha256(fixed).hexdigest()
        stages_b.s7(P, it)
        assert it.state["reason"] == "nonblocking_unfixable:thresholds_stated" and ins.read_bytes() == fixed


@pytest.mark.parametrize("stage", ["S9", "S9R"])
@pytest.mark.parametrize("entry", ["resume", "step"])
def test_migration_adopts_manual_intent_once(tmp_path, monkeypatch, stage, entry):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path, allow_submit=False, review_kw={"est_usd_per_run": 12})
    it = P.store.item(iid)
    it.state.update(stage=stage, status="active", review_round=1)
    it.save()
    P.run(only=[iid])
    sub = "calib" if stage == "S9" else "review"
    rec, = _runs(P, iid, sub)
    assert rec["state"] == "intent"
    write_json(jobs / rec["job_name"] / "config.json", read_json(Path(rec["config"])))
    _finish(jobs, rec["job_name"], cost=7, verdict=V("pass") if stage == "S9R" else None)
    write_json(it.p("gates/attempt-1.json"), {})
    model = rec["model"]
    prior = P.ledger.reserve(model, stage, "earlier", True, est_usd=190)
    P.ledger.settle(prior, 190)
    P.cfg["budget"]["daily_usd_per_model"] = {model: 200}

    def recover():
        return P.resume(only=[iid]) if entry == "resume" else P.step(it.load())

    # Lose the first save after accounting for the manual job. Neither scheduler entry may raise.
    def interrupted(*args):
        raise InterruptedError("interrupted adoption")

    with monkeypatch.context() as m:
        m.setattr(stages_b, "_save_run" if stage == "S9" else "_save_review", interrupted)
        assert recover() == ([] if entry == "resume" else "paused")
    assert it.load().state["status"] == "paused"
    assert P.ledger.summary()["usd_total"] == 197  # settled even before the interrupted state save
    assert recover() == ([iid] if entry == "resume" else "ok")
    adopted, = _runs(P, iid, sub)
    assert P.ledger.summary()["usd_total"] == 197
    from tbsgen.budget import BudgetStop
    with pytest.raises(BudgetStop, match="daily_model"):
        P.ledger.reserve(model, stage, "next", True, est_usd=12)
    assert adopted["recovered"] and adopted["state"] == "launched"
    watch_tick(P, only=[iid], max_steps=0)
    assert it.load().state["stage"] == "S6A_run" and it.state["status"] == "active"
    assert P.item_cost(iid)["usd"] == 7 and P.item_cost(iid)["calls"] == 1


@pytest.mark.parametrize("upgrade", [False, True])
def test_unstarted_review_releases_budget_and_slot(tmp_path, upgrade):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path, allow_submit=False, max_concurrent_jobs=1)
    it = P.store.item(iid)
    _current_probe(it)
    it.state.update(stage="S9R", review_round=1)
    it.save()
    P.run(only=[iid])
    rec, = _runs(P, iid, "review")
    P.cfg["budget"]["run_calls_per_model"] = {rec["model"]: 1}
    call = P.ledger.reserve(rec["model"], "S9R", iid, True, est_usd=30)
    if upgrade:
        rec["review_contract_version"] = "s9r-v5"
    write_json(it.p("review/runs.json"), [dict(rec, state="reserved", ledger_call=call)])
    P.cfg["review"]["enabled"] = upgrade
    it.goto("S9R", "restart")
    assert stages_b.running_jobs(P) == [rec["job_name"]]
    stages_b.s9r(P, it)
    cancelled = _runs(P, iid, "review")[0]
    assert cancelled["state"] == "cancelled_unstarted" and cancelled["cost_usd"] == 0
    assert P.item_cost(iid)["calls"] == 0 and stages_b._slots_full(P) is None
    assert P.ledger.reserve(rec["model"], "S9R", "another-item", True, est_usd=30) != call
    if upgrade:
        assert _runs(P, iid, "review")[1]["review_contract_version"] == review.CURRENT_CONTRACT
    else:
        assert it.state["reason"] == "parked:probe_unreviewed"


def test_packaged_repro_records_fresh_reward_and_verifier_exports_it(tmp_path, monkeypatch):
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    scope = _current_probe(it)
    task = review.build_task(P.cfg, it.dir, it.state, 1)
    logs = tmp_path / "logs"
    verifier = logs / "verifier"
    verifier.mkdir(parents=True)
    reward = verifier / "reward.txt"
    reward.write_text("1")
    # Map the generated image's absolute paths to an isolated directory.
    helper = task / "environment/tbs-repro"
    helper.write_text(helper.read_text().replace('Path("/logs")', f"Path({str(logs)!r})"))
    def run(*argv):
        result = subprocess.run([helper, scope, "--", *argv], capture_output=True, check=False)
        row = json.loads(result.stdout)
        assert row["exit_code"] == result.returncode and row["scope_id"] == scope
        assert row["output_sha256"] == hashlib.sha256(result.stderr).hexdigest()
        return row

    first = run(sys.executable, "-c", "import sys; sys.exit(4)")
    assert first["reward"] is None and first["exit_code"] == 4
    second = run(sys.executable, "-c", f"import sys; print('out', flush=True); print('err', file=sys.stderr); open({str(reward)!r}, 'w').write('0.5')")
    assert second["reward"] == 0.5 and second["repro_id"] != first["repro_id"]
    assert [json.loads(line) for line in (logs / "review/repro.jsonl").read_text().splitlines()] == [first, second]

    output = tmp_path / "review_out"
    finding = dict(repro_id=second["repro_id"], classification="probe_artefact", reason="Synthetic smoke test.",
                   observed_reward=0.5, observation="The optional branch was never executed.")
    write_json(output / "verdict.json", dict(V("pass"), probe_findings={scope: finding}))
    script = (task / "tests/test.sh").read_text().replace("/tests/", str(task / "tests") + "/")
    script = script.replace("/logs/", str(logs) + "/").replace("/app/review_out/", str(output) + "/")
    subprocess.run(["bash", "-c", script], check=True)
    assert reward.read_text().strip() == "0"  # the helper executed a different command
    assert (verifier / "repro.jsonl").read_bytes() == (logs / "review/repro.jsonl").read_bytes()
    assert review.normalize(review.read_verdict(logs), suspected_scopes=[scope],
                            repros=review.read_repros(verifier / "repro.jsonl"))["verdict"] is None

    # Execute the bundle's public-scope command. Translate container paths only at process entry.
    bundle = task / "environment/review"
    records = bundle / "records"
    app = tmp_path / "app"
    app.mkdir()
    (bundle / "pkg/solution/solve.sh").write_text(
        'for i in {1..6}; do echo "$i,$i"; done | (echo id,confidence; cat) > output/results.csv\n')
    (bundle / "pkg/tests/test.sh").write_text(
        f'test "$(wc -l < "{app}/output/results.csv")" -eq 2 && echo 1 > "{reward}"\n')
    probe = records / "probe_outputs.py"
    probe.write_text(probe.read_text().replace('"/app', f'"{app}').replace('"/solution/', f'"{bundle}/pkg/solution/')
                     .replace('"/review/', f'"{bundle}/'))
    python = tmp_path / "python3"
    python.write_text(f'#!/bin/sh\nif [ "$1" = "/review/records/reproduce_scope.py" ]; then\n'
                      f'  shift\n  exec "{sys.executable}" "{records}/reproduce_scope.py" "$@"\nfi\n'
                      f'exec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    argv = read_json(records / "output_probes.json")["repro_commands"][scope]
    scoped = run(*argv)
    assert scoped["command"] == argv and scoped["exit_code"] == 0 and scoped["reward"] == 1
    assert (app / "output/results.csv").read_text().splitlines() == ["id,confidence", "6,6"]
    finding.update(repro_id=scoped["repro_id"], observed_reward=1)
    write_json(output / "verdict.json", dict(V("pass"), probe_findings={scope: finding}))
    subprocess.run(["bash", "-c", script], check=True)
    assert reward.read_text().strip() == "1"
    assert review.normalize(review.read_verdict(logs), suspected_scopes=[scope],
                            repros=review.read_repros(verifier / "repro.jsonl"))["verdict"] == "pass"


def test_received_drop_without_bound_gate_report(tmp_path):
    from tbsgen import stages_b
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    _current_probe(it)
    _to_label(P, iid, jobs)
    rec, = _runs(P, iid, "review")
    _v2_job(it, rec)
    _finish(jobs, rec["job_name"], verdict=V("drop", [BLOCK]))
    ingest(P, it)
    it.p("gates/attempt-1.json").unlink()
    assert stages_b.migrate_probe_report(P, it.load())
    assert it.state["status"] == "dropped" and it.state["reason"].startswith("review:drop")


def test_old_harden_revert_selects_original_report(tmp_path):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    scope = _current_probe(it)
    it.p("gates/attempt-1.json").rename(it.p("gates/attempt-2.json"))
    write_json(it.p("gates/attempt-1.json"), dict(ok=False, probe_policy_version=gates.PROBE_POLICY_VERSION))
    del it.state["gates_attempt"]
    it.state["history"] = [dict(stage="S7", result="ok", attempt=2),
                           dict(stage="S9", result="harden", round=1, to="S5G"),
                           dict(stage="S7", result="ok", attempt=1),
                           dict(stage="PKG", result="harden_revert", to="S9")]
    assert stages_b._probe_scopes(it) == [scope]


@pytest.mark.parametrize("stage", ["S4c", "S5G", "S9", "S9R"])
def test_crash_parking_releases_only_unstarted_jobs(tmp_path, monkeypatch, stage):
    from tbsgen import pipeline, stages_b
    P, iid, jobs = _env(tmp_path, allow_submit=False)
    it = P.store.item(iid)
    it.state.update(stage=stage, review_round=1)
    it.save()
    sub = "calib" if stage == "S9" else "review"
    records = []
    for name in ("reserved", "starting"):
        call = P.ledger.reserve("model", stage, iid, True, est_usd=12)
        records.append(dict(state="reserved", job_dir=str(jobs / name), launch_attempted=name == "starting",
                            job_name=name, outcome=None, ledger_call=call))
    agent_stage = stage in ("S4c", "S5G")
    if agent_stage:
        write_json(it.p(stage, "run.json"), records[0])
        write_json(it.p(sub, "runs.json"), records[1:])
    else:
        write_json(it.p(sub, "runs.json"), records)

    def fail(*args):
        raise ValueError("synthetic failure")

    monkeypatch.setitem(pipeline.HANDLERS, stage, fail)
    for _ in range(2):
        P.step(it)
    assert it.state["status"] == "parked"
    if agent_stage:
        cancelled = read_json(it.p(stage, "run.json"))
        starting, = _runs(P, iid, sub)
    else:
        assert stages_b.running_jobs(P) == ["starting"]
        cancelled, starting = _runs(P, iid, sub)
    assert cancelled["state"] == "cancelled_unstarted" and starting == records[1]
    assert P.item_cost(iid)["calls"] == 1
