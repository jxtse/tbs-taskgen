"""S9R review stage: routing after S9 labels, review job lifecycle (prepare / launch / ingest), verdict contract
(normalize), the S8 review revision (thresholds locked, only allowed files, calibration archived and restarted
from GLM), the second-review rule, manifest notes and withdrawal.  No Docker, no network, no real jobs."""
import json
from pathlib import Path

import pytest

from tbsgen import review
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


def _finish(jobs, job_name, reward=None, cost=5.0, verdict=None, name="trial__a"):
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
    assert review.normalize(V("drop", [BLOCK]))["verdict"] == "revise"          # only a local blocking issue
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
    write_json(it.p("gates", "attempt-9.json"), dict(ok=ok, gates=dict(oracle=ok)))


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


def test_frontier_relabelled_with_review_disabled_withdraws_the_release(tmp_path):
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
    assert st["status"] == "parked" and st["reason"].startswith("frontier:unreviewed"), st
    last = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[-1])
    assert last["item_id"] == iid and last["withdrawn"]


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
