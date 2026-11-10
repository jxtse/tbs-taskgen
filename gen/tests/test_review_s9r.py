"""S9R review stage: routing after S9 labels, review job lifecycle (prepare / launch / ingest), verdict contract
(normalize), the S8 review revision (thresholds locked, only allowed files, calibration archived and restarted
from GLM), the second-review rule, manifest notes and withdrawal.  No Docker, no network, no real jobs."""
import json
from pathlib import Path

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


BLOCK = dict(id="I1", category="a", severity="blocking", title="column name conflict",
             evidence=["instruction.md:1 says spectrum_index; tests/measure.py:1 reads spec"], fix="use spec")
MINOR = dict(id="I2", category="d", severity="minor", title="timing margin small", evidence=[], fix="none")


# ------------------------------------------------------------------ verdict contract
def test_normalize_contract():
    n = review.normalize(V("pass", [MINOR]))
    assert n["verdict"] == "pass" and n["minor"] == ["I2"] and not n["problems"]
    # blocking without evidence -> minor; revise without blocking -> human
    n = review.normalize(V("revise", [dict(BLOCK, evidence=[])]))
    assert n["downgraded"] == ["I1"] and n["blocking"] == [] and n["verdict"] == "human"
    n = review.normalize(V("revise", [BLOCK]))
    assert n["verdict"] == "revise" and n["blocking"] == ["I1"]
    assert review.normalize(V("pass", [BLOCK]))["verdict"] == "human"
    assert review.normalize(V("drop", [MINOR]))["verdict"] == "human"
    assert review.normalize(V("ok"))["verdict"] == "human"
    assert review.normalize(None)["verdict"] == "human"
    n = review.normalize(V("revise", [dict(BLOCK, severity="critical")]))
    assert n["verdict"] == "human" and n["issues"][0]["severity"] == "minor"
    # strict: a structural problem can never yield pass
    assert review.normalize(V("pass", [dict(MINOR, severity="critical")]))["verdict"] == "human"
    assert review.normalize(V("pass", ["not an object"]))["verdict"] == "human"
    assert review.normalize(V("pass"), reward=0.0)["verdict"] == "human"
    assert review.normalize(V("pass"), reward=1.0)["verdict"] == "pass"


def test_reviewable_labels():
    cfg = load_config()
    assert cfg["review"]["enabled"] is True and cfg["review"]["max_revisions"] == 1
    assert review.reviewable(cfg, "hard") and review.reviewable(cfg, "unstable")
    assert review.reviewable(cfg, "parked:teacher_failed")
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
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs, seq=(0.0, 0.0, 0.0))
    it = P.store.item(iid)
    assert it.state["calib_label"] == "parked:teacher_failed" and it.state["stage"] == "S9R"
    [rr] = _runs(P, iid, "review")
    _finish(jobs, rr["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("parked:teacher_failed")
    assert not (P.store.root / "release" / "manifest.jsonl").exists()


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
    assert st["status"] == "parked" and st["reason"] == "review:infra"
    assert _runs(P, iid, "review")[1]["job_name"].endswith("-review1-a2")


def test_drop_and_human(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("drop", [BLOCK]))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "dropped" and st["reason"].startswith("review:drop")
    P, iid, jobs = _env(tmp_path / "h")
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [dict(BLOCK, evidence="")]))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("review:human")   # unevidenced -> human


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
    assert st["status"] == "parked" and "threshold" in st["reason"]
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
    assert review.revision_review(task, {"tests/measure.py": "COL = 'spectrum_index'\n"}, {"nodes": []}, th)["ok"]


def test_revision_bad_path_writes_nothing(tmp_path):
    P, iid, jobs = _env(tmp_path)
    _to_label(P, iid, jobs)
    _finish(jobs, _runs(P, iid, "review")[0]["job_name"], verdict=V("revise", [BLOCK]))
    _revise_client(P, {"../../thresholds.json": "{}", "instruction.md": "x\n"})
    watch_tick(P, only=[iid])
    it = P.store.item(iid)
    assert it.state["status"] == "parked" and "not allowed" in it.state["reason"]
    assert read_json(it.p("thresholds.json"))["thresholds"]["err"]["threshold"] == 1.0e-3
    assert not it.p("repair", "review-1", "patch").exists() and it.p("repair", "review-1", "patch.json").exists()
    assert it.p("task", "instruction.md").read_text().startswith("Write out.csv")


def test_gate_failure_after_revision_goes_to_human_not_ordinary_s8(tmp_path):
    from tbsgen import stages_b
    P, iid, _ = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="S8", status="active", thresholds_locked=True, review_revisions=1,
                    repair_from="gates:oracle")
    it.save()
    stages_b.s8(P, it)
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("review:human (after the review revision")


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


def test_second_review_not_pass_goes_to_human(tmp_path):
    P, iid, jobs = _env(tmp_path)
    it = P.store.item(iid)
    it.state.update(stage="S9R", status="active", calib_label="hard", review_round=2, review_revisions=1)
    it.save()
    P.run(only=[iid])
    rr = _runs(P, iid, "review")[0]
    assert rr["job_name"].endswith("-review2")
    _finish(jobs, rr["job_name"], verdict=V("revise", [BLOCK]))
    watch_tick(P, only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "parked" and st["reason"].startswith("review:human (r2 revise")


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
        # reference now misses a locked threshold -> human
        write_json(it.p("S6A_measure.json"), dict(ref={"0": {"err": 5e-3, "score": 0.95}}, mid={}, naive={}, nop={},
                                                  runs={}))
        it.state.update(stage="S6B", status="active")
        it.save()
        stages_a.s6b(P, it)
        st = P.store.item(iid).state
        assert st["status"] == "parked" and "locked threshold" in st["reason"]
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
