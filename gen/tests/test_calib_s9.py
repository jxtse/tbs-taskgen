"""S9 calibration pipeline fixes: environment-preparation hook, partial calibration entry points (run re-wakes
S9 intents, resume --only/--stage), cost of runs without a Harbor cost (token estimate vs unknown), result types,
`tbsgen calib watch`, teacher-only routing.  No Docker, no network, no real jobs."""
import json
import sqlite3
from pathlib import Path

import pytest

from tbsgen import calib
from tbsgen.budget import Ledger
from tbsgen.config import load_config
from tbsgen.pipeline import Pipeline
from tbsgen.stages_b import ingest, watch_tick

PRICES = {"glm-5.3": {"in": 1.0, "out": 4.0, "cache_read": 0.2}}


# ------------------------------------------------------------------ fixtures
def _env(tmp_path, n_items=1, **calib_kw):
    """Run dir with packaged S9 items; fake tmux runs the launcher inline; the launcher creates the Harbor job dir
    (config.json) like Harbor does at job start."""
    hr = tmp_path / "harbor"
    hr.mkdir(exist_ok=True)
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    counter = tmp_path / "tmux_calls"
    tmux = tmp_path / "fake-tmux"
    # runs the launcher inline; a launcher that started a job is "still running" (its .exit is removed), like a
    # real launcher that returns only when Harbor does
    # has-session: alive unless <tmp>/dead-sessions lists the name
    tmux.write_text("#!/bin/bash\nif [ \"$1\" = has-session ]; then n=${3#=}; "
                    "grep -qx \"$n\" %s/dead-sessions 2>/dev/null && { echo \"can't find session: $n\" >&2; exit 1; }; "
                    "exit 0; fi\necho x >> %s\nshift 4\n\"$@\"\n"
                    "for f in %s/items/*/calib/*.log.exit; do [ -f \"$f\" ] || continue; j=$(basename \"$f\" .log.exit); "
                    "[ -f \"%s/$j/result.json\" ] && rm \"$f\"; done; true\n" % (tmp_path, counter, tmp_path / "run", jobs))
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
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "calib": cal,
                             "budget": {"daily_usd_total": 1000.0, "run_usd_total": 500.0},
                             "models": {"prices": PRICES}})
    P = Pipeline(cfg, clients={})
    ids = P.ingest([dict(doi=f"10.1/s9-{k}", title=f"t{k}", abstract="algorithm estimation") for k in range(n_items)])
    for iid in ids:
        it = P.store.item(iid)
        (it.p("pkg")).mkdir(parents=True, exist_ok=True)
        (it.p("pkg", "task.toml")).write_text('[environment]\nnetwork_mode = "public"\n')
        (it.p("pkg", "instruction.md")).write_text("do it\n")
        it.state.update(stage="S9", status="active")
        it.save()
    return P, ids, counter, jobs


def _trial(jd: Path, reward=0.0, cost=None, exc=None, tokens=None, deliverable=True, name="trial__a"):
    t = jd / name
    (t / "verifier").mkdir(parents=True, exist_ok=True)
    ar = {"cost_usd": cost}
    if tokens:
        ar.update(n_input_tokens=tokens[0], n_cache_tokens=tokens[1], n_output_tokens=tokens[2])
    (t / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": reward}}, "agent_result": ar,
        "exception_info": {"exception_type": exc} if exc else None,
        "agent_execution": {"started_at": "2026-11-09T01:00:00Z", "finished_at": "2026-11-09T03:00:00Z"}}))
    err = "" if deliverable else "/usr/local/bin/python: can't open file '/app/output/solution_cli.py': [Errno 2] No such file or directory\n"
    (t / "verifier" / "metrics.jsonl").write_text(json.dumps(
        {"name": "_solution_cli_run", "value": 1.0, "rc": 0 if deliverable else 2, "stderr_tail": err,
         "aux": True}) + "\n" + json.dumps({"name": "m", "value": 1.0, "passed": reward >= 1}) + "\n")
    return t


def _finish(jobs, job_name, **kw):
    jd = jobs / job_name
    _trial(jd, **kw)
    (jd / "result.json").write_text(json.dumps({"finished_at": "2026-11-09T03:01:00Z"}))
    return jd


def _runs(P, iid):
    return json.loads(P.store.item(iid).p("calib", "runs.json").read_text())


# ------------------------------------------------------------------ 3. cost of runs without a Harbor cost
def test_timeout_without_harbor_cost_is_estimated_from_tokens(tmp_path):
    P, [iid], _, jobs = _env(tmp_path)
    P.run(only=[iid])
    r1 = _runs(P, iid)[0]
    _finish(jobs, r1["job_name"], exc="AgentTimeoutError", tokens=(7_000_000, 6_000_000, 400_000),
            deliverable=False)
    t = ingest(P, P.store.item(iid))
    assert t["cost_kind"] == "tokens" and t["cost_source"] == "agent_result"
    want = (1_000_000 * 1.0 + 6_000_000 * 0.2 + 400_000 * 4.0) / 1e6          # uncached in + cache read + out
    assert t["cost_usd"] == pytest.approx(want)
    rec = _runs(P, iid)[0]
    assert rec["cost_kind"] == "tokens" and rec["cost_usd"] == pytest.approx(want)
    assert calib.actual_spend(P.ledger) == pytest.approx(want)                # not the $30 reservation
    kinds = P.ledger.summary(None)["usd_by_cost_kind"]
    assert kinds["tokens"] == pytest.approx(want) and "reservation" not in kinds
    with P.ledger._db() as db:
        assert db.execute("SELECT in_tok, out_tok, cache_read FROM calls WHERE stage='S9'").fetchone() == \
            (1_000_000, 400_000, 6_000_000)


def test_unknown_cost_keeps_reservation_but_not_as_actual_spend(tmp_path):
    P, [iid], _, jobs = _env(tmp_path)
    P.run(only=[iid])
    _finish(jobs, _runs(P, iid)[0]["job_name"], exc="AgentTimeoutError")          # no cost, no tokens
    t = ingest(P, P.store.item(iid))
    assert t["cost_kind"] == "unknown" and t["cost_usd"] is None
    assert _runs(P, iid)[0]["cost_kind"] == "reservation"
    assert calib.actual_spend(P.ledger) == 0.0
    s = P.ledger.summary(None)
    assert s["usd_by_cost_kind"] == {"reservation": 30.0} and s["usd_total"] == 30.0   # still visible, separate


def test_reported_cost_wins_and_price_missing_is_unknown(tmp_path):
    t = _trial(tmp_path / "j", reward=1.0, cost=12.5, tokens=(10, 0, 10))
    c = calib.trial_cost(t, "glm-5.3", PRICES)
    assert c["cost_kind"] == "reported" and c["cost_usd"] == 12.5
    t2 = _trial(tmp_path / "j2", tokens=(10, 0, 10))
    c = calib.trial_cost(t2, "other-model", PRICES)
    assert c["cost_kind"] == "unknown" and c["usage"]["output_tokens"] == 10 and c["price_missing"] == "other-model"


def test_token_usage_from_trajectory_and_stream_json(tmp_path):
    t = tmp_path / "t1"
    (t / "agent").mkdir(parents=True)
    (t / "agent" / "trajectory.json").write_text(json.dumps({"final_metrics": {
        "total_prompt_tokens": 100, "total_cached_tokens": 60, "total_completion_tokens": 7,
        "extra": {"total_cache_creation_input_tokens": 10}}}))
    u, src = calib.trial_usage(t, {})
    assert src == "trajectory" and u == dict(input_tokens=30, cache_read_input_tokens=60,
                                             cache_creation_input_tokens=10, output_tokens=7)
    t2 = tmp_path / "t2"
    (t2 / "agent").mkdir(parents=True)
    lines = [
        {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 5, "output_tokens": 0}}},
        {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 5, "output_tokens": 9}}},  # same msg
        {"type": "assistant", "message": {"id": "m2", "usage": {"input_tokens": 3, "cache_read_input_tokens": 50,
                                                                "output_tokens": 2}}},
        {"type": "result", "usage": {"input_tokens": 999}}]
    (t2 / "agent" / "claude-code.txt").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    u, src = calib.trial_usage(t2, {})
    assert src == "stream_json" and u["input_tokens"] == 8 and u["output_tokens"] == 11 and \
        u["cache_read_input_tokens"] == 50


def test_ledger_migrates_old_db_without_cost_kind(tmp_path):
    p = tmp_path / "l.sqlite"
    db = sqlite3.connect(p)
    db.executescript("CREATE TABLE calls(id INTEGER PRIMARY KEY, day TEXT, started REAL, ended REAL, model TEXT, "
                     "real INTEGER, stage TEXT, item TEXT, status TEXT, est_usd REAL, usd REAL, in_tok INTEGER, "
                     "out_tok INTEGER, cache_read INTEGER, cache_write INTEGER, error_class TEXT, error TEXT, "
                     "latency REAL, resp_model TEXT, prompt_sha TEXT);"
                     "INSERT INTO calls(day,model,real,stage,status,est_usd,usd) VALUES('d','glm-5.3',1,'S9','ok',30,30);")
    db.commit()
    db.close()
    led = Ledger(p, load_config(None, {"budget": {"daily_usd_total": 100.0}}))
    assert calib.actual_spend(led) == 30.0                     # legacy rows: unchanged meaning
    assert led.summary(None)["usd_by_cost_kind"] == {"token_priced": 30.0}


# ------------------------------------------------------------------ 4. result types
@pytest.mark.parametrize("kw,want", [
    (dict(reward=1.0), "pass"),
    (dict(reward=0.0), "fail_delivered"),
    (dict(reward=0.0, deliverable=False), "fail_no_output"),
    (dict(reward=0.0, exc="AgentTimeoutError"), "timeout_with_output"),
    (dict(reward=0.0, exc="AgentTimeoutError", deliverable=False), "timeout_no_output"),
    (dict(reward=0.0, exc="RateLimitError"), "infra"),
])
def test_result_types(tmp_path, kw, want):
    t = _trial(tmp_path / "j", **kw)
    assert calib.classify_trial(t)["result_type"] == want


def test_unknown_deliverable_is_not_no_output(tmp_path):
    t = _trial(tmp_path / "j", exc="AgentTimeoutError")
    (t / "verifier" / "metrics.jsonl").unlink()
    c = calib.classify_trial(t)
    assert c["delivered"] is None and c["result_type"] == "timeout_with_output"


def _seq(*xs):
    return [dict(who=w, outcome=o, minutes=60, result_type=rt) for w, o, rt in xs]


def test_timeout_no_output_policy_default_unchanged():
    tno = ("glm", "fail", "timeout_no_output")
    cfg = load_config()
    assert calib.next_action(_seq(tno), cfg) == ("run", "teacher")
    assert calib.next_action(_seq(tno, ("teacher", "pass", "pass")), cfg) == ("label", "hard")
    assert cfg["calib"]["keep_labels"] == ["hard", "unstable"]


def test_timeout_no_output_policy_retry_and_park():
    tno = ("glm", "fail", "timeout_no_output")
    cfg = load_config(None, {"calib": {"timeout_no_output": "retry"}})
    assert calib.next_action(_seq(tno), cfg) == ("run", "glm")                    # not decisive: student again
    assert calib.next_action(_seq(tno, ("glm", "fail", "fail_delivered")), cfg) == ("run", "teacher")
    assert calib.next_action(_seq(tno, tno), cfg) == ("run", "teacher")            # retries used up: counts
    assert calib.next_action(_seq(tno, ("glm", "pass", "pass")), cfg) == ("run", "glm")
    assert calib.next_action(_seq(("glm", "pass", "pass"), tno), cfg) == ("run", "glm")
    cfg = load_config(None, {"calib": {"timeout_no_output": "park"}})
    assert calib.next_action(_seq(tno), cfg) == ("label", "parked:timeout_no_output")
    with pytest.raises(ValueError):
        calib.next_action(_seq(tno), load_config(None, {"calib": {"timeout_no_output": "hard"}}))


def test_release_manifest_keeps_result_types(tmp_path):
    P, [iid], _, jobs = _env(tmp_path)
    P.run(only=[iid])
    _finish(jobs, _runs(P, iid)[0]["job_name"], exc="AgentTimeoutError", deliverable=False,
            tokens=(100, 0, 10))
    ingest(P, P.store.item(iid))
    P.run(only=[iid])                                                    # GLM fail -> teacher launched
    assert _runs(P, iid)[1]["who"] == "teacher"
    _finish(jobs, _runs(P, iid)[1]["job_name"], reward=1.0, cost=5.0)
    ingest(P, P.store.item(iid))
    P.run(only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "done" and st["calib_label"] == "hard"
    m = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[0])
    assert m["result_types"] == ["timeout_no_output", "pass"]
    assert [r["cost_kind"] for r in m["calib_runs"]] == ["tokens", "reported"]


# ------------------------------------------------------------------ 2. partial calibration entry points
def test_run_rewakes_prepared_s9_intents_once_submit_is_on(tmp_path):
    P, ids, counter, jobs = _env(tmp_path, n_items=2, allow_submit=False)
    P.run()
    for iid in ids:
        st = P.store.item(iid).state
        assert st["status"] == "waiting" and "submit disabled" in st["reason"]
    P.run()                                                           # submit still off: not re-woken
    assert not counter.exists()
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=dict(P.cfg["calib"], allow_submit=True))), clients={})
    P2.run(only=[ids[0]])                                             # only that item is launched
    assert counter.read_text().count("x") == 1
    assert _runs(P2, ids[0])[0]["state"] == "launched" and _runs(P2, ids[1])[0]["state"] == "intent"
    P2.run(only=[ids[0]])                                             # a running job is not re-woken
    assert counter.read_text().count("x") == 1
    hist = P2.store.item(ids[0]).state["history"]
    assert any(h.get("result") == "rewake" for h in hist)


def test_resume_only_and_stage_filters(tmp_path):
    P, ids, _, _ = _env(tmp_path, n_items=3)
    reasons = ["budget:run_total_actual", "model:config", "budget:daily_total"]
    stages = ["S9", "S5", "S5"]
    for iid, rs, stg in zip(ids, reasons, stages):
        it = P.store.item(iid)
        it.state["stage"] = stg
        it.finish("paused", rs, paused_stage=stg)
    assert P.resume(only=[ids[0]]) == [ids[0]]
    assert P.store.item(ids[1]).state["status"] == "paused"
    assert P.resume(("budget:",), stages=["S5"]) == [ids[2]]
    assert P.store.item(ids[1]).state["status"] == "paused"           # model: not in the prefixes
    assert P.resume() == [ids[1]]


def test_cli_resume_only(tmp_path):
    from tbsgen.cli import main
    P, ids, _, _ = _env(tmp_path, n_items=2)
    for iid in ids:
        P.store.item(iid).finish("paused", "model:config", paused_stage="S5")
    cfgp = tmp_path / "c.toml"
    cfgp.write_text("")
    assert main(["resume", "--run", str(P.store.root), "--config", str(cfgp), "--only", ids[1], "--stage", "S5"]) == 0
    assert [P.store.item(i).state["status"] for i in ids] == ["paused", "active"]


# ------------------------------------------------------------------ 1. environment-preparation hook
def _hook(tmp_path, fail=False):
    s = tmp_path / "prep.sh"
    s.write_text("#!/bin/bash\nset -e\necho \"$@\" >> %s\n%s\nmkdir -p \"$2\"\ncp -r \"$1\"/. \"$2\"/\n"
                 "echo 'docker_image = \"tmpl-'\"$TBSGEN_ITEM_ID\"'\"' >> \"$2/task.toml\"\n"
                 % (tmp_path / "prep_calls", "exit 3" if fail else ""))
    s.chmod(0o755)
    return f"{s} {{pkg}} {{out}} {{item_id}}"


def test_prepare_hook_builds_the_prepared_copy_and_the_job_uses_it(tmp_path):
    prep = tmp_path / "prepared"
    P, [iid], counter, _ = _env(tmp_path, prepared_tasks_dir=str(prep), prepare_command=_hook(tmp_path))
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    assert r["state"] == "launched" and r["task"] == str((prep / iid).resolve())
    conf = json.loads(Path(r["config"]).read_text())
    assert conf["tasks"][0]["path"] == str((prep / iid).resolve())
    assert "docker_image" in (prep / iid / "task.toml").read_text()
    rec = json.loads(P.store.item(iid).p("calib", "prepare.json").read_text())
    assert rec["ok"] and rec["pkg_sha"] == calib.pkg_sha(P.store.item(iid).p("pkg"))
    # same pkg -> reused (no second call); changed pkg -> prepared again
    it = P.store.item(iid)
    assert calib.ensure_prepared(P.cfg, it.dir)["reused"]
    assert (tmp_path / "prep_calls").read_text().count("\n") == 1
    it.p("pkg", "instruction.md").write_text("changed\n")
    assert not calib.ensure_prepared(P.cfg, it.dir).get("reused")
    assert (tmp_path / "prep_calls").read_text().count("\n") == 2


def test_prepare_hook_failure_blocks_without_launch_and_is_not_retried(tmp_path):
    prep = tmp_path / "prepared"
    P, [iid], counter, _ = _env(tmp_path, prepared_tasks_dir=str(prep), prepare_command=_hook(tmp_path, fail=True))
    P.run(only=[iid])
    st = P.store.item(iid).state
    assert st["status"] == "waiting" and "prepare_command failed" in st["reason"] and not counter.exists()
    P.run(only=[iid])                                                 # re-woken, but the same pkg is not re-prepared
    assert (tmp_path / "prep_calls").read_text().count("\n") == 1
    from tbsgen.cli import main
    cfgp = tmp_path / "c.toml"
    cfgp.write_text(f'[calib]\nprepared_tasks_dir = "{prep}"\nprepare_command = "{_hook(tmp_path)}"\n')
    assert main(["calib", "prepare", "--run", str(P.store.root), "--config", str(cfgp), "--item", iid]) == 0
    assert (prep / iid / "task.toml").exists()


def test_intent_written_before_preparation_switches_to_the_prepared_copy(tmp_path):
    """allow_submit=false wrote an intent pointing at pkg/ (no prepared copy yet); once the copy exists the
    launch uses it instead of a run that cannot start on the remote backend."""
    prep = tmp_path / "prepared"
    P, [iid], counter, _ = _env(tmp_path, allow_submit=False)
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    assert r["state"] == "intent" and r["task"].endswith("/pkg")
    cal = dict(P.cfg["calib"], allow_submit=True, prepared_tasks_dir=str(prep), prepare_command=_hook(tmp_path))
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=cal)), clients={})
    P2.run(only=[iid])
    r = _runs(P2, iid)[0]
    assert r["state"] == "launched" and r["task"] == str((prep / iid).resolve())
    assert json.loads(Path(r["config"]).read_text())["tasks"][0]["path"] == r["task"]


# ------------------------------------------------------------------ 6. teacher routing
def test_teacher_has_its_own_environment_hosts_and_launcher_root(tmp_path):
    cfg = load_config(None, {"calib": {"environment": "ext:StudentEnv", "agent_allowed_hosts": ["student.example"],
                                       "teacher_environment": "ext:TeacherEnv",
                                       "teacher_agent_allowed_hosts": ["teacher.example"],
                                       "harbor_root": "/s", "teacher_harbor_root": "/t",
                                       "student_launcher": "a.sh", "teacher_launcher": "b.sh",
                                       "jobs_dir": str(tmp_path / "jobs")}})
    s = calib.job_config(cfg, tmp_path, "glm", "j")
    t = calib.job_config(cfg, tmp_path, "teacher", "j")
    assert s["environment"]["import_path"] == "ext:StudentEnv" and s["agents"][0]["extra_allowed_hosts"] == \
        ["student.example"]
    assert t["environment"]["import_path"] == "ext:TeacherEnv" and t["agents"][0]["extra_allowed_hosts"] == \
        ["teacher.example"]
    item = tmp_path / "items" / "i"
    (item / "pkg").mkdir(parents=True)
    assert calib.prepare(cfg, item, "teacher", 1)["command"].startswith("cd /t && ./b.sh run ")
    assert calib.prepare(cfg, item, "glm", 1)["command"].startswith("cd /s && ./a.sh run ")
    # unset teacher keys fall back to the shared ones
    cfg2 = load_config(None, {"calib": {"environment": "ext:E", "agent_allowed_hosts": ["h"]}})
    t2 = calib.job_config(cfg2, tmp_path, "teacher", "j")
    assert t2["environment"]["import_path"] == "ext:E" and t2["agents"][0]["extra_allowed_hosts"] == ["h"]


def test_teacher_launch_rewrites_its_own_hosts(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path, teacher_agent_allowed_hosts=["teacher.example"])
    P.run(only=[iid])
    _finish(jobs, _runs(P, iid)[0]["job_name"], reward=0.0, cost=1.0)
    ingest(P, P.store.item(iid))
    P.run(only=[iid])
    t = _runs(P, iid)[1]
    assert t["who"] == "teacher" and t["state"] == "launched"
    assert json.loads(Path(t["config"]).read_text())["agents"][0]["extra_allowed_hosts"] == ["teacher.example"]
    assert json.loads(Path(_runs(P, iid)[0]["config"]).read_text())["agents"][0]["extra_allowed_hosts"] == \
        ["models.example.com"]


# ------------------------------------------------------------------ 5. calib watch
def test_watch_ingests_finished_jobs_and_launches_the_teacher(tmp_path):
    P, ids, counter, jobs = _env(tmp_path, n_items=2)
    P.run()
    assert counter.read_text().count("x") == 2
    _finish(jobs, _runs(P, ids[0])[0]["job_name"], reward=0.0, cost=3.0)      # item 0 done, item 1 still running
    out = watch_tick(P)
    assert [a["action"] for a in out["actions"]] == ["ingest"] and out["actions"][0]["result_type"] == \
        "fail_delivered"
    assert _runs(P, ids[0])[1]["who"] == "teacher" and _runs(P, ids[0])[1]["state"] == "launched"
    assert set(out["open"]) == set(ids)
    assert counter.read_text().count("x") == 3
    out = watch_tick(P)                                                        # nothing finished: no-op
    assert out["actions"] == [] and counter.read_text().count("x") == 3
    _finish(jobs, _runs(P, ids[0])[1]["job_name"], reward=1.0, cost=4.0)
    out = watch_tick(P, only=[ids[0]])
    assert out["settled"][ids[0]]["label"] == "hard" and ids[1] not in out["open"]


def test_cli_calib_watch_once(tmp_path, capsys):
    from tbsgen.cli import main
    P, [iid], counter, jobs = _env(tmp_path)
    P.run()
    _finish(jobs, _runs(P, iid)[0]["job_name"], reward=1.0, cost=2.0)
    cal = P.cfg["calib"]
    cfgp = tmp_path / "c.toml"
    cfgp.write_text("[calib]\n" + "\n".join(f"{k} = {json.dumps(cal[k])}" for k in (
        "allow_submit", "harbor_root", "jobs_dir", "student_launcher", "teacher_launcher", "agent_allowed_hosts",
        "tmux_bin")) + "\n[budget]\ndaily_usd_total = 1000.0\n")
    assert main(["calib", "watch", "--run", str(P.store.root), "--config", str(cfgp), "--once"]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert line["actions"][0]["action"] == "ingest" and line["open"][iid]["status"] == "waiting"
    assert _runs(P, iid)[1]["who"] == "glm" and _runs(P, iid)[1]["n"] == 2       # GLM pass -> GLM#2


def test_watch_flags_a_dead_launcher(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path)
    (Path(P.cfg["calib"]["harbor_root"]) / "launch.sh").write_text("#!/bin/bash\nexit 2\n")
    P.run()
    out = watch_tick(P)
    assert out["actions"][0]["action"] == "launcher_failed", out
    r = _runs(P, iid)
    assert r[0]["state"] == "launch_failed" and r[0]["outcome"] == "infra"


# ------------------------------------------------------------------ review round 1
def test_job_that_died_without_finished_at_is_infra(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path)
    P.run()
    r = _runs(P, iid)[0]
    _trial(jobs / r["job_name"], tokens=(1000, 0, 100))                 # partial trial, job result has no finished_at
    Path(r["log"] + ".exit").write_text("1\n")
    out = watch_tick(P)
    assert out["actions"][0]["action"] == "job_died"
    r = _runs(P, iid)[0]
    assert r["state"] == "died" and r["outcome"] == "infra" and r["result_type"] == "infra"
    assert r["cost_kind"] == "tokens" and calib.actual_spend(P.ledger) == pytest.approx((1000 + 400) / 1e6)
    assert _runs(P, iid)[1]["who"] == "glm" and _runs(P, iid)[1]["n"] == 2   # infra does not count: GLM again


def test_running_job_with_exit_file_pending_is_not_dead(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path)
    P.run()
    assert watch_tick(P)["actions"] == []                               # no .exit yet: still running
    assert len(_runs(P, iid)) == 1


def test_deliverable_own_missing_file_error_is_delivered(tmp_path):
    t = _trial(tmp_path / "j", exc="AgentTimeoutError")
    (t / "verifier" / "metrics.jsonl").write_text(json.dumps({
        "name": "_solution_cli_run", "rc": 1, "stderr_tail": 'File "/app/output/solution_cli.py", line 3\n'
        "FileNotFoundError: [Errno 2] No such file or directory: 'solution_cache.npy'"}) + "\n")
    assert calib.classify_trial(t)["result_type"] == "timeout_with_output"


def test_relative_prepared_dir_is_resolved_for_the_hook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    P, [iid], counter, _ = _env(tmp_path, prepared_tasks_dir="prepared", prepare_command=_hook(tmp_path))
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    assert r["state"] == "launched" and r["task"] == str((tmp_path / "prepared" / iid).resolve())


def test_started_intent_keeps_its_task(tmp_path):
    """The printed command of an intent was run by hand (job dir exists): enabling preparation later must not
    rewrite the record's task, or ingest would reject the job as a different task."""
    prep = tmp_path / "prepared"
    P, [iid], counter, jobs = _env(tmp_path, allow_submit=False)
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    (jobs / r["job_name"]).mkdir(parents=True)
    (jobs / r["job_name"] / "config.json").write_text(Path(r["config"]).read_text())
    cal = dict(P.cfg["calib"], allow_submit=True, prepared_tasks_dir=str(prep), prepare_command=_hook(tmp_path))
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=cal)), clients={})
    calib.launch(P2.cfg, dict(r), P2.ledger, lambda x: None)
    rec = calib.launch(P2.cfg, dict(r), None, lambda x: None)
    assert rec["task"] == r["task"] and not counter.exists()


def test_reserved_run_is_not_launched_once_the_run_total_is_reached(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path, allow_submit=False)
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    r["ledger_call"] = P.ledger.reserve(r["model"], "S9", iid, True, est_usd=30.0)
    r["state"] = "reserved"
    P.store.item(iid).p("calib", "runs.json").write_text(json.dumps([r]))
    other = P.ledger.reserve("glm-5.3", "S9", "other", True, est_usd=30.0)
    P.ledger.settle(other, 600.0, kind="reported")                       # actual spend above run_usd_total=500
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=dict(P.cfg["calib"], allow_submit=True))), clients={})
    P2.run(only=[iid])
    st = P2.store.item(iid).state
    assert st["status"] == "paused" and st["reason"].startswith("budget:run_total_actual") and not counter.exists()
    r = _runs(P2, iid)[0]
    assert r["state"] == "intent" and r["ledger_call"] is None
    with P2.ledger._db() as db:
        assert db.execute("SELECT status, usd FROM calls WHERE item=?", (iid,)).fetchone() == ("error", 0.0)


def test_s9_on_a_stale_item_view_does_nothing(tmp_path):
    from tbsgen.stages_b import s9
    P, [iid], counter, jobs = _env(tmp_path)
    stale = P.store.item(iid)                                            # loaded as S9/active
    P.run(only=[iid])                                                    # another process launched it
    assert counter.read_text().count("x") == 1
    s9(P, stale)                                                         # re-reads state under the item lock
    assert counter.read_text().count("x") == 1 and len(_runs(P, iid)) == 1


# ------------------------------------------------------------------ review round 2
def _kill_session(tmp_path, rec):
    with (tmp_path / "dead-sessions").open("a") as f:
        f.write(rec["tmux"] + "\n")


def test_job_whose_session_vanished_without_exit_is_infra(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path)
    P.run()
    r = _runs(P, iid)[0]
    assert watch_tick(P)["actions"] == []                               # session alive: running
    _kill_session(tmp_path, r)                                           # killed / reboot: no .exit written
    out = watch_tick(P)
    assert out["actions"][0]["action"] == "job_died" and _runs(P, iid)[0]["state"] == "died"


def test_finished_job_is_ingested_even_if_session_is_gone(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path)
    P.run()
    r = _runs(P, iid)[0]
    _finish(jobs, r["job_name"], reward=1.0, cost=1.0)
    _kill_session(tmp_path, r)
    assert watch_tick(P)["actions"][0]["action"] == "ingest"


def test_crash_after_tmux_before_job_dir_is_not_relaunched(tmp_path):
    """The launch info is saved before tmux runs; a crash before the job dir exists leaves a reserved record with
    launch_attempted: while the session lives it is taken as launched, never started twice."""
    P, [iid], counter, jobs = _env(tmp_path, allow_submit=False)
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    r.update(state="reserved", ledger_call=P.ledger.reserve(r["model"], "S9", iid, True, est_usd=30.0),
             launch_attempted=1.0, tmux=r["job_name"][-60:], log=str(P.store.item(iid).p("calib", "x.log")))
    P.store.item(iid).p("calib", "runs.json").write_text(json.dumps([r]))
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=dict(P.cfg["calib"], allow_submit=True))), clients={})
    P2.run(only=[iid])
    assert not counter.exists() and _runs(P2, iid)[0]["state"] == "launched"


def test_attempted_launch_with_dead_session_and_no_job_is_launched_again(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path, allow_submit=False)
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    r.update(state="reserved", ledger_call=P.ledger.reserve(r["model"], "S9", iid, True, est_usd=30.0),
             launch_attempted=1.0, tmux=r["job_name"][-60:], log=str(P.store.item(iid).p("calib", "x.log")))
    P.store.item(iid).p("calib", "runs.json").write_text(json.dumps([r]))
    _kill_session(tmp_path, r)
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=dict(P.cfg["calib"], allow_submit=True))), clients={})
    P2.run(only=[iid])
    assert counter.read_text().count("x") == 1 and _runs(P2, iid)[0]["state"] == "launched"


def test_hand_started_intent_is_taken_over_even_with_preparation_enabled(tmp_path):
    prep = tmp_path / "prepared"
    P, [iid], counter, jobs = _env(tmp_path, allow_submit=False)
    P.run(only=[iid])
    r = _runs(P, iid)[0]
    (jobs / r["job_name"]).mkdir(parents=True)
    (jobs / r["job_name"] / "config.json").write_text(Path(r["config"]).read_text())
    cal = dict(P.cfg["calib"], allow_submit=True, prepared_tasks_dir=str(prep), prepare_command=_hook(tmp_path))
    P2 = Pipeline(load_config(None, dict(P.cfg, calib=cal)), clients={})
    P2.run(only=[iid])
    r2 = _runs(P2, iid)[0]
    assert r2["state"] == "launched" and r2["task"] == r["task"] and not counter.exists()
    assert not (tmp_path / "prep_calls").exists()                       # no preparation for an existing job
    _finish(jobs, r["job_name"], reward=1.0, cost=2.0)
    assert ingest(P2, P2.store.item(iid))["outcome"] == "pass"


def test_missing_helper_script_is_not_a_missing_deliverable(tmp_path):
    t = _trial(tmp_path / "j", exc="AgentTimeoutError")
    (t / "verifier" / "metrics.jsonl").write_text(json.dumps({
        "name": "_solution_cli_run", "rc": 2,
        "stderr_tail": "python: can't open file '/app/helper.py': [Errno 2] No such file or directory"}) + "\n")
    assert calib.classify_trial(t)["result_type"] == "timeout_with_output"
    (t / "verifier" / "metrics.jsonl").write_text(json.dumps({
        "name": "_solution_cli_run", "rc": 2, "entry": "/app/output/solution_cli.py", "entry_exists": False,
        "stderr_tail": ""}) + "\n")
    assert calib.classify_trial(t)["result_type"] == "timeout_no_output"


def test_metrics_template_records_entry_existence(tmp_path, monkeypatch):
    import importlib.util
    from tbsgen import templates_metrics
    src = Path(templates_metrics.__file__).read_text()
    mod_path = tmp_path / "_metrics.py"
    mod_path.write_text(src)
    monkeypatch.setenv("TBSGEN_METRICS", str(tmp_path / "m.jsonl"))
    spec = importlib.util.spec_from_file_location("_metrics_t", mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_sandbox_prepare", lambda i, o: None)
    m.run_solution(5, input_dir=str(tmp_path), entry=str(tmp_path / "nope.py"))
    rec = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert rec["entry_exists"] is False and rec["entry"].endswith("nope.py")


def test_launch_failed_run_has_cost_fields(tmp_path):
    P, [iid], counter, jobs = _env(tmp_path, tmux_bin=str(tmp_path / "no-such-tmux"))
    P.run(only=[iid], max_steps=1)
    r = _runs(P, iid)[0]
    assert r["state"] == "launch_failed" and r["cost_usd"] == 0.0 and r["cost_kind"] == "not_launched"
