"""S4c on its own model route (code.runnability_model / runnability_env / runnability_effort / runnability_cli_budget)
and per-stage agent caps (code.agent_job_caps).  No network, fake launcher."""
import json

from tbsgen import codeagent
from tbsgen.state import read_json, write_json

from test_codeagent import _agent_item
from test_difficulty import _files


def _conf(it, stage):
    return json.loads(open(read_json(it.p(stage, "run.json"))["config"]).read())


def test_s4c_runs_on_its_own_model_and_endpoint(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {}, stage="S4c")
    P.cfg["code"].update(runnability_model="deepseek-flash", runnability_effort="max", runnability_cli_budget=False,
                         runnability_env={"ANTHROPIC_BASE_URL": "http://10.0.0.1:13002"})
    P.run(max_steps=1, only=[it.item_id])
    agent = _conf(it, "S4c")["agents"][0]
    assert agent["model_name"] == "deepseek-flash"
    assert agent["env"]["ANTHROPIC_BASE_URL"] == "http://10.0.0.1:13002"
    assert agent["kwargs"]["reasoning_effort"] == "max" and "max_budget_usd" not in agent["kwargs"]
    assert read_json(it.p("S4c", "run.json"))["model"] == "deepseek-flash"
    with P.ledger._db() as db:
        assert db.execute("SELECT model FROM calls WHERE item=? AND stage='S4c'", (it.item_id,)).fetchone()[0] == \
            "deepseek-flash"


def test_s5g_keeps_the_agent_model_when_s4c_has_its_own(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    P.cfg["code"].update(runnability_model="deepseek-flash", runnability_cli_budget=False,
                         runnability_env={"ANTHROPIC_BASE_URL": "http://10.0.0.1:13002"})
    P.run(max_steps=1, only=[it.item_id], until="S5G")
    agent = _conf(it, "S5G")["agents"][0]
    assert agent["model_name"] == P.cfg["calib"]["teacher_model"]
    assert "ANTHROPIC_BASE_URL" not in agent["env"] and "max_budget_usd" in agent["kwargs"]


def test_default_config_is_unchanged(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {}, stage="S4c")
    P.run(max_steps=1, only=[it.item_id])
    agent = _conf(it, "S4c")["agents"][0]
    assert agent["model_name"] == P.cfg["calib"]["teacher_model"] and "max_budget_usd" in agent["kwargs"]
    assert codeagent.stage_model(P.cfg, "S4c") == codeagent.stage_model(P.cfg, "S5G")


def _second_item(P, stage):
    [b] = P.ingest([{"doi": "10.1/b", "title": "Other"}])
    itb = P.store.item(b)
    itb.p("fulltext.txt").write_text("paper")
    write_json(itb.p("S3c_code.json"), {"ok": True, "pin": "p", "remote_dir": "/r/y"})
    itb.state["stage"] = stage
    itb.save()
    return itb


def test_stage_cap_takes_s4c_out_of_the_shared_pool(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {}, stage="S4c")
    itb = _second_item(P, "S4c")
    # a running S5G job elsewhere fills the shared pool of 1 ...
    write_json(it.p("S5G", "run.json"), {"state": "launched", "job_name": "x"})
    P.cfg["code"]["max_concurrent_agent_jobs"] = 1
    P.run(max_steps=1, only=[itb.item_id])
    sb = P.store.item(itb.item_id).state
    assert sb["status"] == "waiting" and "agent slots full (1/1)" in sb["reason"] and not fl.calls
    # ... but S4c with its own cap ignores S5G jobs and counts only S4c jobs
    P.cfg["code"]["agent_job_caps"] = {"S4c": 1}
    itb.load()
    itb.state.update(status="active")
    itb.save()
    P.run(max_steps=1, only=[itb.item_id])
    assert read_json(itb.p("S4c", "run.json"))["state"] == "launched" and len(fl.calls) == 1


def test_stage_cap_counts_its_own_jobs(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, {}, stage="S4c")
    itb = _second_item(P, "S4c")
    write_json(it.p("S4c", "run.json"), dict(read_json(it.p("S4c", "run.json"), {}) or {}, state="launched",
                                             job_name="y"))
    P.cfg["code"]["agent_job_caps"] = {"S4c": 1}
    P.cfg["code"]["max_concurrent_agent_jobs"] = 6
    P.run(max_steps=1, only=[itb.item_id])
    sb = P.store.item(itb.item_id).state
    assert sb["status"] == "waiting" and "agent slots full (1/1)" in sb["reason"]


def test_s5g_pool_excludes_s4c_jobs_when_s4c_has_its_own_cap(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    itb = _second_item(P, "S4c")
    write_json(itb.p("S4c", "run.json"), {"state": "launched", "job_name": "z"})
    P.cfg["code"]["max_concurrent_agent_jobs"] = 1
    P.run(max_steps=1, only=[it.item_id], until="S5G")
    assert P.store.item(it.item_id).state["status"] == "waiting"          # shared pool: the S4c job counts
    P.cfg["code"]["agent_job_caps"] = {"S4c": 10}
    st = P.store.item(it.item_id)
    st.state.update(status="active")
    st.save()
    P.run(max_steps=1, only=[it.item_id], until="S5G")
    assert read_json(it.p("S5G", "run.json"))["state"] in ("launched", "finished")
