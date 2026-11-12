"""Production throttles: per-paper authoring USD cap (budget.item_usd_cap), papers admitted to S4c / S5G per
day (code.daily_new_*), concurrent agent jobs (code.max_concurrent_agent_jobs).  No network, fake launcher."""
import json
from pathlib import Path

import pytest

from tbsgen.budget import BudgetStop, Ledger, item_usd_left
from tbsgen.state import read_json

from test_codeagent import _agent_item
from test_difficulty import _files


def test_item_usd_cap_in_the_ledger(tmp_path):
    from tbsgen.config import load_config
    cfg = load_config(None, {"run": {"root": str(tmp_path)}, "budget": {"daily_usd_total": 100.0, "item_usd_cap": 10.0}})
    L = Ledger(tmp_path / "ledger.sqlite", cfg)
    c = L.reserve("m", "S5G", "a", True, est_usd=6.0)
    L.settle(c, 7.0)
    assert item_usd_left(L, cfg, "a") == pytest.approx(3.0)
    L.reserve("m", "S9", "a", True, est_usd=30.0)                # calibration is not an authoring stage
    assert item_usd_left(L, cfg, "a") == pytest.approx(3.0)
    with pytest.raises(BudgetStop) as e:
        L.reserve("m", "S6C", "a", True, est_usd=4.0)
    assert e.value.reason == "item_usd"
    L.reserve("m", "S6C", "b", True, est_usd=4.0)               # another paper is unaffected


def test_s5g_job_budget_clamped_to_the_paper_cap(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    P.cfg["budget"]["item_usd_cap"] = 10.0
    c = P.ledger.reserve("claude-opus-5-5", "S4c", it.item_id, True, est_usd=2.0)
    P.ledger.settle(c, 2.5)
    P.run(max_steps=1, until="S5G")
    rec = read_json(it.p("S5G", "run.json"))
    conf = json.loads(open(rec["config"]).read())
    assert conf["agents"][0]["kwargs"]["max_budget_usd"] == "7.50"      # 10 - 2.5 (authoring_max_usd is 20)


def test_s5g_dropped_when_the_paper_cap_is_used_up(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    P.cfg["budget"]["item_usd_cap"] = 10.0
    c = P.ledger.reserve("claude-opus-5-5", "S4c", it.item_id, True, est_usd=9.5)
    P.ledger.settle(c, 9.5)
    P.run(max_steps=1, until="S5G")
    st = P.store.item(it.item_id).state
    assert st["status"] == "dropped" and st["reason"] == "s5g_budget_exhausted" and not fl.calls


def test_daily_new_authoring_and_concurrency(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    monkeypatch.setattr("tbsgen.stages_a.validate_spec", lambda s, c: [])
    [b] = P.ingest([{"doi": "10.1/b", "title": "Other"}])
    itb = P.store.item(b)
    itb.p("fulltext.txt").write_text("paper")
    from tbsgen.state import write_json
    write_json(itb.p("S3c_code.json"), {"ok": True, "pin": "p", "remote_dir": "/r/y"})
    itb.state["stage"] = "S5G"
    itb.save()
    P.cfg["code"]["daily_new_authoring"] = 1
    P.run(max_steps=4, until="S5G")
    sa, sb = P.store.item(it.item_id).state, P.store.item(b).state
    admitted = [x for x in (sa, sb) if x["stage"] == "S6A_run"]
    paused = [x for x in (sa, sb) if x["status"] == "paused"]
    assert len(admitted) == 1 and len(paused) == 1 and paused[0]["reason"].startswith("budget:daily_s5g_papers")
    assert len(fl.calls) == 1
    # the admitted paper's second job (re-authoring) is not counted again
    a_id = admitted[0]["item_id"]
    ita = P.store.item(a_id)
    ita.state.update(stage="S5G", status="active", S5G_rerun=True)
    ita.save()
    P.run(max_steps=1, only=[a_id], until="S5G")
    assert len(fl.calls) == 2
    # concurrency: a running job elsewhere blocks a new one (item waits, no launch)
    P.cfg["code"]["daily_new_authoring"] = None
    P.cfg["code"]["max_concurrent_agent_jobs"] = 1
    other = paused[0]["item_id"]
    write_json(ita.p("S5G", "run.json"), dict(read_json(ita.p("S5G", "run.json")), state="launched"))
    ito = P.store.item(other)
    ito.state.update(status="active")
    ito.save()
    P.run(max_steps=1, only=[other], until="S5G")
    so = P.store.item(other).state
    assert so["status"] == "waiting" and "agent slots full" in so["reason"] and len(fl.calls) == 2


def test_reservation_is_the_job_allowance_and_admission_only_on_launch(tmp_path, monkeypatch):
    """With a paper cap the ledger reserves the whole job allowance (an unknown final cost cannot under-count), a
    resumed intent reuses that amount, and a paper is counted for the day only once its job really started."""
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    P.cfg["budget"]["item_usd_cap"] = 10.0
    P.cfg["code"]["daily_new_authoring"] = 1
    from tbsgen import codeagent
    import subprocess as sp
    monkeypatch.setattr(codeagent.subprocess, "run", lambda args, **kw: sp.CompletedProcess(args, 1, "", "no tmux"))
    P.run(max_steps=1, until="S5G")                              # tmux fails: never started
    st = P.store.item(it.item_id).state
    assert st["status"] == "paused" and st["reason"] == "code:S5G:launch_failed"
    assert not (P.store.root / "s5g-admissions.jsonl").exists()  # not admitted: nothing started
    with P.ledger._db() as db:
        rows = db.execute("SELECT est_usd, usd, status FROM calls WHERE item=? AND stage='S5G'", (it.item_id,)).fetchall()
    assert rows == [(10.0, 0.0, "error")]                        # reserved at the allowance, released at 0
    monkeypatch.setattr(codeagent.subprocess, "run", fl)
    P.resume(("code:",))
    P.run(max_steps=1, until="S5G")
    rec = read_json(it.p("S5G", "run.json"))
    assert rec["state"] == "launched" and rec["est_usd"] == 10.0
    assert len((P.store.root / "s5g-admissions.jsonl").read_text().splitlines()) == 1


def test_intent_holds_no_slot_and_resumed_intent_is_readmitted(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    [b] = P.ingest([{"doi": "10.1/b", "title": "Other"}])
    itb = P.store.item(b)
    itb.p("fulltext.txt").write_text("paper")
    from tbsgen.state import write_json
    write_json(itb.p("S3c_code.json"), {"ok": True, "pin": "p", "remote_dir": "/r/y"})
    itb.state["stage"] = "S5G"
    itb.save()
    # item A: budget stop at reservation -> intent left behind
    P.cfg["budget"]["daily_usd_per_model"] = {"claude-opus-5-5": 1.0}
    P.run(max_steps=1, only=[it.item_id], until="S5G")
    assert read_json(it.p("S5G", "run.json"))["state"] == "intent"
    assert P.store.item(it.item_id).state["reason"].startswith("budget:")
    # the intent holds no slot: B starts with max_concurrent_agent_jobs = 1
    P.cfg["budget"]["daily_usd_per_model"] = {}
    P.cfg["code"]["max_concurrent_agent_jobs"] = 1
    P.cfg["code"]["daily_new_authoring"] = 1
    P.stopped_models.clear()
    P.run(max_steps=1, only=[b], until="S5G")
    assert read_json(itb.p("S5G", "run.json"))["state"] == "launched"
    # A resumed: its intent goes through admission again -> daily cap (1) reached by B
    P.resume(("budget:",))
    P.cfg["code"]["max_concurrent_agent_jobs"] = None
    P.run(max_steps=1, only=[it.item_id], until="S5G")
    st = P.store.item(it.item_id).state
    assert st["status"] == "paused" and st["reason"].startswith("budget:daily_s5g_papers"), st
    assert len(fl.calls) == 1


def test_admission_backfilled_after_a_crash(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    P.cfg["code"]["daily_new_authoring"] = 5
    P.run(max_steps=1, until="S5G")
    (P.store.root / "s5g-admissions.jsonl").unlink()               # lost: crash after "launched"
    P.run(max_steps=1, until="S5G")                                # the launched branch re-records it
    rows = (P.store.root / "s5g-admissions.jsonl").read_text().splitlines()
    assert len(rows) == 1 and it.item_id in rows[0]


def test_reserved_without_launch_rechecks_admission_and_admit_uses_launch_day(tmp_path, monkeypatch):
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    P.cfg["code"]["daily_new_authoring"] = 1
    from tbsgen import stages_a
    # another paper already took today's slot
    (P.store.root / "s5g-admissions.jsonl").write_text(json.dumps(dict(day=stages_a._cst_day(), item="x")) + "\n")
    from tbsgen.state import write_json
    (it.p("S5G")).mkdir(parents=True, exist_ok=True)
    write_json(it.p("S5G", "job.json"), {"job_name": "J", "jobs_dir": str(tmp_path), "agents": [{"model_name": "m"}]})
    write_json(it.p("S5G", "run.json"), dict(job_name="J", state="reserved", config=str(it.p("S5G", "job.json")),
                                             job_dir=str(tmp_path / "J"), kind="S5G", est_usd=1.0))
    P.run(max_steps=1, until="S5G")                 # crash before the launch, resumed: admission checked again
    st = P.store.item(it.item_id).state
    assert st["status"] == "paused" and st["reason"].startswith("budget:daily_s5g_papers") and not fl.calls
    # a job launched yesterday whose admission line was lost is recorded on yesterday's date
    rec = read_json(it.p("S5G", "run.json"))
    rec.update(state="launched", launch_attempted=__import__("time").time() - 86400)
    write_json(it.p("S5G", "run.json"), rec)
    with stages_a._agent_slot_lock(P):
        stages_a._agent_admit(P, P.store.item(it.item_id), "S5G")
    rows = [json.loads(x) for x in (P.store.root / "s5g-admissions.jsonl").read_text().splitlines()]
    assert rows[-1]["item"] == it.item_id and rows[-1]["day"] != stages_a._cst_day()


def test_launcher_failed_job_with_its_own_log_is_not_charged(tmp_path, monkeypatch):
    """A relay preflight 503 ends the launcher before Harbor creates the job: no model ran, so the per-paper /
    per-round allowance (spent.json) must not be charged the job's whole max_usd."""
    import subprocess as sp
    from tbsgen import codeagent
    P, it, fl = _agent_item(tmp_path, monkeypatch, _files(design=None), stage="S5G")
    monkeypatch.setattr(codeagent.subprocess, "run", lambda args, **kw: sp.CompletedProcess(args, 0, "", ""))
    P.run(max_steps=1, until="S5G")                 # tmux started; the launcher then fails before Harbor runs
    rec = read_json(it.p("S5G", "run.json"))
    assert rec["state"] == "launched" and Path(rec["log"]).name.startswith(rec["job_name"])
    Path(rec["log"]).write_text("relay preflight http=503\nharbor exit 3\n")
    Path(rec["log"] + ".exit").write_text("3")
    monkeypatch.setattr(codeagent.subprocess, "run", fl)
    P.run(max_steps=1, until="S5G")                 # poll -> launcher_failed -> infra retry
    st = P.store.item(it.item_id).state
    assert st["status"] == "active" and st["stage"] == "S5G", st
    P.run(max_steps=1, until="S5G")                 # next job: the failed one is charged first
    spent = read_json(it.p("S5G", "spent.json"))
    assert spent[rec["job_name"]]["usd"] == 0.0, spent
    assert read_json(it.p("S5G", "run.json"))["job_name"] != rec["job_name"]
