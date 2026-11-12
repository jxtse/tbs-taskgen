"""Production throttles: per-paper authoring USD cap (budget.item_usd_cap), papers admitted to S4c / S5G per
day (code.daily_new_*), concurrent agent jobs (code.max_concurrent_agent_jobs).  No network, fake launcher."""
import json

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
