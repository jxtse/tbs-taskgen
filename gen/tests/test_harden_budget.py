"""Hardening budget and round accounting: budget.item_harden_usd_cap (a hardening round's authoring spend kept apart
from the first version's budget.item_usd_cap), infra-reverted rounds that do not use up harden_max_rounds,
harden_max_attempts, and the profile's preferred levers.  No network, fake launcher."""
import json

import pytest

from tbsgen import difficulty
from tbsgen.budget import BudgetStop, Ledger, harden_scope, item_usd_left
from tbsgen.config import load_config
from tbsgen.state import read_json, write_json

from test_codeagent import SPEC
from test_difficulty import DESIGN, EXAMPLE, _calib_item, _files, _profiled


def _ledger(tmp_path, **b):
    cfg = load_config(None, {"run": {"root": str(tmp_path)},
                             "budget": dict({"daily_usd_total": 1000.0, "item_usd_cap": 10.0}, **b)})
    return cfg, Ledger(tmp_path / "ledger.sqlite", cfg)


# ------------------------------------------------------------------ ledger
def test_harden_cap_unset_keeps_one_shared_cap(tmp_path):
    cfg, L = _ledger(tmp_path)
    L.settle(L.reserve("m", "S5G", "a", True, est_usd=6.0), 6.0)
    with harden_scope(1):
        L.settle(L.reserve("m", "S5G", "a", True, est_usd=3.0), 3.0)
        assert item_usd_left(L, cfg, "a") == pytest.approx(1.0)
        with pytest.raises(BudgetStop) as e:
            L.reserve("m", "S6C", "a", True, est_usd=2.0)
    assert e.value.reason == "item_usd"
    assert item_usd_left(L, cfg, "a") == pytest.approx(1.0)


def test_harden_cap_is_separate_and_covers_s6c_s8_s7(tmp_path):
    cfg, L = _ledger(tmp_path, item_harden_usd_cap=15.0)
    L.settle(L.reserve("m", "S5G", "a", True, est_usd=9.5), 9.5)          # first version: 0.5 left
    assert item_usd_left(L, cfg, "a") == pytest.approx(0.5)
    with harden_scope(1):
        assert item_usd_left(L, cfg, "a") == pytest.approx(15.0)          # the round's own allowance
        L.settle(L.reserve("m", "S5G", "a", True, est_usd=10.0), 10.0)
        L.settle(L.reserve("m", "S6C", "a", True, est_usd=1.0), 1.0)
        L.settle(L.reserve("m", "S8", "a", True, est_usd=1.0), 1.0)
        L.settle(L.reserve("m", "S7", "a", True, est_usd=1.0), 1.0)       # S7 counts in a hardening round
        assert item_usd_left(L, cfg, "a") == pytest.approx(2.0)
        L.reserve("m", "S9", "a", True, est_usd=30.0)                    # calibration: never authoring spend
        with pytest.raises(BudgetStop) as e:
            L.reserve("m", "S6C", "a", True, est_usd=2.5)
        assert e.value.reason == "item_harden_usd"
    with harden_scope(2):                                                # one cap for all rounds together
        assert item_usd_left(L, cfg, "a") == pytest.approx(2.0)
    # the first version's cap ignores the hardening spend
    assert item_usd_left(L, cfg, "a") == pytest.approx(0.5)
    with pytest.raises(BudgetStop) as e:
        L.reserve("m", "S6C", "a", True, est_usd=1.0)
    assert e.value.reason == "item_usd"
    with L._db() as db:
        tags = dict(db.execute("SELECT stage, MAX(COALESCE(harden,0)) FROM calls WHERE item='a' GROUP BY stage"))
    assert tags["S7"] == 1 and tags["S6C"] == 1


def test_s5g_round_budget_is_clamped_to_the_harden_cap(tmp_path, monkeypatch):
    """Pipeline.step tags the round's reservations: the hardening S5G job gets min(harden_authoring_max_usd,
    item_harden_usd_cap left), not the first version's (spent) item_usd_cap."""
    P, it, fl = _profiled(tmp_path, monkeypatch, _files(design=dict(DESIGN, hardening={
        "changes": "c", "why_previous_solution_fails_now": "w"})), harden_authoring_max_usd=8.0)
    P.cfg["budget"].update(item_usd_cap=10.0, item_harden_usd_cap=15.0)
    P.ledger.settle(P.ledger.reserve("claude-opus-5-5", "S5G", it.item_id, True, est_usd=9.8), 9.8)
    with harden_scope(1):
        P.ledger.settle(P.ledger.reserve("claude-opus-5-5", "S6C", it.item_id, True, est_usd=9.0), 9.0)
    from tbsgen.stages_b import _s5g_base
    it.state.update(stage="S5G", status="active", harden_round=1, harden_active=True, harden_base=_s5g_base(it, 1))
    it.save()
    (it.p("harden", "round-1", "prev", "task")).mkdir(parents=True)
    (it.p("harden", "round-1", "prev", "DONE")).write_text("x")
    P.run(max_steps=1, until="S5G")
    rec = read_json(it.p("S5G", "run.json"))
    conf = json.loads(open(rec["config"]).read())
    assert conf["agents"][0]["kwargs"]["max_budget_usd"] == "6.00"        # 15 - 9 (< 8 per round)
    with P.ledger._db() as db:
        assert db.execute("SELECT harden FROM calls WHERE id=?", (rec["ledger_call"],)).fetchone()[0] == 1


# ------------------------------------------------------------------ rounds
def _fail_round(P, iid, reason, result=None):
    it = P.store.item(iid)
    assert it.state["stage"] == "S5G" and it.state.get("harden_active"), it.state
    if result is not None:
        write_json(it.p("S5G_result.json"), result)
    it.finish("dropped", reason)
    P.run(only=[iid], until="S9", max_steps=1)
    return P.store.item(iid)


AGENT_DIED = dict(outcome=None, exception="NonZeroAgentExitCodeError", missing=["_spec/spec.json"])


def test_infra_revert_does_not_use_a_round(tmp_path, monkeypatch):
    from tbsgen.stages_b import watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    P.cfg["difficulty"]["harden_max_rounds"] = 1
    watch_tick(P, only=[iid])
    it = _fail_round(P, iid, "s5g_missing_files", AGENT_DIED)
    st = it.state
    assert not st.get("harden_exhausted") and st["harden_reverted"][0]["infra"] == "s5g:NonZeroAgentExitCodeError"
    assert difficulty.rounds_used(it) == 0 and difficulty.rounds_left(P.cfg, it)
    assert read_json(it.p("harden", "round-1", "record.json"))["reverted"]["infra"]
    ev = [e for e in P.store.events() if e["kind"] == "harden_reverted"][-1]
    assert ev["infra"] and ev["rounds_used"] == 0 and ev["attempts"] == 1
    P.run(only=[iid], until="S9")                       # the old version's pass hardens again: attempt 2
    st = P.store.item(iid).state
    assert st["stage"] == "S5G" and st["harden_round"] == 2 and st["harden_active"]
    assert counter.read_text().count("x") == 1          # no second student run in between


@pytest.mark.parametrize("reason,result", [
    ("s5g_budget_exhausted", None),                                     # cost cap: a real round
    ("item_harden_usd_cap", None),
    ("s5g_missing_files", dict(AGENT_DIED, exception="AgentTimeoutError")),   # ran out of time: real
    ("s5g_missing_files", dict(AGENT_DIED, exception=None)),            # finished without the files: real
])
def test_real_revert_uses_the_round(tmp_path, monkeypatch, reason, result):
    from tbsgen.stages_b import watch_tick
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    watch_tick(P, only=[iid])
    it = _fail_round(P, iid, reason, result)
    assert it.state["harden_exhausted"] and not it.state["harden_reverted"][0]["infra"]
    assert not difficulty.rounds_left(P.cfg, it)


@pytest.mark.parametrize("reason,result", [
    ("s5g_missing_files", dict(outcome="launcher_failed", missing=["x"])),
    ("s5g_missing_files", dict(outcome="no_result", missing=["x"])),
    ("code:S5G:launch_failed", None),
    ("model_error:S6C:network", None),
])
def test_infra_reasons(tmp_path, reason, result):
    from tbsgen.state import Item
    it = Item(tmp_path, "x").load()
    if result:
        write_json(it.p("S5G_result.json"), result)
    stage = "S6C" if "S6C" in reason else "S5G"
    assert difficulty.revert_infra(it, stage, reason)
    assert difficulty.revert_infra(it, "S6B", "not_separable") is None
    assert difficulty.revert_infra(it, "S8", "model_error:S8:other") is None


def test_attempt_cap_stops_an_infra_loop(tmp_path, monkeypatch):
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    it = P.store.item(iid)
    cfg = P.cfg
    it.state.update(harden_round=3, harden_reverted=[dict(round=k, infra="x") for k in (1, 2, 3)])
    assert difficulty.rounds_used(it) == 0 and difficulty.rounds_left(cfg, it)
    it.state.update(harden_round=4, harden_reverted=[dict(round=k, infra="x") for k in (1, 2, 3, 4)])
    assert not difficulty.rounds_left(cfg, it)                          # harden_max_attempts = 4
    cfg["difficulty"]["harden_max_attempts"] = 6
    assert difficulty.rounds_left(cfg, it)
    it.state.update(harden_round=3, harden_reverted=[dict(round=1, infra="x")])
    assert difficulty.rounds_used(it) == 2 and not difficulty.rounds_left(cfg, it)   # 2 real rounds


# ------------------------------------------------------------------ preferred levers
def _pref_profile(tmp_path, pref="EX02"):
    p = tmp_path / "prof.md"
    p.write_text(EXAMPLE.read_text().replace("profile_id: example-weakness-v0",
                                             f"profile_id: example-weakness-v0\npreferred: [{pref}]", 1))
    return p


def test_preferred_levers(tmp_path):
    cfg = load_config(None, {"difficulty": {"profile_path": str(_pref_profile(tmp_path))}})
    prof = difficulty.load_profile(cfg)
    assert prof["preferred"] == ["EX02"]
    assert difficulty.load_profile(load_config(None, {"difficulty": {"profile_path": str(EXAMPLE)}}))["preferred"] == []
    h = dict(DESIGN, hardening={"changes": "c", "why_previous_solution_fails_now": "w"})
    assert difficulty.validate_design(h, prof, SPEC, harden_round=1) == []
    other = dict(h, levers=[dict(DESIGN["levers"][0], weakness_id="EX01")])
    assert difficulty.validate_design(dict(other, hardening=None), prof, SPEC) == []   # first version: free choice
    assert any("preferred" in e for e in difficulty.validate_design(other, prof, SPEC, harden_round=1))
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError):                                     # a preferred id must be a profile entry
        difficulty.load_profile(load_config(None, {"difficulty": {
            "profile_path": str(_pref_profile(tmp_path / "b", pref="ZZ9"))}}))


def test_preferred_levers_reach_the_s5g_prompt(tmp_path, monkeypatch):
    P, it, _ = _profiled(tmp_path, monkeypatch, _files())
    P.cfg["difficulty"]["profile_path"] = str(_pref_profile(tmp_path))
    P.run(max_steps=1, until="S5G")
    ins = it.p("S5G", "task", "instruction.md").read_text()
    assert "Preferred levers for this solver: EX02" in ins


def test_s7_hardening_round_needs_a_shown_preferred_lever(tmp_path):
    from test_difficulty import _item_dir
    th = {"rmse": {"threshold": 0.3, "direction": "<="}}
    prof = dict(id="example-weakness-v0", ids=["EX01", "EX02", "EX03"], preferred=["EX02"])
    shown = difficulty.check_lever_separation(_item_dir(tmp_path / "a"), SPEC, th)
    assert difficulty.preferred_problem(shown, prof, 1) == []
    flat = difficulty.check_lever_separation(_item_dir(tmp_path / "b", mid=0.2, naive=0.25), SPEC, th)
    assert difficulty.preferred_problem(flat, prof, 1)                  # EX02 named but not measured
    assert difficulty.preferred_problem(flat, prof, 0) == []            # first version: not required
    assert difficulty.preferred_problem(flat, dict(prof, preferred=[]), 1) == []
    other = dict(shown, levers=[dict(shown["levers"][0], weakness_id="EX01")])
    assert difficulty.preferred_problem(other, prof, 2)


def test_s7_rejects_a_hardened_version_without_a_preferred_lever(tmp_path, monkeypatch):
    """End to end at S7: a hardening round whose only shown lever is not preferred goes back to S5G."""
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    P.cfg["difficulty"]["profile_path"] = str(_pref_profile(tmp_path, pref="EX01"))
    it = P.store.item(iid)
    for rel in ("generator/generate.py", "solution/solve.py", "tests/measure.py", "baselines/mid.py"):
        p = it.p("harden", "round-1", "prev", "task", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("old " + rel)
    write_json(it.p("S6A_measure.json"), {w: {s: {"rmse": v} for s in ("0", "1")}
                                          for w, v in (("ref", 0.1), ("mid", 0.5), ("naive", 0.9))})
    write_json(it.p("thresholds.json"), {"thresholds": {"rmse": {"threshold": 0.3, "direction": "<="}}})
    it.state.update(stage="S7", status="active", harden_round=1, harden_active=True)
    it.save()
    monkeypatch.setattr("tbsgen.stages_a._agent_reauthor", lambda P_, it_, why, force=False: False)
    monkeypatch.setattr("tbsgen.codeagent.env_ok", lambda cfg, agent=False: None)
    from tbsgen import stages_b
    stages_b.s7(P, P.store.item(iid))
    g = read_json(it.p("gates", "difficulty.json"))
    assert any("preferred lever (EX01)" in x for x in g["hardening"]), g
    assert P.store.item(iid).state["reason"] == "difficulty_design_unproven"
