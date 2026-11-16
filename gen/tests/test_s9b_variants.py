"""S9b: harder variants for tasks the calibration labels easy.

An easy version (the student passed both decisive runs) is re-authored harder (the S5G hardening design, with the
weakness profile and the evidence of the decisive pass) and re-calibrated under the same label rules.  Unlike the
first-pass hardening loop it has its own counters (calib.variant_depth, state.variant_round), its own per-item
budget (budget.item_variant_usd_cap, calls.variant) and works for S9R-reviewed items (a fresh review round per
version, so review job names / verdict records never collide and an earlier verdict is never reused).
No Docker, no network, no real jobs.
"""
import json
from pathlib import Path

import pytest

from tbsgen import difficulty, review
from tbsgen.budget import BudgetStop, Ledger, harden_scope, item_usd_left, variant_scope
from tbsgen.config import load_config
from tbsgen.state import read_json, write_json

from test_calib_s9 import _runs
from test_difficulty import _calib_item
from test_harden_budget import AGENT_DIED, _fail_round
from test_review_s9r import V, _env as _review_env, _finish as _finish_job, _to_label


# ------------------------------------------------------------------ helpers
def _easy_variant_item(tmp_path, monkeypatch, depth=2):
    """S9 item, S9b on, the first-pass hardening loop off (so the easy label itself starts the round)."""
    P, iid, counter, jobs = _calib_item(tmp_path, monkeypatch)
    P.cfg["difficulty"]["harden"] = False
    P.cfg["calib"]["variant_depth"] = depth
    return P, iid, counter, jobs


def _s9(P, iid, **state):
    it = P.store.item(iid)
    it.load()
    if state:
        it.state.update(state)
        it.save()
    return it


def _finish_last(P, iid, jobs, reward=1.0, cost=5.0):
    _finish_job(jobs, _runs(P, iid)[-1]["job_name"], reward=reward, cost=cost)


def _to_easy(P, iid, jobs):
    """The student passes both decisive runs -> the item is labelled easy (and S9b reacts)."""
    from tbsgen.stages_b import watch_tick
    watch_tick(P, only=[iid])                   # ingest run 1 (pass), launch run 2
    _finish_last(P, iid, jobs)
    watch_tick(P, only=[iid])                   # easy -> an S9b variant round instead of finishing
    return _s9(P, iid)


def _simulate_packaged_version(it, epoch, runs):
    """What PKG does at the end of a round, then a fresh calibration epoch with these decisive runs."""
    it.state.pop("harden_active", None)
    it.state.pop("harden_kind", None)
    write_json(it.p("calib", "runs.json"), runs)
    it.state.update(stage="S9", status="active", calib_epoch=epoch)
    it.save()
    return it


def _two_passes():
    return [dict(who="glm", outcome="pass", minutes=30, job_name="agent-1", result_type="pass", n=1),
            dict(who="glm", outcome="pass", minutes=30, job_name="agent-2", result_type="pass", n=2)]


HARD = [dict(who="glm", outcome="fail", minutes=60, job_name="agent-3", result_type="fail_delivered", n=1),
        dict(who="teacher", outcome="pass", minutes=60, job_name="teacher-1", result_type="pass", n=1)]


# ------------------------------------------------------------------ the trigger
def test_easy_label_starts_a_variant_round(tmp_path, monkeypatch):
    it = None
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch)
    from tbsgen.stages_b import watch_tick
    watch_tick(P, only=[iid])                   # run 1 passed; the first-pass loop is off -> a second run
    it = _s9(P, iid)
    assert it.state["stage"] == "S9" and counter.read_text().count("x") == 2
    job2 = _runs(P, iid)[-1]["job_name"]                        # the second decisive run (it passed)
    _finish_last(P, iid, jobs)
    watch_tick(P, only=[iid])
    it.load()
    st = it.state
    # (watch_tick's run continues into S5G, which pauses here: this config has no build host)
    assert st["stage"] == "S5G" and st.get("harden_active") and st["status"] in ("active", "paused"), st
    assert st["harden_kind"] == "variant" and st["harden_round"] == 1 and st["variant_round"] == 1
    assert st.get("variant_reopened") is None       # the S9 label path, not the CLI
    # the evidence of the decisive pass (the second run), not the first one
    fb = read_json(it.p("harden", "round-1", "feedback.json"))
    assert fb["job"] == job2 and fb["outcome"] == "pass"
    assert fb["metric_margins"] and fb["metric_margins"][0]["metric"] == "m"
    rec = read_json(it.p("harden", "round-1", "record.json"))
    assert rec["kind"] == "variant" and rec["round"] == 1 and rec["archived_epoch"] == 0
    assert rec["trigger"] == job2
    ev = [e for e in P.store.events() if e["kind"] == "harden_start"][-1]
    assert ev["rkind"] == "variant" and ev["item"] == iid
    # the variant's authoring round has its own instruction (S9b, not the first-pass hardening wording)
    seen = []
    monkeypatch.setattr("tbsgen.stages_a._agent_stage", lambda P_, it_, k, ins, *a, **kw: seen.append(ins) or None)
    P.resume(("code:",))                        # the test config cannot run the S5G agent
    P.run(only=[iid], max_steps=1)
    assert "S9b variant round 1 of at most 2" in seen[0]


def test_variant_rounds_are_separate_from_the_hardening_rounds(tmp_path, monkeypatch):
    """An item that used up harden_max_rounds still gets its S9b rounds (own counters, own depth)."""
    from tbsgen.stages_b import variant_cmd
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch)
    it = _s9(P, iid, harden_round=2, harden_exhausted=True)
    write_json(it.p("calib", "runs.json"), _two_passes())        # an easy version (no run pending)
    assert not difficulty.rounds_left(P.cfg, it) and difficulty.variant_rounds_left(P.cfg, it)
    out = variant_cmd(P, it, "S9b re-open")
    assert out["round"] == 3 and out["variant_round"] == 1 and out["stage"] == "S5G"
    it = _s9(P, iid)
    assert it.state["harden_kind"] == "variant"
    assert difficulty.attempts_used(it, "harden") == 2                  # the hardening counter is untouched
    assert difficulty.rounds_used_kind(it, "variant") == 1
    assert difficulty.rounds_used_kind(it, "harden") == 2


# ------------------------------------------------------------------ the outcome
def test_still_easy_after_the_variant_depth_is_s9b_exhausted(tmp_path, monkeypatch):
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch, depth=1)
    it = _to_easy(P, iid, jobs)
    assert it.state["stage"] == "S5G" and it.state["variant_round"] == 1
    # the variant version is built and re-calibrated: still easy
    _simulate_packaged_version(it, 1, _two_passes())
    P.run(only=[iid])
    it.load()
    st = it.state
    assert st["status"] == "done" and st["calib_label"] == "easy", st
    assert st["reason"] == "label:easy (S9b exhausted)" and st["variant_round"] == 1


def test_easy_with_s9b_off_keeps_the_old_reason(tmp_path, monkeypatch):
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch, depth=0)
    it = _to_easy(P, iid, jobs)
    assert it.state["status"] == "done" and it.state["reason"] == "label:easy (not kept; S9b variants off)"


def test_a_variant_that_comes_out_hard_is_kept(tmp_path, monkeypatch):
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch)
    it = _to_easy(P, iid, jobs)
    _simulate_packaged_version(it, 1, HARD)
    P.run(only=[iid])
    it.load()
    assert it.state["calib_label"] == "hard" and it.state["stage"] == "S10" and it.state["status"] == "done"


# ------------------------------------------------------------------ revert rules
def test_infra_failed_variant_round_does_not_use_up_a_variant(tmp_path, monkeypatch):
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch)
    it = _to_easy(P, iid, jobs)
    assert it.state["harden_kind"] == "variant"
    it = _fail_round(P, iid, "s5g_missing_files", AGENT_DIED)    # dropped -> reverted by the scheduler
    it = _s9(P, iid)
    rev = (it.state.get("harden_reverted") or [])[0]
    assert rev["kind"] == "variant" and rev["infra"] == "s5g:NonZeroAgentExitCodeError"
    assert not it.state.get("variant_exhausted") and not it.state.get("harden_exhausted")
    assert difficulty.rounds_used_kind(it, "variant") == 0 and difficulty.variant_rounds_left(P.cfg, it)
    assert it.state["stage"] == "S9"


def test_real_failed_variant_round_exhausts_s9b(tmp_path, monkeypatch):
    P, iid, counter, jobs = _easy_variant_item(tmp_path, monkeypatch)
    it = _to_easy(P, iid, jobs)
    _s9(P, iid, stage="S6B").finish("dropped", "not_separable")
    P.run(only=[iid], until="S9", max_steps=1)                   # the scheduler reverts the failed version
    it = _s9(P, iid)
    assert it.state["variant_exhausted"] and not it.state.get("harden_exhausted")
    assert not difficulty.variant_rounds_left(P.cfg, it)
    assert it.state["stage"] == "S9"
    assert (it.p("task", "solution", "solve.py")).read_text().startswith("v1 ")


# ------------------------------------------------------------------ budget: its own per-item cap
def _cfg(caps):
    return load_config(None, {"budget": dict({"daily_usd_total": 1000.0, "est_usd_per_call": 1.0,
                                              "item_usd_cap_stages": ["S4c", "S5G", "S6A", "S6M", "S6C", "S8"]},
                                             **caps)})


def test_variant_cap_is_separate_from_the_harden_cap(tmp_path):
    cfg = _cfg(dict(item_usd_cap=10.0, item_harden_usd_cap=5.0, item_variant_usd_cap=7.0))
    L = Ledger(tmp_path / "l.sqlite", cfg)
    L.settle(L.reserve("m", "S5G", "a", True, est_usd=5.0), 5.0)                  # first version: 5 of 10
    with harden_scope(1):
        L.settle(L.reserve("m", "S5G", "a", True, est_usd=5.0), 5.0)              # hardening round: 5 of 5
        with pytest.raises(BudgetStop) as e:
            L.reserve("m", "S5G", "a", True, est_usd=0.5)
        assert e.value.reason == "item_harden_usd"
        assert item_usd_left(L, cfg, "a") == pytest.approx(0.0)
    # the variant rounds have their own bucket: the hardening spend does not count against it
    with variant_scope(1):
        assert item_usd_left(L, cfg, "a") == pytest.approx(7.0)
        L.settle(L.reserve("m", "S5G", "a", True, est_usd=6.0), 6.0)
        assert item_usd_left(L, cfg, "a") == pytest.approx(1.0)
        with pytest.raises(BudgetStop) as e:
            L.reserve("m", "S6C", "a", True, est_usd=2.0)
        assert e.value.reason == "item_variant_usd"
    with L._db() as db:
        assert db.execute("SELECT harden, variant FROM calls ORDER BY id DESC LIMIT 1").fetchone() == (None, 1)
    # the first version's bucket keeps the rounds' spend out (both caps set): 5 of 10 left, not exhausted
    assert item_usd_left(L, cfg, "a") == pytest.approx(5.0)


def test_variant_spend_shares_item_cap_when_no_variant_cap(tmp_path):
    cfg = _cfg(dict(item_usd_cap=10.0, item_variant_usd_cap=None))                # original rule
    L = Ledger(tmp_path / "l.sqlite", cfg)
    L.settle(L.reserve("m", "S5G", "a", True, est_usd=9.0), 9.0)
    with variant_scope(1):
        assert item_usd_left(L, cfg, "a") == pytest.approx(1.0)
        with pytest.raises(BudgetStop) as e:
            L.reserve("m", "S5G", "a", True, est_usd=2.0)
        assert e.value.reason == "item_usd"


# ------------------------------------------------------------------ reviewed items: a fresh review round per version
def _reviewed_easy_item(tmp_path):
    """Item reviewed at round 1 (verdict pass, released), then recalibrated easy (as after a review revision)."""
    from tbsgen.stages_b import watch_tick
    P, iid, jobs = _review_env(tmp_path)
    P.cfg["calib"]["variant_depth"] = 2
    it = P.store.item(iid)
    write_json(it.p("_spec", "author_code.json"), {"package_names": ["fastsmoother"]})
    it.save()
    _to_label(P, iid, jobs)                                     # glm fail, teacher pass -> hard -> S9R
    it = _s9(P, iid)
    assert it.state["stage"] == "S9R" and it.state["review_round"] == 1
    [rr] = json.loads(it.p("review", "runs.json").read_text())
    _finish_job(jobs, rr["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])                                   # released at review round 1
    it = _s9(P, iid)
    assert it.state["stage"] == "S10" and it.state["status"] == "done" and it.state["review_round"] == 1
    old = read_json(it.p("review", "verdict.json"), {})
    assert old["round"] == 1 and old["pkg_sha"]
    return P, iid, jobs, old["pkg_sha"]


def test_reviewed_item_gets_a_fresh_review_round_after_a_variant(tmp_path):
    from tbsgen.stages_b import review_gate, variant_cmd, watch_tick
    P, iid, jobs, old_sha = _reviewed_easy_item(tmp_path)
    it = _s9(P, iid)
    # the version is now easy (both decisive runs pass): an S9b variant instead of discarding it
    write_json(it.p("calib", "runs.json"), _two_passes())
    out = variant_cmd(P, it, "S9b re-open")
    it = _s9(P, iid)
    assert out["kind"] == "variant" and it.state["harden_kind"] == "variant" and it.state["variant_round"] == 1
    assert it.state["review_round"] == 1                        # not yet: the variant version is not at S9
    # it builds and is recalibrated: this time hard
    it.p("pkg", "instruction.md").write_text("do it, harder\n")
    _simulate_packaged_version(it, 1, HARD)
    P.run(only=[iid])
    it = _s9(P, iid)
    assert it.state["calib_label"] == "hard" and it.state["review_round"] == 2, it.state
    assert not it.p("review", "r2", "verdict.json").exists()
    # the earlier version's verdict cannot release the new one (another round / package)
    write_json(it.p("review", "verdict.json"), read_json(it.p("review", "r1", "verdict.json")))
    assert "another round" in (review_gate(P, it) or "")
    [rr] = [r for r in json.loads(it.p("review", "runs.json").read_text()) if r["n"] == 2]
    assert rr["job_name"] == f"tbsgen-{iid}-review2" and rr["pkg_sha"] != old_sha
    bundle = Path(rr["task"]) / "environment" / "review" / "records"
    assert (bundle / "previous_reviews" / "r1-verdict.json").exists()      # provenance, never reused
    assert read_json(bundle / "pipeline_history.json")["review_round"] == 2
    _finish_job(jobs, rr["job_name"], verdict=V("pass"))
    watch_tick(P, only=[iid])
    it = _s9(P, iid)
    assert it.state["stage"] == "S10" and it.state["status"] == "done"
    m = json.loads((P.store.root / "release" / "manifest.jsonl").read_text().splitlines()[-1])
    assert m["review"]["round"] == 2 and m["difficulty"]["variant_round"] == 1


def test_variant_cmd_is_idempotent_and_evented(tmp_path):
    from tbsgen.stages_b import variant_cmd
    P, iid, jobs, _ = _reviewed_easy_item(tmp_path)
    it = _s9(P, iid)
    write_json(it.p("calib", "runs.json"), _two_passes())
    first = variant_cmd(P, it, "S9b re-open")
    second = variant_cmd(P, P.store.item(iid), "S9b re-open")     # never a second round from the same re-open
    assert second.get("skipped") and second["round"] == first["round"] == 1
    ev = [e for e in P.store.events() if e["kind"] == "s9b_reopen"]
    assert len(ev) == 1 and ev[0]["item"] == iid and ev[0]["round"] == 1


def test_variant_cmd_refuses_a_non_code_grounded_item(tmp_path):
    from tbsgen.stages_b import variant_cmd
    P, iid, jobs = _review_env(tmp_path)
    P.cfg["calib"]["variant_depth"] = 2
    it = P.store.item(iid)
    (it.p("_spec", "author_code.json")).unlink(missing_ok=True)
    with pytest.raises(ValueError) as e:
        variant_cmd(P, it, "S9b re-open")
    assert "not code-grounded" in str(e.value) or "excluded" in str(e.value)


# ------------------------------------------------------------------ config / review helpers
def test_variant_config_defaults_and_fresh_review_rounds(tmp_path):
    cfg = load_config(None, {})
    assert cfg["calib"]["variant_depth"] == 0 and cfg["calib"]["variant_max_attempts"] == 4
    assert cfg["budget"]["item_variant_usd_cap"] == 15.0
    d = tmp_path / "item"
    (d / "review" / "r1").mkdir(parents=True)
    (d / "review" / "r2").mkdir()
    assert review.next_round(d, {"review_round": 2}) == 3       # above the current round and every verdict dir
    assert review.next_round(d, {"review_round": 1, "review_revisions": 2}) == 3
    assert review.next_round(tmp_path / "fresh", {}) == 1
    assert review.runs_of_version(tmp_path / "fresh", {}) == []
