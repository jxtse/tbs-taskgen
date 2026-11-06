"""Integration tests: the whole pipeline on the toy fixture with the fake model, task code executed in Docker
(tbsgen-base:py311-sci-v2).  Skipped when Docker or the image is unavailable."""
import json
from pathlib import Path

import pytest

from tbsgen import sandbox
from tbsgen.budget import Ledger
from tbsgen.config import load_config
from tbsgen.dryrun import FIX, make_toy_candidate, simulate_harbor
from tbsgen.fakes import toy_script
from tbsgen.models import FakeClient
from tbsgen.package import validate_pkg
from tbsgen.pipeline import Pipeline
from tbsgen.report import daily_report
from tbsgen.state import read_json

IMAGE = load_config()["sandbox"]["image"]
pytestmark = pytest.mark.skipif(not (sandbox.docker_available() and sandbox.image_exists(IMAGE)),
                                reason="docker / sandbox image unavailable")


def mk(tmp_path, fake_kw=None, budget=None, script=None, roles=("spec", "gen", "tests", "repair")):
    cfg = load_config(None, {"run": {"root": str(tmp_path)},
                             "fulltext": {"cache_dirs": [str(tmp_path / "fulltext_cache")], "allow_network": False},
                             "budget": budget or {}})
    led = Ledger(tmp_path / "ledger.sqlite", cfg)
    fake = FakeClient(led, cfg, script or toy_script(FIX, **(fake_kw or {})), name="fake-opus")
    return Pipeline(cfg, clients={r: fake for r in roles}), fake


def test_happy_path_to_harbor_package_and_calibration(tmp_path):
    P, fake = mk(tmp_path)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert it.state["stage"] == "S9" and it.state["status"] == "waiting", it.state
    gate = read_json(sorted(it.p("gates").glob("*.json"))[-1])
    assert gate["ok"], gate["gates"]
    s0 = gate["details"]["seeds"]["0"]
    assert s0["ref"]["passed"] and not s0["nop"]["passed"] and not s0["mid"]["passed"]
    assert 1 <= s0["mid"]["n_failed"] <= 2 and s0["naive"]["n_failed"] >= 2
    assert validate_pkg(it.p("pkg"))["ok"]
    assert not any(p.name in ("generate.py", "mid.py", "naive.py") for p in it.p("pkg").rglob("*"))
    runs = read_json(it.p("calib", "runs.json"))
    conf = json.loads(Path(runs[0]["config"]).read_text())
    assert conf["agents"][0]["model_name"] == "glm-5.3" and conf["agents"][0]["kwargs"]["version"] == "2.1.280"
    assert runs[0]["submitted"] is False
    # GLM pass, GLM fail -> unstable -> kept -> S10
    st = simulate_harbor(P, iid, ["pass", "fail"])
    assert st["stage"] == "S10" and st["status"] == "done" and st["calib_label"] == "unstable"
    assert (tmp_path / "release" / "manifest.jsonl").exists()
    rep = daily_report(P).read_text()
    assert "fake-opus" in rep and "日报" in rep
    # re-running a finished run does nothing and makes no model calls
    n = len(fake.log)
    P.run()
    assert len(fake.log) == n


def test_resume_after_crash_mid_stage(tmp_path):
    """Kill-like interruption: run stops after S6A_run; a new Pipeline object continues from state.json."""
    P, fake = mk(tmp_path)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run(max_steps=7)        # S1..S6A_run
    st = P.store.item(iid).state
    assert st["stage"] == "S6B", st["stage"]
    calls_before = len(fake.log)
    P2, fake2 = mk(tmp_path)
    P2.run()
    st = P2.store.item(iid).state
    assert st["stage"] == "S9" and st["status"] == "waiting"
    assert calls_before == 2 and len(fake2.log) == 1      # S5+S6A before; only S6C after resume


def test_budget_breaker_pauses_and_resume(tmp_path):
    P, fake = mk(tmp_path, budget={"daily_calls_per_model": {"fake-opus": 2}})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    st = P.store.item(iid).state
    assert st["status"] == "paused" and st["reason"] == "budget:daily_calls" and st["stage"] == "S6C"
    assert P.ledger.summary()["calls_total"] == 2
    # raise the cap (new config) and resume
    P2, _ = mk(tmp_path, budget={"daily_calls_per_model": {"fake-opus": 10}})
    assert P2.resume() == [iid]
    P2.run()
    assert P2.store.item(iid).state["status"] == "waiting"


def test_quota_error_opens_breaker_and_pauses(tmp_path):
    def script(stage, system, user):
        return {"error": "quota", "text": "HTTP 429 API key 额度已用完"}
    P, _ = mk(tmp_path, script=script)
    ids = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    st = P.store.item(ids[0]).state
    assert st["status"] == "paused" and st["reason"] == "model:quota"
    assert P.ledger.breaker("fake-opus")["open"]


def test_reference_crash_goes_to_repair_and_recovers(tmp_path):
    P, _ = mk(tmp_path, fake_kw={"break_reference": True})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert it.state["status"] == "waiting", it.state
    assert it.state["repair_rounds"] == 1
    assert read_json(it.p("repair", "round-1", "review.json"))["ok"]


def test_injected_leak_blocked_then_repaired(tmp_path):
    P, _ = mk(tmp_path, fake_kw={"leak_in_public": True})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    g1 = read_json(it.p("gates", "attempt-1.json"))
    assert not g1["gates"]["leak"] and "leak" in g1["blocking"], g1["gates"]
    assert "L2" in g1["details"]["leak"]["classes"]
    assert it.state["status"] == "waiting" and read_json(it.p("gates", "attempt-2.json"))["ok"]


def test_unrepairable_is_dropped_after_two_rounds(tmp_path):
    P, _ = mk(tmp_path, fake_kw={"leak_in_public": True, "repair_fixes": False})
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    st = P.store.item(iid).state
    assert st["status"] == "dropped" and st["reason"].startswith("repair_limit"), st
    assert st["repair_rounds"] == 2


def test_s6a_truncated_answer_gets_one_continuation(tmp_path):
    """e2e-iter1 regression: a real S6A answer stopped at max_tokens after generator/generate.py and the item was
    dropped (s6a_missing_files).  Now: partial last block discarded, one continuation asks for the missing files."""
    base = toy_script(FIX)
    seen = {"S6A": 0}

    def script(stage, system, user):
        out = base(stage, system, user)
        if stage != "S6A":
            return out
        seen["S6A"] += 1
        if seen["S6A"] == 1:      # cut inside solution/solve.py
            cut = out.index("=== FILE: solution/solve.py ===") + 60
            return {"text": out[:cut], "stop": "max_tokens", "usage": {"output_tokens": 32000}}
        assert "Files already received" in user and "solution/solve.py" in user.split("missing files")[-1]
        return out[out.index("=== FILE: solution/solve.py ==="):]
    P, fake = mk(tmp_path, script=script)
    [iid] = P.ingest([make_toy_candidate(tmp_path)])
    P.run()
    it = P.store.item(iid)
    assert seen["S6A"] == 2
    calls = read_json(it.p("S6A_calls.json"))
    assert calls[0]["stop"] == "max_tokens" and "solution/solve.py" not in calls[0]["got"] and calls[1]["cont"]
    assert len(list(it.p("raw").glob("S6A-*.txt"))) == 2
    assert it.state["status"] == "waiting", it.state


def test_intake_failure_routes(tmp_path):
    P, _ = mk(tmp_path)
    cands = [dict(doi="10.1/review", title="A review of things", abstract="This review surveys perspectives."),
             dict(doi="10.1/nofulltext", title="Robust estimator algorithm",
                  abstract="We propose an estimation algorithm with simulation benchmark; code on github in python.")]
    ids = P.ingest(cands)
    P.run()
    a, b = (P.store.item(i).state for i in ids)
    assert a["status"] == "dropped" and a["reason"] == "low_screen_score"
    assert b["status"] == "parked" and b["reason"] == "fulltext_unavailable"
    assert (P.store.item(ids[1]).p("S3_queue.json")).exists()
