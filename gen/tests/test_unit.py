"""Unit tests: budget ledger, model layer, spec rules, thresholds, calibration logic, leak-gate pieces,
repair review, packaging validation.  No Docker, no network."""
import json
from pathlib import Path

import numpy as np
import pytest

from tbsgen import calib, repair
from tbsgen.budget import BudgetStop, Ledger
from tbsgen.config import load_config
from tbsgen.leakgate import FORBID, join_probe, numeric_scan
from tbsgen.leaves import Leaf, file_leaves, norm_name
from tbsgen.models import FakeClient, ModelError, classify_http, extract_json, parse_files
from tbsgen.spec import set_thresholds, validate_spec
from tbsgen.fakes import SPEC


def cfg_with(tmp_path, **budget):
    return load_config(None, {"run": {"root": str(tmp_path)}, "budget": budget})


# ------------------------------------------------------------------ ledger
def test_real_call_refused_without_budget(tmp_path):
    led = Ledger(tmp_path / "l.sqlite", cfg_with(tmp_path))
    with pytest.raises(BudgetStop) as e:
        led.reserve("claude-opus-5-5", "S5", "x", real=True)
    assert e.value.reason == "budget_unset"
    led.reserve("fake", "S5", "x", real=False)       # fakes exempt


def test_daily_budget_and_call_caps(tmp_path):
    cfg = cfg_with(tmp_path, daily_usd_total=1.0, est_usd_per_call=0.4, daily_calls_per_model={"claude-opus-5-5": 5})
    led = Ledger(tmp_path / "l.sqlite", cfg)
    a = led.reserve("claude-opus-5-5", "S5", "i1", True)
    led.reserve("claude-opus-5-5", "S5", "i1", True)                # 0.8 reserved
    with pytest.raises(BudgetStop) as e:
        led.reserve("claude-opus-5-5", "S5", "i1", True)            # 1.2 > 1.0
    assert e.value.reason == "daily_total"
    usd = led.finish(a, {"input_tokens": 1000, "output_tokens": 1000})  # actual $0.03 replaces the 0.4 estimate
    assert usd == pytest.approx((1000 * 5 + 1000 * 25) / 1e6)
    led.reserve("claude-opus-5-5", "S5", "i1", True)                # now fits
    s = led.summary()
    assert s["calls_total"] == 3


def test_breaker_opens_on_quota_and_blocks(tmp_path):
    cfg = cfg_with(tmp_path, daily_usd_total=100.0)
    led = Ledger(tmp_path / "l.sqlite", cfg)
    cid = led.reserve("claude-opus-5-5", "S5", "i", True)
    led.finish(cid, error_class="quota", error="HTTP 429 API key 额度已用完")
    assert led.breaker("claude-opus-5-5")["open"]
    with pytest.raises(BudgetStop) as e:
        led.reserve("claude-opus-5-5", "S5", "i", True)
    assert e.value.reason == "breaker_open"
    led.reset_breaker("claude-opus-5-5")
    led.reserve("claude-opus-5-5", "S5", "i", True)


def test_breaker_consecutive_errors(tmp_path):
    cfg = cfg_with(tmp_path, daily_usd_total=100.0, breaker_consecutive_errors=3)
    led = Ledger(tmp_path / "l.sqlite", cfg)
    for _ in range(3):
        led.finish(led.reserve("m", "S5", "i", False), error_class="server", error="500")
    assert led.breaker("m")["open"]


def test_classify_http():
    assert classify_http(429, '{"error":"API key 额度已用完"}') == "quota"
    assert classify_http(429, "slow down") == "rate"
    assert classify_http(401, "") == "auth"
    assert classify_http(529, "") == "overload"


# ------------------------------------------------------------------ model layer
def test_fake_client_records_and_errors(tmp_path):
    cfg = cfg_with(tmp_path)
    led = Ledger(tmp_path / "l.sqlite", cfg)
    fc = FakeClient(led, cfg, [{"stage": "S5", "text": '{"a": 1}'}, {"stage": "S5", "error": "quota"}])
    r = fc.chat("s", "u", stage="S5", item="i")
    assert extract_json(r.text) == {"a": 1} and r.usd == 0
    with pytest.raises(ModelError) as e:
        fc.chat("s", "u", stage="S5", item="i")
    assert e.value.cls == "quota"
    assert led.breaker("fake")["open"]


def test_parse_files_and_extract_json():
    t = "x\n=== FILE: a/b.py ===\n```python\nprint(1)\n```\n=== FILE: c.md ===\nhi\n=== END ===\n"
    f = parse_files(t)
    assert f["a/b.py"].strip() == "print(1)" and f["c.md"].strip() == "hi"
    assert extract_json("blah ```json\n{\"k\": [1,2]}\n``` tail") == {"k": [1, 2]}


# ------------------------------------------------------------------ spec + thresholds
def test_validate_spec_ok_and_bad():
    cfg = load_config()
    assert validate_spec(SPEC, cfg) == []
    bad = json.loads(json.dumps(SPEC))
    bad["form"] = "INTERACT"
    bad["depgraph"]["mid_drop"] = []
    errs = validate_spec(bad, cfg)
    assert any("form" in e for e in errs) and any("mid_drop" in e for e in errs)


def test_thresholds_from_measurement():
    cfg = load_config()
    meas = {"ref": {"0": {"amp_median_rel_error": 0.0077, "rate_p90_rel_error": 0.09, "coverage": 1.0},
                    "1": {"amp_median_rel_error": 0.0063, "rate_p90_rel_error": 0.14, "coverage": 1.0}},
            "mid": {"0": {"amp_median_rel_error": 0.0218, "rate_p90_rel_error": 0.159, "coverage": 1.0},
                    "1": {"amp_median_rel_error": 0.0152, "rate_p90_rel_error": 0.76, "coverage": 1.0}},
            "naive": {"0": {"amp_median_rel_error": 0.09, "rate_p90_rel_error": 0.23, "coverage": 1.0}}}
    th = set_thresholds(SPEC, meas, cfg)
    t = th["thresholds"]
    assert th["status"] == "ok", th
    assert 0.0077 < t["amp_median_rel_error"]["threshold"] < 0.0152
    assert t["rate_p90_rel_error"]["separating"] and 0.14 < t["rate_p90_rel_error"]["threshold"] < 0.159
    assert not t["coverage"]["separating"] and t["coverage"]["threshold"] <= 1.0
    assert th["mid_fail"] == {"0": 2, "1": 2}


def test_thresholds_not_separable():
    cfg = load_config()
    same = {"amp_median_rel_error": 0.01, "rate_p90_rel_error": 0.1, "coverage": 1.0}
    th = set_thresholds(SPEC, {"ref": {"0": same}, "mid": {"0": dict(same)}, "naive": {}}, cfg)
    assert th["status"] == "not_separable"


# ------------------------------------------------------------------ calibration logic
@pytest.mark.parametrize("runs,expect", [
    ([], ("run", "glm")),
    ([("glm", "fail")], ("run", "teacher")),
    ([("glm", "fail"), ("teacher", "pass")], ("label", "hard")),
    ([("glm", "fail"), ("teacher", "fail")], ("run", "teacher")),
    ([("glm", "fail"), ("teacher", "fail"), ("teacher", "fail")], ("label", "parked:teacher_failed")),
    ([("glm", "pass")], ("run", "glm")),
    ([("glm", "pass"), ("glm", "fail")], ("label", "unstable")),
    ([("glm", "pass"), ("glm", "pass")], ("label", "easy")),
    ([("glm", "infra"), ("glm", "fail")], ("run", "teacher")),
    ([("glm", "infra"), ("glm", "infra"), ("glm", "infra")], ("label", "parked:infra")),
])
def test_calibration_sequence(runs, expect):
    cfg = load_config()
    assert calib.next_action([dict(who=w, outcome=o, minutes=60) for w, o in runs], cfg) == expect


def test_quick_pass_option():
    cfg = load_config(None, {"calib": {"quick_pass_minutes": 30}})
    assert calib.next_action([dict(who="glm", outcome="pass", minutes=12)], cfg) == ("label", "easy")


def test_classify_trial_quota_is_infra_but_reward1_is_pass(tmp_path):
    for name, reward, status in (("a", 0.0, 429), ("b", 1.0, 429), ("c", 0.0, None)):
        t = tmp_path / name
        (t / "agent").mkdir(parents=True)
        (t / "result.json").write_text(json.dumps({
            "exception_info": {"exception_type": "UnknownApiError"} if status else None,
            "verifier_result": {"rewards": {"reward": reward}}}))
        (t / "agent" / "claude-code.txt").write_text(json.dumps(
            {"type": "result", "api_error_status": status, "result": "API Error: Request rejected (429) · API key 额度已用完"
             if status else "done"}).replace(" ", "") + "\n")
    got = {r["trial"]: r["outcome"] for r in calib.collect(tmp_path)}
    assert got == {"a": "infra", "b": "pass", "c": "fail"}


# ------------------------------------------------------------------ leak-gate pieces
def test_forbid_regex():
    assert FORBID.search("- `machines/<id>.npz` holds the internal machine state used by the oracle.")
    assert FORBID.search("Do not decode the blob or reimplement the simulator.")
    assert FORBID.search("Your code must never read it or the private attributes of `BlackBox`.")
    assert not FORBID.search("It must not use the network or read anything outside `<input_dir>`.")
    assert not FORBID.search("prints a binary mask of opaque lines on a clear background")


def test_numeric_scan_finds_hidden_copy_and_ignores_echo():
    rng = np.random.default_rng(0)
    secret = rng.normal(size=8)
    truth = [Leaf("truth/t.json", "x_star", secret, "x_star"), Leaf("truth/t.json", "x0", np.array([1.2345678, 2.5]), "x0")]
    public = [Leaf("data/m.npz", "ustar", secret.copy(), "ustar"), Leaf("data/p.json", "x0", np.array([1.2345678, 2.5]), "x0")]
    f, echoes = numeric_scan(truth, public, declared={"x0"})
    assert echoes == 1 and len(f) == 1 and f[0]["truth_name"] == "x_star" and f[0]["severity"] == "block"


def test_join_probe_detects_invertible_variance():
    rng = np.random.default_rng(1)
    doms = [f"D{i:03d}" for i in range(40)]
    p = rng.uniform(0.05, 0.45, 40)
    n = rng.integers(50, 500, 40)
    pub = [dict(domain=d, n=str(int(k)), var_prop=str(1.6 * pp * (1 - pp) / k), y=str(pp + rng.normal(0, .05)))
           for d, pp, k in zip(doms, p, n)]
    tru = {"truth/t.csv": [dict(domain=d, mu_prop=str(pp)) for d, pp in zip(doms, p)]}
    f = join_probe(tru, {"panel.csv": pub})
    assert f and f[0]["truth"].endswith("mu_prop") and f[0]["severity"] == "block"


def test_join_probe_limit_non_monotone():
    """Known limit (documented in README): rank correlation misses non-monotone inversions such as
    p(1-p) when p spans both sides of 0.5.  L4b (cheap-agent attack, 待定) is meant to cover this."""
    rng = np.random.default_rng(1)
    doms = [f"D{i:03d}" for i in range(40)]
    p = rng.uniform(0.05, 0.95, 40)
    pub = [dict(domain=d, n="100", var_prop=str(pp * (1 - pp) / 100)) for d, pp in zip(doms, p)]
    tru = {"truth/t.csv": [dict(domain=d, mu_prop=str(pp)) for d, pp in zip(doms, p)]}
    assert join_probe(tru, {"panel.csv": pub}) == []


def test_file_leaves_formats(tmp_path):
    np.savez(tmp_path / "a.npz", Q=np.eye(3) * 1.234567, key=np.array("abc"))
    (tmp_path / "b.csv").write_text("id,v\nA,1.5\nB,2.25\n")
    (tmp_path / "c.json").write_text(json.dumps({"P01": {"f_star": 1.0251609}}))
    names = {lf.name for p in tmp_path.iterdir() for lf in file_leaves(p, p.name)}
    assert {"q", "v", "f_star"} <= names
    assert norm_name("problems[3].P01_xstar") == "xstar"


# ------------------------------------------------------------------ repair review
def test_repair_review_rejects_threshold_change(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_outputs.py").write_text("check('a', m['a'], 0.011, '<=')\ncheck('b', m['b'], 0.5, '>=')\n")
    (tmp_path / "instruction.md").write_text("task")
    dg = {"nodes": [{"id": "H", "provision": "hidden_truth", "secret_names": ["params.csv"]}]}
    ok = repair.review(tmp_path, {"tests/test_outputs.py": "check('a', m['a'], 0.011, '<=')\ncheck('b', m['b'], 0.5, '>=')\n# fix\n"}, dg)
    assert ok["ok"]
    loose = repair.review(tmp_path, {"tests/test_outputs.py": "check('a', m['a'], 0.05, '<=')\ncheck('b', m['b'], 0.5, '>=')\n"}, dg)
    assert not loose["ok"] and "threshold literals changed" in loose["reasons"]
    fewer = repair.review(tmp_path, {"tests/test_outputs.py": "check('a', m['a'], 0.011, '<=')\n"}, dg)
    assert not fewer["ok"]
    new = repair.review(tmp_path, {"extra.py": "x"}, dg)
    assert not new["ok"]
    sec = repair.review(tmp_path, {"instruction.md": "see params.csv"}, dg)
    assert not sec["ok"]
