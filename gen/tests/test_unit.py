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


def test_run_caps_and_s9_reservation_settle(tmp_path):
    cfg = cfg_with(tmp_path, daily_usd_total=500.0, run_usd_total=30.0, run_calls_per_model={"claude-opus-5-5": 2})
    led = Ledger(tmp_path / "l.sqlite", cfg)
    led.reserve("claude-opus-5-5", "S5", "i", True)
    led.reserve("claude-opus-5-5", "S5", "i", True)
    with pytest.raises(BudgetStop) as e:
        led.reserve("claude-opus-5-5", "S5", "i", True)
    assert e.value.reason == "run_calls"
    cid = led.reserve("glm-5.3", "S9", "i", True, est_usd=20.0)          # 3.0 + 20 <= 30
    with pytest.raises(BudgetStop) as e:
        led.reserve("glm-5.3", "S9", "i", True, est_usd=20.0)
    assert e.value.reason == "run_total"
    led.settle(cid, 4.5)                                                  # actual Harbor cost replaces the estimate
    led.reserve("glm-5.3", "S9", "i", True, est_usd=20.0)                 # 3.0 + 4.5 + 20 <= 30
    cid2 = led.reserve("fake", "S9", "i", False)
    led.settle(cid2, None)
    assert led.summary(None)["usd_total"] == pytest.approx(3.0 + 4.5 + 20.0)


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


def test_parse_files_truncated_drops_partial_last_block():
    # e2e-iter1: S6A hit max_tokens=32000 mid-file; the partial file must not be written as if complete
    t = "=== FILE: a.py ===\nprint(1)\n=== FILE: b.py ===\ndef f(:\n    x = ["
    assert set(parse_files(t)) == {"a.py", "b.py"}
    assert set(parse_files(t, truncated=True)) == {"a.py"}
    done = t + "1]\n=== END ===\n"
    assert set(parse_files(done, truncated=True)) == {"a.py", "b.py"}


@pytest.mark.parametrize("tail,keep", [
    ("=== FILE: b.py", {"a.py"}),            # cut inside the next file header (review #10)
    ("=== FI", {"a.py"}),
    ("=== FILE: b.py ===\nx = [", {"a.py"}),   # cut inside the next body
    ("=== END", {"a.py"}),                   # cut inside END
    ("=== END ===\n", {"a.py"}),
    ("x = [1,", set()),                      # cut inside a.py itself
])
def test_parse_files_truncation_boundaries(tail, keep):
    f = parse_files("=== FILE: a.py ===\nprint(1)\n" + tail, truncated=True)
    assert set(f) == keep
    if keep:
        assert f["a.py"] == "print(1)\n"


# ------------------------------------------------------------------ intake screen
def test_screen_keeps_method_papers_found_in_e2e():
    from tbsgen.intake import screen
    cfg = load_config()
    ismv = dict(title="An iterative spherical mean value method for background field removal in MRI",
                abstract="The proposed background field removal method, termed iterative SMV (iSMV), consists of "
                         "applying the SMV operation repeatedly on the field map. It was validated in a phantom; "
                         "truncation threshold needed for deconvolution.")
    assert screen(ismv, cfg)["keep"]
    noabs = dict(title="Thermal drift in room temperature nanoindentation experiments: Measurement and correction")
    r = screen(noabs, cfg)
    assert r["keep"] and r["deferred"]
    r2 = screen(noabs, cfg, fulltext_head="We propose a correction algorithm; drift is estimated by least-squares "
                                          "fitting of hold segments and validated by simulation.")
    assert r2["keep"] and not r2.get("deferred")
    assert not screen(dict(title="A review", abstract="This review surveys perspectives."), cfg)["keep"]


# ------------------------------------------------------------------ spec + thresholds
def test_non_separating_threshold_rejects_clearly_worse_naive():
    # e2e-iter1 numbers: ratio metric, ref ~1.00, mid ~1.08 (not separating at 10%), naive ~1.5 -> was 1.6 (naive passed)
    cfg = load_config()
    spec = {"metrics": [{"name": "r", "direction": "<="}]}
    meas = {"ref": {"0": {"r": 1.0024}, "1": {"r": 1.0037}}, "mid": {"0": {"r": 1.089}, "1": {"r": 1.084}},
            "naive": {"0": {"r": 1.478}, "1": {"r": 1.487}}}
    t = set_thresholds(spec, meas, cfg)["thresholds"]["r"]
    assert not t["separating"] and t["naive_capped"]
    assert 1.0037 * 1.1 <= t["threshold"] < 1.478
    # e2e-iter1 DecontX: correlation (>=), ref 0.9962, mid equal, naive 0.89 -> loose rule gave 0.49
    spec = {"metrics": [{"name": "p", "direction": ">="}]}
    meas = {"ref": {"0": {"p": 0.99623}, "1": {"p": 0.99692}}, "mid": {"0": {"p": 0.99623}, "1": {"p": 0.99692}},
            "naive": {"0": {"p": 0.8908}, "1": {"p": 0.9238}}}
    t = set_thresholds(spec, meas, cfg)["thresholds"]["p"]
    # capped at the reference margin floor (0.99623*0.9 = 0.8966 -> 0.89 after loose rounding), not at 0.49;
    # the naive (0.89-0.92) is inside the 10% reference margin here, so it is rejected by its other metrics
    assert t["naive_capped"] and 0.85 < t["threshold"] <= 0.99623 * 0.9, t

def test_reference_equal_to_empty_output_value_is_missing():
    """Review #7: missing output is detected against the nop measurement, not by a global magnitude."""
    cfg = load_config()
    spec = {"metrics": [{"name": "rmse_public", "direction": "<="}, {"name": "corr", "direction": ">="},
                        {"name": "neg_ll", "direction": "<="}, {"name": "big", "direction": "<="}]}
    nop = {"rmse_public": 1e6, "corr": -1e6, "neg_ll": 0.0, "big": 1e12}
    ok_ref = {"rmse_public": 0.3, "corr": 0.9, "neg_ll": -1e6, "big": 2e7}     # legit: negative and huge values
    mid = {"rmse_public": 0.6, "corr": 0.5, "neg_ll": -5e5, "big": 4e7}
    th = set_thresholds(spec, {"ref": {"0": ok_ref}, "mid": {"0": mid}, "nop": {"0": nop}}, cfg)
    assert all(t["threshold"] is not None for t in th["thresholds"].values()), th
    bad_ref = dict(ok_ref, corr=-1e6)                                           # ">=" metric missing
    th = set_thresholds(spec, {"ref": {"0": bad_ref}, "mid": {"0": mid}, "nop": {"0": nop}}, cfg)
    assert th["status"] == "ref_metric_missing" and th["thresholds"]["corr"]["threshold"] is None
    bad_ref = dict(ok_ref, rmse_public=1e6)                                     # "<=" metric missing
    th = set_thresholds(spec, {"ref": {"0": bad_ref}, "mid": {"0": mid}, "nop": {"0": nop}}, cfg)
    assert th["status"] == "ref_metric_missing" and th["thresholds"]["rmse_public"]["threshold"] is None


def test_reference_worse_than_naive_is_flagged():
    cfg = load_config()
    spec = {"metrics": [{"name": "public_rmse", "direction": "<="}, {"name": "hidden_rmse", "direction": "<="}]}
    meas = {"ref": {"0": {"public_rmse": 0.29, "hidden_rmse": 4.19}},
            "mid": {"0": {"public_rmse": 3.77, "hidden_rmse": 4.36}},
            "naive": {"0": {"public_rmse": 3.70, "hidden_rmse": 3.52}}}
    th = set_thresholds(spec, meas, cfg)
    assert th["status"] == "ref_worse_than_naive" and th["ref_worse_than_naive"] == ["hidden_rmse"], th


def test_resource_metric_slower_than_naive_is_not_ref_worse():
    """e2e-prod-20261109 (phase unwrapping): accurate reference 0.58 s vs a trivial naive 0.40 s on the hidden runtime
    metric -> was ref_worse_than_naive -> S8 x2 -> dropped.  A resource metric does not trigger it; quality metrics do."""
    from tbsgen.spec import is_resource_metric
    cfg = load_config()
    spec = {"metrics": [{"name": "hidden_frac_correct", "direction": ">="},
                        {"name": "hidden_mae", "direction": "<="},
                        {"name": "hidden_runtime_sec", "direction": "<="},
                        {"name": "working_set_kib", "direction": "<=", "kind": "resource"}]}
    meas = {"ref": {"0": {"hidden_frac_correct": 0.96, "hidden_mae": 0.20, "hidden_runtime_sec": 0.58, "working_set_kib": 900}},
            "mid": {"0": {"hidden_frac_correct": 0.79, "hidden_mae": 1.35, "hidden_runtime_sec": 0.66, "working_set_kib": 800}},
            "naive": {"0": {"hidden_frac_correct": 0.82, "hidden_mae": 1.17, "hidden_runtime_sec": 0.40,
                            "working_set_kib": 100}}}
    th = set_thresholds(spec, meas, cfg)
    assert th["ref_worse_than_naive"] == [] and th["status"] != "ref_worse_than_naive", th
    meas["ref"]["0"]["hidden_mae"] = 1.50                                       # a quality metric still counts
    assert set_thresholds(spec, meas, cfg)["ref_worse_than_naive"] == ["hidden_mae"]
    spec["metrics"][3].pop("kind")                                             # kind is what exempts it
    assert set_thresholds(spec, meas, cfg)["ref_worse_than_naive"] == ["hidden_mae", "working_set_kib"]
    assert [is_resource_metric({"name": n}) for n in
            ("hidden_runtime_per_spectrum", "wall_time", "hidden_runtime_sec", "peak_mem_mb", "peak_rss", "runtime",
             "membrane_err", "second_moment", "hidden_rmse", "hidden_rss", "rss", "size_mb", "sec_moment",
             "hidden_delay_mae_seconds", "hidden_arrival_time_s_rmse", "latency_err", "elapsed_s")] == \
        [True] * 6 + [False] * 11


def test_resource_metric_exempt_item_passes_with_its_threshold_unchanged():
    """mid fails 1-2 quality criteria, naive >= 2: status ok, and the runtime threshold is the plain loose rule."""
    cfg = load_config()
    spec = {"metrics": [{"name": "hidden_frac_correct", "direction": ">="}, {"name": "hidden_mae", "direction": "<="},
                        {"name": "hidden_runtime_sec", "direction": "<="}]}
    meas = {"ref": {"0": {"hidden_frac_correct": 0.96, "hidden_mae": 0.20, "hidden_runtime_sec": 0.58}},
            "mid": {"0": {"hidden_frac_correct": 0.79, "hidden_mae": 1.35, "hidden_runtime_sec": 0.60}},
            "naive": {"0": {"hidden_frac_correct": 0.50, "hidden_mae": 1.90, "hidden_runtime_sec": 0.40}}}
    th = set_thresholds(spec, meas, cfg)
    assert th["status"] == "ok" and th["ref_worse_than_naive"] == [], th
    rt = th["thresholds"]["hidden_runtime_sec"]
    assert rt["threshold"] == pytest.approx(1.8) and rt["ref_passes"] and not rt["separating"]   # ref x3, rounded
    spec["metrics"][2]["name"] = "hidden_delay_mae_seconds"                     # quality metric in seconds: not exempt
    meas = {w: {"0": {("hidden_delay_mae_seconds" if k == "hidden_runtime_sec" else k): v for k, v in d["0"].items()}}
            for w, d in meas.items()}
    assert set_thresholds(spec, meas, cfg)["status"] == "ref_worse_than_naive"


@pytest.mark.parametrize("d,ref,naive", [("<=", 1.04, 1.15), (">=", 0.96, 0.85)])
def test_naive_cap_survives_rounding(d, ref, naive):
    """Review #9: ref=mid=1.04, naive=1.15 -> floor 1.144 used to round to 1.2 and let naive pass."""
    cfg = load_config()
    spec = {"metrics": [{"name": "r", "direction": d}]}
    th = set_thresholds(spec, {"ref": {"0": {"r": ref}}, "mid": {"0": {"r": ref}}, "naive": {"0": {"r": naive}}},
                        cfg)["thresholds"]["r"]
    thr = th["threshold"]
    assert th["naive_capped"]
    assert (naive > thr) if d == "<=" else (naive < thr), th
    assert (ref <= thr) if d == "<=" else (ref >= thr), th


def test_mid_fail_counts_public_hidden_twins_once():
    # e2e-iter1 RCESPP: mid fails optimal_frac on public AND hidden + mean_rel_gap = 2 criteria (was 3 -> dropped)
    cfg = load_config()
    spec = {"metrics": [{"name": n, "direction": d} for n, d in
                        (("hidden_feasible_frac", ">="), ("hidden_optimal_frac", ">="), ("hidden_mean_rel_gap", "<="),
                         ("public_optimal_frac", ">="))]}
    ref = {"hidden_feasible_frac": 1.0, "hidden_optimal_frac": 1.0, "hidden_mean_rel_gap": 0.01,
           "public_optimal_frac": 1.0}
    mid = {"hidden_feasible_frac": 1.0, "hidden_optimal_frac": 0.7, "hidden_mean_rel_gap": 0.067,
           "public_optimal_frac": 0.667}
    th = set_thresholds(spec, {"ref": {"0": ref}, "mid": {"0": mid}, "naive": {"0": mid}}, cfg)
    assert th["mid_fail"] == {"0": 2} and th["status"] == "ok", th


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


def test_calibration_teacher_unavailable_parks():
    cfg = load_config(None, {"calib": {"teacher_available": False}})
    assert calib.next_action([dict(who="glm", outcome="fail", minutes=60)], cfg) == \
        ("label", "parked:teacher_unavailable")


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


# ------------------------------------------------------------------ S9 submission robustness (review #3-#6)
def _s9_env(tmp_path, monkeypatch=None, launcher_rc=0, make_job=True, **calib_kw):
    import os
    from tbsgen.pipeline import Pipeline
    hr = tmp_path / "harbor"
    hr.mkdir()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    counter = tmp_path / "tmux_calls"
    tmux = tmp_path / "fake-tmux"
    tmux.write_text("#!/bin/bash\necho x >> %s\nshift 4\n\"$@\"\n" % counter)   # runs `bash -lc INNER` inline
    tmux.chmod(0o755)
    mk = ('mkdir -p "%s/$(python3 -c \'import json,sys;print(json.load(open(sys.argv[1]))[\"job_name\"])\' $2)" && '
          'cp $2 "%s/$(python3 -c \'import json,sys;print(json.load(open(sys.argv[1]))[\"job_name\"])\' $2)/config.json"'
          % (jobs, jobs)) if make_job else "true"
    (hr / "launch.sh").write_text("#!/bin/bash\n%s\nexit %d\n" % (mk, launcher_rc))
    (hr / "launch.sh").chmod(0o755)
    cal = dict(allow_submit=True, harbor_root=str(hr), jobs_dir=str(jobs), student_launcher="launch.sh",
               agent_allowed_hosts=["models.example.com"],
               tmux_bin=str(tmux), est_usd_per_run=10.0)
    cal.update(calib_kw)
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "calib": cal,
                             "budget": {"daily_usd_total": 1000.0, "run_usd_total": 100.0}})
    P = Pipeline(cfg, clients={})
    [iid] = P.ingest([dict(doi="10.1/s9", title="t", abstract="algorithm estimation simulation benchmark")])
    it = P.store.item(iid)
    it.state.update(stage="S9", status="active")
    it.save()
    return P, iid, counter, jobs


def _finish_job(jobs, job_name, reward=1.0, cost=4.0):
    jd = jobs / job_name
    (jd / "trial__a").mkdir(parents=True, exist_ok=True)
    (jd / "trial__a" / "result.json").write_text(json.dumps(
        {"verifier_result": {"rewards": {"reward": reward}}, "agent_result": {"cost_usd": cost}}))
    (jd / "result.json").write_text(json.dumps({"finished_at": "2026-11-06T12:00:00"}))
    return jd


def test_s9_crash_after_launch_does_not_resubmit(tmp_path, monkeypatch):
    from tbsgen import stages_b
    P, iid, counter, jobs = _s9_env(tmp_path)
    real = stages_b._save_run

    def crashing(it, rec):
        if rec.get("state") == "launched":
            raise KeyboardInterrupt("crash right after launch")
        real(it, rec)
    monkeypatch.setattr(stages_b, "_save_run", crashing)
    with pytest.raises(KeyboardInterrupt):
        P.run(only=[iid])
    monkeypatch.setattr(stages_b, "_save_run", real)
    it = P.store.item(iid)
    it.state.update(stage="S9", status="active")
    it.save()
    P.run(only=[iid])
    runs = json.loads((P.store.item(iid).p("calib", "runs.json")).read_text())
    assert len(runs) == 1 and runs[0]["state"] == "launched" and runs[0].get("recovered"), runs
    assert counter.read_text().count("x") == 1                 # launched exactly once


def test_ingest_checks_identity_and_consumes_once(tmp_path):
    from tbsgen.stages_b import ingest
    P, iid, counter, jobs = _s9_env(tmp_path)
    P.run(only=[iid])
    it = P.store.item(iid)
    r1 = json.loads(it.p("calib", "runs.json").read_text())[0]
    jd1 = _finish_job(jobs, r1["job_name"], reward=1.0, cost=4.0)
    t = ingest(P, it, None)
    assert t["outcome"] == "pass"
    P.run(only=[iid])                                           # GLM#1 pass -> GLM#2 submitted
    it = P.store.item(iid)
    runs = json.loads(it.p("calib", "runs.json").read_text())
    assert len(runs) == 2 and runs[1]["n"] == 2
    with pytest.raises(ValueError, match="not finished|identity|consumed"):
        ingest(P, it, str(jd1))                                 # replaying the GLM#1 job for GLM#2
    other = jobs / "tbsgen-doi-other-glm2"
    (other).mkdir()
    (other / "config.json").write_text(json.dumps({"job_name": "tbsgen-doi-other-glm2", "agents": [{}], "tasks": [{}]}))
    _finish_job(jobs, "tbsgen-doi-other-glm2")
    with pytest.raises(ValueError, match="identity"):
        ingest(P, it, str(other))                               # another item's job
    jd2 = _finish_job(jobs, runs[1]["job_name"], reward=0.0, cost=6.0)
    assert ingest(P, it, None)["outcome"] == "fail"
    with pytest.raises(ValueError):
        ingest(P, it, str(jd2))                                 # consumed once
    s = P.ledger.summary(None)
    assert s["usd_total"] == pytest.approx(10.0)                # 4 + 6 actual, no estimate left


def test_s9_launch_exception_settles_reservation(tmp_path):
    P, iid, counter, jobs = _s9_env(tmp_path, tmux_bin=str(tmp_path / "no-such-tmux"))
    P.run(only=[iid], max_steps=1)
    runs = json.loads(P.store.item(iid).p("calib", "runs.json").read_text())
    assert runs[0]["state"] == "launch_failed" and runs[0]["outcome"] == "infra", runs
    with P.ledger._db() as db:
        st, usd = db.execute("SELECT status, usd FROM calls WHERE stage='S9'").fetchone()
    assert st == "error" and usd == 0.0


def test_s9_launcher_immediate_failure_is_detected(tmp_path):
    P, iid, counter, jobs = _s9_env(tmp_path, launcher_rc=2, make_job=False)
    P.run(only=[iid], max_steps=1)
    it = P.store.item(iid)
    it.state.update(stage="S9", status="active")
    it.save()
    P.run(only=[iid], max_steps=1)                              # reconcile: launcher exited, no job
    runs = json.loads(P.store.item(iid).p("calib", "runs.json").read_text())
    assert runs[0]["state"] == "launch_failed" and "launcher exit 2" in runs[0]["error"], runs


def test_s9_stops_when_actual_spend_reaches_run_cap(tmp_path):
    from tbsgen.stages_b import ingest
    P, iid, counter, jobs = _s9_env(tmp_path)
    P.run(only=[iid])
    it = P.store.item(iid)
    r1 = json.loads(it.p("calib", "runs.json").read_text())[0]
    _finish_job(jobs, r1["job_name"], reward=1.0, cost=150.0)  # real cost far above the $10 reservation
    ingest(P, it, None)
    P.run(only=[iid])
    it = P.store.item(iid)
    assert it.state["status"] == "paused" and it.state["reason"].startswith("budget:run_total_actual"), it.state
    assert counter.read_text().count("x") == 1


# ------------------------------------------------------------------ defaults without machine-specific paths
def test_job_config_builtin_agent_and_env_by_default(tmp_path):
    cfg = load_config()
    conf = calib.job_config(cfg, tmp_path, "glm", "j")
    assert conf["agents"][0]["name"] == "claude-code" and "import_path" not in conf["agents"][0]
    assert conf["environment"]["type"] == "docker" and "import_path" not in conf["environment"]
    cfg = load_config(None, {"calib": {"student_agent": "my_ext:Agent", "environment": "my_ext:Env"}})
    conf = calib.job_config(cfg, tmp_path, "glm", "j")
    assert conf["agents"][0]["import_path"] == "my_ext:Agent" and "name" not in conf["agents"][0]
    assert conf["environment"]["import_path"] == "my_ext:Env" and "type" not in conf["environment"]


def test_prepare_without_launcher_prints_plain_harbor_command(tmp_path):
    item = tmp_path / "items" / "x"
    (item / "pkg").mkdir(parents=True)
    rec = calib.prepare(load_config(), item, "glm", 1)
    assert rec["command"].startswith("harbor run --config ") and "blocked" not in rec
    rec = calib.prepare(load_config(None, {"calib": {"allow_submit": True}}), item, "glm", 1)
    assert "harbor_root" in rec["blocked"]
    rec = calib.prepare(load_config(None, {"calib": {"harbor_root": "/h", "student_launcher": "run.sh"}}), item, "glm", 1)
    assert rec["command"].startswith("cd /h && ./run.sh run ")


def test_teacher_url_unset_is_a_config_error(tmp_path):
    from tbsgen.models import TeacherClient
    cfg = load_config(None, {"models": {"teacher": {"url": None}}, "budget": {"daily_usd_total": 1.0}})
    led = Ledger(tmp_path / "l.sqlite", cfg)
    with pytest.raises(ModelError) as e:
        TeacherClient(led, cfg)
    assert e.value.cls == "config" and "models.teacher.url" in str(e.value)


def test_teacher_role_resolves_to_the_teacher_client(tmp_path):
    from tbsgen.config import resolve_role
    from tbsgen.models import TeacherClient, get_client
    cfg = load_config(None, {"models": {"teacher": {"url": "http://127.0.0.1:9/v1/messages"}},
                             "budget": {"daily_usd_total": 1.0}})
    assert resolve_role(cfg, "spec") == "teacher"
    assert isinstance(get_client("teacher", cfg, Ledger(tmp_path / "l.sqlite", cfg)), TeacherClient)
    assert "opus" not in cfg["models"]


def test_external_locations_come_from_config_only(monkeypatch):
    """No environment variable points at an earlier pipeline: cache dirs and the TBS dedup root are config keys."""
    import importlib
    import tbsgen.config as gc
    monkeypatch.setenv("TBS_OLD_PIPE", "/data/old")
    monkeypatch.delenv("TBS_HARBOR_ROOT", raising=False)
    c = importlib.reload(gc).load_config()
    assert c["fulltext"]["cache_dirs"] == [] and c["probe"]["tbs_tasks_root"] is None and c["calib"]["harbor_root"] is None
    monkeypatch.undo()
    importlib.reload(gc)


# ------------------------------------------------------------------ teacher: thinking used the whole output budget
class _SSEResp:
    def __init__(self, lines):
        self.lines = [x.encode() for x in lines]

    def __iter__(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse(text, stop, out_tok):
    ev = [{"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 1000}}}]
    if text:
        ev.append({"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}})
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": out_tok}})
    return _SSEResp(["data: " + json.dumps(e) for e in ev])


def _teacher(tmp_path, monkeypatch, responses, **teacher):
    import urllib.request
    from tbsgen.models import TeacherClient
    sent = []

    class Opener:
        def open(self, req, timeout=None):
            sent.append(json.loads(req.data))
            return responses.pop(0)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: Opener())
    cfg = load_config(None, {"run": {"root": str(tmp_path)}, "budget": {"daily_usd_total": 50.0},
                             "models": {"teacher": dict({"url": "http://127.0.0.1:9/v1/messages"}, **teacher)}})
    led = Ledger(tmp_path / "l.sqlite", cfg)
    return TeacherClient(led, cfg), led, sent


def _calls(led):
    with led._db() as db:
        return db.execute("SELECT status, error_class, usd, out_tok FROM calls ORDER BY id").fetchall()


def test_teacher_empty_max_tokens_is_billed_and_retried_at_low_effort(tmp_path, monkeypatch):
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("", "max_tokens", 64000), _sse("=== FILE: a ===\nx\n", "end_turn", 500)])
    r = c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
    assert r.text.startswith("=== FILE: a")
    assert [s["output_config"]["effort"] for s in sent] == ["medium", "low"]
    rows = _calls(led)
    assert [(x[0], x[1], x[3]) for x in rows] == [("error", "empty", 64000), ("ok", None, 500)]
    assert rows[0][2] == pytest.approx((1000 * 5 + 64000 * 25) / 1e6)     # the failed call is billed, not $0


def test_teacher_empty_max_tokens_retries_only_once(tmp_path, monkeypatch):
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("", "max_tokens", 64000), _sse("", "max_tokens", 64000)])
    with pytest.raises(ModelError) as e:
        c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
    assert e.value.thinking_exhausted and len(sent) == 2
    assert all(x[2] > 0 for x in _calls(led))


def test_teacher_empty_retry_counts_against_item_cap(tmp_path, monkeypatch):
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("", "max_tokens", 64000), _sse("ok", "end_turn", 5)])
    led.cfg["budget"]["max_calls_per_item"] = 1
    with pytest.raises(BudgetStop) as e:
        c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
    assert e.value.reason == "item_calls" and len(sent) == 1


def test_teacher_effort_by_stage_and_no_retry_when_disabled(tmp_path, monkeypatch):
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("ok", "end_turn", 5), _sse("", "max_tokens", 9000)],
                            effort_by_stage={"S6A": "low"}, effort_on_thinking_exhausted=None)
    c.chat("sys", "user", stage="S6A", item="i1", max_tokens=9000)
    with pytest.raises(ModelError) as e:
        c.chat("sys", "user", stage="S5", item="i1", max_tokens=9000)
    assert [s["output_config"]["effort"] for s in sent] == ["low", "medium"] and e.value.cls == "empty"


def test_teacher_other_empty_answers_are_not_retried(tmp_path, monkeypatch):
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("", "end_turn", 3)])
    with pytest.raises(ModelError) as e:
        c.chat("sys", "user", stage="S5", item="i1", max_tokens=9000)
    assert not e.value.thinking_exhausted and len(sent) == 1


def test_teacher_no_retry_when_already_at_fallback_effort(tmp_path, monkeypatch):
    # the retry only lowers the effort: a call already at the fallback effort is not repeated
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("", "max_tokens", 64000), _sse("ok", "end_turn", 5)],
                            effort_by_stage={"S6A": "low"})
    with pytest.raises(ModelError) as e:
        c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
    assert e.value.thinking_exhausted and len(sent) == 1 and _calls(led)[0][2] > 0


def test_teacher_stream_failure_bills_usage_already_reported(tmp_path, monkeypatch):
    start = {"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 1000}}}
    delta = {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 4000}}
    err = {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}

    class Broken(_SSEResp):
        def __iter__(self):
            yield from self.lines
            raise TimeoutError("read timeout")

    for resp, cls in [(_SSEResp(["data: " + json.dumps(e) for e in (start, delta, err)]), "stream"),
                      (_SSEResp(["data: " + json.dumps(start)]), "stream"),
                      (Broken(["data: " + json.dumps(e) for e in (start, delta)]), "network")]:
        c, led, sent = _teacher(tmp_path / cls / str(id(resp)), monkeypatch, [resp])
        with pytest.raises(ModelError) as e:
            c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
        assert e.value.cls == cls
        assert _calls(led)[0][2] > 0                     # input (and any reported output) tokens are billed


def test_teacher_fallback_never_raises_effort_and_fake_matches(tmp_path, monkeypatch):
    c, led, sent = _teacher(tmp_path, monkeypatch, [_sse("", "max_tokens", 64000), _sse("ok", "end_turn", 5)],
                            effort_by_stage={"S6A": "low"}, effort_on_thinking_exhausted="high")
    with pytest.raises(ModelError):
        c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
    assert [s["output_config"]["effort"] for s in sent] == ["low"]
    # the fake client resolves the same effective effort (stage override) and makes the same decision
    cfg = load_config(None, {"run": {"root": str(tmp_path)}, "budget": {"daily_usd_total": 50.0},
                             "models": {"teacher": {"url": "http://x/v1/messages", "effort_by_stage": {"S6A": "low"}}}})
    f = FakeClient(Ledger(tmp_path / "f.sqlite", cfg), cfg,
                   [{"stage": "S6A", "error": "empty", "stop": "max_tokens", "usage": {"output_tokens": 9}},
                    {"stage": "S6A", "text": "ok"}])
    with pytest.raises(ModelError):
        f.chat("sys", "user", stage="S6A", item="i1")
    assert [x["effort"] for x in f.log] == ["low"]


def test_teacher_malformed_sse_event_bills_reported_usage(tmp_path, monkeypatch):
    start = {"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 1000}}}
    bad = {"type": "content_block_delta", "delta": None}
    c, led, sent = _teacher(tmp_path, monkeypatch, [_SSEResp(["data: " + json.dumps(e) for e in (start, bad)])])
    with pytest.raises(ModelError) as e:
        c.chat("sys", "user", stage="S5", item="i1", max_tokens=9000)
    assert e.value.cls == "stream" and _calls(led)[0][2] > 0


@pytest.mark.parametrize("bad", ['data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": nul',
                                 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": null}}',
                                 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": 7}}',
                                 'data: [1, 2]'])
def test_teacher_corrupt_sse_is_a_billed_stream_error_not_a_retry(tmp_path, monkeypatch, bad):
    start = {"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 1000}}}
    end = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 4000}}
    resp = _SSEResp(["data: " + json.dumps(start), bad, "data: " + json.dumps(end)])
    c, led, sent = _teacher(tmp_path, monkeypatch, [resp, _sse("ok", "end_turn", 5)])
    with pytest.raises(ModelError) as e:
        c.chat("sys", "user", stage="S6A", item="i1", max_tokens=64000)
    assert e.value.cls == "stream" and len(sent) == 1 and _calls(led)[0][2] > 0


def test_resource_metric_threshold_tolerates_timing_noise():
    """e2e-prod-20261109: ref runtime 0.58 s in S6A, 0.93 s re-measured in S7; threshold ref x1.5 = 0.88 -> oracle gate
    failed -> 2 repair rounds -> dropped.  Non-separating resource metrics get ref x(1+resource_loose_margin)."""
    cfg = load_config()
    spec = {"metrics": [{"name": "hidden_mae", "direction": "<="}, {"name": "hidden_runtime_sec", "direction": "<="}]}
    meas = {"ref": {"0": {"hidden_mae": 0.2, "hidden_runtime_sec": 0.58}},
            "mid": {"0": {"hidden_mae": 1.3, "hidden_runtime_sec": 0.60}},
            "naive": {"0": {"hidden_mae": 1.9, "hidden_runtime_sec": 0.40}}}
    th = set_thresholds(spec, meas, cfg)["thresholds"]
    assert th["hidden_runtime_sec"]["threshold"] >= 0.58 * 3 - 1e-9 > 0.93
    # a much slower naive: the naive cap never goes below ref x3 (here the loose rule is already tighter)
    meas["naive"]["0"]["hidden_runtime_sec"] = 2.0
    t2 = set_thresholds(spec, meas, cfg)["thresholds"]["hidden_runtime_sec"]
    assert t2["threshold"] >= 0.58 * 3 - 1e-9 and t2["threshold"] < 2.0
    # quality metrics unchanged (ref x1.5 loose rule)
    spec2 = {"metrics": [{"name": "hidden_mae", "direction": "<="}, {"name": "hidden_err2", "direction": "<="}]}
    m2 = {w: {"0": {"hidden_mae": v["0"]["hidden_mae"], "hidden_err2": 0.58 if w == "ref" else 0.6 if w == "mid" else 0.4}}
          for w, v in meas.items()}
    assert set_thresholds(spec2, m2, cfg)["thresholds"]["hidden_err2"]["threshold"] == pytest.approx(0.87)


def test_resource_loose_margin_is_configurable():
    cfg = load_config(None, {"thresholds": {"resource_loose_margin": 0.8}})
    spec = {"metrics": [{"name": "hidden_mae", "direction": "<="}, {"name": "hidden_runtime_sec", "direction": "<="}]}
    meas = {"ref": {"0": {"hidden_mae": 0.2, "hidden_runtime_sec": 0.58}},
            "mid": {"0": {"hidden_mae": 1.3, "hidden_runtime_sec": 0.60}},
            "naive": {"0": {"hidden_mae": 1.9, "hidden_runtime_sec": 2.0}}}
    assert set_thresholds(spec, meas, cfg)["thresholds"]["hidden_runtime_sec"]["threshold"] == pytest.approx(1.1)


def test_metrics_at_the_floating_point_noise_floor():
    """e2e-prod-20261109 (one-loop power spectrum): sigma metrics ref 4.3e-11 / mid 2.7e-11 / naive 4.3e-11 -> was
    ref_worse_than_naive with threshold 6.5e-11 -> S8 x2 -> dropped."""
    cfg = load_config()
    spec = {"metrics": [{"name": "hidden_relerr", "direction": "<="}, {"name": "hidden_l2", "direction": "<="},
                        {"name": "hidden_sigma_relerr", "direction": "<="}]}
    meas = {"ref": {"0": {"hidden_relerr": 6.1e-9, "hidden_l2": 3.1e-10, "hidden_sigma_relerr": 4.3e-11}},
            "mid": {"0": {"hidden_relerr": 0.41, "hidden_l2": 3e-10, "hidden_sigma_relerr": 2.7e-11}},
            "naive": {"0": {"hidden_relerr": 0.29, "hidden_l2": 0.47, "hidden_sigma_relerr": 2.7e-11}}}
    th = set_thresholds(spec, meas, cfg)
    assert th["ref_worse_than_naive"] == [] and th["status"] == "ok", th
    sig = th["thresholds"]["hidden_sigma_relerr"]
    assert sig["threshold"] == pytest.approx(1e-9) and not sig["separating"] and sig["ref_passes"]
    assert th["thresholds"]["hidden_relerr"]["separating"]                       # real separation unaffected
    meas["ref"]["0"]["hidden_sigma_relerr"] = 5e-3                              # a real regression still counts
    assert set_thresholds(spec, meas, cfg)["ref_worse_than_naive"] == ["hidden_sigma_relerr"]
    # tiny-scale metric where one baseline is above the floor: plain rules (ref x1.5), separation kept
    meas["ref"]["0"]["hidden_sigma_relerr"] = 1e-12
    meas["mid"]["0"]["hidden_sigma_relerr"] = 1e-12
    meas["naive"]["0"]["hidden_sigma_relerr"] = 5e-9
    t3 = set_thresholds(spec, meas, cfg)["thresholds"]["hidden_sigma_relerr"]
    assert t3["threshold"] == pytest.approx(1.5e-12, rel=1e-6, abs=0)          # plain loose rule, no floor
    meas["mid"]["0"]["hidden_sigma_relerr"] = 2e-9                              # mid above the floor: separating
    assert set_thresholds(spec, meas, cfg)["thresholds"]["hidden_sigma_relerr"]["separating"]
    cfg0 = load_config(None, {"thresholds": {"noise_floor_abs": 0}})             # disabled: old behaviour
    meas = {"ref": {"0": {"hidden_relerr": 6.1e-9, "hidden_l2": 3.1e-10, "hidden_sigma_relerr": 4.3e-11}},
            "mid": {"0": {"hidden_relerr": 0.41, "hidden_l2": 3e-10, "hidden_sigma_relerr": 2.7e-11}},
            "naive": {"0": {"hidden_relerr": 0.29, "hidden_l2": 0.47, "hidden_sigma_relerr": 2.7e-11}}}
    assert set_thresholds(spec, meas, cfg0)["ref_worse_than_naive"] == ["hidden_sigma_relerr"]


def test_noise_floor_separation_branch_and_nonfinite_values():
    spec = {"metrics": [{"name": "hidden_a", "direction": "<="}, {"name": "hidden_b", "direction": "<="}]}
    meas = {"ref": {"0": {"hidden_a": 1e-10, "hidden_b": 0.1}}, "mid": {"0": {"hidden_a": 2e-10, "hidden_b": 0.9}},
            "naive": {"0": {"hidden_a": 4e-10, "hidden_b": 0.9}}}
    on = set_thresholds(spec, meas, load_config())
    off = set_thresholds(spec, meas, load_config(None, {"thresholds": {"noise_floor_abs": 0}}))
    assert off["thresholds"]["hidden_a"]["separating"] and not on["thresholds"]["hidden_a"]["separating"]
    assert on["mid_fail"]["0"] == 1 and off["mid_fail"]["0"] == 2
    # a non-finite mid value on one seed: not "all at the floor" -> plain loose rule, not the floor
    meas = {"ref": {"0": {"hidden_a": 1e-10, "hidden_b": 0.1}, "1": {"hidden_a": 1e-10, "hidden_b": 0.1}},
            "mid": {"0": {"hidden_a": 2e-10, "hidden_b": 0.9}, "1": {"hidden_a": float("inf"), "hidden_b": 0.9}},
            "naive": {"0": {"hidden_a": 1e-10, "hidden_b": 0.9}, "1": {"hidden_a": 1e-10, "hidden_b": 0.9}}}
    assert set_thresholds(spec, meas, load_config())["thresholds"]["hidden_a"]["threshold"] < 1e-9


def test_noise_floor_threshold_is_exact_for_non_default_floor():
    cfg = load_config(None, {"thresholds": {"noise_floor_abs": 1e-10}})
    spec = {"metrics": [{"name": "hidden_a", "direction": "<="}, {"name": "hidden_b", "direction": "<="}]}
    meas = {"ref": {"0": {"hidden_a": 1e-11, "hidden_b": 0.1}}, "mid": {"0": {"hidden_a": 1e-11, "hidden_b": 0.9}},
            "naive": {"0": {"hidden_a": 1e-11, "hidden_b": 0.9}}}
    assert set_thresholds(spec, meas, cfg)["thresholds"]["hidden_a"]["threshold"] == 1e-10


def test_load_candidates_accepts_tbsretr_per_target_files(tmp_path):
    """`tbsretr run` writes one file per target whose `target` is the id string (crashed ingest before)."""
    from tbsgen.intake import load_candidates
    one = {"target": "task:x", "papers": [{"doi": "10.1/a", "title": "A"}]}
    (tmp_path / "t.json").write_text(json.dumps(one))
    assert load_candidates(tmp_path / "t.json") == [{"doi": "10.1/a", "title": "A", "_target": "task:x"}]
    (tmp_path / "l.json").write_text(json.dumps([one, {"target": {"id": "task:y"}, "papers": [{"doi": "10.1/b"}]}]))
    assert [c["_target"] for c in load_candidates(tmp_path / "l.json")] == ["task:x", "task:y"]
    (tmp_path / "e.json").write_text(json.dumps({"target": {"id": "task:z"}, "selected": [{"doi": "10.1/c"}]}))
    assert load_candidates(tmp_path / "e.json")[0]["_target"] == "task:z"
