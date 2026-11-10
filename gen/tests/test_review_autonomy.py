"""Review autonomy PR: machine-speed scaled timing gates, resource-metric gating policy, blocklist auto-recheck,
downloader fixes (B1 OA breaker per site, B2 Wiley prefixes, B3 cooldown on the final site, B4 S4-first order,
B5 fetch --max default), `tbsgen review summary`.  Offline: no Docker, no network."""
import importlib.util
import json
import time
from types import SimpleNamespace

import pytest

from tbsgen import blocklist, fulltext, review, runner
from tbsgen.config import load_config
from tbsgen.dlledger import DownloadLedger
from tbsgen.pipeline import Pipeline
from tbsgen.state import read_json, write_json


# ------------------------------------------------------------------ timing: speed scaling + gating policy
def _metrics_module(tmp_path, cfg, spec, item_dir=None, bench=None):
    """The verifier helper as a task would get it, loaded in-process with a fixed speed_bench()."""
    p = tmp_path / f"_metrics_{abs(hash((str(item_dir), bench, json.dumps(spec, sort_keys=True))))}.py"
    p.write_text(runner.metrics_helper_text(cfg, spec, item_dir))
    s = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m._PATH = str(tmp_path / f"{p.stem}.jsonl")
    if bench is not None:
        m._CACHE["speed_bench"] = bench
    return m


FASTPT_SPEC = dict(metrics=[dict(name="hidden_p1loop_relerr_med", direction="<="),
                            dict(name="hidden_runtime_per_spectrum", direction="<=")])


def _item_with_base(tmp_path, base):
    d = tmp_path / f"item-{base}"
    d.mkdir(exist_ok=True)
    write_json(d / "thresholds.json", dict(thresholds={}, speed_bench_sec=base))
    return d


@pytest.mark.parametrize("host,value,bench", [("aliyun", 0.0128, 0.30), ("gu26-a", 0.0630, 1.20),
                                              ("gu26-b", 0.0663, 1.20)])
def test_fastpt_runtime_gate_scales_with_machine_speed(tmp_path, host, value, bench):
    """FAST-PT: threshold 0.056 s set on the fast host (bench 0.30 s); gu26 (bench 4x slower) oracle 0.063-0.066 s
    passes at 0.056 x 4; the Aliyun oracle 0.0128 s passes unscaled.  With gating on (speed is the contribution)."""
    cfg = load_config()
    spec = dict(FASTPT_SPEC, resource_metric_gating=True)
    m = _metrics_module(tmp_path, cfg, spec, _item_with_base(tmp_path, 0.30), bench=bench)
    m.check("hidden_runtime_per_spectrum", value, 0.056, "<=")          # no AssertionError = reward 1 for this
    rec = [json.loads(x) for x in open(m._PATH)][-1]
    assert rec["passed"] is True and rec["threshold"] == 0.056               # the locked literal is recorded
    assert rec["threshold_effective"] == pytest.approx(0.056 * max(1.0, bench / 0.30) if bench > 0.33 else 0.056)


def test_unscaled_slow_host_fails_and_ratio_is_clamped(tmp_path):
    cfg = load_config()
    spec = dict(FASTPT_SPEC, resource_metric_gating=True)
    # no speed base recorded (old thresholds): no scaling -> the gu26 oracle fails, as observed before this change
    m = _metrics_module(tmp_path, cfg, spec, None, bench=1.20)
    with pytest.raises(AssertionError):
        m.check("hidden_runtime_per_spectrum", 0.0630, 0.056, "<=")
    # ratio clamped to [0.25, 8]
    m = _metrics_module(tmp_path, cfg, spec, _item_with_base(tmp_path, 0.01), bench=10.0)
    m.check("hidden_runtime_per_spectrum", 0.40, 0.056, "<=")
    rec = [json.loads(x) for x in open(m._PATH)][-1]
    assert rec["speed_ratio"] == 8.0 and rec["threshold_effective"] == pytest.approx(0.448)
    m = _metrics_module(tmp_path, cfg, spec, _item_with_base(tmp_path, 10.0), bench=0.01)
    with pytest.raises(AssertionError):
        m.check("hidden_runtime_per_spectrum", 0.02, 0.056, "<=")               # 0.056 * 0.25 = 0.014
    # a same-host bench within the deadband never moves the threshold
    m = _metrics_module(tmp_path, cfg, spec, _item_with_base(tmp_path, 1.0), bench=1.05)
    m.check("hidden_runtime_per_spectrum", 0.05, 0.056, "<=")
    assert [json.loads(x) for x in open(m._PATH)][-1]["speed_ratio"] == 1.0


def test_resource_metrics_recorded_not_gated_by_default(tmp_path):
    """Default (resource_metric_gating absent/false): the runtime check records but never fails; accuracy metrics
    still assert exactly; S7/S6 readers skip gated=False rows."""
    cfg = load_config()
    m = _metrics_module(tmp_path, cfg, FASTPT_SPEC, _item_with_base(tmp_path, 0.30), bench=1.20)
    m.check("hidden_runtime_per_spectrum", 5.0, 0.056, "<=")                  # way over: still no failure
    rec = [json.loads(x) for x in open(m._PATH)][-1]
    assert rec["gated"] is False and rec["passed_raw"] is False and "passed" not in rec
    with pytest.raises(AssertionError):
        m.check("hidden_p1loop_relerr_med", 2e-6, 1.5e-6, "<=")
    # a memory metric is never speed-scaled; a quality metric with "time" in its name is not a resource metric
    spec = dict(metrics=[dict(name="hidden_peak_rss_mb", direction="<="), dict(name="delay_time_err", direction="<=")],
                resource_metric_gating=True)
    assert runner.resource_metric_kinds(spec) == {"hidden_peak_rss_mb": "memory"}
    m = _metrics_module(tmp_path, cfg, spec, _item_with_base(tmp_path, 0.30), bench=1.20)
    with pytest.raises(AssertionError):
        m.check("hidden_peak_rss_mb", 120.0, 100.0, "<=")
    with pytest.raises(AssertionError):
        m.check("delay_time_err", 1.2, 1.0, "<=")
    assert [json.loads(x) for x in open(m._PATH)][-1].get("speed_ratio") is None


def test_speed_bench_runs_and_marker_lines_are_filled(tmp_path):
    cfg = load_config(None, {"thresholds": {"speed_deadband": 0.2}})
    txt = runner.metrics_helper_text(cfg, dict(FASTPT_SPEC, resource_metric_gating=True),
                                     _item_with_base(tmp_path, 0.31))
    assert "SPEED_BASE = 0.31  # tbsgen:speed_base" in txt and "RESOURCE_GATING = True" in txt
    assert "SPEED_DEADBAND = 0.2  # tbsgen:speed_deadband" in txt
    assert "'hidden_runtime_per_spectrum': 'time'" in txt
    m = _metrics_module(tmp_path, cfg, FASTPT_SPEC)
    t0 = time.time()
    b = m.speed_bench(repeats=1)
    assert b is not None and 0 < b < 30 and time.time() - t0 < 60


def test_review_checklist_asks_whether_timing_should_gate():
    assert "h_timing_gate" in review.CHECKLIST
    import inspect
    assert "resource_metric_gating" in inspect.getsource(review)


# ------------------------------------------------------------------ blocklist auto-recheck
def _bl_env(tmp_path, n_blocked=3, n_other=7, **rc):
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "models": {"roles": {"screen": "rules"}},
                             "screen": {"blocklist_recheck": rc} if rc else {}})
    P = Pipeline(cfg)
    papers = [dict(doi=f"10.1177/x{i}", title=f"sage {i}", abstract="We propose an algorithm.", _target="T")
              for i in range(n_blocked)]
    papers += [dict(doi=f"10.9999/y{i}", title=f"other {i}", abstract="We propose an algorithm.", _target="T")
               for i in range(n_other)]
    ids = P.ingest(papers)
    return P, ids


def test_blocklist_trial_quota_and_share(tmp_path):
    P, ids = _bl_env(tmp_path, n_blocked=3, n_other=7, weekly_quota=2)       # 30% of target T is sage
    rules = P.cfg["screen"]["blocklist"]
    sage = next(r for r in rules if r["name"] == "sage")
    ev = []
    acts = []
    for iid in ids[:3]:
        c = dict(read_json(P.store.item(iid).p("S1_candidate.json")), item_id=iid)
        acts.append(blocklist.decide(P.store, P.cfg, c, rules, sage, lambda k, **d: ev.append((k, d)))["action"])
    assert acts == ["trial", "trial", "block"]                                 # weekly quota 2
    assert [k for k, _ in ev] == ["blocklist_trial", "blocklist_trial"]
    # a second S2 pass of a trial paper does not take another slot
    c = dict(read_json(P.store.item(ids[0]).p("S1_candidate.json")), item_id=ids[0])
    assert blocklist.decide(P.store, P.cfg, c, rules, sage, lambda *a, **k: None).get("repeat")
    # below the share threshold: always blocked
    P2, ids2 = _bl_env(tmp_path / "low", n_blocked=1, n_other=9)              # 10%: not more than 10%
    c = dict(read_json(P2.store.item(ids2[0]).p("S1_candidate.json")), item_id=ids2[0])
    assert blocklist.decide(P2.store, P2.cfg, c, rules, sage, lambda *a, **k: None)["action"] == "block"


def _trial(P, iid, ok: bool):
    it = P.store.item(iid)
    it.state["blocklist_trial"] = "sage"
    if ok:
        it.p("fulltext.txt").write_text("text")
    else:
        it.state.update(stage="S3", status="parked", reason="fulltext_unavailable")
    it.save()
    blocklist.resolve(P.store, it)


def test_blocklist_lift_and_keep_decisions(tmp_path):
    P, ids = _bl_env(tmp_path, n_blocked=6, n_other=4, weekly_quota=6)
    rules = P.cfg["screen"]["blocklist"]
    sage = next(r for r in rules if r["name"] == "sage")
    for iid in ids[:6]:
        c = dict(read_json(P.store.item(iid).p("S1_candidate.json")), item_id=iid)
        assert blocklist.decide(P.store, P.cfg, c, rules, sage, P.store.event)["action"] == "trial"
    for iid, ok in zip(ids[:4], (True, False, True, False)):
        _trial(P, iid, ok)
    assert blocklist.reevaluate(P.store, P.cfg, P.store.event) == {}        # 4 decided < min_trials 5
    _trial(P, ids[4], False)
    assert blocklist.reevaluate(P.store, P.cfg, P.store.event) == {"sage": "kept"}     # 2/5 = 40% < 60%
    assert blocklist.reevaluate(P.store, P.cfg, P.store.event) == {"sage": "kept"}     # no new event
    assert sum(e["kind"] == "blocklist_kept" for e in P.store.events()) == 1
    # a paper S3 failed on is delivered later by the fetch queue: upgraded; 6th trial ok -> 4/6 >= 60% -> lifted
    P.store.item(ids[1]).p("fulltext.txt").write_text("late")
    _trial(P, ids[5], True)
    assert blocklist.reevaluate(P.store, P.cfg, P.store.event) == {"sage": "lifted"}
    assert "sage" in blocklist.lifted_rules(blocklist.load_state(P.store))
    assert any(e["kind"] == "blocklist_lifted" for e in P.store.events())
    # lifted: the rule no longer matches
    assert blocklist.match({"doi": "10.1177/z"}, rules, {"sage"}) is None
    # a trial paper dropped by the S2 screen never reached the download: void, not counted
    P3, ids3 = _bl_env(tmp_path / "void", n_blocked=2, n_other=3)
    c = dict(read_json(P3.store.item(ids3[0]).p("S1_candidate.json")), item_id=ids3[0])
    blocklist.decide(P3.store, P3.cfg, c, rules, sage, P3.store.event)
    blocklist.resolve(P3.store, P3.store.item(ids3[0]), void="S2 screen: off topic")
    assert blocklist.load_state(P3.store)["rules"]["sage"]["trials"][0]["outcome"] == "void"


def test_s2_trial_paper_goes_through_screen(tmp_path):
    P, ids = _bl_env(tmp_path, n_blocked=3, n_other=7)
    P.run(only=ids[:1], max_steps=3)
    it = P.store.item(ids[0])
    assert it.state.get("blocklist_trial") == "sage"
    assert read_json(it.p("S2_screen.json"))["method"] != "blocklist"


# ------------------------------------------------------------------ downloader fixes
def test_b1_oa_breaker_counts_distinct_sites(tmp_path):
    led = DownloadLedger(tmp_path / "l.sqlite")
    for _ in range(15):                                   # one site blocking over and over: its own cooldown only
        led.block("www.mdpi.com", "cf", channel="oa")
    assert led.breaker("oa") is None
    for i in range(8):
        led.block(f"site{i}.example.org", "cf", channel="oa")
    assert led.breaker("oa") is None                      # 9 distinct sites; the 10th trips (oa limit 10)
    led.block("site8.example.org", "cf", channel="oa")
    assert "blocked sites" in led.breaker("oa")
    # other channels keep counting block rows
    led2 = DownloadLedger(tmp_path / "l2.sqlite", global_limits={"breaker_blocks_per_day": 2})
    led2.block("chrome:10.1021", "cf", channel="browser")
    led2.block("chrome:10.1021", "cf", channel="browser")
    assert "block signals" in led2.breaker("browser")


def test_b1_b3_one_block_counted_once_and_cools_the_final_site():
    # doi.org redirected to the blocking publisher: the publisher is cooled and counted, doi.org is not cooled
    t = fulltext.block_targets("doi.org", "https://doi.org/10.1/x", "https://pub.example.com/a")
    assert t == [("pub.example.com", True)]
    # a direct link redirected to another site: final counted, starting host cooled uncounted
    t = fulltext.block_targets("repo.example.org", "https://repo.example.org/x", "https://cdn.example.net/x")
    assert t[0] == ("cdn.example.net", True) and ("repo.example.org", False) in t and len(t) == 2
    # no redirect: one host, counted once
    assert fulltext.block_targets("a.org", "https://a.org/x", "https://a.org/x") == [("a.org", True)]
    # a ledger key that differs from the URL host (API channel key) is cooled but not counted again
    t = fulltext.block_targets("api.wiley.com", "https://api.wiley.com/x", "https://api.wiley.com/x")
    assert t == [("api.wiley.com", True)]


def test_b2_wiley_prefixes_match_publisher_aliases():
    cfg = load_config()
    want = set(fulltext.PUBLISHERS["wiley"]["prefixes"])
    got = {p.rstrip("/") for p in cfg["fulltext"]["wiley"]["prefixes"]}
    assert "10.1029" in got and got == want
    want = set(fulltext.PUBLISHERS["elsevier"]["prefixes"])
    assert {p.rstrip("/") for p in cfg["fulltext"]["elsevier"]["prefixes"]} == want


def test_b4_order_prefers_s4_jev_score_within_target(tmp_path):
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}})
    P = Pipeline(cfg)
    ids = P.ingest([dict(doi=f"10.9/{k}", title=f"paper number {k} about {k * 7} things", abstract="a",
                         _target="T") for k in range(3)])
    s2 = {ids[0]: 9, ids[1]: 5, ids[2]: 7}
    s4 = {ids[0]: 0.2, ids[1]: 0.9}                       # ids[2] has no S4 score
    for iid in ids:
        it = P.store.item(iid)
        write_json(it.p("S2_screen.json"), dict(score=s2[iid]))
        if iid in s4:
            write_json(it.p("S4_probe.json"), dict(jev=dict(score=s4[iid])))
    order = [i.item_id for i in P.order(P.store.items())]
    assert order == [ids[1], ids[0], ids[2]]


def test_b5_fetch_max_defaults_to_remote_batch():
    from tbsgen.cli import fetch_max_default
    assert fetch_max_default(load_config(), None) == 10
    on = load_config(None, {"fulltext": {"remote": {"enabled": True, "batch_max": 50}}})
    assert fetch_max_default(on, None) == 50 and fetch_max_default(on, 7) == 7


# ------------------------------------------------------------------ review summary
def test_review_summary_counts_sample_and_alerts(tmp_path):
    cfg = load_config(None, {"run": {"root": str(tmp_path / "run")}, "review": {"summary_alerts": {"drop_rate": 0.2}}})
    P = Pipeline(cfg)
    ids = P.ingest([dict(doi=f"10.9/{k}", title=f"t{k}", abstract="a") for k in range(4)])
    blk = dict(id="I1", category="a", severity="blocking", problem="statement", evidence=["x" * 20])
    fund = dict(blk, problem="reference_wrong")
    plan = {ids[0]: [("pass", [])], ids[1]: [("revise", [blk]), ("pass", [])], ids[2]: [("drop", [fund])],
            ids[3]: [("pass", []), ("revise", [blk])]}
    for iid, rounds in plan.items():
        it = P.store.item(iid)
        runs = []
        for k, (v, issues) in enumerate(rounds, 1):
            raw = dict(verdict=v, issues=issues)
            body = dict(raw, _normalized=review.normalize(raw), job=f"j{k}", pkg_sha="s1", calib_epoch=0)
            write_json(it.p("review", f"r{k}", f"j{k}.verdict.json"), body)
            runs.append(dict(n=k, job_name=f"j{k}", outcome="done", verdict=v, pkg_sha="s1"))
            write_json(it.p("review", "verdict.json"), dict(body, round=k))
        write_json(it.p("review", "runs.json"), runs)
        it.state.update(review_round=len(rounds), review_revisions=len(rounds) - 1,
                        status="dropped" if iid == ids[2] else "done")
        it.save()
    s = review.summary(P, sample_n=2, seed=1)
    assert s["verdicts"] == {"pass": 2, "drop": 1, "revise": 1}
    assert s["per_item"][ids[1]]["revisions"] == 1 and s["blocking_problems"] == {"reference_wrong": 1, "statement": 1}
    assert s["revisions_histogram"] == {0: 2, 1: 2}
    assert len(s["pass_sample"]) == 2 and all(x["records"] for x in s["pass_sample"])
    kinds = {a["kind"] for a in s["alerts"]}
    assert "drop_rate" in kinds and "contradiction" in kinds                    # ids[3]: pass then revise, same pkg
    assert "blocklist" in s


def test_blocklist_weekly_revival_and_share_denominator(tmp_path, monkeypatch):
    """Papers dropped by a rule in an earlier week are the next week's trial pool; the share counts every
    candidate of the target (DOI or not)."""
    P, ids = _bl_env(tmp_path, n_blocked=4, n_other=6, weekly_quota=2)
    P.run(only=None, max_steps=40)
    st = blocklist.load_state(P.store)
    assert len(st["rules"]["sage"]["trials"]) == 2
    dropped = [i for i in ids[:4] if P.store.item(i).state.get("reason") == "blocklist:sage"]
    assert len(dropped) == 2
    monkeypatch.setattr(blocklist, "_iso_week", lambda: "2099-W01")
    out = blocklist.revive_for_trials(P.store, P.cfg, P.store.event)
    assert sorted(out) == sorted(dropped) and all(P.store.item(i).state["status"] == "active" for i in out)
    # no DOI candidates still count in the denominator: 1 blocked of 11 = 9% -> no trial
    P2, ids2 = _bl_env(tmp_path / "nodoi", n_blocked=1, n_other=0)
    P2.ingest([dict(title=f"no doi {k}", abstract="a", _target="T") for k in range(10)])
    sage = next(r for r in P2.cfg["screen"]["blocklist"] if r["name"] == "sage")
    c = dict(read_json(P2.store.item(ids2[0]).p("S1_candidate.json")), item_id=ids2[0])
    blocklist._SHARE_CACHE.clear()
    assert blocklist.decide(P2.store, P2.cfg, c, P2.cfg["screen"]["blocklist"], sage, lambda *a, **k: None)["action"] == "block"


def test_s7_margin_uses_effective_threshold_and_skips_ungated():
    """S7 oracle_margin: a recorded-only resource metric demands no margin; a gated time metric is compared with
    its machine-scaled threshold (the projection keeps gated / threshold_effective)."""
    import inspect
    from tbsgen import gates
    src = inspect.getsource(gates)
    assert '"gated", "threshold_effective"' in src and 'm.get("threshold_effective", m["threshold"])' in src


def test_locked_s6b_applies_gating_policy_and_speed_ratio():
    cfg = load_config()
    assert runner.speed_ratio(cfg, 1.2, 0.3) == 4.0 and runner.speed_ratio(cfg, 1.05, 1.0) == 1.0
    assert runner.speed_ratio(cfg, 100.0, 1.0) == 8.0 and runner.speed_ratio(cfg, None, 1.0) == 1.0


def test_validator_accepts_human_only_when_opted_in(tmp_path):
    from tbsgen.review import ALLOWED_HUMAN, VERDICTS, VALIDATOR
    off = VALIDATOR.replace("{problems}", "[]").replace("{verdicts}", repr(list(VERDICTS)))
    on = VALIDATOR.replace("{problems}", "[]").replace("{verdicts}", repr(list(VERDICTS) + list(ALLOWED_HUMAN)))
    assert "'human'" not in off.split("VERDICTS = ")[1].splitlines()[0]
    assert "'human'" in on.split("VERDICTS = ")[1].splitlines()[0]
    # odd field types never crash normalize(): invalid verdict
    n = review.normalize(dict(verdict="revise", issues=[dict(id="I1", severity="blocking", problem=["x"],
                                                              evidence=["long enough evidence"])]))
    assert n["verdict"] is None
    assert review.normalize(dict(verdict=["pass"]))["verdict"] is None


def test_codex_r2_memory_never_scaled_and_ungated_never_decides_s6b():
    from tbsgen.spec import set_thresholds
    assert runner.resource_metric_kinds(dict(metrics=[dict(name="hidden_rss_mb", kind="resource"),
                                                      dict(name="hidden_runtime_sec", direction="<="),
                                                      dict(name="solver_cost", kind="resource")])) == \
        {"hidden_rss_mb": "memory", "hidden_runtime_sec": "time", "solver_cost": "memory"}
    cfg = load_config()
    spec = dict(metrics=[dict(name="err", direction="<="), dict(name="hidden_peak_rss", direction="<=")])
    meas = dict(ref={"0": dict(err=0.1, hidden_peak_rss=50.0)}, mid={"0": dict(err=1.0, hidden_peak_rss=50.0)},
                naive={"0": dict(err=5.0, hidden_peak_rss=50.0)}, nop={"0": dict(err=1e6, hidden_peak_rss=50.0)})
    th = set_thresholds(spec, meas, cfg)
    assert th["status"] == "ok", th                 # ref RSS == nop RSS: recorded only, decides nothing
    meas["ref"]["0"]["hidden_peak_rss"] = float("nan")
    assert set_thresholds(spec, meas, cfg)["status"] == "ok"
    assert set_thresholds(dict(spec, resource_metric_gating=True), meas, cfg)["status"] == "ref_metric_missing"


def test_codex_r2_corrupt_verdict_is_invalid_and_gating_revision(tmp_path):
    trial = tmp_path / "t"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "verdict.json").write_text("{not json")
    raw = review.read_verdict(trial)
    assert raw is not None and review.normalize(raw)["verdict"] is None
    # a timing_gate fix flips spec.resource_metric_gating through the pseudo-path, scoring-only
    task = tmp_path / "task"
    task.mkdir()
    ok = review.revision_review(task, {review.GATING_PATH: "true"}, {})
    assert ok["ok"] and review.scoring_only({review.GATING_PATH: "false"})
    assert not review.revision_review(task, {review.GATING_PATH: "maybe"}, {})["ok"]
    assert "fail_no_output" in review.NO_OUTPUT_TYPES
