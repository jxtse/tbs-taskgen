"""Verifier time budget (timeout-fix PR): the agent program's scored runs are capped in total so that scored runs +
timed repeats + verifier overhead always end before the task's verifier timeout (e2e-batch2 review: a task's
measure.py called run_solution(1800) under a 1200 s verifier timeout).  Offline: no Docker, no network."""
import importlib.util
import json
import sys
import time

from tbsgen import package, runner
from tbsgen.config import load_config


def _helper(tmp_path, cfg, spec, name="m"):
    p = tmp_path / f"_metrics_{name}.py"
    p.write_text(runner.metrics_helper_text(cfg, spec))
    s = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m._PATH = str(tmp_path / f"{name}.jsonl")
    return m


RES = dict(metrics=[dict(name="hidden_runtime_sec")])
ACC = dict(metrics=[dict(name="hidden_rmse")])


def test_scored_budget_defaults_fit_verifier_timeout():
    cfg = load_config(None)
    sc = runner.scored_budget(cfg)
    assert sc == min(cfg["package"]["verifier_timeout"], cfg["gen"]["test_timeout"]) - \
        cfg["package"]["verifier_overhead_sec"]
    for spec in (RES, ACC):
        vt = cfg["package"]["verifier_timeout"] + runner.timing_budget(cfg, spec)   # what package() writes
        assert sc + runner.timing_budget(cfg, spec) + cfg["package"]["verifier_overhead_sec"] <= vt
    txt = runner.metrics_helper_text(cfg, RES)
    assert f"SCORED_BUDGET_SEC = {sc!r}  # tbsgen:scored_budget" in txt


def test_scored_budget_uses_smaller_local_test_timeout():
    cfg = load_config(None, {"gen": {"test_timeout": 400}, "package": {"verifier_timeout": 900}})
    assert runner.scored_budget(cfg) == 400 - cfg["package"]["verifier_overhead_sec"]


def test_scored_budget_rejects_overhead_eating_everything():
    cfg = load_config(None, {"package": {"verifier_timeout": 100, "verifier_overhead_sec": 100}})
    try:
        runner.scored_budget(cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _pkg(tmp_path, cfg, spec, verifier_sec):
    pkg = tmp_path / "pkg"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "tests" / "_metrics.py").write_text(runner.metrics_helper_text(cfg, spec))
    (pkg / "task.toml").write_text(f"[verifier]\ntimeout_sec = {verifier_sec}\n")
    return pkg


def test_pkg_check_accepts_consistent_and_rejects_short_verifier(tmp_path):
    cfg = load_config(None)
    ov = cfg["package"]["verifier_overhead_sec"]
    vt = cfg["package"]["verifier_timeout"] + runner.timing_budget(cfg, RES)
    assert package.verifier_time_errors(_pkg(tmp_path / "a", cfg, RES, vt), ov) == []
    errs = package.verifier_time_errors(_pkg(tmp_path / "b", cfg, RES, vt - 1), ov)
    assert errs and "> verifier timeout" in errs[0], errs


def test_pkg_check_flags_old_helper_without_cap(tmp_path):
    cfg = load_config(None)
    pkg = _pkg(tmp_path, cfg, RES, 1200)
    src = (pkg / "tests" / "_metrics.py").read_text()
    import re
    (pkg / "tests" / "_metrics.py").write_text(re.sub(r"^SCORED_BUDGET_SEC = .*$", "", src, flags=re.M))
    errs = package.verifier_time_errors(pkg, 120)
    assert errs and "SCORED_BUDGET_SEC" in errs[0], errs


def test_run_solution_timeout_clamped_to_scored_budget(tmp_path):
    """measure.py asks for 1800 s; the scored budget (here 2 s) wins: the sleeping program is killed at ~2 s, and
    a second hidden set with nothing left fails at once (rc -9, inf) instead of running past the verifier."""
    cfg = load_config(None)
    m = _helper(tmp_path, cfg, ACC)
    m.SCORED_BUDGET_SEC = 2.0
    entry = tmp_path / "cli.py"
    entry.write_text("import time\ntime.sleep(30)\n")
    d1, d2 = tmp_path / "h1", tmp_path / "h2"
    d1.mkdir()
    d2.mkdir()
    t0 = time.time()
    _o, sec, rc = m.run_solution(1800, str(d1), entry=str(entry))
    assert rc != 0 and sec < 10 and time.time() - t0 < 20, (rc, sec)
    _o2, sec2, rc2 = m.run_solution(1800, str(d2), entry=str(entry))
    assert rc2 == -9 and sec2 == float("inf")
    recs = [json.loads(x) for x in open(m._PATH)]
    runs = [r for r in recs if r["name"] == "_solution_cli_run"]
    assert runs[0]["timeout_effective"] <= 2.0 and runs[0]["threshold"] == 1800.0
    assert runs[1].get("budget_exhausted") is True and runs[1]["passed"] is False


def test_run_solution_fast_program_unaffected(tmp_path):
    cfg = load_config(None)
    m = _helper(tmp_path, cfg, ACC, "fast")
    entry = tmp_path / "ok.py"
    entry.write_text("import os, sys\nos.makedirs(sys.argv[2], exist_ok=True)\n")
    d = tmp_path / "h"
    d.mkdir()
    _o, sec, rc = m.run_solution(1800, str(d), entry=str(entry))
    assert rc == 0 and sec < 30
    assert m._SCORED_SPENT[0] > 0
