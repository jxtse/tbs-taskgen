"""Verifier time budget (timeout-fix PR): the agent program's scored runs are capped in total so that scored runs +
timed repeats + verifier overhead always end before the task's verifier timeout (e2e-batch2 review: a task's
measure.py called run_solution(1800) under a 1200 s verifier timeout).  Offline: no Docker, no network."""
import importlib.util
import json
import sys
import time
from pathlib import Path

from tbsgen import package, runner
from tbsgen.config import load_config


def _helper(tmp_path, cfg, spec, name="m"):
    p = tmp_path / f"_metrics_{name}.py"
    p.write_text(runner.metrics_helper_text(cfg, spec))
    s = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m._PATH = str(tmp_path / f"{name}.jsonl")
    # offline unit tests: never touch the host's users / dirs even when pytest runs as root (Codex r1); the
    # root/nobody path is covered by the Docker integration tests
    m._sandbox_prepare = lambda input_dir, out: None
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


def test_scored_budget_rejects_negative_or_nonfinite_overhead():
    for ov in (-120, float("nan"), float("inf")):
        cfg = load_config(None, {"package": {"verifier_overhead_sec": ov}})
        try:
            runner.scored_budget(cfg)
        except ValueError:
            continue
        raise AssertionError(ov)
    assert package.verifier_time_errors(Path("/nonexistent"), 120)          # unreadable package -> error


def test_validate_pkg_with_cfg_runs_the_time_check(tmp_path):
    cfg = load_config(None)
    pkg = _pkg(tmp_path, cfg, RES, 500)
    for r in package.REQUIRED:
        q = pkg / r
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            if not r.endswith("/"):
                q.write_text("x\n")
    (pkg / "task.toml").write_text('[task]\n[agent]\n[environment]\nnetwork_mode = "no-network"\n'
                                   '[verifier]\ntimeout_sec = 500\n')
    v = package.validate_pkg(pkg, cfg)
    assert not v["ok"] and any("> verifier timeout" in e for e in v.get("errors", [])), v
    assert not any("verifier time" in e for e in package.validate_pkg(pkg).get("errors", []))  # no cfg: structural


def test_nan_timeout_cannot_disable_the_cap(tmp_path):
    cfg = load_config(None)
    m = _helper(tmp_path, cfg, ACC, "nan")
    m.SCORED_BUDGET_SEC = 1.5
    entry = tmp_path / "cli.py"
    entry.write_text("import time\ntime.sleep(30)\n")
    d = tmp_path / "h"
    d.mkdir()
    t0 = time.time()
    _o, _sec, rc = m.run_solution(float("nan"), str(d), entry=str(entry))
    assert rc != 0 and time.time() - t0 < 15


def test_budget_is_shared_across_hidden_sets(tmp_path):
    """Two 1 s runs under a 1.6 s budget: the first runs fully, the second gets the 0.6 s left and times out."""
    cfg = load_config(None)
    m = _helper(tmp_path, cfg, ACC, "shared")
    m.SCORED_BUDGET_SEC = 1.6
    entry = tmp_path / "cli.py"
    entry.write_text("import os, sys, time\ntime.sleep(1.0)\nos.makedirs(sys.argv[2], exist_ok=True)\n")
    d1, d2 = tmp_path / "h1", tmp_path / "h2"
    d1.mkdir()
    d2.mkdir()
    assert m.run_solution(600, str(d1), entry=str(entry))[2] == 0
    _o, _s, rc2 = m.run_solution(600, str(d2), entry=str(entry))
    assert rc2 != 0
    runs = [json.loads(x) for x in open(m._PATH) if '"_solution_cli_run"' in x]
    assert 0 < runs[1]["timeout_effective"] < 0.7, runs[1]


def test_resource_task_repeats_capped_by_clamped_first_timeout():
    """Static check of the template: a timed repeat's timeout is min(clamped first timeout, timing budget left),
    never the raw timeout_sec measure.py asked for."""
    src = (Path(runner.__file__).parent / "templates_metrics.py").read_text()
    assert "min(t_first, t_rep)" in src and "min(float(timeout_sec), t_rep)" not in src
