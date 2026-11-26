"""Static verifier-budget correction and PKG routing, using only synthetic tasks."""
import hashlib
import shutil
import tomllib
from types import SimpleNamespace

import pytest

from tbsgen import calib, package, stages_b
from tbsgen.config import load_config
from tbsgen.state import Item, read_json, write_json


def _hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def _setup(tmp_path, source, *, locked=False, timeout=None):
    it = Item(tmp_path, "sample").load()
    it.state.update(stage="PKG", gates_attempt=1)
    if locked:
        it.state.update(thresholds_locked=True, post_regate=True, review_round=2,
                        review_revisions=1, calib_label="hard", calib_epoch=2)
        write_json(it.p("calib/runs.json"), [
            dict(epoch=2, who="teacher", outcome="pass" if not timeout else "infra", trial=dict(exception=timeout))])
    it.save()
    for rel, text in (("task/tests/measure.py", source), ("task/tests/test_outputs.py", "# tests\n"),
                      ("task/solution/solve.py", "# reference\n"), ("task/instruction.md", "Count rows.\n"),
                      ("runs/s7-1/detA/environment/data/sample.csv", "batch\none\n"),
                      ("runs/s7-1/detA/tests/truth/sample", "1"),
                      ("runs/s7-1/logs-ref0/metrics.jsonl", "")):
        path = it.p(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    write_json(it.p("_spec/spec.json"), dict(form="FILE"))
    write_json(it.p("thresholds.json"), dict(thresholds={}))
    write_json(it.p("gates/attempt-1.json"), dict(attempt=1, gates={}))
    events = []
    pipeline = SimpleNamespace(cfg=load_config(), item_cost=lambda _: 0,
                               store=SimpleNamespace(root=tmp_path, event=lambda *args, **kw: events.append((args, kw))))
    if locked:
        out = package.package(pipeline.cfg, it.dir, it.p("runs/s7-1/detA"), read_json(it.p("_spec/spec.json")), {})
        it.state["regates"] = [dict(pkg_sha=calib.pkg_sha(out))]
        it.save()
    return pipeline, it, events


def test_fix_changes_only_verifier_timeout_and_survives_rebuild(tmp_path, monkeypatch):
    pipeline, it, events = _setup(tmp_path, "run_solution(900)")
    source = it.p("task/task.toml")
    source.write_bytes(b'[agent]\r\ntimeout_sec = 4\r\n[verifier] # local\r\ntimeout_sec = 900 # budget\r\n')
    before = _hashes(it.p("task"))
    packaged = {}
    build = package.package

    def capture(*args):
        pkg = build(*args)
        packaged.update(_hashes(pkg))
        return pkg

    monkeypatch.setattr(package, "package", capture)
    stages_b.pkg(pipeline, it)
    after = _hashes(it.p("pkg"))
    assert after.keys() == packaged.keys()
    assert [p for p in after if after[p] != packaged[p]] == ["task.toml"]
    assert [p for p, h in _hashes(it.p("task")).items() if before[p] != h] == ["task.toml"]
    assert source.read_bytes() == b'[agent]\r\ntimeout_sec = 4\r\n[verifier] # local\r\ntimeout_sec = 1200.0 # budget\r\n'
    assert it.state["stage"] == "S9"
    assert it.state["require_prepared_receipt"]
    check = read_json(it.p("PKG_check.json"))
    assert check["ok"] and check["verifier_timeout_fix"] == dict(old=900, new=1200)
    assert any(name == ("verifier_timeout_fixed",) and data["new"] == 1200 for name, data in events)
    build(pipeline.cfg, it.dir, it.p("runs/s7-1/detA"), {}, {})
    assert tomllib.loads(it.p("pkg/task.toml").read_text())["verifier"]["timeout_sec"] == 1200
    assert package.validate_pkg(it.p("pkg"), pipeline.cfg)["ok"]


@pytest.mark.parametrize(("first", "second", "overhead", "required", "fails"), [
    ("400", "400", 120, 1100, True), ("100", "350", 120, 750, False),
    ("200 * 3", "180", 120, 1080, True), ("200", "400", 300, 901, True),
    ("400", "limit()", 120, 700, True)])
def test_timeout_gate_and_fix_use_all_test_files(tmp_path, first, second, overhead, required, fails):
    pipeline, it, _ = _setup(tmp_path, f"run_solution({first})")
    pipeline.cfg["package"]["verifier_overhead_sec"] = overhead
    out = package.package(pipeline.cfg, it.dir, it.p("runs/s7-1/detA"), {}, {})
    (out / "tests/extra.py").write_text(f"from _metrics import run_solution as run\nrun({second})")
    report = package.validate_pkg(out, pipeline.cfg)
    assert report["ok"] == (not fails)
    assert report["run_solution_timeouts"]["required"] == required
    checked, parked = stages_b._repair_pkg_timeout(pipeline, it, out, report)
    assert parked == ("limit" in second)
    assert checked["ok"] == (not parked)
    if fails and not parked:
        assert checked["verifier_timeout_fix"]["new"] == required


@pytest.mark.parametrize(("source", "enabled", "cap"), [
    ("run_solution(900)\nrun_solution(dynamic())", True, 7200),
    ("run_solution(dynamic())", True, 7200), ("run_solution(900)", True, 1100),
    ("run_solution(900)", False, 7200)])
@pytest.mark.parametrize("locked", [False, True])
def test_unfixable_timeout_parks_without_edits_or_revision(tmp_path, source, enabled, cap, locked, monkeypatch):
    pipeline, it, _ = _setup(tmp_path, source, locked=locked)
    pipeline.cfg["package"].update(autofix_verifier_timeout=enabled, max_verifier_timeout_s=cap)
    monkeypatch.setattr(stages_b, "_revision_retry", lambda *a: pytest.fail("unexpected S8 retry"))
    stages_b.pkg(pipeline, it)
    assert it.state["status"] == "parked" and it.state["reason"] == "package_invalid"
    assert not it.p("task/task.toml").exists()
    assert read_json(it.p("PKG_check.json"))["ok"] is False
    assert tomllib.loads(it.p("pkg/task.toml").read_text())["verifier"]["timeout_sec"] == 900


@pytest.mark.parametrize(("exception", "stage"), [(None, "S9R"), ("AgentTimeoutError", "S9R"),
                                                   ("VerifierTimeoutError", "S9")])
def test_locked_fix_preserves_epoch_unless_verifier_timed_out(tmp_path, monkeypatch, exception, stage):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)", locked=True, timeout=exception)
    before_runs = it.p("calib/runs.json").read_bytes()
    monkeypatch.setattr(stages_b, "_revision_retry", lambda *a: pytest.fail("unexpected S8 retry"))
    stages_b.pkg(pipeline, it)
    assert it.state["stage"] == stage and it.state["review_revisions"] == 1
    assert read_json(it.p("PKG_check.json"))["ok"]
    if stage == "S9R":
        assert it.state["status"] == "active" and it.state["review_round"] == 3
        assert it.state["calib_epoch"] == 2 and it.p("calib/runs.json").read_bytes() == before_runs
    else:
        assert it.state["calib_epoch"] == 3 and not it.state.get("calib_label")
        assert it.p("calib/epoch-2/runs.json").read_bytes() == before_runs


def test_syntax_error_is_independent_of_timeout_only_errors(tmp_path):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)")
    it.p("task/tests/test_outputs.py").write_text("def incomplete(")
    stages_b.pkg(pipeline, it)
    report = read_json(it.p("PKG_check.json"))
    assert not report["ok"] and report["run_solution_timeouts"]["invalid"]
    assert not it.p("task/task.toml").exists() and it.state["status"] == "parked"


def test_other_pkg_failure_blocks_timeout_fix(tmp_path, monkeypatch):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)")
    from tbsgen import prerelease
    monkeypatch.setattr(prerelease, "readymade_check", lambda *args: dict(ok=False, findings=["synthetic"]))
    stages_b.pkg(pipeline, it)
    assert it.state["status"] == "parked" and not it.p("task/task.toml").exists()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("changed_path", [None, "task/tests/test_outputs.py", "task/solution/solve.py",
                                          "runs/s7-1/detA/tests/truth/README_labels.json",
                                          "runs/s7-1/detA/tests/hidden/README.md"])
def test_regate_keeps_calibration_only_when_timeout_is_the_only_edit(tmp_path, changed_path, legacy):
    from tbsgen import calib
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)", locked=True)
    spec = read_json(it.p("_spec/spec.json"))
    out = package.package(pipeline.cfg, it.dir, it.p("runs/s7-1/detA"), spec, {})
    it.state["regates"] = [dict(pkg_sha=calib.pkg_sha(out))]
    it.state["rescore_pending"] = dict(reason="regate")
    if not legacy:
        it.state["rescore_pending"]["inputs_sha"] = stages_b._timeout_independent_sha(out)
    if changed_path:
        it.p(changed_path).parent.mkdir(parents=True, exist_ok=True)
        it.p(changed_path).write_text("assert 4 + 4 == 8\n")
    it.save()
    stages_b.pkg(pipeline, it)
    if changed_path and changed_path != "task/tests/test_outputs.py":
        assert it.state["status"] == "dropped" and it.state["reason"].startswith("review:gate_failed")
        assert "locked reference" in it.state["reason"]
        return
    assert ("rescore_pending" in it.state) == bool(changed_path)
    assert it.state["stage"] == ("S9" if changed_path else "S9R")
    assert it.state["calib_epoch"] == 2 and it.state["review_revisions"] == 1
    pipeline.cfg["review"]["enabled"] = False
    assert stages_b.review_gate(pipeline, it) == "no review verdict"


def test_timeout_only_regate_without_rescoring_preserves_calibration(tmp_path):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)", locked=True)
    spec = read_json(it.p("_spec/spec.json"))
    package.package(pipeline.cfg, it.dir, it.p("runs/s7-1/detA"), spec, {})
    pipeline.cfg["review"]["rescore_scoring_only"] = False
    it.state["stage"] = "S9R"
    it.save()
    before = it.p("calib/runs.json").read_bytes()
    stages_b.regate(pipeline, it, "updated verifier budget")
    assert it.state["calib_epoch"] == 2
    it.state["stage"] = "PKG"
    it.save()
    stages_b.pkg(pipeline, it)
    assert it.state["stage"] == "S9R" and it.state["calib_epoch"] == 2
    assert it.p("calib/runs.json").read_bytes() == before
    assert "rescore_pending" not in it.state


@pytest.mark.parametrize(("source", "total", "unknown"), [
    ("run_solution()", 0, True),
    ("run_solution(**options)", 0, True),
    ("run_solution(**{'timeout_sec': 60, **options})", 0, True),
    ("run_solution(**{'timeout_sec': 200, 'timeout_sec': 900})", 900, False),
    ("invoke = m.run_solution\ninvoke(*(900,))", 900, False),
    ("t = 30\nfrom options import t\nrun_solution(t)", 0, True),
    ("t = 30\n[run_solution(t) for t in options]", 0, True),
    ("if enabled:\n    t = 900\nelse:\n    t = 900\nrun_solution(t)", 900, False),
    ("t=30\nif enabled:\n    t=900\n    class C:\n        run_solution(t)", 900, False),
    ("for d in dirs:\n    t=900\n    run_solution(t, d)", 900, False),
    ("def setup():\n    t = 900\n    def go(): run_solution(t)\n    go()\n    t = 3", 0, True),
    ("t = 5\ndef go(): run_solution(t)\nt = 900\ngo()\nt = 5", 0, True),
    ("t = 5\ndef update():\n    global t\n    t = choose()\nupdate()\nrun_solution(t)", 0, True),
])
def test_timeout_bindings_and_call_forms(tmp_path, source, total, unknown):
    (tmp_path / "test_outputs.py").write_text(source)
    result = package.verifier_timeout_check(tmp_path, 900, 120)
    assert result["sum_sec"] == total
    assert bool(result["unknown"]) is unknown
    assert bool(result["errors"] or result["unknown"])


def test_s7_timeout_projection_matches_packaged_sources(tmp_path):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)")
    # The pipeline injects the metrics helper and copies the authored measure and output tests.
    it.p("task/tests/_metrics.py").write_text("run_solution(3000)")
    it.p("task/tests/scratch.py").write_text("run_solution(4000)")
    helper = package.metrics_helper_text(pipeline.cfg, {}, it.dir)
    source = package.verifier_timeout_check(it.p("task/tests"), 900, 120, packaged_helper=helper)
    out = package.package(pipeline.cfg, it.dir, it.p("runs/s7-1/detA"), {}, {})
    assert package.validate_pkg(out)["run_solution_timeouts"] == source
    assert source["sum_sec"] == 900


@pytest.mark.parametrize("interrupt", ["verifier_timeout_fixed", "partial_move", "calib_archived", "final_state"])
def test_retry_completes_timeout_fix_once(tmp_path, monkeypatch, interrupt):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)", locked=True, timeout="VerifierTimeoutError")
    runs = it.p("calib/runs.json").read_bytes()
    event = pipeline.store.event
    save = it.save

    def fail_event(name, **kwargs):
        if name == interrupt:
            raise OSError("synthetic interruption")
        event(name, **kwargs)

    def fail_move(src, dst):
        if interrupt == "partial_move":
            shutil.move(src / "runs.json", dst / "runs.json")
            raise OSError("synthetic interruption")
        move(src, dst)

    move = stages_b._move_records

    def fail_transition():
        if interrupt == "final_state" and it.state["stage"] == "S9":
            raise OSError("synthetic interruption")
        save()

    with monkeypatch.context() as m:
        m.setattr(pipeline.store, "event", fail_event)
        m.setattr(it, "save", fail_transition)
        m.setattr(stages_b, "_move_records", fail_move)
        with pytest.raises(OSError, match="synthetic interruption"):
            stages_b.pkg(pipeline, it)
    it.bump("crash:PKG")
    it.load()
    stages_b.pkg(pipeline, it)
    it.load()
    assert it.state["stage"] == "S9" and it.state["calib_epoch"] == 3
    assert it.state["requires_formal_review_sha"] == calib.pkg_sha(it.p("pkg"))
    assert not it.state.get("verifier_timeout_pending") and not it.state.get("calib_label")
    assert [record["epoch"] for record in it.state["calib_history"]] == [2]
    assert it.p("calib/epoch-2/runs.json").read_bytes() == runs


@pytest.mark.parametrize("legacy", [True, False])
def test_later_trial_timeout_archives_calibration(tmp_path, legacy):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)", locked=True)
    pipeline.ledger = None
    jd = tmp_path / "jobs" / "retry"
    write_json(jd / "result.json", dict(finished_at="today"))
    write_json(jd / "config.json", dict(job_name="retry"))
    for name, exc in (("attempt-a", "EnvironmentStartTimeoutError"), ("attempt-b", "VerifierTimeoutError")):
        write_json(jd / name / "result.json", dict(exception_info=dict(exception_type=exc)))
    rec = dict(job_dir=str(jd), job_name="retry", epoch=2)
    stages_b._ingest_run(pipeline, it, rec, "S9", jd)
    if legacy:
        runs = read_json(it.p("calib/runs.json"))
        runs[-1].pop("trial_verifier_timeouts", None)
        write_json(it.p("calib/runs.json"), runs)
    else:
        assert rec["trial_verifier_timeouts"] == {"attempt-a": False, "attempt-b": True}
        # Proof survives job-result retention.
        for path in jd.glob("*/result.json"):
            path.unlink()
    it.state["stage"] = "PKG"
    it.save()
    stages_b.pkg(pipeline, it)
    assert it.state["stage"] == "S9" and it.state["calib_epoch"] == 3


@pytest.mark.parametrize("enabled", [True, False])
def test_unapproved_source_timeout_cannot_override_configuration(tmp_path, enabled):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)")
    it.p("task/task.toml").write_text("[verifier]\ntimeout_sec=10000\n")
    pipeline.cfg["package"]["autofix_verifier_timeout"] = enabled
    stages_b.pkg(pipeline, it)
    config = tomllib.loads(it.p("pkg/task.toml").read_text())
    assert config["verifier"]["timeout_sec"] == (1200 if enabled else 900)
    assert it.state["status"] == ("active" if enabled else "parked")


@pytest.mark.parametrize("source", [
    "t = 30\n(t := 900, run_solution(t))",
    "t = 30\nignored = (flag and (t := 900))\nrun_solution(t)",
    "invoke = run_solution if flag else int\ninvoke(10)",
    "if flag:\n    invoke = run_solution\nelse:\n    invoke = int\ninvoke(10)",
    "((invoke := run_solution), invoke(10))",
    "t=30\ndef choose(value=(t:=900)): pass\nrun_solution(t)",
    "def check(invoke=run_solution): invoke(10)",
])
def test_ambiguous_timeout_or_alias_parks(tmp_path, source):
    pipeline, it, _ = _setup(tmp_path, source)
    stages_b.pkg(pipeline, it)
    assert it.state["status"] == "parked"
    assert read_json(it.p("PKG_check.json"))["run_solution_timeouts"]["unknown"]


def test_timeout_alias_is_local_to_its_function(tmp_path):
    source = ("def first():\n    invoke = run_solution\n    invoke(30)\n"
              "def second():\n    invoke = int\n    invoke('42')\n"
              "def third(invoke):\n    invoke('43')\n")
    pipeline, it, _ = _setup(tmp_path, source)
    stages_b.pkg(pipeline, it)
    assert it.state["stage"] == "S9"
    report = read_json(it.p("PKG_check.json"))["run_solution_timeouts"]
    assert report["sum_sec"] == 30 and not report["unknown"]


@pytest.mark.parametrize("snapshot_present", [True, False])
def test_regate_never_adopts_drifted_live_reference(tmp_path, snapshot_present):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)", locked=True)
    sha = calib.pkg_sha(it.p("pkg"))
    write_json(it.p("review/verdict.json"), dict(round=2, calib_epoch=2, pkg_sha=sha))
    if snapshot_present:
        shutil.copytree(it.p("pkg"), it.p("review/r2/task/environment/review/pkg"))
    for tree in ("pkg", "task"):
        it.p(tree, "solution/solve.py").write_text("# drifted reference\n")
    it.state.pop("thresholds_locked")
    it.state.update(stage="S10", status="done")
    it.save()
    stages_b.regate(pipeline, it, "updated verifier")
    it.state["stage"] = "PKG"
    it.save()
    stages_b.pkg(pipeline, it)
    assert it.state["status"] == "dropped" and it.state["reason"].startswith("review:gate_failed")
