"""PKG states the verifier time limit for PROG tasks (e2e-batch2 review: the 780 s scored cap was unstated)."""
from tbsgen import package, runner
from tbsgen.config import load_config


def test_prog_statement_gets_the_time_limit_once():
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\nDo it.\n")
    assert f"{runner.scored_budget(cfg):g} seconds" in t1 and package.TIME_NOTE_MARK in t1
    assert f"on {package.TASK_CPUS} CPU cores" in t1 and "once on each hidden input set" in t1
    assert package.instruction_text(cfg, {"form": "PROG"}, t1) == t1          # idempotent
    assert t1.count("## Verifier time limit") == 1


def test_non_prog_statement_unchanged_and_old_note_removed():
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\n")
    assert package.instruction_text(cfg, {"form": "FILE"}, t1) == "# Task\n"


def test_text_after_the_marked_block_is_kept():
    """Codex r1: a marked statement copied back into task/ and then revised must keep the revision's text."""
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\n")
    revised = t1 + "\n## Pass criteria\n- `m` <= 0.5\nRun `python /app/output/solution_cli.py <input_dir> <output_dir>`\n"
    t2 = package.instruction_text(cfg, {"form": "PROG"}, revised)
    assert "## Pass criteria" in t2 and "solution_cli.py <input_dir> <output_dir>" in t2
    assert t2.count("## Verifier time limit") == 1


def test_fractional_budget_kept_and_task_toml_cpus_match():
    cfg = load_config(None, {"package": {"verifier_overhead_sec": 120.5}})
    assert "779.5 seconds" in package.instruction_text(cfg, {"form": "PROG"}, "x\n")
    assert f"cpus = {package.TASK_CPUS}\n" in package.TASK_TOML.format(
        task_id="t", desc="d", keywords="[]", domain="", field="", form="", expert_h=1.0, verifier_sec=1.0,
        agent_sec=1.0, cpus=package.TASK_CPUS, verifier_network="", agent_network="", env_network="")



def _pkg_item(tmp_path, monkeypatch, old_ins, new_ins, rescore=True):
    """An item at PKG with a kept calibration and (optionally) a pending re-score; package.package is replaced by a
    stub that writes `new_ins` as the packaged statement."""
    import json
    from types import SimpleNamespace
    from tbsgen import stages_b
    d = tmp_path / "run" / "items" / "doi-x"
    for sub in ("_spec", "gates", "pkg", "calib", "runs/s7-1/detA"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "_spec" / "spec.json").write_text(json.dumps(dict(form="PROG")))
    (d / "gates" / "attempt-1.json").write_text(json.dumps(dict(attempt=1, gates={})))
    (d / "thresholds.json").write_text(json.dumps(dict(thresholds={})))
    (d / "pkg" / "instruction.md").write_text(old_ins)
    (d / "calib" / "runs.json").write_text(json.dumps([dict(job_name="j1", who="glm", outcome="fail")]))
    state = dict(stage="PKG", status="active", calib_label="hard", calib_epoch=0, history=[])
    if rescore:
        state["rescore_pending"] = dict(reason="regate")
    (d / "state.json").write_text(json.dumps(state))
    from tbsgen.state import Item
    it = Item(tmp_path / "run", "doi-x").load()

    def fake_package(cfg, item_dir, gen0, spec, prov):
        (item_dir / "pkg" / "instruction.md").write_text(new_ins)
        return item_dir / "pkg"
    monkeypatch.setattr(stages_b.package, "package", fake_package)
    monkeypatch.setattr(stages_b.package, "validate_pkg", lambda out, cfg=None: dict(ok=True))
    monkeypatch.setattr(stages_b, "_sha_file", lambda p: "x")
    from tbsgen import prerelease
    monkeypatch.setattr(prerelease, "readymade_check", lambda *a, **k: dict(ok=True, findings=[]))
    monkeypatch.setattr(prerelease, "signals", lambda *a, **k: dict(repos=[]))
    events = []
    P = SimpleNamespace(cfg=load_config(None), item_cost=lambda i: 0.0,
                        store=SimpleNamespace(event=lambda *a, **k: events.append((a, k))))
    stages_b.pkg(P, it)
    return Item(tmp_path / "run", "doi-x").load(), events


def test_pkg_recalibrates_when_the_statement_changed_under_a_pending_rescore(tmp_path, monkeypatch):
    """Codex r2: an old PROG package re-packaged after a scoring-only revision / regate gets the time note, so the
    kept calibration ran on another statement -> archive (epoch+1), no re-score."""
    it, ev = _pkg_item(tmp_path, monkeypatch, "# T\n", "# T\n\nnote\n")
    assert it.state["calib_epoch"] == 1 and "rescore_pending" not in it.state and "calib_label" not in it.state
    assert any(a[0] == "calib_archived" for a, _ in ev)


def test_pkg_keeps_the_rescore_when_the_statement_is_unchanged(tmp_path, monkeypatch):
    it, ev = _pkg_item(tmp_path, monkeypatch, "# T\n", "# T\n")
    assert it.state.get("calib_epoch", 0) == 0 and it.state.get("rescore_pending") and it.state["calib_label"] == "hard"
