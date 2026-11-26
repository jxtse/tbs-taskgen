"""Deploy handoffs for packages and previously issued calibration intents."""
import importlib.util
import os
import tomllib
from pathlib import Path

import pytest

from tbsgen import calib, gates, stages_b
from tbsgen.state import read_json, write_json
from test_calib_s9 import _env, _hook, _runs
from test_verifier_timeout_autofix import _setup


def _upgrade_from(module):
    """Exercise the deployed writer directly when its local sources are provided."""
    source = os.getenv("TBSGEN_UPGRADE_FROM_SRC")
    if not source:
        return module
    name = module.__name__.split(".")[-1]
    spec = importlib.util.spec_from_file_location(f"tbsgen._previous_{name}", Path(source) / f"{name}.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    return previous


def test_unpacked_migration_then_first_package_repairs_timeout(tmp_path):
    pipeline, it, _ = _setup(tmp_path, "run_solution(900)")
    assert not it.p("pkg").exists()
    assert _upgrade_from(stages_b).migrate_probe_report(pipeline, it)
    assert it.load().state["stage"] == "S6A_run" and it.state["thresholds_locked"]
    assert it.state["revision_data_hash"] is None and it.state["regates"][-1]["pkg_sha"] is None
    # Supply the passing S7 report and verifier log for the migrated item.
    write_json(it.p("gates/attempt-1.json"), dict(attempt=1, gates={}, probe_policy_version=gates.PROBE_POLICY_VERSION))
    it.state.update(stage="PKG", gates_attempt=1)
    it.save()
    stages_b.pkg(pipeline, it)
    assert (it.load().state["stage"], it.state["status"]) == ("S9", "active")
    assert it.state["requires_formal_review_sha"] == calib.pkg_sha(it.p("pkg"))
    assert "timeout_fix_requires_review" not in it.state
    assert read_json(it.p("PKG_check.json"))["verifier_timeout_fix"] == dict(old=900, new=1200)
    assert tomllib.loads(it.p("pkg/task.toml").read_text())["verifier"]["timeout_sec"] == 1200


@pytest.mark.parametrize("changed", ["instruction.md", "tests/check.py", "solution/solve.py"])
def test_successful_hook_returning_previous_copy_cannot_issue_receipt(tmp_path, changed):
    root = tmp_path / "prepared"
    pipeline, [iid], launches, _ = _env(tmp_path, prepared_tasks_dir=str(root), prepare_command=_hook(tmp_path))
    it = pipeline.store.item(iid)
    src = it.p("pkg", changed)
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("version one\n")
    assert calib.ensure_prepared(pipeline.cfg, it.dir)["ok"]
    src.write_text("version two\n")
    pipeline.cfg["calib"]["prepare_command"] = "test -f {out}/task.toml"
    pipeline.run(only=[iid])
    receipt = read_json(it.p("calib/prepare.json"))
    assert not receipt["ok"] and "prepared_sha" not in receipt
    assert it.load().state["status"] == "waiting" and not launches.exists()
    assert (root / iid / changed).read_text() == "version one\n"


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("keep_hook", [False, True])
def test_previous_prepared_intent_upgrades_in_place(tmp_path, manual, keep_hook):
    root = tmp_path / "prepared"
    pipeline, [iid], _, jobs = _env(tmp_path, prepared_tasks_dir=str(root), prepare_command=_hook(tmp_path),
                                    allow_submit=False)
    it = pipeline.store.item(iid)
    previous = _upgrade_from(calib)
    receipt = previous.ensure_prepared(pipeline.cfg, it.dir)
    assert receipt["ok"]
    if previous is calib:
        # Source distributions can exercise the historical receipt format without a checkout.
        receipt.pop("prepared_sha")
        write_json(it.p("calib/prepare.json"), receipt)
    assert "prepared_sha" not in receipt
    intent = previous.prepare(pipeline.cfg, it.dir, "glm", 1)
    write_json(it.p("calib/runs.json"), [intent])
    it.finish("waiting", "awaiting manual run")
    if not keep_hook:
        pipeline.cfg["calib"]["prepare_command"] = None
    if manual:
        assert stages_b.manual_launch(pipeline, it, intent["job_name"], intent["n"]) == 0
    else:
        pipeline.cfg["calib"]["allow_submit"] = True
        pipeline.run(only=[iid])
    [launched] = _runs(pipeline, iid)
    assert launched["state"] == "launched" and launched["job_name"] == intent["job_name"]
    assert (jobs / intent["job_name"] / "config.json").is_file() and launched["ledger_call"]
    assert read_json(it.p("calib/prepare.json")) == dict(receipt, prepared_sha=calib.pkg_sha(root / iid))
    assert (tmp_path / "prep_calls").read_text().count("\n") == 1


@pytest.mark.parametrize("has_target_hash", [False, True])
def test_previous_receipt_binding_stale_copy_is_rejected(tmp_path, has_target_hash):
    root = tmp_path / "prepared"
    pipeline, [iid], launches, _ = _env(tmp_path, prepared_tasks_dir=str(root), prepare_command=_hook(tmp_path))
    it = pipeline.store.item(iid)
    receipt = calib.ensure_prepared(pipeline.cfg, it.dir)
    assert receipt["ok"]
    if not has_target_hash:
        receipt.pop("prepared_sha")
    # A previous hook signed the new source hash while reusing its old destination.
    it.p("pkg/instruction.md").write_text("A different task.\n")
    receipt["pkg_sha"] = calib.pkg_sha(it.p("pkg"))
    write_json(it.p("calib/prepare.json"), receipt)
    pipeline.cfg["calib"]["prepare_command"] = None
    assert calib.stale_prepared(pipeline.cfg, it.dir, "glm", 0)
    pipeline.run(only=[iid])
    assert it.load().state["status"] == "waiting" and not launches.exists()
    assert read_json(it.p("calib/prepare.json")) == receipt
