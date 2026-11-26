"""Advisory statement revisions through the normal stage handlers, using synthetic data and model replies."""
import json
import shutil

import pytest

from tbsgen import calib, gates, package, review, runner, stages_b
from tbsgen.models import FakeClient
from tbsgen.state import read_json, write_json

from test_review_s9r import _env


FINDING = dict(category="a", statement_quote="spec", fix="Use profile_id. advisory-sentinel",
               evidence=dict(file="environment/data/table.csv", column="profile_id", quote="profile_id"))


@pytest.fixture
def flow(tmp_path, monkeypatch):
    P, item_id, _ = _env(tmp_path, allow_submit=False)
    it = P.store.item(item_id)
    P.cfg["prereview"] = dict(mode="enforce", allowed_contexts=["base", "review_revision"])
    state = {"change_data": False}

    def synthetic_generator(cfg, item_dir, mounts, *args):
        out = mounts[0][0]
        (out / "environment/data/table.csv").write_text("profile_id\n" + ("8" if state["change_data"] else "7"))
        (out / "environment/data/ReadMe.md").write_text("Sample format notes.\n")
        (out / "tests/hidden/example.json").write_text('{"rows": 2}')
        (out / "tests/truth/expected.json").write_text("[7]")
        return dict(rc=0, err="")

    def run_gates(cfg, item_dir, spec, th, attempt):
        runner.generate(cfg, item_dir / "task", 1, item_dir / "runs" / f"s7-{attempt}" / "detA")
        blocking = ["statement_clean"] if gates.statement_clean((item_dir / "task/instruction.md").read_text()) else []
        return dict(ok=not blocking, blocking=blocking, nonblocking=[], gates={"statement_clean": not blocking},
                    details={}, attempt=attempt, probe_policy_version=gates.PROBE_POLICY_VERSION)

    monkeypatch.setattr(runner.taskimage, "run", synthetic_generator)
    monkeypatch.setattr(gates, "run_gates", run_gates)
    it.goto("S7", "synthetic package")
    P.step(it)
    assert it.load().state["stage"] == "PKG"
    P.step(it)
    assert it.load().state["stage"] == "S9"
    P._clients["prereview"] = FakeClient(P.ledger, P.cfg, [
        {"stage": "S9P", "sticky": True, "text": json.dumps({"findings": [FINDING]})}])
    assert package.validate_pkg(it.p("pkg"), P.cfg)["ok"]
    return P, it, state


def _repair(P, edits):
    text = "\n".join(f"=== FILE: {rel} ===\n{body}\n=== END ===" for rel, body in edits.items())
    def reply(stage, system, user):
        assert stage == "S8" and "advisory-sentinel" in user
        return text
    client = P._clients["repair"] = FakeClient(P.ledger, P.cfg, reply)
    return client


def _expect_step(P, it, stage):
    P.step(it)
    assert it.load().state["stage"] == stage, it.state
    assert it.state["status"] in {"active", "waiting"}, it.state


def test_enforce_regates_then_calibrates_and_keeps_review_blind(flow):
    P, it, _ = flow
    edits = {"instruction.md": "Read the profile_id column.", "environment/data/ReadMe.md": "Column: profile_id."}
    client = _repair(P, edits)
    protected = calib.pkg_sha(it.p("pkg"), exclude=set(edits))
    thresholds = it.p("thresholds.json").read_bytes()
    source = calib.pkg_sha(it.p("task"), exclude=set(edits))
    for stage in ["S8", "S7", "PKG", "S9", "S9"]:
        _expect_step(P, it, stage)
    rec = read_json(it.p("prereview/e0-r1.json"))
    assert rec["revision_applied"] and rec["validated"] == [FINDING]
    assert it.state.get("review_revisions", 0) == 0 and it.state.get("calib_epoch", 0) == 0
    assert len(client.log) == len(P.client("prereview").log) == 1
    assert calib.pkg_sha(it.p("pkg"), exclude=set(edits)) == protected
    assert calib.pkg_sha(it.p("task"), exclude=set(edits)) == source
    assert it.p("thresholds.json").read_bytes() == thresholds
    for rel, body in edits.items():
        source = it.p("task") if rel == "instruction.md" else it.p("_spec/format_notes")
        assert it.p("pkg", rel).read_text().strip() == (source / rel).read_text().strip() == body
    assert [r["who"] for r in read_json(it.p("calib/runs.json"))] == ["glm"]
    task = review.build_task(P.cfg, it.dir, it.state, 1)
    for f in task.rglob("*"):
        assert "prereview" not in f.relative_to(task).parts
        assert not f.is_file() or b"advisory-sentinel" not in f.read_bytes()

    # A later Opus revision can change generator-authored format notes without an old overlay winning.
    (it.p("task/generator/generate.py")).write_text("# revised generator\n")
    output = it.p("runs/later-version")
    runner.generate(P.cfg, it.p("task"), 1, output)
    assert (output / "environment/data/ReadMe.md").read_text() == "Sample format notes.\n"


@pytest.mark.parametrize("failure", ["forbidden_patch", "gate_failure", "changed_data"])
def test_edit_rejection_restores_original_package(flow, failure):
    P, it, state = flow
    edits = {"instruction.md": "Read the profile_id column."}
    if failure == "gate_failure":
        edits["instruction.md"] += "\n<!-- internal note -->"
    if failure == "forbidden_patch":
        edits["tests/test_outputs.py"] = "assert True"
    client = _repair(P, edits)
    original = calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))
    attempt = it.state["gates_attempt"]
    _expect_step(P, it, "S8")
    if failure != "forbidden_patch":
        _expect_step(P, it, "S7")
        state.update(change_data=failure == "changed_data")
        if failure == "changed_data":
            _expect_step(P, it, "PKG")
    _expect_step(P, it, "S9")
    assert (calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))) == original
    assert it.state["gates_attempt"] == attempt
    assert not read_json(it.p("prereview/e0-r1.json"))["revision_applied"]
    _expect_step(P, it, "S9")
    assert it.state["status"] == "waiting"
    assert len(client.log) == len(P.client("prereview").log) == 1


@pytest.mark.parametrize("case", ["shadow", "model_error", "malformed", "off", "hardening", "variant"])
def test_first_run_advisory_modes(flow, case):
    P, it, _ = flow
    P.cfg["prereview"]["mode"] = case if case in {"off", "shadow"} else "enforce"
    if case in {"hardening", "variant"}:
        it.state["harden_round" if case == "hardening" else "variant_round"] = 1
        it.save()
    if case in {"model_error", "malformed"}:
        rule = {"error": "network"} if case == "model_error" else {"text": "invalid JSON"}
        P._clients["prereview"] = FakeClient(P.ledger, P.cfg, [dict(stage="S9P", **rule)])
    before = calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))
    _expect_step(P, it, "S9")
    assert it.state["status"] == "waiting"
    rec = read_json(it.p("prereview/e0-r1.json"))
    if case in {"off", "hardening", "variant"}:
        assert rec is None and not P.client("prereview").log
    else:
        assert not rec["revision_applied"] and bool(rec["error"]) == (case != "shadow")
        assert len(rec["validated"]) == (1 if case == "shadow" else 0)
    assert (calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))) == before


def test_once_per_epoch_round_and_revised_context(flow, monkeypatch):
    P, it, _ = flow
    P.cfg["prereview"]["mode"] = "shadow"
    monkeypatch.setattr(calib, "preflight_wait", lambda *a: "defer first run")
    _expect_step(P, it, "S9")
    assert it.state["status"] == "waiting" and not it.p("calib/runs.json").exists()
    it.goto("S9", "retry")
    _expect_step(P, it, "S9")
    assert len(P.client("prereview").log) == 1
    it.state.update(review_round=1, review_revisions=1)
    stages_b._archive_calibration(it)
    it.goto("S9", "recalibrate")
    _expect_step(P, it, "S9")
    assert len(P.client("prereview").log) == 2
    assert read_json(it.p("prereview/e1-r2.json"))["context"] == "review_revision"



@pytest.mark.parametrize("failure", ["unwritable", "invalid_utf8", "invalid_json"])
def test_prereview_record_error_still_prepares_calibration(flow, failure):
    P, it, _ = flow
    if failure == "unwritable":
        it.p("prereview").write_text("unavailable")
    else:
        it.p("prereview").mkdir()
        it.p("prereview/e0-r1.json").write_bytes(b"\xff" if failure == "invalid_utf8" else b"[")
    _expect_step(P, it, "S9")
    assert [r["who"] for r in read_json(it.p("calib/runs.json"))] == ["glm"]
    assert not P.client("prereview").log


def test_base_revision_requires_fresh_prepared_copy(flow):
    P, it, _ = flow
    _repair(P, {"instruction.md": "Read the profile_id column."})
    prepared = it.root / "prepared"
    P.cfg["calib"]["prepared_tasks_dir"] = str(prepared)
    shutil.copytree(it.p("pkg"), prepared / it.item_id)
    receipt = it.p("calib/prepare.json")
    write_json(receipt, {"ok": True, "pkg_sha": calib.pkg_sha(it.p("pkg"))})
    for stage in ["S8", "S7", "PKG", "S9", "S9"]:
        _expect_step(P, it, stage)
    assert not it.p("calib/runs.json").exists()
    assert "current pkg" in it.state["reason"]
    shutil.copytree(it.p("pkg"), prepared / it.item_id, dirs_exist_ok=True)
    write_json(receipt, {"ok": True, "pkg_sha": calib.pkg_sha(it.p("pkg"))})
    it.goto("S9", "refreshed prepared copy")
    _expect_step(P, it, "S9")
    [run] = read_json(it.p("calib/runs.json"))
    assert run["task"] == str(prepared / it.item_id)
    assert len(P.client("prereview").log) == 1
    receipt.unlink()
    with pytest.raises(ValueError, match="current pkg"):
        stages_b.manual_launch(P, it, run["job_name"], run["n"])


@pytest.mark.parametrize("interruption", ["request", "rollback", "commit"])
def test_interrupted_revision_is_resumable(flow, monkeypatch, interruption):
    from tbsgen import prereview_flow

    P, it, _ = flow
    edited = "Read the profile_id column."
    client = _repair(P, {"instruction.md": edited})
    original = it.p("pkg/instruction.md").read_bytes()
    write_json(it.p("gates/difficulty.json"), {"version": "original"})
    _expect_step(P, it, "S8")
    if interruption == "request":
        key = it.p("prereview/e0-r1.json")
        write_json(key, dict(read_json(key), status="requesting"))
    else:
        _expect_step(P, it, "S7")
        if interruption == "rollback":
            write_json(it.p("gates/difficulty.json"), {"version": "candidate"})
            with monkeypatch.context() as m:
                def interrupt(*args):
                    raise KeyboardInterrupt()
                m.setattr(prereview_flow, "_rollback", interrupt)
                with pytest.raises(KeyboardInterrupt):
                    prereview_flow.reject(it, "incomplete candidate")
        else:
            _expect_step(P, it, "PKG")
            with monkeypatch.context() as m:
                def interrupt(item, rec, **kwargs):
                    raise KeyboardInterrupt()
                m.setattr(prereview_flow, "_finish", interrupt)
                with pytest.raises(KeyboardInterrupt):
                    P.step(it)
    _expect_step(P, it, "S9")
    rec = read_json(it.p("prereview/e0-r1.json"))
    assert rec["revision_applied"] == (interruption == "commit")
    if interruption == "commit":
        assert it.p("pkg/instruction.md").read_text().strip() == edited
    else:
        assert it.p("pkg/instruction.md").read_bytes() == it.p("task/instruction.md").read_bytes() == original
        assert read_json(it.p("gates/difficulty.json")) == {"version": "original"}
    _expect_step(P, it, "S9")
    assert len(client.log) == (interruption != "request")
    assert len(P.client("prereview").log) == 1 and it.p("calib/runs.json").exists()


@pytest.mark.parametrize("reply", ["applied", "forbidden", "exhausted"])
def test_revision_cost_includes_rejected_and_billed_failed_calls(flow, reply):
    P, it, _ = flow
    model = "fixture-reviser"
    P.cfg["models"]["prices"][model] = {"out": 3}
    P.cfg["models"]["teacher"].update(effort="high", effort_on_thinking_exhausted="low")
    path = "instruction.md" if reply == "applied" else "tests/measure.py"
    answer = {"text": f"=== FILE: {path} ===\nRead profile_id.\n=== END ===", "usage": {"output_tokens": 400}}
    if reply == "exhausted":
        answer.update(error="empty", stop="max_tokens")
    client = FakeClient(P.ledger, P.cfg, [dict(answer, stage="S8")], name=model)
    client.real = True  # price the scripted reply through the normal ledger
    P._clients["repair"] = client
    _expect_step(P, it, "S8")
    for stage in (["S7", "PKG", "S9"] if reply == "applied" else ["S9"]):
        _expect_step(P, it, stage)
    rec = read_json(it.p("prereview/e0-r1.json"))
    assert rec["revision_usd"] == pytest.approx(0.0012)
    assert rec["revision_applied"] == (reply == "applied") and len(client.log) == 1
