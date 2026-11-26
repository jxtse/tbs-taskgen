"""Advisory statement revisions through the normal stage handlers, using synthetic data and model replies."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tbsgen import calib, gates, package, prereview, review, runner, stages_b
from tbsgen.models import FakeClient
from tbsgen.state import read_json, write_json

from test_review_s9r import _env


FINDING = dict(category="a", statement_quote="spec", fix="Use profile_id. advisory-sentinel",
               evidence=dict(file="environment/data/table.csv", column="profile_id", quote="profile_id"))


@pytest.fixture
def flow(tmp_path, monkeypatch, request):
    P, item_id, _ = _env(tmp_path, allow_submit=False)
    it = P.store.item(item_id)
    P.cfg["prereview"] = dict(mode="enforce", allowed_contexts=["base", "review_revision"])
    state = {"change_data": False}
    toy = getattr(request, "param", None) == "toy"

    def synthetic_generator(cfg, item_dir, mounts, *args):
        out = mounts[0][0]
        if toy:
            subprocess.run([sys.executable, "generator/generate.py", "1"], cwd=out, check=True)
            return dict(rc=0, err="")
        (out / "environment/data/table.csv").write_text("profile_id\n" + ("8" if state["change_data"] else "7"))
        (out / "environment/data/readmeasurements.txt").write_text("7\n")
        (out / "environment/data/ReadMe.md").write_text("Sample format notes.\n")
        (out / "tests/hidden/example.json").write_text('{"rows": 2}')
        (out / "tests/truth/expected.json").write_text('[1758.125, "reference-phase"]')
        return dict(rc=0, err="")

    def run_gates(cfg, item_dir, spec, th, attempt):
        runner.generate(cfg, item_dir / "task", 1, item_dir / "runs" / f"s7-{attempt}" / "detA")
        blocking = ["statement_clean"] if gates.statement_clean((item_dir / "task/instruction.md").read_text()) else []
        return dict(ok=not blocking, blocking=blocking, nonblocking=[], gates={"statement_clean": not blocking},
                    details={}, attempt=attempt, probe_policy_version=gates.PROBE_POLICY_VERSION)

    monkeypatch.setattr(runner.taskimage, "run", synthetic_generator)
    monkeypatch.setattr(gates, "run_gates", run_gates)
    if toy:
        shutil.copytree(Path(__file__).parent / "fixtures/toy_task", it.p("task"), dirs_exist_ok=True)
        it.p("task/instruction.md").write_text("Estimate amp and rate for each id in series.csv.")
    else:
        it.p("task/tests/measure.py").write_text("COL = 'profile_id'\n")
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


def _review_history(P, it):
    task = review.build_task(P.cfg, it.dir, it.state, 1)
    return read_json(task / "environment/review/records/pipeline_history.json")["history"]


@pytest.mark.parametrize("flow", ["toy"], indirect=True)
def test_toy_column_correction_is_installed(flow):
    P, it, _ = flow
    corrected = "Estimate amp and rate for each series_id in series.csv."
    finding = dict(FINDING, statement_quote="each id", fix="Use series_id. advisory-sentinel",
                   evidence=dict(file="environment/data/series.csv", column="series_id", quote="series_id"))
    P._clients["prereview"] = FakeClient(P.ledger, P.cfg, [
        {"stage": "S9P", "text": json.dumps({"findings": [finding]})}])
    _repair(P, {"instruction.md": corrected})
    for stage in ["S8", "S7", "PKG", "S9", "S9"]:
        _expect_step(P, it, stage)
    record = read_json(it.p("prereview/e0-r1.json"))
    assert record["validated"] == [finding] and record["revision_applied"]
    assert it.p("pkg/instruction.md").read_text().strip() == corrected
    assert [r["who"] for r in read_json(it.p("calib/runs.json"))] == ["glm"]


@pytest.mark.parametrize("document", ["instruction.md", "environment/data/FORMAT.md"])
def test_headerless_candidate_is_protected_in_enforce(flow, document):
    P, it, _ = flow
    it.p("pkg/environment/data/candidates.csv").write_text("ochre-sample\nsilver-sample\n")
    it.p("pkg/environment/data/FORMAT.md").write_text("candidates.csv has no header.\n")
    it.p("pkg/tests/truth/phase.json").write_text('{"phase": "ochre-sample"}')
    _repair(P, {document: 'The expected phase for the supplied signals is "ochre-sample".'})
    original = calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))
    _expect_step(P, it, "S8")
    _expect_step(P, it, "S9")
    record = read_json(it.p("prereview/e0-r1.json"))
    assert record["revision_error"] == "edit_leak" and not record["revision_applied"]
    assert (calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))) == original


@pytest.mark.parametrize("description", ["is headerless", "has no header"])
def test_documented_headerless_table_cannot_exempt_grader_literal(tmp_path, description):
    data = tmp_path / "environment/data"
    data.mkdir(parents=True)
    (data / "candidates.csv").write_text("ochre-sample\nsilver-sample\n")
    (data / "signals.csv").write_text("signal_id,value\nblue,7\n")
    (data / "FORMAT.md").write_text(
        f"candidates.csv {description}.\nsignals.csv header: `signal_id,value`\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/measure.py").write_text("PHASE = 'ochre-sample'\nFIELD = 'signal_id'\n")
    (tmp_path / "instruction.md").write_text("Choose a phase.")
    assert prereview.edit_adds_answer(tmp_path, {"instruction.md": "Use ochre-sample."})
    assert not prereview.edit_adds_answer(tmp_path, {"instruction.md": "Read signal_id."})


def test_public_schema_exemptions_respect_row_and_truth(tmp_path):
    data = tmp_path / "environment/data"
    data.mkdir(parents=True)
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    (truth / "labels.json").write_text('["ochre-sample", "silver-sample"]')
    (tmp_path / "tests/measure.py").write_text("FIELD = 'signal_id'\n")
    (tmp_path / "instruction.md").write_text("Choose a phase.")
    table = data / "signals.tsv"
    generator = tmp_path / "generator"
    generator.mkdir()
    notes = generator / "README.md"
    patch = {"instruction.md": "Read signal_id."}
    table.write_text("signal_id\tochre-sample\n")
    assert prereview.edit_adds_answer(tmp_path, patch)
    notes.write_text("signals.tsv header: `signal_id,ochre-sample`")
    assert not prereview.edit_adds_answer(tmp_path, patch)
    assert not prereview.edit_adds_answer(tmp_path, {"instruction.md": "Use ochre-sample."})
    assert prereview.edit_adds_answer(tmp_path, {"instruction.md": "Use silver-sample."})
    table.write_text("signal_id\t8173.625\n")
    notes.write_text("signals.tsv header: `signal_id,8173.625`")
    assert prereview.edit_adds_answer(tmp_path, patch)


@pytest.mark.parametrize("suffix", ["csv", "tsv"])
def test_private_table_headers_are_schema_and_data_remain_protected(tmp_path, suffix):
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    (truth / f"labels.{suffix}").write_text("phase\nstable\n")
    (truth / f"empty.{suffix}").write_text("series_id\n")
    data = tmp_path / "environment/data"
    data.mkdir(parents=True)
    (data / "series.csv").write_text("series_id,value\n")
    (data / "FORMAT.md").write_text("series.csv header: `series_id,value`")
    doc = tmp_path / "instruction.md"
    doc.write_text("Estimate the result.")
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "Read phase for each series_id."})
    assert prereview.edit_adds_answer(tmp_path, {doc.name: "The result is stable."})
    (data / "FORMAT.md").unlink()
    (data / "series.csv").write_text("series_id,value\nS01,1\n")
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "Read series_id."})
    (data / "FORMAT.md").write_text("series.csv has no header.")
    assert prereview.edit_adds_answer(tmp_path, {doc.name: "Read series_id."})


def test_enforce_regates_then_calibrates_and_keeps_review_blind(flow):
    P, it, _ = flow
    # The grader's column reference agrees with the public header.
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


@pytest.mark.parametrize(("file", "source", "answer", "document"), [
    ("tests/measure.py", "expected = 8173.625", "`8173.625`", "instruction.md"),
    ("tests/test_outputs.py", "label = 'ochre-sample'", "`ochre-sample`", "instruction.md"),
    ("tests/truth/phase.csv", "ochre-sample", "`ochre-sample`", "environment/data/ReadMe.md"),
    ("tests/measure.py", "expected = 8173.625", "8,173.625", "instruction.md"),
    ("tests/truth/mass.json", "8173.625", "8.173625 × 10^3", "environment/data/FORMAT.md"),
    ("tests/test_outputs.py", "label = 'ochre-sample'", "_ochre-sample_", "instruction.md"),
    ("tests/truth/phase.csv", "ochre-sample", "_ochre-sample_", "environment/data/FORMAT.md"),
    ("tests/truth/phase.txt", '"ochre-sample"', "ochre-sample", "environment/data/FORMAT.md"),
])
def test_enforce_discards_answer_added_by_validated_finding(flow, file, source, answer, document):
    P, it, _ = flow
    with it.p("pkg", file).open("a") as f:
        f.write(source + "\n")
    if document == "environment/data/FORMAT.md":
        it.p("pkg", document).write_text("Sample format notes.\n")
    finding = dict(FINDING, fix=f"Include {answer}. advisory-sentinel")
    P._clients["prereview"] = FakeClient(P.ledger, P.cfg, [
        {"stage": "S9P", "text": json.dumps({"findings": [finding]})}])
    client = _repair(P, {document: f"Use {answer}."})
    original = calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))
    _expect_step(P, it, "S8")
    _expect_step(P, it, "S9")
    rec = read_json(it.p("prereview/e0-r1.json"))
    assert rec["validated"] == [finding] and rec["revision_error"] == "edit_leak"
    assert not rec["revision_applied"] and len(client.log) == len(P.client("prereview").log) == 1
    assert (calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))) == original
    _expect_step(P, it, "S9")
    assert [r["who"] for r in read_json(it.p("calib/runs.json"))] == ["glm"]


def test_edit_compares_labels_across_markup(tmp_path):
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    (truth / "phase.txt").write_text(r'"ochre\u002dsample"' + "\n'green phase'\n")
    statement = tmp_path / "instruction.md"
    statement.write_text("Compare ochre-samples.")
    for body in ("_ochre-sample_", "*ochre-sample*", "**ochre-sample**", "`ochre-sample`",
                 "~~ochre-sample~~", "[‘ochre-sample’]", "green phase"):
        assert prereview.edit_adds_answer(tmp_path, {"instruction.md": f"Use {body}."}), body
    statement.write_text("Use _ochre-sample_.")
    assert not prereview.edit_adds_answer(tmp_path, {"instruction.md": 'Use "ochre-sample".'})


def test_edit_distinguishes_identifiers_from_wrapped_labels(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/measure.py").write_text("LABEL = 'stable'\nFIELD = 'rate'\n")
    doc = tmp_path / "instruction.md"
    original = "Read stable_signal."
    doc.write_text(original)
    for token in ("stable", "_stable_", "**stable**", "`stable`"):
        assert prereview.edit_adds_answer(tmp_path, {doc.name: original + f" The regime is {token}."})
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "Read /app/data/decay_rate_series.csv."})
    doc.write_text("The regime is _stable_.")
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "The regime is stable."})


def test_edit_compares_answer_tokens_per_document(tmp_path):
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    (truth / "values.tsv").write_text("first-label\t-12.345\t1.20\n")
    (truth / "values.json").write_text('{"nested": [{"label": "second label"}]}')
    (truth / "labels.txt").write_text("third-label QXZ\nthe R 0.012 2300")
    np.save(truth / "label.npy", np.array("scalar-label"))
    np.savez_compressed(truth / "arrays.npz", small=np.full(64, "small-label"), large=np.full(65, "large-label"))
    (tmp_path / "instruction.md").write_text("Report first-label at -12.345.")
    notes = "environment/data/FORMAT.md"
    (tmp_path / notes).parent.mkdir(parents=True)
    (tmp_path / notes).write_text("Note the format.")
    for token in ("first-label", "-1.2345e1", "1.2", "second label", "third-label", "QXZ", "scalar-label", "small-label"):
        assert prereview.edit_adds_answer(tmp_path, {notes: f"Use `{token}`."}), token
    assert not prereview.edit_adds_answer(tmp_path, {
        "instruction.md": "Use first-label and -1.2345e1.",
        notes: "Use the R column at 0.012 or 2300. Compare large-label with third-labels.",
    })


@pytest.mark.parametrize("rendered_answer", ["1758.125", r"1.758125\times10^{3}"])
def test_enforce_compares_rendered_statement_before_install(flow, monkeypatch, rendered_answer):
    P, it, _ = flow
    _repair(P, {"instruction.md": "Read the profile_id column."})
    original = calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))
    for stage in ["S8", "S7", "PKG"]:
        _expect_step(P, it, stage)
    render = package.instruction_text
    monkeypatch.setattr(package, "instruction_text", lambda *args: render(*args) + f"Use {rendered_answer}.\n")
    _expect_step(P, it, "S9")
    assert read_json(it.p("prereview/e0-r1.json"))["revision_error"] == "edit_leak"
    assert (calib.pkg_sha(it.p("pkg")), calib.pkg_sha(it.p("task"))) == original


@pytest.mark.parametrize("spelling", [
    "8,173.625", "8\u2009173.625", "8\u202f173.625", r"8\,173.625",
    "8.173625 × 10^3", "8.173625x 10^3", "8.173625·10³", "8.173625 × 10^{3}",
    "8.173625e3", "8.173625E+03", r"\(8.173625\times10^{3}\)",
])
def test_numeric_edit_compares_normalized_values_in_each_document(tmp_path, spelling):
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    (truth / "mass.json").write_text('{"mass": 8173.625}')
    doc = tmp_path / "instruction.md"
    doc.write_text("Estimate the mass.")
    assert prereview.edit_adds_answer(tmp_path, {doc.name: f"Use {spelling}."})
    doc.write_text(f"The measured mass is {spelling}.")
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "Use 8173.625."})


def test_numeric_edit_uses_answer_sign_and_precision(tmp_path):
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    mass = truth / "mass.json"
    mass.write_text("-0.00817")
    doc = tmp_path / "instruction.md"
    doc.write_text("Estimate the mass.")
    for spelling in ("−8.17 × 10^{−3}", "−8.17·10⁻³", "−8.17E−03", "-0.008171"):
        assert prereview.edit_adds_answer(tmp_path, {doc.name: f"Use {spelling}."})
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "Use 0.00817."})
    mass.write_text("-0.00817000")
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: "Use -0.008171."})


@pytest.mark.parametrize(("answer", "original", "revised"), [
    ("8173.625", "Vector [8,173.625].", "Use 8173.625."),
    ("8173.625", "Estimate the mass.", "Use [8,173.625]."),
    ("173.625", "Estimate the mass.", "Use [8,173.625]."),
    ("8173.625", "Read id_8173.625.", "Use 8173.625."),
    ("8173.625", "Read id_8173.625.", "Use __8173.625__."),
    ("173.625", "Estimate the mass.", "```csv\nid,mass_mg\n7,173.625\n```"),
    ("173.625", "Estimate the mass.", "id,mass_mg\n7,173.625"),
    ("173.625", "Estimate the mass.", "trial A,7,173.625"),
    ("173.625", "Estimate the mass.", "7,173.625"),
    ("7173.625", "Estimate the mass.", "7,173.625"),
    ("7173.625", "Estimate the mass.", "Please, read carefully.\n7,173.625"),
    ("7173.625", "id,mass_mg\n7,173.625", "Use 7173.625."),
])
def test_numeric_edit_keeps_ambiguous_readings_separate(tmp_path, answer, original, revised):
    truth = tmp_path / "tests/truth"
    truth.mkdir(parents=True)
    (truth / "mass.json").write_text(answer)
    doc = tmp_path / "instruction.md"
    doc.write_text(original)
    assert not prereview.edit_adds_answer(tmp_path, {doc.name: original + " Report the mass."})
    assert prereview.edit_adds_answer(tmp_path, {doc.name: revised})


def test_legacy_partial_install_cannot_supply_its_own_baseline(flow):
    P, it, _ = flow
    _expect_step(P, it, "S8")
    record = it.p("prereview/e0-r1.json")
    approved = {"instruction.md": "Use 1758.125.", "environment/data/ReadMe.md": "Use profile_id."}
    write_json(record, dict(read_json(record), status="committing", approved=approved, patch=approved))
    it.p("pkg/instruction.md").write_text(approved["instruction.md"])
    P.step(it)
    assert it.load().state["status"] == "parked" and "restoration" in it.state["reason"]
    assert it.p("pkg/environment/data/ReadMe.md").read_text() == "Sample format notes.\n"


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


@pytest.mark.parametrize("applied", [True, False])
def test_prereview_transitions_are_private_and_preview_preserves_version(flow, applied):
    P, it, _ = flow
    original_history = [{k: h[k] for k in ("stage", "result", "to", "reason") if k in h}
                        for h in it.state["history"]]
    client = _repair(P, {"instruction.md" if applied else "tests/measure.py": "Use profile_id."})
    for stage in (["S8", "S7", "PKG", "S9"] if applied else ["S8", "S9"]):
        _expect_step(P, it, stage)
    assert _review_history(P, it) == original_history
    review.build_task(P.cfg, it.dir, it.state, 3)
    assert review.next_round(it.dir, it.state) == 1
    _expect_step(P, it, "S9")
    assert it.state["status"] == "waiting"
    assert len(client.log) == len(P.client("prereview").log) == 1


def test_watch_resumes_base_prereview_transaction(flow):
    P, it, _ = flow
    _repair(P, {"instruction.md": "Use profile_id."})
    _expect_step(P, it, "S8")
    for stage in ["S7", "PKG", "S9", "S9"]:
        action = stages_b.watch_tick(P, [it.item_id], max_steps=1)
        assert action["open"][it.item_id]["stage"] == stage
    assert it.load().state["status"] == "waiting" and it.p("calib/runs.json").is_file()


@pytest.mark.parametrize("kind", ["harden", "variant"])
def test_formal_revision_after_hardening_has_its_own_prereview(flow, kind):
    P, it, _ = flow
    it.state.update(harden_round=1, variant_round=int(kind == "variant"), review_round=1)
    it.state["history"].append(dict(result=kind, stage="S9", to="S5G", round=1))
    it.goto("S8", "review revision")
    d = it.p("repair/review-1")
    write_json(d / "patch.json", {"instruction.md": "Revised spec."})
    stages_b._finish_revision(P, it, dict(dir=str(d), epoch=1, n=1, round=1))
    it.goto("S9", "synthetic rebuilt version")
    _expect_step(P, it, "S8")
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


@pytest.mark.parametrize("origin", ["shadow", "enforce", "timeout_fix"])
def test_new_flow_parks_before_unreviewed_release(flow, tmp_path, origin):
    from test_review_s9r import _to_label

    P, it, _ = flow
    P.cfg["review"]["enabled"] = False
    P.cfg["calib"]["allow_submit"] = True
    if origin == "timeout_fix":
        P.cfg["prereview"]["mode"] = "off"
        it.p("task/tests/measure.py").write_text("run_solution(900)\n")
        it.goto("PKG", "package a new task")
        _expect_step(P, it, "S9")
        assert read_json(it.p("PKG_check.json"))["verifier_timeout_fix"]
    else:
        P.cfg["prereview"]["mode"] = origin
        if origin == "enforce":
            _repair(P, {"instruction.md": "Read the profile_id column."})
    _to_label(P, it.item_id, tmp_path / "jobs")
    it.load()
    assert it.state["calib_label"] == "hard"
    assert it.state["status"] == "parked"
    assert not (P.store.root / "release/manifest.jsonl").exists()
    assert it.state["requires_formal_review_sha"] == calib.pkg_sha(it.p("pkg"))


def test_timeout_fix_rejects_reused_prepared_copy(flow):
    P, it, _ = flow
    P.cfg["prereview"]["mode"] = "off"
    root = it.root / "prepared"
    copy = ('mkdir -p {out} && cp -R {pkg}/. {out}/ && '
            'echo \'docker_image = "fixture-template"\' >> {out}/task.toml')
    P.cfg["calib"].update(allow_submit=True, prepared_tasks_dir=str(root), prepare_command=copy)
    for directory in ("task", "pkg"):
        it.p(directory, "tests/measure.py").write_text("run_solution(900)\n")
    original = calib.ensure_prepared(P.cfg, it.dir)
    assert original["ok"]
    P.cfg["calib"]["prepare_command"] = "test -f {out}/task.toml"
    it.goto("PKG", "fix verifier timeout")
    _expect_step(P, it, "S9")
    assert read_json(it.p("PKG_check.json"))["verifier_timeout_fix"] == dict(old=900, new=1200)
    _expect_step(P, it, "S9")
    receipt = read_json(it.p("calib/prepare.json"))
    assert not receipt["ok"] and "prepared_sha" not in receipt
    assert not it.p("calib/runs.json").exists()

    # A receipt already signed by an older deployment cannot authorize the stale copy either.
    write_json(it.p("calib/prepare.json"), dict(original, pkg_sha=calib.pkg_sha(it.p("pkg"))))
    assert calib.stale_prepared(P.cfg, it.dir, "glm", 0)
    P.cfg["calib"]["prepare_command"] = copy
    it.goto("S9", "retry with the corrected preparation hook")
    _expect_step(P, it, "S9")
    [run] = read_json(it.p("calib/runs.json"))
    assert run["state"] == "launched" and run["task"] == str(root / it.item_id)


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
    P.cfg["calib"]["prepare_command"] = 'cp -R {pkg}/. {out}/'
    assert calib.ensure_prepared(P.cfg, it.dir)["ok"]
    P.cfg["calib"]["prepare_command"] = None
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


@pytest.mark.parametrize(("stage", "retry"), [("S9P", True), ("S9P", False), ("S8", False)])
def test_resume_recovers_settled_cost_by_call_id(flow, monkeypatch, stage, retry):
    from tbsgen.report import prereview_summary

    P, it, _ = flow
    role = "repair" if stage == "S8" else "prereview"
    model = f"fixture-{role}"
    P.cfg["models"]["prices"][model] = {"out": 4}
    reply = dict(stage=stage, text='{"findings":[]}', usage={"output_tokens": 500})
    script = [dict(reply, error="stream"), reply] if retry else [reply]
    client = FakeClient(P.ledger, P.cfg, script, name=model)
    client.real = True
    P._clients[role] = client
    if stage == "S8":
        _expect_step(P, it, "S8")
    finish = P.ledger.finish

    def crash_after_settlement(*args, **kwargs):
        usd = finish(*args, **kwargs)
        if len(client.log) == len(script):
            raise KeyboardInterrupt
        return usd

    with monkeypatch.context() as m:
        m.setattr(P.ledger, "finish", crash_after_settlement)
        m.setattr(prereview.time, "sleep", lambda _: None)
        with pytest.raises(KeyboardInterrupt):
            P.step(it)
    record_path = it.p("prereview/e0-r1.json")
    field = "revision_usd" if stage == "S8" else "usd"
    rec = read_json(record_path)
    assert rec[field] == 0
    if stage == "S9P" and not retry:
        rec.pop("call_ids")  # Legacy interruptions only persisted call_id.
        write_json(record_path, rec)
    P.cfg["prereview"]["mode"] = "off"  # disabling future calls still settles the in-flight check/edit
    _expect_step(P, it, "S9")
    assert read_json(record_path)[field] == pytest.approx(.002 * len(script))
    assert prereview_summary(it.root)["enforce"]["usd"] == pytest.approx(.002 * len(script))
    assert len(client.log) == len(script)


def test_unpriced_restricted_reviser_is_refused(flow):
    P, it, _ = flow
    client = _repair(P, {"instruction.md": "Use profile_id."})
    client.name = "unpriced-fixture"
    client.real = True
    _expect_step(P, it, "S8")
    _expect_step(P, it, "S9")
    rec = read_json(it.p("prereview/e0-r1.json"))
    assert rec["revision_error"] == "S8: config" and not rec["revision_applied"]
    assert not client.log and "revision_call_id" not in rec


@pytest.mark.parametrize("stage", ["S9P", "S8"])
def test_report_counts_settled_call_after_record_write_error(flow, monkeypatch, stage):
    from tbsgen import prereview_flow
    from tbsgen.report import prereview_summary

    P, it, _ = flow
    role = "repair" if stage == "S8" else "prereview"
    if stage == "S8":
        _expect_step(P, it, "S8")
    model = f"fixture-{role}"
    P.cfg["models"]["prices"][model] = {"out": 4}
    client = FakeClient(P.ledger, P.cfg, [dict(stage=stage, text='{"findings":[]}', usage={"output_tokens": 500})],
                        name=model)
    client.real = True
    P._clients[role] = client
    field = "revision_usd" if stage == "S8" else "usd"
    failed = False

    def fail_once(path, record):
        nonlocal failed
        if not failed and record.get(field):
            failed = True
            raise OSError("synthetic record write failure")
        write_json(path, record)

    monkeypatch.setattr(prereview_flow, "write_json", fail_once)
    _expect_step(P, it, "S9")
    if stage == "S8":
        _expect_step(P, it, "S9")
    assert failed and it.state["status"] == "waiting" and it.p("calib/runs.json").exists()
    assert len(client.log) == 1
    assert prereview_summary(it.root)["enforce"]["usd"] == pytest.approx(.002)


def test_prereview_rejects_readme_prefix_data_edit(flow):
    P, it, _ = flow
    data = "environment/data/readmeasurements.txt"
    _repair(P, {data: "8"})
    original = calib.pkg_sha(it.p("pkg"))
    _expect_step(P, it, "S8")
    _expect_step(P, it, "S9")
    assert it.p("pkg", data).read_text() == "7\n" and calib.pkg_sha(it.p("pkg")) == original
    assert not read_json(it.p("prereview/e0-r1.json"))["revision_applied"]
