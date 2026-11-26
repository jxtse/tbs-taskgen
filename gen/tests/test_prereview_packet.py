"""Synthetic packages and an in-process Messages transport for advisory review."""
import gzip
import io
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tbsgen import package, prereview
from tbsgen.budget import Ledger
from tbsgen.config import load_config
from tbsgen.models import FakeClient, get_client


def put(root, path, text):
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest


def test_packet_is_deterministic_bounded_and_excludes_private_files(tmp_path):
    put(tmp_path, "instruction.md", "Group samples by profile.")
    put(tmp_path, "generator/README.md", "Data sections.")
    put(tmp_path, "environment/data/README.txt", "Tabular fields.")
    put(tmp_path, "environment/data/table.csv", "profile_id,q\na,1\nb,2\nc,3\nd,4")
    put(tmp_path, "environment/data/schema.json", '{"units":"raw-value","samples":[99]}')
    put(tmp_path, "environment/data/blob.bin", "binary-value")
    put(tmp_path, "tests/hidden/README", "Hidden schema.")
    put(tmp_path, "tests/test_outputs.py", "abc\n" * prereview.TEXT_BYTES)
    put(tmp_path, "tests/measure.py", "def measure(sample): pass")
    np.save(tmp_path / "environment/data/layers.npy", np.array([[913]]))
    np.savez_compressed(tmp_path / "environment/data/grid.npz", pixels=np.zeros((3, 2)), labels=np.zeros(2))
    for rel in ["tests/truth/data.csv", "solution/solve.py", "generator/generate.py"]:
        put(tmp_path, rel, "private-content")
    put(tmp_path, "provenance.json", json.dumps({"thresholds": {
        "mean": {"direction": ">=", "threshold": 0.75, "reference": "private-content"}},
        "history": "private-content"}))
    packet = prereview.build_packet(tmp_path)
    assert packet == prereview.build_packet(tmp_path)
    enc = json.dumps(packet)
    assert len(enc.encode()) <= prereview.PACKET_BYTES
    assert not any(v in enc for v in ["private-content", "binary-value", "raw-value", "913", "solution/solve.py"])
    files = packet["files"]
    assert files["generator/README.md"]["text"] == "Data sections."
    assert files["environment/data/README.txt"]["text"] == "Tabular fields."
    assert files["tests/test_outputs.py"]["text"].endswith(prereview.TRUNCATED)
    assert files["environment/data/blob.bin"] == {"size": 12}
    assert files["environment/data/table.csv"]["columns"] == ["profile_id", "q"]
    assert files["environment/data/table.csv"]["sample"].splitlines() == ["profile_id,q", "a,1", "b,2", "c,3"]
    assert files["environment/data/schema.json"]["keys"] == {"samples": "array", "units": "string"}
    assert files["environment/data/grid.npz"]["arrays"] == {"pixels": [3, 2], "labels": [2]}
    assert files["environment/data/layers.npy"]["arrays"] == {"layers": [1, 1]}
    assert files["provenance.json"]["thresholds"] == {"mean": {"direction": ">=", "threshold": .75}}


def test_validator_binds_statement_file_quote_and_exact_locator(tmp_path):
    put(tmp_path, "instruction.md", "Group samples by profile.")
    put(tmp_path, "environment/data/table.csv", "profile_id,score\na,1")
    put(tmp_path, "tests/measure.py", "import math\nmetric = 'median'\n")
    put(tmp_path, "environment/data/schema.json", '{"cutoff": 100}')
    put(tmp_path, "environment/data/batch.json", '[1,2,3]')
    packet = prereview.build_packet(tmp_path)
    evidence = {"file": "environment/data/table.csv", "column": "profile_id", "quote": "profile_id,score"}
    good = {"category": "a", "statement_quote": "Group samples by profile.", "evidence": evidence,
            "fix": "Group samples by profile_id."}
    omission = dict(good, category="b", statement_quote="",
                    evidence={"file": "tests/measure.py", "line": 2, "quote": "metric = 'median'"})
    key = dict(omission, evidence={"file": "environment/data/schema.json", "key": "cutoff", "quote": "cutoff"})
    top_level = dict(omission, evidence={"file": "environment/data/batch.json", "key": "type", "quote": "array"})
    bad = [dict(good, statement_quote="Sort by score."), dict(good, evidence=dict(evidence, quote="not-a-header")),
           dict(good, evidence=dict(evidence, column="profile")), dict(good, evidence=dict(evidence, file="absent")),
           dict(omission, evidence=dict(omission["evidence"], line=1)), dict(key, evidence=dict(key["evidence"], key="cut")),
           {"category": "a"}, None]
    assert prereview.validate_findings(packet, [good, *bad, omission, key, top_level]) == (
        [good, omission, key, top_level], len(bad))


def test_compressed_schemas_format_notes_and_partial_csv(tmp_path):
    put(tmp_path, "instruction.md", "Describe the samples.")
    notes = put(tmp_path, "source/generator/FORMAT.md", "Format notes.")
    put(tmp_path, "source/generator/README-helper.py", "private-source")
    put(tmp_path, "environment/data/README.md", "Tabular notes.")
    put(tmp_path, "environment/data/table.csv", "id,value\ra,one\rb," + "x" * prereview.TEXT_BYTES)
    with gzip.open(tmp_path / "environment/data/counts.npy.gz", "wb") as fp:
        np.save(fp, np.array([8341]))
    packet = prereview.build_packet(tmp_path, generator_dir=notes.parent)
    assert packet["format_notes"] == {"generator/FORMAT.md": "Format notes."}
    files = packet["files"]
    assert files["environment/data/README.md"]["text"] == "Tabular notes."
    sample = files["environment/data/table.csv"]
    assert sample["sample"].splitlines() == ["id,value", "a,one"] and sample["truncated"]
    assert files["environment/data/counts.npy.gz"]["arrays"] == {"counts": [1]}
    assert "8341" not in json.dumps(packet)


@pytest.mark.parametrize("case", ["threshold", "provenance_size", "inventory", "npz_index", "parent_link"])
def test_packet_rejects_unbounded_or_out_of_scope_content_before_model(tmp_path, monkeypatch, case):
    pkg = tmp_path / "pkg"
    put(pkg, "instruction.md", "Inspect the samples.")
    generator = tmp_path / "task/generator"
    if case == "threshold":
        put(pkg, "provenance.json", json.dumps({"thresholds": {"score": {
            "direction": ">=", "threshold": {"value": "private-value"}}}}))
    elif case == "provenance_size":
        monkeypatch.setattr(prereview, "JSON_INPUT_BYTES", 32)
        put(pkg, "provenance.json", json.dumps({"other": "x" * 64}))
    elif case == "inventory":
        monkeypatch.setattr(prereview, "PACKET_BYTES", 80)
        for n in range(10):
            put(pkg, f"environment/data/{n}.bin", "data")
    elif case == "npz_index":
        monkeypatch.setattr(prereview, "NPZ_INDEX_BYTES", 512)
        data = pkg / "environment/data"
        data.mkdir(parents=True)
        np.savez(data / "samples.npz", **{f"part_{n}": np.zeros(1) for n in range(48)})
    else:
        target = put(tmp_path, "private/generator/README.md", "private-notes")
        generator.parent.symlink_to(target.parents[1], target_is_directory=True)
    call = prereview.run_prereview(pkg, lambda: pytest.fail("unexpected call"), item="sample", generator_dir=generator)
    assert call["error"] == "ValueError" and call["packet_hash"] is None


def capture_messages(monkeypatch, output, stop="end_turn"):
    sent = []

    class Transport:
        def open(self, req, timeout):
            sent.append((json.loads(req.data), req.get_header("X-api-key")))
            return io.BytesIO(json.dumps({"model": "synthetic-reviewer", "content": [{"type": "text", "text": output}],
                                          "usage": {"input_tokens": 200, "output_tokens": 100},
                                          "stop_reason": stop}).encode())

    monkeypatch.setenv("TEST_REVIEW_KEY", "test-only")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Transport())
    return sent


@pytest.mark.parametrize(("output", "stop", "failure"), [
    ('{"findings":[]}', "end_turn", None), ("not JSON", "end_turn", "JSONDecodeError"),
    ('{"findings":["\\ud800"]}', "end_turn", "UnicodeEncodeError"),
    ('{"findings":[NaN]}', "end_turn", "ValueError"), ('{"findings":[]}', "tool_use", "ValueError"),
    ("", "max_tokens", "empty")])
def test_prereview_uses_one_plain_authenticated_model_call_and_its_own_price(
        tmp_path, monkeypatch, output, stop, failure):
    put(tmp_path, "instruction.md", "Group samples by profile.")
    cfg = load_config(overrides={"budget": {"daily_usd_total": 30},
                                 "models": {"prereview": {"url": "https://example.invalid/v1/messages",
                                                          "model": "synthetic-reviewer", "key_env": "TEST_REVIEW_KEY",
                                                          "max_tokens": 1000, "effort_on_thinking_exhausted": "low"},
                                            "prices": {"synthetic-reviewer": {"in": 2, "out": 3}}}})
    ledger = Ledger(tmp_path / "cost.sqlite", cfg)
    sent = capture_messages(monkeypatch, output, stop)
    call = prereview.run_prereview(tmp_path, lambda: get_client("prereview", cfg, ledger), item="sample")
    assert call["error"] == failure
    assert call["raw"] == call["validated"] == []
    assert call["usd"] == pytest.approx(.0007)
    assert len(sent) == 1
    body, key = sent[0]
    assert body["system"] == prereview.PROMPT and key == "test-only"
    assert {"tools", "thinking", "output_config"}.isdisjoint(body)
    with ledger._db() as db:
        assert tuple(db.execute("SELECT model, stage, item, usd FROM calls").fetchone()) == (
            "synthetic-reviewer", "S9P", "sample", pytest.approx(.0007))


def test_prereview_rejects_link_before_reading(tmp_path):
    put(tmp_path, "instruction.md", "Inspect the input.")
    data = tmp_path / "environment/data"
    data.mkdir(parents=True)
    target = put(tmp_path, "tests/truth/private.csv", "private-content")
    (data / "alias.csv").symlink_to(target)
    result = prereview.run_prereview(tmp_path, lambda: pytest.fail("should not call"), item="sample")
    assert result["error"] == "ValueError" and result["packet_hash"] is None


def test_missing_prereview_price_fails_open_before_transport(tmp_path, monkeypatch):
    put(tmp_path, "instruction.md", "Inspect the input.")
    cfg = load_config(overrides={"models": {"prereview": {"model": "unpriced", "url": "https://example.invalid",
                                                           "key_env": "TEST_REVIEW_KEY"}}})
    monkeypatch.setenv("TEST_REVIEW_KEY", "test-only")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("unexpected transport"))
    ledger = Ledger(tmp_path / "cost.sqlite", cfg)
    call = prereview.run_prereview(tmp_path, lambda: get_client("prereview", cfg, ledger), item="sample")
    assert call["error"] == "config" and call["usd"] == 0 and call["raw"] == call["validated"] == []
    with ledger._db() as db:
        assert db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


def test_hidden_input_schema_supports_evidence_without_rows(tmp_path):
    put(tmp_path, "instruction.md", "Read the public inputs.")
    put(tmp_path, "tests/hidden/README.md", "private format notes")
    put(tmp_path, "tests/truth/sample.csv", "answer\ncobalt")
    put(tmp_path, "tests/hidden/nested/sample.csv", "signal,strength\nprivate-row,854321")
    put(tmp_path, "tests/hidden/types.json", '{"samples": ["hidden-value"]}')
    np.savez(tmp_path / "tests/hidden/arrays.npz", field=np.zeros((2, 3)))
    packet = prereview.build_packet(tmp_path)
    payload = json.dumps(packet)
    assert all(token not in payload for token in ("README.md", "tests/truth", "private", "854321", "hidden-value", "cobalt"))
    files = packet["files"]
    assert files["tests/hidden/types.json"]["keys"] == {"samples": "array"}
    assert files["tests/hidden/arrays.npz"]["arrays"] == {"field": [2, 3]}
    path = "tests/hidden/nested/sample.csv"
    assert files[path]["columns"] == ["signal", "strength"]
    good = dict(category="b", statement_quote="", fix="Describe strength.",
                evidence=dict(file=path, column="strength", quote="strength"))
    bad = dict(good, evidence=dict(file=path, column="strength", quote="854321"))
    assert prereview.validate_findings(packet, [good, bad]) == ([good], 1)


@pytest.mark.parametrize(("name", "truth"), [
    ("data.json", '{"mass": 1758.125}'), ("data.csv", "mass\n1758.125"),
])
def test_grader_may_read_truth_field_name(tmp_path, name, truth):
    put(tmp_path, "instruction.md", "Compute the measurement.")
    put(tmp_path, "tests/truth/" + name, truth)
    source = 'def measure(stream): return json.load(stream)["mass"]'
    put(tmp_path, "tests/measure.py", source)
    assert prereview.build_packet(tmp_path)["files"]["tests/measure.py"]["text"] == source


@pytest.mark.parametrize("grader", ["toy", "generated"])
def test_grader_control_constants_allow_prereview_call(tmp_path, grader):
    source = (Path(__file__).parent / "fixtures/toy_task/tests/measure.py").read_text() if grader == "toy" else """
import json, sys

def compute(row):
    dims = row.shape
    argc = len(sys.argv)
    max_args = 3
    expected_shape = (4, 2)
    expected_columns = ["mass", "label"]
    out_dir, elapsed, rc = metrics.run_solution(600)
    if argc > max_args or dims != expected_shape or rc != 0:
        return {}
    if out_dir is None or out_dir == "unused" or out_dir is False:
        return {}
    if row.columns != expected_columns or row.mode == "invalid" or row.status_code != 200:
        return {}
    if row["flag"] is False or row["label"] == "ignored":
        return {}
    return {"mass": row[-1]["mass"]}

if __name__ == "__main__":
    print(json.dumps(compute(load())))
"""
    put(tmp_path, "instruction.md", "Report the measured mass.")
    put(tmp_path, "tests/measure.py", source)
    put(tmp_path, "tests/truth/mass.json", '{"mass": 8173.625}')
    cfg = load_config()
    client = FakeClient(Ledger(tmp_path / "cost.sqlite", cfg), cfg, [{"stage": "S9P", "text": '{"findings":[]}'}])
    result = prereview.run_prereview(tmp_path, lambda: client, item="sample")
    assert result["error"] is None
    assert [call["stage"] for call in client.log] == ["S9P"]


def make_review_package(P, it, case):
    task = it.p("task")
    generated = it.p("generated")
    generated.mkdir()
    if case == "toy":
        fixture = Path(__file__).parent / "fixtures/toy_task"
        shutil.copytree(fixture, task, dirs_exist_ok=True)
        subprocess.run([sys.executable, task / "generator/generate.py", "1"], cwd=generated, check=True)
        put(task, "instruction.md", "Estimate amp and rate for each series_id in series.csv.")
        put(task, "tests/test_outputs.py",
            'from measure import metrics\n\ndef test_coverage():\n    assert metrics()["coverage"] == 1\n')
    else:
        put(generated, "environment/data/samples.csv", "sample_id,signal\nalpha,1.25\n")
        put(generated, "tests/truth/other.json", '{"sample_id": "alpha"}')
        if case == "shape":
            np.save(generated / "tests/truth/matrix.npy", np.array([[73.4, 95.6]]))
            body = ('    a = np.load("/app/output/matrix.npy")\n'
                    '    t = np.load("/tests/truth/matrix.npy")\n'
                    '    if a.shape != (len(t), 2):\n        return 1.0\n'
                    '    return float(np.mean(np.abs(a - t)))')
        else:
            read_row = '    with open("/app/output/report.csv") as f:\n        row = next(csv.DictReader(f))\n'
            if case == "numeric":
                body = read_row + '    expected = 8173.625\n    return abs(float(row["value"]) - expected)'
            elif case == "label":
                body = read_row + '    return float(row["phase"] == "ochre-sample")'
            else:
                put(generated, "tests/truth/phase.csv", "ochre-sample\n")
                put(task, "generator/README.md", "Example output label: ochre-sample.")
                body = read_row + ('    with open("/tests/truth/phase.csv") as f:\n'
                                   '        return float(row["phase"] == next(csv.reader(f))[0])')
        put(task, "tests/measure.py", 'import csv, json\nimport numpy as np\n\ndef compute():\n' + body
            + '\n\nif __name__ == "__main__":\n    print(json.dumps(compute()))\n')
        target = 0 if case in {"numeric", "shape"} else 1
        put(task, "tests/test_outputs.py", f"from measure import compute\n\ndef test_score():\n    assert compute() == {target}\n")
    pkg = package.package(P.cfg, it.dir, generated, {}, {})
    validation = package.validate_pkg(pkg, P.cfg)
    assert validation["ok"], validation
    return pkg


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
@pytest.mark.parametrize("case", ["toy", "shape", "numeric", "label", "headerless"])
def test_review_packages_each_send_one_request(tmp_path, monkeypatch, mode, case):
    from test_review_s9r import _env

    P, item_id, _ = _env(tmp_path, allow_submit=False)
    it = P.store.item(item_id)
    pkg = make_review_package(P, it, case)
    P.cfg["prereview"]["mode"] = mode
    P.cfg["models"]["prereview"].update(url="https://example.invalid/v1/messages", model="synthetic-reviewer",
                                        key_env="TEST_REVIEW_KEY", max_tokens=1000)
    P.cfg["models"]["prices"]["synthetic-reviewer"] = {"in": 2, "out": 3}
    sent = capture_messages(monkeypatch, '{"findings":[]}')

    P.step(it)
    [(request, _key)] = sent
    assert request["system"] == prereview.PROMPT
    packet = json.loads(request["messages"][0]["content"])
    files = packet["files"]
    assert not any(rel.startswith(("solution/", "tests/truth/")) for rel in files)
    assert files["tests/measure.py"]["text"] == (pkg / "tests/measure.py").read_text()
    if case == "toy":
        assert "S000" in files["environment/data/series.csv"]["sample"]
        assert "S000" in (pkg / "tests/truth/params.csv").read_text()
    if case == "headerless":
        assert packet["format_notes"]["generator/README.md"] == "Example output label: ochre-sample."


@pytest.mark.parametrize("header", ["signal,signal", "123,0.75", 'signal,"unterminated', '"signal\nvalue",label'])
def test_uncertain_hidden_header_skips_before_call(tmp_path, header):
    put(tmp_path, "instruction.md", "Read the inputs.")
    put(tmp_path, "tests/hidden/inputs.csv", header + "\n")
    result = prereview.run_prereview(tmp_path, lambda: pytest.fail("unexpected client"), item="sample")
    assert result["error"] and result["packet_hash"] is None


def test_header_only_hidden_csv(tmp_path):
    put(tmp_path, "instruction.md", "Read the inputs.")
    put(tmp_path, "tests/hidden/inputs.csv", "signal,strength\n")
    entry = prereview.build_packet(tmp_path)["files"]["tests/hidden/inputs.csv"]
    assert entry["columns"] == ["signal", "strength"]


def test_evidence_must_match_named_member_or_cell(tmp_path):
    put(tmp_path, "instruction.md", "Read the fields.")
    put(tmp_path, "environment/data/types.json", '{"count": 8, "label": "cobalt"}')
    put(tmp_path, "environment/data/records.tsv", 'count\tlabel\n8\tcobalt')
    packet = prereview.build_packet(tmp_path)
    base = dict(category="b", statement_quote="", fix="Name the field.")
    csv = dict(file="environment/data/records.tsv", column="count")
    key = dict(file="environment/data/types.json", key="count")
    bad = [dict(base, evidence=dict(csv, quote="cobalt")), dict(base, evidence=dict(csv, quote="label")),
           dict(base, evidence=dict(key, quote='"label":"string"'))]
    good = [dict(base, evidence=dict(csv, quote="8")), dict(base, evidence=dict(key, quote='"count":"number"'))]
    assert prereview.validate_findings(packet, bad + good) == (good, len(bad))
