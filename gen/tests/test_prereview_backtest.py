"""Offline snapshot replay and mode reporting on a synthetic two-round item."""
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tbsgen import cli
from tbsgen.calib import pkg_sha
from tbsgen.report import round_verdict
from tbsgen.state import read_json, write_json


ISSUE = {"id": "C1", "severity": "blocking", "category": "a_consistency", "title": "Group column",
         "evidence": ["environment/data/groups.csv:1 lists `profile_id`"]}
FINDING = {"category": "a", "statement_quote": "Group by profile.", "fix": "Group by profile_id.",
           "evidence": {"file": "environment/data/groups.csv", "column": "profile_id", "quote": "profile_id"}}


def test_snapshot_backtest_and_flip_rate(tmp_path):
    run = tmp_path / "experiment"
    item = run / "items/sample"
    directory = item / "review/r1"
    bundle = directory / "task/environment/review"
    pkg = bundle / "pkg"
    (pkg / "environment/data").mkdir(parents=True)
    (pkg / "environment/data/groups.csv").write_text("profile_id,value\na,5\n")
    (pkg / "instruction.md").write_text("Group by profile.")
    (directory / "task/instruction.md").write_text("Review the whole package.")
    (item / "pkg").mkdir()
    (item / "pkg/instruction.md").write_text("Revised statement.")
    write_json(directory / "verdict.json", {"pkg_sha": pkg_sha(pkg), "_normalized": {"verdict": "revise",
               "issues": [ISSUE, dict(ISSUE, id="C2", severity="minor")]}})
    # The fake answers only the snapshot prompt, then reverses its blocking decision on the repeat.
    script = tmp_path / "fake.json"
    write_json(script, [{"stage": "S9P", "match": "Group by profile.", "text": json.dumps({"findings": findings})}
                        for findings in ([FINDING], [])])
    (run / "config.toml").write_text("[models.roles]\nprereview = " + json.dumps("fake:" + str(script)) + "\n")
    write_json(item / "prereview/e0-r1.json", {"validated": ["live record"]})
    original = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}

    out = tmp_path / "results"
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/prereview_backtest.py"),
           "--run", str(run), "--out", str(out), "--round", "1", "--items", "sample", "--repeat", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    summary = json.loads(proc.stdout)
    assert summary["items"] == summary["repeat_comparisons"] == summary["blocking_flip_rate"] == 1
    assert summary["attempts"] == 2 and summary["errors"] == 0
    with (out / "comparison.csv").open(newline="") as f:
        first, second = csv.DictReader(f)
    assert first["opus_verdict"] == "revise"
    assert first["reviewer_model_source"] == "configured"
    assert json.loads(first["opus_blocking_ab"]) == [
        {k: ISSUE[k] for k in ("id", "category", "title", "evidence")}]
    assert json.loads(first["prereview_validated"]) == [FINDING]
    assert json.loads(first["tentative_matches"]) == [{"finding": 1, "opus_issue": "C1"}]
    assert second["blocking_flip"] == second["evidence_flip"] == "True"
    assert read_json(out / "items/sample.json")["attempts"][0]["validated"] == [FINDING]
    with sqlite3.connect(out / "ledger.sqlite") as db:
        assert db.execute("SELECT stage, count(*) FROM calls GROUP BY stage").fetchall() == [("S9P", 2)]
    assert {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()} == original


def test_mode_report_joins_frozen_round_and_epoch(tmp_path, capsys):
    item = tmp_path / "items/sample"
    rec = dict(mode="enforce", epoch=1, round=2, raw=[FINDING, {}], validated=[FINDING],
               usd=.031, revision_applied=True)
    write_json(item / "prereview/e1-r2.json", dict(rec, revision_usd=.12))
    write_json(item / "prereview/e0-r1.json", dict(rec, epoch=0, round=1, usd=0, revision_applied=False))
    write_json(item / "prereview/e2-r3.json", dict(rec, mode="shadow", epoch=2, round=3))
    verdict = {"_normalized": {"verdict": "revise", "issues": [ISSUE, dict(ISSUE, category="d")]}}
    write_json(item / "review/r2/verdict.json", dict(verdict, calib_epoch=1))
    write_json(item / "review/r1/verdict.json", dict(verdict, calib_epoch=8))
    assert cli.main(["report", "--run", str(tmp_path), "--prereview"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows["enforce"] == dict(items_checked=1, checks=2, findings_raw=4, findings_validated=2,
                                   revisions_applied=1, prereview_usd=.031, revision_usd=.12, usd=.151,
                                   errors=0, opus_verdicts=1, opus_blocking_ab=1)
    assert rows["shadow"]["opus_verdicts"] == rows["off"]["items_checked"] == 0


def test_archived_verdict_uses_bound_job_and_routed_judgment(tmp_path):
    directory = tmp_path / "review/r2"
    record = dict(job="sample-review", pkg_sha="sample-hash", calib_epoch=1,
                  _normalized={"verdict": "revise", "issues": [ISSUE]})
    write_json(directory / "verdict.json", record)
    write_json(directory.parent / "verdict.json", dict(record, round=2, _normalized={"verdict": "human"}))
    job = dict(job_name="sample-review", model="sample-opus", pkg_sha="sample-hash", calib_epoch=1, outcome="done")
    write_json(directory.parent / "runs.json", [job])
    verdict, issues = round_verdict(directory)
    assert (verdict["verdict"], verdict["model"], issues) == ("human", "sample-opus", [])
    assert verdict["model_source"] == "recorded"
    write_json(directory.parent / "runs.json", [dict(job, model="sample-other")])
    assert round_verdict(directory) is None


@pytest.mark.parametrize(("citation", "matches"), [
    ("copy/environment/data/groups.csv:1 lists profile_id", False),
    ("environment/data/groups.csv:1; `tests/measure.py` mentions profile_id", False),
    ("environment/data/groups.csv:1\n/review/pkg/environment/data/groups.csv:2 uses profile_id", True),
    ({"file": "/app/data/groups.csv", "column": "profile_id"}, True),
    ({"file": "environment/data/groups.csv", "column": "profile_id_suffix"}, False),
])
def test_tentative_matches_require_evidence_from_same_file(citation, matches):
    import runpy
    helpers = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prereview_backtest.py"))
    assert helpers["evidence_overlaps"](FINDING["evidence"], citation) == matches
