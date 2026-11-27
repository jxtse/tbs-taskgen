"""Stage-contract replay tests: the inter-stage contracts that broke in production are asserted mechanically on
small synthetic-but-valid fixtures (`fixtures/contracts/`), using the real producer/gate functions.

  (a) S5G `generator_assumptions[stated]` quotes survive an S6C-like rewrite into the final instruction.md, and the
      S6C prompt names them (`statement.quote_block`) — 2026-11-27: S6C paraphrased a disclosed assumption and S7's
      `assumptions_stated` gate dropped the item.
  (b) the S6C file set is exactly what S7 needs, and the statement's declared CSV header matches the reference's
      (`gates.output_contract`).
  (c) PKG's `instruction_text` keeps the statement gates passing (and its verifier-time note is never evidence).
  (d) S4c deliverables are the set S5G harvests, and S5G's /work receives HOWTO.md + runnability.json.
"""
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbsgen import codeagent, gates, package, runner, spec as specmod, stages_b, statement
from tbsgen.config import load_config
from tbsgen.state import read_json, write_json

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"
SPEC = json.loads((FIXTURES / "spec_s5g.json").read_text())
THRESHOLDS = json.loads((FIXTURES / "thresholds.json").read_text())["thresholds"]
CANONICAL = (FIXTURES / "s6c" / "task" / "instruction.md").read_text()
PARAPHRASED = (FIXTURES / "s6c_paraphrased" / "task" / "instruction.md").read_text()
STATED = statement.stated_quotes(SPEC["generator_assumptions"])


def _cfg():
    return load_config(None)


def _s6c_files():
    base = FIXTURES / "s6c" / "task"
    return {"instruction.md": (base / "instruction.md").read_text(),
            "tests/test_outputs.py": (base / "tests" / "test_outputs.py").read_text(),
            "baselines/exploit_truth_table.py": (base / "baselines" / "exploit_truth_table.py").read_text()}


def _item(tmp_path):
    """The item directory as S5G leaves it for S6C: spec, thresholds, measurements, task/ without tests yet."""
    (tmp_path / "_spec").mkdir(parents=True)
    write_json(tmp_path / "_spec" / "spec.json", SPEC)
    write_json(tmp_path / "thresholds.json", {"thresholds": THRESHOLDS})
    write_json(tmp_path / "S6A_measure.json", dict(ref={}, mid={}, naive={}))
    (tmp_path / "task").mkdir()
    saved: dict = {}
    it = SimpleNamespace(state={}, item_id="doi-10.1/smoke-crystallite",
                         p=lambda *a: tmp_path.joinpath(*a), save=lambda: None,
                         finish=lambda *a, **k: saved.__setitem__("finish", a),
                         advance=lambda *a: saved.__setitem__("advance", a))
    return it, saved


def _run_s6c(tmp_path, monkeypatch, files, capture=None):
    it, seen = _item(tmp_path)

    def chat_files(P, it_, role, stage, user, need, **kw):
        if capture is not None:
            capture.update(user=user, need=need, kw=kw)
        return files, [dict(role=role, stage=stage)]
    monkeypatch.setattr(stages_b, "chat_files", chat_files)
    stages_b.s6c(SimpleNamespace(cfg=_cfg()), it)
    return it, seen


# ----------------------------------------------------------------------------------------------- (0) the fixture
def test_fixture_spec_is_valid():
    assert specmod.validate_spec(SPEC, _cfg(), "3.0") == []


# ----------------------------------------------------------------------------------------------- (a) assumption quotes
def test_the_s6c_prompt_names_every_stated_quote(tmp_path, monkeypatch):
    cap = {}
    _run_s6c(tmp_path, monkeypatch, _s6c_files(), capture=cap)
    assert len(STATED) == 2
    for q in STATED:
        assert q in cap["user"], q
    assert "word for word" in cap["user"]
    block = cap["user"][cap["user"].index("The S7 gate assumptions_stated requires"):]
    assert "recovered from data:" not in block, "the quote block lists stated quotes only"


def test_faithful_s6c_rewrite_keeps_the_gate_green(tmp_path, monkeypatch):
    task = tmp_path
    it, seen = _run_s6c(tmp_path, monkeypatch, _s6c_files())
    assert seen["advance"] == ("S6C", "S7"), seen
    ins = statement.strip_time_note((task / "task" / "instruction.md").read_text())
    assert statement.assumptions_stated(ins, SPEC["generator_assumptions"], require_long_quote=True) == []
    assert statement.statement_clean(ins) == []
    assert statement.threshold_format(ins, THRESHOLDS) == []
    assert gates.output_contract(ins, FIXTURES / "ref_out")["ok"] is True


def test_paraphrased_rewrite_fails_the_gate_by_name():
    ins = statement.strip_time_note(PARAPHRASED)
    missing = statement.assumptions_stated(ins, SPEC["generator_assumptions"], require_long_quote=True)
    assert missing == STATED, "both reworded quotes must be reported"
    for q in STATED:
        assert q not in PARAPHRASED
    # the rest of the statement is untouched: only the quotes broke
    assert statement.threshold_format(ins, THRESHOLDS) == []
    assert gates.output_contract(ins, FIXTURES / "ref_out")["ok"] is True


def test_the_quote_block_is_the_thing_that_makes_the_fixture_survive():
    """Contract in one line: quotes(S5G) -> quote_block -> S6C prompt -> S7 gate."""
    block = statement.quote_block(SPEC["generator_assumptions"])
    for q in STATED:
        assert json.dumps(q, ensure_ascii=False) in block   # one exact line per stated quote
    assert statement.quote_block([]) == ""


@pytest.mark.parametrize("rel", ["instruction.md", "tests/test_outputs.py", "baselines/exploit_truth_table.py"])
def test_s6c_without_one_required_file_drops_the_item(tmp_path, monkeypatch, rel):
    files = {k: v for k, v in _s6c_files().items() if k != rel}
    it, seen = _run_s6c(tmp_path, monkeypatch, files)
    assert seen["finish"] == ("dropped", "s6c_missing_files"), seen
    got = read_json(tmp_path / "S6C_error.json")["got"]
    assert rel not in got and got, "the error names what the model did return"


# ----------------------------------------------------------------------------------------------- (b) output contract
def test_statement_header_must_match_the_reference_csv(tmp_path):
    drifted = CANONICAL.replace("Header: `crystallite_id,orientation_deg,size_nm`",
                                "Header: `crystallite,orientation,size_nm`")
    (tmp_path / "instruction.md").write_text(drifted)
    oc = gates.output_contract(drifted, FIXTURES / "ref_out")
    assert oc["ok"] is False
    assert oc["mismatch"]["output/matches.csv"]["stated"] == ["crystallite", "orientation", "size_nm"]
    assert oc["mismatch"]["output/matches.csv"]["reference"] == ["crystallite_id", "orientation_deg", "size_nm"]


def test_reference_outputs_are_what_the_verifier_checks():
    with open(FIXTURES / "ref_out" / "output" / "matches.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == ["crystallite_id", "orientation_deg", "size_nm"]
    verifier = (FIXTURES / "s6c" / "task" / "tests" / "test_outputs.py").read_text()
    # the verifier's own assertion on the header is the same column set as the reference's
    assert '{"crystallite_id", "orientation_deg", "size_nm"}' in verifier
    assert SPEC["output_contract"]["output/matches.csv"] == header


# ----------------------------------------------------------------------------------------------- (c) PKG statement
def test_pkg_instruction_text_is_idempotent_and_keeps_the_gates_green():
    cfg = _cfg()
    packed = package.instruction_text(cfg, SPEC, CANONICAL)
    assert packed == package.instruction_text(cfg, SPEC, packed), "a copied-back statement must not grow a 2nd note"
    assert package.TIME_NOTE_MARK in packed and package.TIME_NOTE_END in packed
    assert package.strip_time_note(packed) == package.strip_time_note(CANONICAL).rstrip() + "\n"
    ins = package.strip_time_note(packed)
    assert statement.assumptions_stated(ins, SPEC["generator_assumptions"], require_long_quote=True) == []
    assert statement.statement_clean(ins) == []
    assert statement.threshold_format(ins, THRESHOLDS) == []
    assert gates.output_contract(ins, FIXTURES / "ref_out")["ok"] is True
    # the time limit itself is stated (that was the e2e-batch2 finding)
    assert f"{runner.scored_budget(cfg):g}" in packed


def test_the_verifier_time_note_is_never_evidence_for_a_threshold():
    cfg = _cfg()
    packed = package.instruction_text(cfg, SPEC, CANONICAL)
    budget = runner.scored_budget(cfg)
    th = dict(THRESHOLDS, note_budget=dict(threshold=budget, direction="max"))
    assert budget in gates.numbers_in(packed), "the note does contain the number ..."
    assert budget not in gates.numbers_in(package.strip_time_note(packed)), "... but never as the author's text"
    assert statement.threshold_format(package.strip_time_note(packed), th) == []


def test_a_PROG_task_without_the_note_does_not_state_the_verifier_budget():
    cfg = _cfg()
    bare = package.instruction_text(cfg, dict(SPEC, form="ANS"), CANONICAL)
    assert package.TIME_NOTE_MARK not in bare and package.strip_time_note(bare) == CANONICAL.rstrip() + "\n"


# ----------------------------------------------------------------------------------------------- (d) S4c -> S5G
def _s4c_trial(tmp_path, drop=None):
    trial = tmp_path / "trial"
    out = trial / "verifier" / "out"
    shutil.copytree(FIXTURES / "s4c", out)
    if drop:
        (out / drop).unlink()
    (trial / "result.json").write_text(json.dumps({
        "agent_result": {"cost_usd": 1.2, "n_input_tokens": 10, "n_output_tokens": 5},
        "verifier_result": {"rewards": {"reward": 1}},
        "agent_execution": {"started_at": "2026-11-26T10:00:00Z", "finished_at": "2026-11-26T10:20:00Z"}}))
    return trial, out


def test_s4c_deliverables_are_the_ones_s5g_harvests(tmp_path):
    cfg = _cfg()
    trial, out = _s4c_trial(tmp_path)
    kind, required = codeagent._required_outputs(cfg, {"ledger_stage": "S4c", "task": str(tmp_path / "task")})
    assert kind == "S4c" and required == ["runnability.json", "HOWTO.md"]
    result = json.loads((trial / "result.json").read_text())
    assert codeagent._agent_trial_complete(trial, result, kind, required) is True
    verdict = json.loads(codeagent.safe_read(out / "runnability.json", root=trial))
    assert verdict["runnable"] is True and verdict["reason"] == "ok" and str(verdict["entry"]).strip()
    [receipt] = codeagent.run_receipts(out)
    assert receipt["rc"] == 0 and receipt["cmd"] == verdict["entry"]
    assert codeagent.safe_read(out / receipt["log"], root=out) is not None, "the run log travels with the receipt"


@pytest.mark.parametrize("drop,why", [("HOWTO.md", "the how-to is required"), ("runs/run1.log", "a receipt needs its log")])
def test_s4c_missing_deliverable_is_not_harvested(tmp_path, drop, why):
    trial, _ = _s4c_trial(tmp_path, drop=drop)
    kind, required = codeagent._required_outputs(_cfg(), {"ledger_stage": "S4c", "task": str(tmp_path / "task")})
    result = json.loads((trial / "result.json").read_text())
    assert codeagent._agent_trial_complete(trial, result, kind, required) is False, why


def test_s5g_work_dir_gets_the_howto_and_runnability(tmp_path):
    item = tmp_path / "item"
    shutil.copytree(FIXTURES / "s4c", item / "S4c")
    (item / "fulltext.txt").write_text("paper text\n")
    write_json(item / "S1_candidate.json", dict(title="Crystallite matching", doi="10.1/x"))
    write_json(item / "S3c_code.json", dict(remote_dir="/r/x", pin="abc", source="github", dep_files=["requirements.txt"]))
    it = SimpleNamespace(state={}, item_id="doi-x", p=lambda *a: item.joinpath(*a))
    codeagent.stage_inputs(_cfg(), it, tmp_path / "work", with_contract=False)
    work = tmp_path / "work"
    assert (work / "HOWTO.md").read_text() == (FIXTURES / "s4c" / "HOWTO.md").read_text()
    assert json.loads((work / "runnability.json").read_text())["runnable"] is True
    assert json.loads((work / ".tbsgen-remote-repo").read_text())["pin"] == "abc"


def test_s5g_required_files_are_a_superset_of_the_tutorial_deliverables():
    s4c = codeagent.agent_output_files("S4c")
    s5g = codeagent.agent_output_files("S5G")
    assert s4c == ["runnability.json", "HOWTO.md"]
    assert "_spec/spec.json" in s5g and "task/instruction.md" in s5g
    assert not set(s4c) & set(s5g)
