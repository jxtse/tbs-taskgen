"""Offline regressions for the review of the hard-task share."""
import csv
import json
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tbsgen import budget, cli, codeagent, diversity, gates, package, probe_outputs, review, runner, stages_a, stages_b
from tbsgen import templates_metrics
from tbsgen.config import load_config
from tbsgen.fakes import SPEC
from tbsgen.prompts import PROMPT_VERSION
from tbsgen.prompts_code import CODE_PROMPT_VERSION
from tbsgen.spec import set_thresholds, validate_spec
from tbsgen.state import Store, write_json
from tbsgen.statement import assumptions_stated, statement_clean, threshold_format


def test_statement_hygiene_real_statements():
    fixture = json.loads((Path(__file__).parent / "fixtures/review_statements.json").read_text())
    assert len(fixture) == 3
    for name, text in fixture.items():
        assert not statement_clean(text), name
    for heading in ("## Output-contract clarification (audit copy)", "## Additional hidden-output checks (audit revision)"):
        assert statement_clean(heading) == [heading]


def test_hygiene_forms_and_exceptions():
    for word in ("todo", "FIXME", "xxx", "audit revision", "reviewer note", "review round", "S9R", "S7 gate",
                 "S8 repair", "quality gate", "tbsgen", "internal note", "patch notes", "fix log", "hardening round",
                 "round-12", "difficulty lever", "naive baseline", "mid-baseline", "# exploit", "<!-- private -->"):
        assert statement_clean(word), word
    assert not statement_clean("Investigate, navigate and aggregate the hidden outputs.\nFit the baseline level.")
    # scientific homonyms from real statements of the 2026-11-23 share stay clean
    for line in ("# CMB polarization amplitude behind Galactic foregrounds, patch by patch",
                 "- more spurious and more missed H&E detections, including a contiguous patch where H&E nuclei",
                 "# Locating damage on a CFRP plate from a single guided-wave scan (no baseline)",
                 "# Baseline-free damage localisation from time-reversal Lamb-wave reconstructions",
                 "# Glass content of quartz–zincite–glass powders by internal-standard XRD"):
        assert not statement_clean(line), line
    assert not statement_clean(package.instruction_text(load_config(), {"form": "PROG"}, "Hidden shift"))


def test_threshold_literals_and_unreadable_separation():
    thresholds = {"a": {"threshold": 0.1234567}, "b": {"threshold": 0.123456}}
    assert threshold_format("| cut | 12.34567% |\nT ≤ `1.234567e-4`", thresholds) == ["12.34567", "1.234567e-4"]
    assert threshold_format("≥ 0.8665767420502813") == ["0.8665767420502813"]
    assert threshold_format("value 0.8031095863545515 here") == ["0.8031095863545515"]   # float repr anywhere
    assert not threshold_format("At most 0.123456 or 1e-9. Catalogue ID 12345678.", thresholds)
    # physical constants and small exact cutoffs in real statements stay allowed (st-batch2-023, doi-47f3c851...)
    assert not threshold_format("- GM = 398600.4418 km^3/s^2, equatorial radius = 6378.137 km, J2 = 1.08262668e-3.")
    assert not threshold_format("fitted slope **≥ −0.0000000001**, difference **≤ 0.000000001**")
    # prod-code doi-7f1e4ce020185a83: naive capped at the ref-margin floor and naive (0.912) clears even the exact
    # floor 0.8666: the old code printed `>= 0.8665767420502813`; now a readable 2-digit threshold
    ms = {w: {"0": {"p": a}, "1": {"p": b}} for w, a, b in (("ref", .9628630467225348, .9667504236545317),
                                                          ("mid", .9628630467225348, .9667504236545317),
                                                          ("naive", .8902762012864169, .9120052338894341))}
    t = set_thresholds({"metrics": [{"name": "p", "direction": ">="}]}, ms, load_config())["thresholds"]["p"]
    assert t["threshold"] == 0.86 and t["ref_passes"]
    # a separating metric whose mid sits right next to the midpoint still gets a readable threshold that rejects mid
    result = set_thresholds({"metrics": [{"name": "p", "direction": "<="}]},
                            {w: {"0": {"p": v}} for w, v in (("ref", 1.0), ("mid", 1.0000004), ("naive", 9.0))},
                            dict(load_config(), thresholds=dict(load_config()["thresholds"], min_ref_margin=1e-7)))
    t = result["thresholds"]["p"]
    assert t["threshold"] is not None and t["threshold"] < 1.0000004 and len(repr(t["threshold"])) <= 8


def test_versioned_spec_and_quotes():
    cfg = load_config()
    old = {k: v for k, v in SPEC.items() if k != "generator_assumptions"}
    assert not validate_spec(old, cfg, "cg-v3-2026-11-12")
    for version in (PROMPT_VERSION, CODE_PROMPT_VERSION, "cg-v10", "v2.10"):
        assert validate_spec(dict(old, prompt_version=version), cfg)
        assert validate_spec(old, cfg, version)
    a = {"stated": True, "assumption": "noise", "where": "Additive Gaussian noise"}
    spec = dict(old, generator_assumptions=[a], shared_inputs=["cal.csv"], data_formats=["csv"])
    assert not validate_spec(spec, cfg)
    assert not assumptions_stated("Additive   Gaussian\nnoise", [a])
    assert assumptions_stated("Other noise", [a]) == [a["where"]]
    assert assumptions_stated("Gaussian", [dict(a, where="Gaussian")])
    for bad in ("string", [None], [dict(a, stated=0)], [dict(a, stated=False)], [dict(a, where="")]):
        assert validate_spec(dict(spec, generator_assumptions=bad), cfg)
    assert not validate_spec(dict(spec, generator_assumptions=[
        dict(a, stated=False, where="recovered from data: fit replicate variance")]), cfg)
    for field, value in (("data_formats", "csv"), ("shared_inputs", [0]), ("extra_pip", ["x>=5"]),
                         ("extra_pip", ["x==2;echo"]), ("extra_pip", ["x==1"] * 9)):
        assert validate_spec(dict(spec, **{field: value}), cfg)


def test_hidden_parity_only_counts_measured_metrics():
    spec = {"metrics": [{"name": "mae", "on": "hidden"}, {"name": "public_mae"}, {"name": "hidden_bias"},
                        {"name": "public_time", "public_only": True, "public_only_reason": "startup"},
                        {"name": "unrecorded"}]}
    rows = [{"name": n} for n in ("mae", "public_mae", "public_time", "hidden_bias", "extra_public")]
    rows += [{"name": "_solution_cli_run", "aux": True}]
    assert gates.hidden_parity(rows, spec) == {"missing_public": ["bias"], "missing_hidden": ["extra"]}
    spec["metrics"][3]["public_only_reason"] = " "
    assert gates.hidden_parity(rows, spec)["missing_hidden"] == ["extra", "time"]


def test_hidden_duplicates(tmp_path):
    pub, hidden = tmp_path / "data", tmp_path / "hidden"
    one, two = hidden / "one", hidden / "two"
    for d in (pub, one, two):
        d.mkdir(parents=True)
        (d / "small").write_bytes(b"x" * 1024)
        (d / "cal.csv").write_bytes(b"y" * 1025)
    (pub / "large").write_bytes(b"z" * 2000)
    (one / "renamed").write_bytes((pub / "large").read_bytes())
    (two / "renamed").write_bytes((pub / "large").read_bytes())
    dup = gates.hidden_duplicates(pub, hidden, ["cal.csv"])
    assert dup["copied_files"] == {"one/renamed": ["large"], "two/renamed": ["large"]}
    assert dup["identical_sets"] == [["one", "two"]]
    for d in (one, two):
        (d / "renamed").write_text(d.name)
    assert gates.hidden_duplicates(pub, hidden, ["cal.csv"])["ok"]
    for d in (pub, one, two):
        (d / "cal.csv").write_bytes(b"y" * 4000)
    rep = gates.hidden_duplicates(pub, hidden)              # cal.csv = 4000 of 5027 bytes per set: mostly public
    assert not rep["ok"] and set(rep["mostly_public"]) == {"one", "two"}
    assert gates.hidden_duplicates(pub, hidden, ["cal.csv"])["ok"]   # declared shared input
    # one shared file inside an otherwise fresh hidden set is fine (training table / time grid)
    for d in (one, two):
        (d / "fresh").write_bytes(d.name.encode() * 4000)
    rep = gates.hidden_duplicates(pub, hidden)
    assert rep["ok"] and rep["copied_files"] and not rep["mostly_public"]


def test_realism_uses_s6a_and_notes(tmp_path):
    (tmp_path / "data.json").write_text("1")
    runs = {"ref0": {"solve_rc": 0, "solve_sec": 29}, "mid0": {"solve_rc": 0, "solve_sec": 40}}
    spec = {"expert_hours": 2}
    r = gates.realism(spec, tmp_path, runs)
    assert not r["effort_floor"][0] and not r["data_realism"][0]
    assert all(x[0] for x in gates.realism(dict(spec, data_realism_note="compact", effort_note="derivation"),
                                          tmp_path, runs).values())
    (tmp_path / "signal.fits").touch()
    runs["ref1"] = {"solve_rc": 0, "solve_sec": 30}
    assert all(x[0] for x in gates.realism(spec, tmp_path, runs).values())


def test_pkg_checks_and_sandbox_inventory(tmp_path, monkeypatch):
    cfg = load_config()
    assert package.run_solution_timeouts("T = 780\nm.run_solution(timeout_sec=T)\nrun_solution(779.99)", 780) == [
        "line 2: run_solution timeout 780s >= verifier budget 780s"]
    assert not package.run_solution_timeouts("run_solution(dynamic(2000))", 780)
    assert not package.run_solution_timeouts("T=1200\ndef test(T): run_solution(T)", 780)
    assert package.run_solution_timeouts("from _metrics import run_solution as run\nrun(900)", 780)
    assert "h5py==3.12.1" in package.dockerfile(cfg, "selfcontained", ["h5py==3.12.1"])
    for p in package.REQUIRED:
        file = tmp_path / p
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    (tmp_path / "tests/_metrics.py").write_text(runner.metrics_helper_text(cfg, {}))
    (tmp_path / "tests/measure.py").write_text("T: int = 1200\nrun_solution(T)\n")
    (tmp_path / "tests/test_outputs.py").write_text("run_solution(timeout_sec=780)\n")
    (tmp_path / "task.toml").write_text('[task]\n[environment]\nnetwork_mode="public"\n[agent]\n'
                                        '[verifier]\ntimeout_sec=900\n')
    messages = package.validate_pkg(tmp_path, cfg)["errors"]
    assert any("measure.py" in e and "1200s >= verifier budget 780s" in e for e in messages)
    assert any("test_outputs.py" in e and "780s >=" in e for e in messages)
    assert any("[verifier].network_mode" in e for e in messages)
    assert any("[agent].network_mode" in e for e in messages)
    policy = {"agent": {"network_mode": "public"}, "verifier": {"network_mode": "no-network"}}
    errors, warnings = package.network_errors(policy, "public")
    assert not errors and warnings
    assert package.network_errors(policy, "agent-allowlist")[0]
    pins = ["h5py==3.12.1"]
    write_json(tmp_path / "provenance.json", {"extra_pip": pins})
    monkeypatch.setattr(package.sandbox, "run", lambda *args: dict(rc=1, err="not installed"))
    assert any("extra_pip_unsupported" in e for e in package.validate_pkg(tmp_path, cfg)["errors"])
    monkeypatch.setattr(package.sandbox, "run", lambda *args: dict(rc=0, err=""))
    assert package.sandbox_pin_errors(cfg, pins) == []



def test_array_json_and_small_csv(tmp_path):
    (tmp_path / "small.csv").write_text("label,count\nb,1\na,2\nb,3\n")
    np.save(tmp_path / "a.npy", np.arange(4))
    np.savez(tmp_path / "b.npz", score=np.arange(6), labels=np.array(["a", "a", "b"]))
    (tmp_path / "c.json").write_text('[{"x":1,"color":"red"},{"x":2,"color":"blue"},{"x":8,"color":"red"}]')
    (tmp_path / "d.txt").write_text("x value\n0 1.2\n3 4\n")
    assert probe_outputs.mutate(tmp_path, "one_confident") == 0
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == 5
    assert not np.load(tmp_path / "a.npy").any()
    with np.load(tmp_path / "b.npz") as data:
        assert not data["score"].any() and data["labels"].tolist() == ["a"] * 3
    assert json.loads((tmp_path / "c.json").read_text()) == [{"x": 0, "color": "red"}] * 3
    assert (tmp_path / "d.txt").read_text() == "x value\n3 4\n0 1.2\n"


@pytest.fixture
def wired_gates(tmp_path, monkeypatch):
    cfg = load_config()
    spec = {"form": "PROG", "expert_time_hours": 2, "metrics": [
        {"name": "mae_hidden", "on": "hidden"}, {"name": "hidden_score"}, {"name": "mae_public"}, {"name": "score"}]}
    def generate(cfg, task, seed, path):
        for rel in ("environment/data/a.csv", "tests/hidden/d/a.csv"):
            p = path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(rel)
        return dict(rc=0, hash_env="a", hash_tests="b", err="")
    def solve(cfg, task, gen, prog, work, tag):
        out = work / f"out-{tag}"
        out.mkdir()
        if prog:
            (out / "out.csv").write_text("x\n1\n")
        return dict(rc=0, sec=1, err="", outdir=str(out))
    def verifier(cfg, task, gen, out, work, tag):
        ref = tag.startswith("ref") or tag == "probe-ref0"
        value = .93 if ref else 2 if tag.startswith("nop") else 1.5
        rows = [dict(name=m["name"], value=value, direction="<=", threshold=1, passed=ref) for m in spec["metrics"]]
        rows.append(dict(name="_solution_cli_run", rc=0, aux=True))
        return dict(metrics=rows, passed=ref, n_checks=4, n_failed=0 if ref else 4,
                    failed=[] if ref else [r["name"] for r in rows[:-1]])
    monkeypatch.setattr(runner, "generate", generate)
    monkeypatch.setattr(runner, "solve", solve)
    monkeypatch.setattr(runner, "test", verifier)
    monkeypatch.setattr(runner, "truth_probe", lambda *args: {"ok": True})
    arms = []
    def probe(cfg, task, gen, ref, work, arm, prog):
        arms.append((arm, gen.name))
        return {"ok": True, "na": arm == "one_confident"}
    monkeypatch.setattr(runner, "output_probe", probe)
    monkeypatch.setattr(gates, "leak_report", lambda *a, **k: {"verdict": "ok"})
    task = tmp_path / "task"
    (task / "baselines").mkdir(parents=True)
    (task / "baselines/exploit_echo.py").touch()
    (task / "instruction.md").write_text("python /app/output/solution_cli.py <input_dir> <output_dir>\n")
    write_json(tmp_path / "S6A_measure.json", {"runs": {"ref0": {"solve_rc": 0, "solve_sec": 1}}})
    return cfg, spec, arms, task


def test_gate_wiring(tmp_path, monkeypatch, wired_gates):
    cfg, spec, arms, task = wired_gates
    cfg["thresholds"]["min_ref_margin_hidden"] = .10   # stricter than the default 0.05: the 7 % margin fails
    rep = gates.run_gates(cfg, tmp_path, spec, {}, 1)
    assert rep["blocking"] == ["oracle_margin"]
    assert rep["details"]["oracle_margin"][0] == ["mae_hidden", "hidden_score"]
    assert rep["details"]["coverage"] == "na" and rep["gates"]["hidden_parity"]
    assert arms == [("one_confident", "detA"), ("hidden_garbage", "detA")]
    cfg["thresholds"]["min_ref_margin_hidden"] = .07  # exactly the measured margin
    rep = gates.run_gates(cfg, tmp_path, spec, {}, 2)
    assert rep["ok"] and set(rep["nonblocking"]) == {"effort_floor", "data_realism"}
    cfg["gen"]["seeds"] = [1]
    arms.clear()
    assert gates.run_gates(cfg, tmp_path, spec, {}, 4)["ok"]
    assert arms == [("one_confident", "probe-gen0"), ("hidden_garbage", "probe-gen0")]
    monkeypatch.setattr(runner, "output_probe", lambda *a: {"ok": False})
    (task / "instruction.md").write_text("## Audit copy\n≤ 0.8665767420502813")
    spec["generator_assumptions"] = [dict(assumption="a", stated=True, where="Noise is Gaussian")]
    rep = gates.run_gates(cfg, tmp_path, spec, {}, 3)
    assert {"exploit", "hidden_garbage", "coverage", "threshold_format", "statement_clean", "assumptions_stated"} <= set(rep["blocking"])


def test_diversity_signatures_and_components():
    assert len(diversity.TAXONOMY) == 14
    assert diversity.classify("chemistry", "computing", "") == "chemistry"
    assert diversity.classify("", "hydrodynamics", "") == "fluids & continuum mechanics"
    assert diversity.classify("unknown", "", "") == "other"
    def task(id, words, target="", metric="public_rmse"):
        return diversity.template(id, {"competencies": [words], "metrics": [{"name": metric}]}, {"target": target})
    a, b, c = task("a", "residual bootstrap"), task("b", "bootstrap residual nystrom"), task("c", "nystrom bootstrap")
    assert diversity.related(a, b) and not diversity.related(a, c)
    assert diversity.clusters([c, a, b]) == [["a", "b", "c"]]
    a, b = task("a", "near infrared spectrum calibration", "x"), task("b", "solve orthogonal matrices", "x", "hidden_rmse")
    assert diversity.related(a, b)
    assert not diversity.related(a, task("b", "solve orthogonal matrices", "x", "hidden_bias"))
    assert not diversity.related(task("x", ""), task("y", ""))


def test_diversity_inventory_and_both_admissions(tmp_path, capsys):
    run = tmp_path / "run"
    store = Store(run)
    def item(id, status="done", area="chemistry", target="method:spectrum-fit"):
        it = store.item(id)
        it.state["status"] = status
        it.save()
        write_json(it.p("S1_candidate.json"), {"title": f"{area} work", "_target": target})
        write_json(it.p("_spec/spec.json"), {"domain": area, "competencies": ["spectral correction"],
                                             "depgraph": {"nodes": [{"id": "smooth_spectrum"}]},
                                             "metrics": [{"name": "public_rmse"}]})
        write_json(it.p("_spec/provenance.json"), {"target": target})
        return it
    a, b = item("a"), item("b")
    candidate = item("c", "active")
    item("d", "dropped")
    item("e", "active")
    (run / "release").mkdir()
    (run / "release/manifest.jsonl").write_text('{"item_id":"a"}\n{"item_id":"a"}\n{"item_id":"b"}\n'
                                               '{"item_id":"e"}\n{"item_id":"e","withdrawn":true}\n')
    P = SimpleNamespace(cfg=load_config(), store=store)
    P.cfg["diversity"]["enforce"] = True
    # 2 released tasks: an area share of 100 % says nothing yet (min_released 20) -> only the template cap applies
    assert stages_a._agent_admission(P, candidate, "S5G") == "diversity:template_full:a"
    P.cfg["diversity"].update(enforce=False, min_released=2)
    assert stages_a._agent_admission(P, candidate, "S5G") is None
    why = "diversity:area_full:chemistry"
    assert candidate.state["diversity"]["S5G"]["reason"] == why
    P.cfg["diversity"]["enforce"] = True
    assert stages_a._agent_admission(P, candidate, "S5G") == why
    assert stages_a._admission_status(why) == "parked"
    P.cfg["diversity"]["max_area_share"] = 1
    assert stages_a._agent_admission(P, candidate, "S5G") == "diversity:template_full:a"
    stages_b.s10(P, candidate)
    assert candidate.state["status"] == "parked" and candidate.state["reason"] == "diversity:template_full:a"
    assert diversity.released_ids(run) == {a.item_id, b.item_id}
    assert store.events()[-1]["stage"] == "S10"
    second = tmp_path / "second"
    write_json(second / "items/f/state.json", {"status": "waiting"})
    assert cli.main(["diversity", "--run", str(run), "--runs-root", str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["released"] == 2 and report["in_flight"] == 2
    assert report["released_by_area"]["chemistry"] == 2 and "materials" in report["gaps"]
    assert report["reskin_clusters"] == [["a", "b", "e"]]
    assert "in flight" in diversity.format_report(report)
    assert not (second / "ledger.sqlite").exists()



def test_statement_fix_kind_routes_to_local_revision():
    raw = {"verdict": "revise", "checklist": {k: {"status": "ok", "note": "checked"} for k in review.CHECKLIST},
           "issues": [{"id": "B1", "severity": "blocking", "fix_kind": "statement", "problem": "statement",
                       "title": "missing formula", "category": "missing_conditions", "evidence": "generate.py:12"}]}
    out = review.normalize(raw)
    assert out["issues"][0]["fix_kind"] == "local" and out["verdict"] == "revise"


def test_readonly_audit_cli(tmp_path):
    share = tmp_path / "share"
    pkg = share / "1-hard/demo"
    (pkg / "tests").mkdir(parents=True)
    (share / "INDEX.csv").write_text("task_id,tbsgen_run\ndemo,\n")
    (pkg / "task.toml").write_text('[task]\ndescription="Spectrum"\n[metadata]\ndomain="chemistry"\n'
                                   '[verifier]\ntimeout_sec=900\n[agent]\n')
    (pkg / "instruction.md").write_text("## Audit revision\nScore ≥ 0.12345678")
    (pkg / "tests/test_outputs.py").write_text("run_solution(800)\n")
    before = {p: p.stat().st_mtime_ns for p in share.rglob("*")}
    script = Path(__file__).resolve().parents[1] / "scripts/audit_share.py"
    result = subprocess.run([sys.executable, str(script), str(pkg.parent)], check=True, capture_output=True, text=True)
    headers, row, summary = result.stdout.splitlines()
    assert headers.startswith("item\t") and row.startswith("demo\tFAIL")
    data = json.loads(summary)
    assert data["flagged"] == dict(statement_clean=1, threshold_format=1, hidden_duplicates=0,
                                   run_solution_timeout=1, network_explicit=1)
    assert data["diversity"]["released_by_area"]["chemistry"] == 1
    assert before == {p: p.stat().st_mtime_ns for p in share.rglob("*")}


def test_diversity_parking_releases_unlaunched_reservation(tmp_path):
    store = Store(tmp_path)
    it = store.item("next")
    calls = []
    P = SimpleNamespace(store=store, cfg=load_config(), ledger=SimpleNamespace(
        settle=lambda *args, **kwargs: calls.append((args, kwargs))))
    P.cfg["diversity"].update(enforce=True, min_released=1)
    for item in ("released", "next"):
        write_json(store.item(item).p("S1_candidate.json"), {"title": "astronomy"})
    (tmp_path / "release").mkdir()
    (tmp_path / "release/manifest.jsonl").write_text('{"item_id":"released"}\n')
    path = it.p("S5G/run.json")
    write_json(path, {"state": "reserved", "ledger_call": 13, "job_dir": str(tmp_path / "unstarted")})
    stages_a._agent_stage(P, it, "S5G", "", [], 30, 1, 1, True)
    assert it.state["status"] == "parked" and it.state["reason"].startswith("diversity:area_full:")
    rec = json.loads(path.read_text())
    assert rec["state"] == "intent" and rec["ledger_call"] is None and rec["orphan_calls"] == []
    assert calls[0][0] == (13, 0) and calls[0][1]["kind"] == "diversity_parked"


def test_hidden_parity_blocks_only_public_checks_missing_on_hidden():
    spec = {"form": "PROG", "metrics": [{"name": "public_rmse"}, {"name": "hidden_rmse"}, {"name": "hidden_drift"}]}
    rows = [{"name": n} for n in ("public_rmse", "hidden_rmse", "hidden_drift")]
    parity = gates.hidden_parity(rows, spec)
    assert parity == {"missing_hidden": [], "missing_public": ["drift"]}     # recorded, not blocking (see run_gates)




# ---------------------------------------------------------------- Codex review of PR #42: regression tests

def _probe_fixture(tmp_path, monkeypatch, sets=("A", "B")):
    """Reference writes <set>.csv (id,label,confidence; 7 rows) per hidden set and a 7-row public.csv."""
    cfg = load_config()
    ref = tmp_path / "ref"
    ref.mkdir()
    table = "id,label,confidence\n" + "".join(f"{i},{'rare' if i == 2 else 'usual'},{i/10}\n" for i in range(7))
    (ref / "public.csv").write_text(table)
    (ref / "solution_cli.py").write_text("from pathlib import Path\nimport sys\n"
        "inp, out = map(Path, sys.argv[1:])\n(out/(inp.name + '.csv')).write_text((inp/'data').read_text())\n"
        "(out/'plot.png').write_bytes(b'png')\n")
    gen = tmp_path / "gen"
    for name in sets:
        d = gen / "tests/hidden" / name
        d.mkdir(parents=True)
        (d / "data").write_text(table)
    def prepare(cfg, mounts, cmd, timeout, name):
        arm = "one_confident" if "one_confident" in cmd else "hidden_garbage"
        return dict(rc=0, out=str(probe_outputs.mutate(mounts[0][0], arm, identity="identity=True" in cmd)), err="")
    monkeypatch.setattr(runner.sandbox, "run", prepare)
    return cfg, ref, gen, table


def _verifier(tmp_path, gen, table, score_sets=("A", "B"), stop_early=False, public_count=True):
    """A fake verifier: runs the CLI on every hidden set (like run_solution), then checks the sets in score_sets.
    A set is correct when its rows equal the reference table; coverage = row count.  stop_early: the first failing
    set stops scoring (a pytest assert)."""
    ref_rows = list(csv.reader(table.splitlines()))
    def verify(cfg, task, g, out, work, tag):
        metrics, failed = [], []
        pub = list(csv.reader((out / "public.csv").read_text().splitlines()))
        if public_count and len(pub) != len(ref_rows):
            failed.append("public_count")
        outs = {}
        for d in sorted((gen / "tests/hidden").iterdir()):
            res = tmp_path / f"res-{tag}-{d.name}"
            res.mkdir()
            # Exercise the generated wrapper on this host with the sandbox account simulated in this child only.
            harness = ("import os, pwd, runpy, sys; acct = pwd.getpwnam('nobody'); "
                       "os.geteuid = lambda: acct.pw_uid; os.getegid = lambda: acct.pw_gid; "
                       "sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name='__main__')")
            run = subprocess.run([sys.executable, "-c", harness, out / "solution_cli.py", d, res],
                                 capture_output=True, text=True)
            metrics.append(dict(name="_solution_cli_run", rc=run.returncode, stderr_tail=run.stderr, aux=True))
            outs[d.name] = res
        for name in score_sets:
            rows = list(csv.reader((outs[name] / f"{name}.csv").read_text().splitlines()))
            if rows != ref_rows:
                failed.append(f"hidden_{name}")
                if stop_early:
                    break
        return dict(metrics=metrics, rc=0, passed=not failed, reward="0" if failed else "1", failed=failed)
    return verify


def test_probe_early_stopping_verifier_is_not_a_failure(tmp_path, monkeypatch):
    # finding 1: A fails first, B is never scored in that run -> still a valid rejection per scope
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "test", _verifier(tmp_path, gen, table, stop_early=True))
    for arm in ("hidden_garbage", "one_confident"):
        (tmp_path / arm).mkdir()
        r = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path / arm, arm, True)
        assert r["ok"] and not r["accepted_scopes"], r


def test_probe_catches_an_unscored_hidden_set(tmp_path, monkeypatch):
    # finding 2: verifier runs A and B but scores only A -> the B-scoped run is accepted -> probe fails
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "test", _verifier(tmp_path, gen, table, score_sets=("A",)))
    for arm in ("hidden_garbage", "one_confident"):
        (tmp_path / arm).mkdir()
        r = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path / arm, arm, True)
        assert not r["ok"] and r["accepted_scopes"] == ["hidden:B"], r


def test_probe_public_count_does_not_hide_missing_hidden_coverage(tmp_path, monkeypatch):
    # finding 2 (coverage): public row count checked, hidden coverage never checked (hidden scored on row content
    # only for A... here: hidden not scored at all) -> hidden scopes accepted
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "test", _verifier(tmp_path, gen, table, score_sets=()))
    r = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "one_confident", True)
    assert not r["ok"] and set(r["accepted_scopes"]) == {"hidden:A", "hidden:B"} and r["scopes"]["public"]["reward"] == "0"


def test_probe_side_files_do_not_skip_the_check(tmp_path, monkeypatch):
    # finding 3: plot.png next to the scorable CSV used to turn the whole probe into n/a
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "test", _verifier(tmp_path, gen, table, score_sets=("A",)))
    r = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "hidden_garbage", True)
    assert not r["na"] and r["scopes"]["hidden:A"]["reward"] == "0" and r["accepted_scopes"] == ["hidden:B"]


def test_probe_requires_every_hidden_set_to_be_run(tmp_path, monkeypatch):
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch)
    inner = _verifier(tmp_path, gen, table)
    def skip_b(cfg_, task, g, out, work, tag):
        (gen / "tests/hidden/B").rename(gen / "tests/hidden_B")
        try:
            return _verifier(tmp_path, gen, table, score_sets=("A",))(cfg_, task, g, out, work, tag)
        finally:
            (gen / "tests/hidden_B").rename(gen / "tests/hidden/B")
    monkeypatch.setattr(runner, "test", skip_b)
    r = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "hidden_garbage", True)
    assert not r["ok"] and r["error"] == "hidden set not run by the verifier"


def test_probe_ids_stay_valid_while_answers_move():
    # finding 10: an id-only verifier must not reject the garbage table; ids keep their values, answers rotate
    rows = [[str(i), f"lab{i}", str(i / 10)] for i in range(6)]
    out = probe_outputs._rotate_rows(rows, ["id", "label", "score"])
    assert [r[0] for r in out] == [r[0] for r in rows]
    assert sorted(r[1] for r in out) == sorted(r[1] for r in rows)
    assert all(o[1] != r[1] and o[2] != r[2] for o, r in zip(out, rows))
    # key in column 1 instead: its association with column 2 is destroyed too (different offsets)
    assert all((o[1], o[2]) not in {(r[1], r[2]) for r in rows} for o in out)


def test_unsupported_files_are_skipped_not_fatal(tmp_path):
    (tmp_path / "x.fits").write_bytes(b"SIMPLE  =")
    (tmp_path / "t.csv").write_text("id,v\n1,2\n2,3\n3,4\n")
    skipped = []
    assert probe_outputs.mutate(tmp_path, "hidden_garbage", skipped) == 1 and "x.fits" in skipped[0]
    (tmp_path / "t.csv").unlink()
    assert probe_outputs.mutate(tmp_path, "hidden_garbage", []) == 0


@pytest.mark.parametrize("isolated", [True, False])
def test_probes_use_isolated_seed_zero(tmp_path, monkeypatch, wired_gates, isolated):
    cfg, spec, _, _ = wired_gates
    cfg["gen"]["seeds"] = [7, 0]
    checked, used = [], []
    def truth(cfg, task, gen, ref, work, tag):
        checked.append((gen.name, ref.name))
        return {"ok": isolated}
    def probe(cfg, task, gen, ref, work, arm, prog):
        used.append((gen.name, ref.name))
        return {"ok": True}
    monkeypatch.setattr(runner, "truth_probe", truth)
    monkeypatch.setattr(runner, "output_probe", probe)
    rep = gates.run_gates(cfg, tmp_path, spec, {}, 1)
    assert checked == [("gen0", "out-ref0")]
    assert used == (checked * 2 if isolated else [])
    assert rep["gates"]["truth_isolation"] == isolated
    assert {"threshold_format", "statement_clean", "hidden_parity", "hidden_garbage", "coverage",
            "assumptions_stated"} <= gates._blocking({"gates": {"blocking": ["oracle"]}})   # finding 5


def test_parity_ignores_recorded_only_resource_metrics():
    # finding 11
    rows = [{"name": "public_rmse"}, {"name": "hidden_rmse"}, {"name": "public_runtime", "gated": False},
            {"name": "public_wall_time_sec"}]
    assert gates.hidden_parity(rows, {"metrics": []})["missing_hidden"] == []


def test_hidden_margin_defaults_to_the_old_rule():
    # finding 12: unset -> half of min_ref_margin, whatever a run config sets min_ref_margin to
    assert load_config()["thresholds"]["min_ref_margin_hidden"] is None


def test_statement_science_words_stay_clean():
    # finding 13 (+ real share lines: lever arm, convergence audit)
    for line in ("Measure gate voltage in volts.", "Estimate the strain hardening exponent.",
                 "the LiDAR is mounted with a different lever arm;", "every case must pass a convergence audit.",
                 "See Fig. S7 for the layout.", "Use the 2019 data revision."):
        assert not statement_clean(line), line
    for line in ("## Output-contract clarification (audit copy)", "## Additional hidden-output checks (audit revision)",
                 "The S7 gate rejects constant outputs.", "Hardening round 2 added noise.", "difficulty lever DSW03"):
        assert statement_clean(line), line


def test_timeout_constants_follow_source_order():
    # finding 14
    assert not package.run_solution_timeouts("T=60\nrun_solution(T)\nT=1200\n", 780)
    assert package.run_solution_timeouts("T=60\ndef t():\n    run_solution(T)\nT=1200\n", 780)
    assert package.run_solution_timeouts("T=1200\nrun_solution(T)\n", 780)


def test_small_copied_files_count_as_public(tmp_path):
    # finding 9: an 800-byte case file copied into the hidden set
    pub, hid = tmp_path / "data", tmp_path / "hidden"
    (hid / "A").mkdir(parents=True)
    pub.mkdir()
    (pub / "cases.csv").write_bytes(b"c" * 800)
    (pub / "README").write_text("public")
    (hid / "A/cases.csv").write_bytes(b"c" * 800)
    r = gates.hidden_duplicates(pub, hid)
    assert not r["ok"] and r["mostly_public"] == {"A": 1.0}


def test_s10_keeps_an_existing_release(tmp_path, monkeypatch):
    # finding 6: manifest already holds the item (crash before "done"): no re-decision, no withdrawal
    store = Store(tmp_path)
    it = store.item("x")
    (tmp_path / "release").mkdir()
    (tmp_path / "release/manifest.jsonl").write_text('{"item_id":"x"}\n')
    P = SimpleNamespace(store=store, cfg=load_config())
    monkeypatch.setattr(diversity, "admission", lambda *a: pytest.fail("re-decided an existing release"))
    stages_b.s10(P, it)
    assert it.state["status"] == "done"


def test_orphan_reservation_settled_before_relaunch(tmp_path, monkeypatch):
    # finding 4: crash after run.json went back to intent but before the settle -> settled on the next pass,
    # as diversity_parked (finding 7: not counted as a model call)
    calls = []
    P = SimpleNamespace(ledger=SimpleNamespace(settle=lambda *a, **k: calls.append((a, k))))
    recp = tmp_path / "run.json"
    rec = {"state": "intent", "ledger_call": None, "orphan_calls": [13]}
    write_json(recp, rec)
    stages_a._settle_orphans(P, recp, rec, "x")
    assert calls[0][0] == (13, 0.0) and calls[0][1]["kind"] == "diversity_parked"
    assert json.loads(recp.read_text())["orphan_calls"] == []


def test_parity_matches_renamed_hidden_metrics():
    # released prod-code names: public_rr10 vs hidden_rr10_set_a, public_rel_rmse vs hidden_large_rel_rmse_max
    rows = [{"name": n} for n in ("public_rr10", "hidden_rr10_set_a", "public_rel_rmse", "hidden_large_rel_rmse_max",
                                  "public_mean_clean_error", "hidden_max_clean_error", "public_host_accuracy")]
    assert gates.hidden_parity(rows, {"metrics": []})["missing_hidden"] == ["host_accuracy"]


def test_public_probe_command_survives_the_shell(tmp_path, monkeypatch):
    # The public-scope command runs as `bash -lc <cmd>` in the sandbox. A fake sandbox that only greps the command
    # hid a quoting bug (repr(arm) closed the single-quoted python -c argument: NameError one_confident). Run the
    # real command through bash with the probe module on PYTHONPATH.
    import shutil as _sh
    cfg = load_config()
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "public.csv").write_text("id,label,confidence\n" + "".join(f"{i},x,{i/10}\n" for i in range(7)))
    seen = []
    def run(cfg, mounts, cmd, timeout, name, *a, **k):
        root = tmp_path / ("root-" + str(len(seen)))
        out = root / "app/output"
        _sh.copytree(mounts[0][0], out)
        _sh.copy(mounts[1][0], root / "probe_outputs.py")
        p = subprocess.run(["bash", "-c", cmd.replace("cd / ", f"cd {root} ").replace("/app/output", str(out))],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root)))
        seen.append((cmd, p.returncode, p.stderr))
        if p.returncode == 0:
            _sh.rmtree(mounts[0][0]); _sh.copytree(out, mounts[0][0])
        return dict(rc=p.returncode, out=p.stdout, err=p.stderr)
    monkeypatch.setattr(runner.sandbox, "run", run)
    monkeypatch.setattr(runner, "test", lambda *a, **k: dict(metrics=[], rc=0, passed=False, reward="0", failed=["x"]))
    gen = tmp_path / "gen"
    gen.mkdir()
    for arm in ("one_confident", "hidden_garbage"):
        (tmp_path / arm).mkdir()
        runner.output_probe(cfg, tmp_path, gen, ref, tmp_path / arm, arm, True)
    public = [s for s in seen if "probe_outputs import mutate" in s[0]]
    assert public and all(rc == 0 for _, rc, _ in public), public


def test_wrapper_requires_dropped_privileges(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["wrapper", "A", str(tmp_path)])
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("started reference"))
    with pytest.raises(PermissionError):
        probe_outputs.hidden_main("hidden_garbage", "A")


@pytest.mark.parametrize("version,enforced,per_set", [
    ("v2.7", False, False), ("v2.8", True, False), (PROMPT_VERSION, True, True),
    ("cg-v3", False, False), ("cg-v4", True, False), (CODE_PROMPT_VERSION, True, True),
])
@pytest.mark.parametrize("has_bias", [True, False])
def test_runtime_parity_by_hidden_set(tmp_path, monkeypatch, wired_gates, version, enforced, per_set, has_bias):
    cfg, spec, _, _ = wired_gates
    cfg["gen"]["seeds"] = [0]
    spec.update(prompt_version=version, metrics=[{"name": n} for n in
                ("public_error", "public_bias", "hidden_error", "hidden_bias")])
    generate, verify = runner.generate, runner.test
    def two_sets(cfg, task, seed, path):
        result = generate(cfg, task, seed, path)
        (path / "tests/hidden/d").rename(path / "tests/hidden/A")
        (path / "tests/hidden/B").mkdir()
        return result
    def scores(cfg, task, gen, out, work, tag):
        result = verify(cfg, task, gen, out, work, tag)
        if tag.startswith("ref"):
            log = work / "metrics.jsonl"
            monkeypatch.setattr(templates_metrics, "_PATH", str(log))
            for name, scope in [("public_error", None), ("public_bias", None),
                                ("hidden_error", "A"), ("hidden_bias", "B")]:
                if not has_bias and name == "hidden_bias":
                    continue
                value = 0.8
                if scope and per_set:
                    other = "A" if scope == "B" else "B"
                    templates_metrics.hidden_value(gen / "tests/hidden" / other, name, value)
                    value = templates_metrics.hidden_value(gen / "tests/hidden" / scope, name, value)
                templates_metrics.check(name, value, 1., "<=")
            result["metrics"] = [json.loads(line) for line in log.read_text().splitlines()]
        return result
    monkeypatch.setattr(runner, "generate", two_sets)
    monkeypatch.setattr(runner, "test", scores)
    rep = gates.run_gates(cfg, tmp_path, spec, {}, 1)
    parity = rep["details"]["hidden_parity"][0]
    if per_set:
        assert parity["missing_hidden"] == {"A": ["bias"], "B": ["error"] if has_bias else ["bias", "error"]}
        assert parity["evaluated_hidden"] == {"A": ["error"], "B": ["bias"] if has_bias else []}
    else:
        assert parity["missing_hidden"] == ([] if has_bias else ["bias"])
    assert parity["enforced"] == enforced
    assert ("hidden_parity" in rep["blocking"]) == (enforced and (per_set or not has_bias))


@pytest.mark.parametrize("rows,expected_count", [(["1,7", "2,7", "3,7"], 0), (["1,7", "2,9"], 1)])
def test_garbage_preserves_keys_and_counts_changes(tmp_path, rows, expected_count):
    path = tmp_path / "results.csv"
    original = ("id,value\n" + "\n".join(rows) + "\n").encode()
    path.write_bytes(original)
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == expected_count
    if expected_count:
        changed = list(csv.reader(path.read_text().splitlines()))[1:]
        assert [r[0] for r in changed] == ["1", "2"]
        assert [r[1] for r in changed] != ["7", "9"]
    else:
        assert path.read_bytes() == original


def test_failed_identity_control_is_inconclusive(tmp_path, monkeypatch):
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch, sets=("A",))
    verify = _verifier(tmp_path, gen, table, score_sets=("A",))
    calls = []
    def wrapper_fails(cfg, task, gen, out, work, tag):
        result = verify(cfg, task, gen, out, work, tag)
        calls.append(result["reward"])
        return dict(result, reward="0", passed=False, failed=["runtime"])
    monkeypatch.setattr(runner, "test", wrapper_fails)
    result = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "hidden_garbage", True)
    assert not result["ok"] and result["inconclusive"]
    assert calls.count("1") == 1 and len(calls) <= 2


@pytest.mark.parametrize("rc,reward", [(124, "1"), (0, None)])
def test_zero_changes_still_requires_verifier_result(tmp_path, monkeypatch, rc, reward):
    cfg, ref, gen, _ = _probe_fixture(tmp_path, monkeypatch, sets=("A",))
    (ref / "solution_cli.py").write_text("import pathlib, sys\n(pathlib.Path(sys.argv[2])/'x.fits').touch()")
    verify = _verifier(tmp_path, gen, "", score_sets=(), public_count=False)
    def timeout(*args):
        return dict(verify(*args), rc=rc, reward=reward)
    monkeypatch.setattr(runner, "test", timeout)
    result = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "hidden_garbage", True)
    assert not result["ok"] and not result.get("na")


def test_hidden_scope_named_public(tmp_path, monkeypatch):
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch, sets=("public",))
    monkeypatch.setattr(runner, "test", _verifier(tmp_path, gen, table, score_sets=()))
    result = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "one_confident", True)
    assert result["scopes"]["public"]["reward"] == "0"
    assert result["accepted_scopes"] == ["hidden:public"]


@pytest.mark.parametrize("version", [PROMPT_VERSION, CODE_PROMPT_VERSION])
def test_spec_quote_length_after_normalization(version):
    def errors(where, v):
        entry = dict(assumption="noise", stated=True, where=where)
        return validate_spec(dict(SPEC, generator_assumptions=[entry]), load_config(), v)
    assert errors("  A   Gaussian\n", version)
    assert not errors("  A   Gaussian\n", "cg-v3")
    assert not errors("  12345\n  678901", version)  # 12 normalized characters


def test_diversity_resume_reuses_committed_reservation(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["budget"].update(daily_usd_total=10, run_calls_per_model={"model": 1})
    ledger = budget.Ledger(tmp_path / "ledger.sqlite", cfg)
    store = Store(tmp_path)
    it = store.item("x")
    P = SimpleNamespace(ledger=ledger, cfg=cfg, store=store)
    job = "tbsgen-x-s5g1"
    conf = dict(job_name=job, agents=[{"model_name": "model"}], jobs_dir=str(tmp_path))
    config = it.p("S5G", "job.json")
    write_json(config, conf)
    old = ledger.reserve("model", "S5G", it.item_id, True, est_usd=1.)
    recp = it.p("S5G", "run.json")
    write_json(recp, dict(state="reserved", job_name=job, job_dir=str(tmp_path / job),
                          config=str(config), ledger_call=old))
    def step():
        stages_a._agent_stage(P, it, "S5G", "", [], 30, 1, 1, True)
    monkeypatch.setattr(stages_a, "_agent_admission", lambda *a: "diversity:template_full:x")
    step()
    assert it.state["status"] == "parked"
    monkeypatch.setattr(stages_a, "_agent_admission", lambda *a: None)
    # Stand-in for Harbor's already-created config; no external launcher is needed.
    write_json(tmp_path / job / "config.json", conf)
    def crash(path, data):
        if path == recp and data.get("state") == "reserved":
            raise InterruptedError("after reserve commit")
        write_json(path, data)
    with monkeypatch.context() as m:
        m.setattr(codeagent, "write_json", crash)
        with pytest.raises(InterruptedError):
            step()
    with ledger._db() as db:
        held = db.execute("SELECT id FROM calls WHERE status='reserved'").fetchall()
    assert len(held) == 1
    step()
    rec = json.loads(recp.read_text())
    assert rec["state"] == "launched" and rec["ledger_call"] == held[0][0]
    with ledger._db() as db:
        assert db.execute("SELECT count(*) FROM calls").fetchone() == (2,)


def test_release_retry_after_append_is_idempotent(tmp_path, monkeypatch):
    store = Store(tmp_path)
    P = SimpleNamespace(cfg=load_config(), store=store)
    it = store.item("x")
    it.state.update(stage="S10", calib_epoch=1)
    it.save()
    write_json(it.p("pkg", "data.json"), 1)
    with monkeypatch.context() as m:
        save = it.save
        def crash():
            if it.state["status"] == "done":
                raise InterruptedError("before done save")
            save()
        m.setattr(it, "save", crash)
        with pytest.raises(InterruptedError):
            stages_b.s10(P, it)
    manifest = tmp_path / "release/manifest.jsonl"
    assert len(manifest.read_text().splitlines()) == 1
    it.load()
    stages_b.s10(P, it)
    assert it.state["status"] == "done" and len(manifest.read_text().splitlines()) == 1
    it.state["calib_epoch"] += 1
    stages_b.s10(P, it)
    write_json(it.p("pkg", "data.json"), 2)
    stages_b.s10(P, it)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len({(r["pkg_sha"], r["calib_epoch"]) for r in records}) == len(records) == 3


def test_class_body_uses_definition_time_timeout():
    src = "T=60\nclass C:\n    run_solution(T)\n    def method(self): run_solution(T)\nT=1200"
    assert package.run_solution_timeouts(src, 780) == ["line 4: run_solution timeout 1200s >= verifier budget 780s"]


def test_hidden_measurements_reach_check_record(tmp_path, monkeypatch):
    log = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(templates_metrics, "_PATH", str(log))
    values = [templates_metrics.hidden_value(d, "hidden_error", v) for d, v in
              [("/tests/hidden/A", 0.5), ("/tests/hidden/B", 0.8)]]
    templates_metrics.check("hidden_error", templates_metrics.hidden_aggregate("max", values), 1., "<=")
    row = json.loads(log.read_text())
    assert row["hidden_sets"] == ["A", "B"] and row["value"] == 0.8

def test_constant_answer_scope_is_na(tmp_path, monkeypatch):
    cfg, ref, gen, _ = _probe_fixture(tmp_path, monkeypatch, sets=("A",))
    table = "value,case_id,trial_key\n7,1,a\n7,2,b\n7,3,c\n"
    (gen / "tests/hidden/A/data").write_text(table)
    monkeypatch.setattr(runner, "test", _verifier(tmp_path, gen, table, score_sets=("A",), public_count=False))
    result = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "hidden_garbage", True)
    assert result["na"] and result["scopes"] == {"hidden:A": "na"}


def test_check_credits_only_its_argument(tmp_path, monkeypatch):
    log = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(templates_metrics, "_PATH", str(log))
    a = templates_metrics.hidden_value("hidden/A", "hidden_error", 0.5)
    templates_metrics.hidden_value("hidden/B", "hidden_error", 0.5)
    for value in (a, 0.5):
        templates_metrics.check("hidden_error", value, 1., "<=")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r.get("hidden_sets", []) for r in rows] == [["A"], []]


def test_single_set_requires_provenance(tmp_path, wired_gates):
    cfg, spec, _, _ = wired_gates
    cfg["gen"]["seeds"] = [0]
    write_json(tmp_path / "_spec/provenance.json", {"prompt_version": CODE_PROMPT_VERSION})
    rep = gates.run_gates(cfg, tmp_path, spec, {}, 1)
    assert rep["details"]["hidden_parity"][0]["missing_hidden"] == {"d": ["mae", "score"]}
    assert "hidden_parity" in rep["blocking"]


@pytest.mark.parametrize("values", [["0.0"], ["1", "1.0"], ["NaN", "nan"]])
@pytest.mark.parametrize("ext", ["csv", "txt"])
def test_rotation_compares_numeric_values(tmp_path, values, ext):
    path = tmp_path / ("out." + ext)
    original = "value\n" + "\n".join(values) + "\n"
    path.write_text(original)
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == 0
    assert path.read_text() == original


@pytest.mark.parametrize("answer,values", [
    ("predicted_class_id", ["7", "8", "7"]), ("predicted_class_id", ["7", "8"]),
    ("site_id", ["7", "8", "7"]),
])
def test_id_like_answer_is_rotated(tmp_path, answer, values):
    path = tmp_path / "out.csv"
    path.write_text(f"id,{answer}\n" + "".join(f"{i},{v}\n" for i, v in enumerate(values)))
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == 1
    rows = list(csv.reader(path.read_text().splitlines()))[1:]
    assert [r[0] for r in rows] == [str(i) for i in range(len(values))]
    assert [r[1] for r in rows] != values


@pytest.mark.parametrize("arm", ["hidden_garbage", "one_confident"])
def test_control_rewrites_and_covers_all_sets(tmp_path, monkeypatch, arm):
    cfg, ref, gen, table = _probe_fixture(tmp_path, monkeypatch)
    verify = _verifier(tmp_path, gen, table)
    controls = []

    def timed_verifier(cfg, task, gen, out, work, tag):
        result = verify(cfg, task, gen, out, work, tag)
        if tag.endswith("__control"):
            controls.append(tag)
            for name in ("A", "B"):
                data = (tmp_path / f"res-{tag}-{name}" / f"{name}.csv").read_bytes()
                # csv.writer emits CRLF after actually parsing and rewriting the LF reference.
                assert b"\r\n" in data and data.decode().splitlines() == table.splitlines()
        else:
            # The all-set control also needs the same timed I/O as a trial on non-target sets.
            for name in ("A", "B"):
                assert b"\r\n" in (tmp_path / f"res-{tag}-{name}" / f"{name}.csv").read_bytes()
        return result

    monkeypatch.setattr(runner, "test", timed_verifier)
    assert runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, arm, True)["ok"]
    assert len(controls) == 1


def test_ans_uses_scored_reference_as_control(tmp_path, monkeypatch):
    cfg, ref, gen, _ = _probe_fixture(tmp_path, monkeypatch)
    calls = []

    def verify(cfg, task, gen, out, work, tag):
        calls.append(tag)
        return dict(rc=0, reward="0", failed=["count"], metrics=[])

    monkeypatch.setattr(runner, "test", verify)
    result = runner.output_probe(cfg, tmp_path, gen, ref, tmp_path, "one_confident", False)
    assert len(calls) == 1 and result["ok"]


@pytest.mark.parametrize("version", ["cg-v3", "v2.7"])
def test_legacy_explicit_assumption_passes_both_stages(tmp_path, wired_gates, version):
    cfg, spec, _, task = wired_gates
    entry = dict(assumption="noise distribution", stated=True, where="Gaussian")
    assert not validate_spec(dict(SPEC, generator_assumptions=[entry]), cfg, version)
    spec.update(generator_assumptions=[entry])
    write_json(tmp_path / "_spec/provenance.json", {"prompt_version": version})
    with (task / "instruction.md").open("a") as f:
        f.write("Gaussian noise.\n")
    assert gates.run_gates(cfg, tmp_path, spec, {}, 1)["gates"]["assumptions_stated"]
    (task / "instruction.md").write_text("The noise distribution is unknown.")
    assert not gates.run_gates(cfg, tmp_path, spec, {}, 2)["gates"]["assumptions_stated"]


def test_resume_b88e514_reservation(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["budget"].update(daily_usd_total=10, run_calls_per_model={"model": 1})
    it = Store(tmp_path).item("x")
    job = "tbsgen-x-s5g1"
    conf = dict(job_name=job, agents=[{"model_name": "model"}], jobs_dir=str(tmp_path))
    config = it.p("S5G", "job.json")
    write_json(config, conf)
    rec = dict(job_name=job, state="intent", ts=100., kind="S5G", job_dir=str(tmp_path / job),
               config=str(config), budget=dict(timeout_s=30, max_usd=1.), est_usd=1., prompt_version="cg-v4-2026-11-24")
    write_json(it.p("S5G", "run.json"), rec)
    # Unmigrated b88e514 calls table and the exact columns its reserve() inserted.
    path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(budget.SCHEMA)
        db.execute("ALTER TABLE calls ADD COLUMN cost_kind TEXT")
        assert "reservation_key" not in {c[1] for c in db.execute("PRAGMA table_info(calls)")}
        old = db.execute("INSERT INTO calls(day,started,model,real,stage,item,status,est_usd,prompt_sha,harden,variant) "
                         "VALUES('2026-11-24',101,'model',1,'S5G','x','reserved',1,'',NULL,NULL)").lastrowid
    ledger = budget.Ledger(path, cfg)
    cfg["code"].update(agent_harbor_root=str(tmp_path), agent_launcher="launch")
    launches = []

    def launch(cmd, **kwargs):
        launches.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(codeagent.subprocess, "run", launch)
    P = SimpleNamespace(ledger=ledger, cfg=cfg, store=Store(tmp_path))

    def step():
        stages_a._agent_stage(P, it, "S5G", "", [], 30, 1, 1, True)

    # Kill the upgrade after it committed the key, before that key made it to disk in run.json.
    def crash(path, data):
        if data.get("state") == "reserved":
            raise InterruptedError
        write_json(path, data)

    with monkeypatch.context() as m:
        m.setattr(codeagent, "write_json", crash)
        with pytest.raises(InterruptedError):
            step()
    step()
    assert len(launches) == 1 and job in launches[0]
    resumed = json.loads(it.p("S5G", "run.json").read_text())
    assert resumed["state"] == "launched" and resumed["ledger_call"] == old
    with ledger._db() as db:
        assert db.execute("SELECT id,reservation_key FROM calls").fetchall() == [(old, resumed["reservation_key"])]


def test_resume_b88e514_manifest(tmp_path):
    store = Store(tmp_path)
    it = store.item("x")
    it.state.update(stage="S10", calib_epoch=2, calib_label="hard")
    cfg = load_config()
    cfg["review"]["enabled"] = False
    manifest = tmp_path / "release/manifest.jsonl"
    manifest.parent.mkdir()
    previous = dict(item_id="x", pkg=str(it.p("pkg")), label="hard", channel="main", calib_runs=[], result_types=[],
                    calib_epoch=2, review=None, difficulty=None, ts=42.)
    manifest.write_text(json.dumps(previous) + "\n")
    stages_b.s10(SimpleNamespace(cfg=cfg, store=store), it)
    assert it.state["status"] == "done"
    assert manifest.read_text().splitlines() == [json.dumps(previous)]


def test_release_records_changed_review_and_label(tmp_path):
    store = Store(tmp_path)
    it = store.item("x")
    it.state.update(stage="S10", calib_epoch=2, calib_label="hard")
    P = SimpleNamespace(cfg=load_config(), store=store)
    P.cfg["review"]["enabled"] = False
    stages_b.s10(P, it)
    P.cfg["review"]["enabled"] = True
    it.state["review_round"] = 1
    write_json(it.p("review/verdict.json"), dict(_normalized=dict(verdict="pass"), round=1,
               calib_epoch=2, pkg_sha=stages_b.calib.pkg_sha(it.p("pkg"))))
    stages_b.s10(P, it)
    stages_b.s10(P, it)
    it.state["calib_label"] = "medium"
    stages_b.s10(P, it)
    records = [json.loads(line) for line in (tmp_path / "release/manifest.jsonl").read_text().splitlines()]
    assert [(r["label"], (r["review"] or {}).get("verdict")) for r in records] == [
        ("hard", None), ("hard", "pass"), ("medium", "pass")]


def test_nested_class_uses_enclosing_function_or_module():
    src = ("T=60\nclass Outer:\n    T=1200\n    class Inner:\n        run_solution(T)\n"
           "def build():\n    T=900\n    class Outer:\n        T=60\n        class Inner:\n"
           "            run_solution(T)\nT=700\n")
    assert package.run_solution_timeouts(src, 780) == ["line 11: run_solution timeout 900s >= verifier budget 780s"]


@pytest.mark.parametrize("ext", ["json", "npy", "npz"])
def test_numeric_zero_format_is_unchanged(tmp_path, ext):
    path = tmp_path / ("scores." + ext)
    if ext == "json":
        path.write_text('["0.0", "-0"]')
    elif ext == "npy":
        np.save(path, np.array([-0.0]))
    else:
        np.savez(path, score=np.array(["0.0", "-0"]))
    original = path.read_bytes()
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("ext", ["json", "npy", "npz"])
def test_key_uniqueness_across_formats(tmp_path, ext):
    path = tmp_path / ("scores." + ext)
    fields = dict(id=[1, 2, 3], site_id=[7, 8, 7], predicted_class_id=[7, 8, 9])
    if ext == "json":
        path.write_text(json.dumps([dict(zip(fields, row)) for row in zip(*fields.values())]))
    elif ext == "npy":
        np.save(path, np.array(list(zip(*fields.values())), dtype=[(k, int) for k in fields]))
    else:
        np.savez(path, **fields)
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == 1
    if ext == "json":
        rows = json.loads(path.read_text())
        result = {k: [r[k] for r in rows] for k in fields}
    elif ext == "npy":
        a = np.load(path)
        result = {k: a[k].tolist() for k in fields}
    else:
        with np.load(path) as z:
            result = {k: z[k].tolist() for k in fields}
    assert result == dict(id=[1, 2, 3], site_id=[0, 0, 0], predicted_class_id=[0, 0, 0])


def test_headerless_first_column_is_not_an_identifier(tmp_path):
    path = tmp_path / "answers.txt"
    path.write_text("1 7\n2 7\n")
    assert probe_outputs.mutate(tmp_path, "hidden_garbage") == 1
    assert path.read_text() == "2 7\n1 7\n"
