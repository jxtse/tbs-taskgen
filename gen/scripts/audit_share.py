#!/usr/bin/env python3
"""Audit released packages from a share without importing or executing their code."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbsgen import diversity, package, review
from tbsgen.gates import hidden_duplicates
from tbsgen.state import read_json
from tbsgen.statement import statement_clean, threshold_format


def audit(root, runs_root):
    with (root.parent / "INDEX.csv").open(newline="") as f:
        index = {r["task_id"]: r for r in csv.DictReader(f)}
    rows, records = [], []
    for pkg in sorted(p.parent for p in root.glob("*/task.toml")):
        item = pkg.name
        meta = tomllib.loads((pkg / "task.toml").read_text())
        prov = read_json(pkg / "provenance.json", {})
        spec = {}
        run = index[item].get("tbsgen_run")
        if run and runs_root:
            spec = read_json(runs_root / run / "items" / prov["item_id"] / "_spec/spec.json", {})
        # Older standalone packages expose competencies in TOML keywords and metric names in literal check calls.
        source = (pkg / "tests/test_outputs.py").read_text()
        calls = review.check_calls(source) or []
        thresholds = prov.get("thresholds") or {n: {"threshold": value} for n, value, _ in calls}
        metadata = dict(meta.get("metadata", {}), title=meta["task"]["description"],
                        competencies=meta["task"].get("keywords", []), metrics=[{"name": n} for n in thresholds])
        records.append(diversity.template(item, {**metadata, **spec}, prov))
        text = (pkg / "instruction.md").read_text()
        budget = meta["verifier"]["timeout_sec"] - 120
        time_errors, time_warnings = [], []
        for name in ("measure.py", "test_outputs.py"):
            path = pkg / "tests" / name
            if path.is_file():
                time_errors.extend(package.run_solution_timeouts(path.read_text(), budget, warnings=time_warnings))
        network, _ = package.network_errors(meta, (prov.get("network") or {}).get("mode", "agent-allowlist"))
        dup = hidden_duplicates(pkg / "environment/data", pkg / "tests/hidden", spec.get("shared_inputs", []))
        rows.append(dict(item=item, statement_clean=statement_clean(text), threshold_format=threshold_format(text, thresholds),
                         run_solution_timeout=time_errors, timeout_warnings=time_warnings, network_explicit=network,
                         hidden_duplicates=any(dup[k] for k in ("identical_sets", "identical_public", "mostly_public"))))
    return rows, diversity.summarize(records, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hard_dir", type=Path, help="share's 1-hard directory")
    parser.add_argument("--runs-root", type=Path, help="read the full specs for pipeline-authored packages")
    args = parser.parse_args()
    rows, report = audit(args.hard_dir, args.runs_root)
    keys = ("statement_clean", "threshold_format", "run_solution_timeout", "network_explicit", "hidden_duplicates")
    print("\t".join(("item", *keys)))
    for row in rows:
        status = {k: "FAIL" if row[k] else "ok" for k in keys}
        if row["timeout_warnings"] and not row["run_solution_timeout"]:
            status["run_solution_timeout"] = "WARN"
        print("\t".join([row["item"], *(status[k] for k in keys)]))
    print(json.dumps(dict(packages=len(rows), flagged={k: sum(bool(r[k]) for r in rows) for k in keys},
                          timeout_warnings={r["item"]: r["timeout_warnings"] for r in rows if r["timeout_warnings"]},
                          diversity=report), ensure_ascii=False))


if __name__ == "__main__":
    main()
