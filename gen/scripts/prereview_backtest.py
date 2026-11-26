#!/usr/bin/env python3
"""Replay S9P against frozen S9R packages, with an independent ledger in the output directory."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbsgen.budget import Ledger
from tbsgen.calib import pkg_sha
from tbsgen.config import load_config, resolve_role
from tbsgen.models import get_client
from tbsgen.prereview import run_prereview
from tbsgen.report import configured_reviewer, round_verdict
from tbsgen.state import write_json


def _cited_snippets(filename: str, citation):
    paths = {filename, "./" + filename, "/review/pkg/" + filename}
    if filename.startswith("environment/data/"):
        paths.add("/app/data/" + filename.removeprefix("environment/data/"))
    if filename.startswith("tests/"):
        paths.add("/" + filename)
    if isinstance(citation, dict):
        if str(citation.get("file", "")) in paths:
            yield " ".join(str(citation.get(k, "")) for k in ("quote", "column", "key"))
    elif isinstance(citation, str):
        # A reviewer can cite several files in one string, including repeated references
        # to one file. Attach each excerpt only to its explicit, complete package path.
        names = "|".join(re.escape(path) for path in sorted(paths))
        pattern = rf"(?<![\w./-])(?:{names})(?![\w./-])"
        for match in re.finditer(pattern, citation):
            tail = citation[match.end():]
            yield re.split(r";|\n|(?<![\w./-])(?:[\w./-]+/)*[\w.-]+\.(?:py|md|txt|tsv|csv|json|npz|npy)\b",
                           tail, maxsplit=1)[0]


def evidence_overlaps(evidence: dict, citation) -> bool:
    """Suggest a pair only when the same file has an overlapping quote or exact identifier."""
    terms = [evidence.get(k) for k in ("quote", "column", "key")]
    for snippet in _cited_snippets(evidence["file"], citation):
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", snippet)
               for term in terms if isinstance(term, str) and term.strip()):
            return True
        quoted = re.findall(r"[`'\"“]([^`'\"”]+)[`'\"”]", snippet)
        if isinstance(citation, dict) and isinstance(citation.get("quote"), str):
            quoted.append(citation["quote"])
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", evidence["quote"])
               for term in quoted if len(term.strip()) >= 3):
            return True
    return False


def matches(findings: list, issues: list) -> list:
    pairs = []
    for i, finding in enumerate(findings, 1):
        for issue in issues:
            citations = issue.get("evidence") or []
            if not isinstance(citations, list):
                citations = [citations]
            if any(evidence_overlaps(finding["evidence"], ref) for ref in citations):
                pairs.append(dict(finding=i, opus_issue=issue["id"]))
    return pairs


def replay(run: Path, out: Path, rnd: int, items, repeat: int, config=None):
    if not (run / "items").is_dir():
        raise ValueError("run must contain an items directory")
    out = out.resolve()
    run = run.resolve()
    if (run.is_relative_to(out) or out.is_relative_to((run / "items").resolve())
            or (out.exists() and (not out.is_dir() or any(out.iterdir())))):
        raise ValueError("choose a dedicated empty output directory outside the items tree")
    cfg = load_config(config or run / "config.toml", {"run": {"root": str(out)}})
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger.sqlite", cfg)
    alias = resolve_role(cfg, "prereview")
    client = None

    def client_factory():
        nonlocal client
        if client is None:
            client = get_client(alias, cfg, ledger)
        return client

    selected = set(items or [])
    fallback_model = configured_reviewer(run)
    rows = []
    eligible = compared = flips = evidence_flips = errors = 0
    skipped = {}
    for directory in sorted((run / "items").glob(f"*/review/r{rnd}")):
        item = directory.parents[1].name
        if selected and item not in selected:
            continue
        task = directory / "task"
        bundle = task / "environment/review"
        package = bundle / "pkg" if bundle.is_dir() else task
        result = round_verdict(directory, fallback_model)
        if not result or not (package / "instruction.md").is_file():
            skipped[item] = "no eligible S9R verdict or package snapshot"
            continue
        verdict, issues = result
        if verdict["pkg_sha"] and verdict["pkg_sha"] != pkg_sha(package):
            skipped[item] = "verdict belongs to a different package"
            continue
        eligible += 1
        attempts = []
        baseline = None
        for trial in range(1, (repeat if selected or eligible <= 5 else 1) + 1):
            record = run_prereview(package, client_factory, item=item, generator_dir=bundle / "extra/generator")
            findings = record["validated"]
            pairs = matches(findings, issues)
            # Compare evidence anchors independently of the model's order or proposed phrasing.
            anchors = {(f["category"], f["evidence"]["file"], locator, f["evidence"][locator])
                       for f in findings for locator in ("line", "column", "key") if locator in f["evidence"]}
            changed = anchors_changed = None
            if record.get("error"):
                errors += 1
            elif trial == 1:
                baseline = anchors
            elif baseline is not None:
                changed = bool(anchors) != bool(baseline)
                anchors_changed = anchors != baseline
                compared += 1
                flips += changed
                evidence_flips += anchors_changed
            attempts.append(dict(record, trial=trial, tentative_matches=pairs, blocking_flip=changed,
                                 evidence_flip=anchors_changed))
            rows.append(dict(item=item, round=rnd, trial=trial, opus_verdict=verdict["verdict"],
                             reviewer_model=verdict["model"], reviewer_model_source=verdict["model_source"],
                             opus_blocking_ab=json.dumps(issues, ensure_ascii=False),
                             prereview_validated=json.dumps(findings, ensure_ascii=False),
                             tentative_matches=json.dumps(pairs), blocking_flip=changed, evidence_flip=anchors_changed,
                             error=record.get("error")))
        write_json(out / "items" / f"{item}.json", dict(item=item, round=rnd, opus=verdict, issues=issues,
                                                        attempts=attempts))
    fields = ("item", "round", "trial", "opus_verdict", "reviewer_model", "reviewer_model_source",
              "opus_blocking_ab", "prereview_validated", "tentative_matches", "blocking_flip", "evidence_flip", "error")
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(items=eligible, attempts=len(rows), errors=errors, repeat_comparisons=compared, blocking_flips=flips,
                   blocking_flip_rate=flips / compared if compared else None,
                   evidence_flip_rate=evidence_flips / compared if compared else None, skipped=skipped)
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="empty result directory (default: RUN/reports/prereview-backtest-rN)")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--items", nargs="+", metavar="ITEM")
    parser.add_argument("--repeat", type=int, default=1, help="attempts per selected item (first five if unfiltered)")
    parser.add_argument("--config", type=Path, help="model and budget settings (default: RUN/config.toml)")
    args = parser.parse_args(argv)
    if min(args.repeat, args.round) < 1:
        parser.error("--repeat and --round must be positive")
    out = args.out or args.run / "reports" / f"prereview-backtest-r{args.round}"
    try:
        result = replay(args.run, out, args.round, args.items, args.repeat, args.config)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(result, out=str(out)), indent=2))


if __name__ == "__main__":
    main()
