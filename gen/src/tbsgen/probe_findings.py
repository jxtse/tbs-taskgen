"""Shared by the review-task verifier and the host when ingesting a verdict."""

import json
import logging
import math
from pathlib import Path
import re


CURRENT_CONTRACT = "s9r-v8"
JOURNAL_CONTRACTS = ("s9r-v6", "s9r-v7", CURRENT_CONTRACT)


def repro_command(scope):
    return ["python3", "/review/records/reproduce_scope.py", scope]


def has_evidence(ev) -> bool:
    if isinstance(ev, str):
        return len(ev.strip()) >= 8
    if isinstance(ev, list):
        return any(has_evidence(x) for x in ev)
    if isinstance(ev, dict):
        return any(has_evidence(v) for v in ev.values())
    return False


def read_repros(path):
    """Index the trial's helper records. A duplicate identifier cannot identify a run."""
    records = {}
    skipped = 0
    try:
        with Path(path).open("rb") as journal:
            for line in journal:
                try:
                    row = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                if not isinstance(row, dict) or not isinstance(row.get("repro_id"), str):
                    continue
                key = row["repro_id"]
                records[key] = None if key in records else row
    except FileNotFoundError:
        return {}
    if skipped:
        logging.warning("Skipped %d unparseable reproduction journal lines in %s", skipped, path)
    return records


def _matches_run(finding, scope, records, contract):
    key = finding.get("repro_id")
    row = records.get(key) if isinstance(key, str) and key else None
    reward = finding["observed_reward"]
    if isinstance(reward, str):
        reward = int(reward)  # only the legacy spellings "0" and "1" are accepted
    return (isinstance(row, dict) and row.get("scope_id") == scope
            and type(row.get("exit_code")) is int
            and (contract == "s9r-v6" or row["exit_code"] == 0 and row.get("command") == repro_command(scope))
            and isinstance(row.get("output_sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", row["output_sha256"])
            and type(row.get("reward")) in (int, float)
            and row["reward"] == reward)


def _optional_repro_errors(entry, records, location, scope=None):
    if "repro_id" in entry:
        key = entry["repro_id"]
        row = records.get(key) if isinstance(key, str) and key else None
        if not isinstance(row, dict):
            return [f"{location}: unknown or ambiguous repro_id"]
        errors = []
        if scope is not None and row.get("scope_id") not in (scope, None, ""):
            errors.append(f"{location}: cited repro record belongs to scope {row.get('scope_id')!r}")
        reward = entry.get("observed_reward")
        if reward is not None:
            try:
                if float(reward) != row.get("reward"):
                    errors.append(f"{location}: observed_reward does not match the cited repro record")
            except (TypeError, ValueError):
                errors.append(f"{location}: invalid observed_reward")
        return errors
    return []


def finding_errors(verdict, suspected_scopes, contract=CURRENT_CONTRACT, repros=None):
    if contract == "s9r-v1":
        return []
    if contract not in ("s9r-v2", "s9r-v3", "s9r-v4", "s9r-v5", *JOURNAL_CONTRACTS):
        return ["unknown review contract"]
    findings = verdict.get("probe_findings", {})
    if not isinstance(findings, dict):
        return ["probe_findings must be an object keyed by scope ID"]
    issues = verdict.get("issues")
    blocking_ids = set()
    errors = []
    records = repros or {}
    for i, issue in enumerate(issues if isinstance(issues, list) else ()):
        if (isinstance(issue, dict) and issue.get("severity") == "blocking"
                and isinstance(issue.get("id"), str) and has_evidence(issue.get("evidence"))):
            blocking_ids.add(issue["id"])
        if contract == "s9r-v8" and isinstance(issue, dict):
            errors.extend(_optional_repro_errors(issue, records, f"issues[{i}]"))
    scopes = suspected_scopes if contract == "s9r-v2" else [*suspected_scopes, *findings]
    for scope in dict.fromkeys(scopes):
        finding = findings.get(scope)
        if not isinstance(finding, dict):
            errors.append(f"missing probe finding: {scope}")
            continue
        extra = contract == "s9r-v8" and scope not in suspected_scopes
        if finding.get("classification") == "grader_gap":
            issue = finding.get("issue_id")
            if not isinstance(issue, str) or issue not in blocking_ids:
                errors.append(f"{scope}: grader_gap requires an evidenced blocking issue")
        elif extra:
            errors.append(f"{scope}: additional findings must link an evidenced blocking issue as grader_gap")
        elif finding.get("classification") != "probe_artefact":
            errors.append(f"{scope}: expected grader_gap or probe_artefact")
        if extra:
            errors.extend(_optional_repro_errors(finding, records, scope, scope))
            continue
        if contract == "s9r-v2":
            if not isinstance(finding.get("reproduction"), str) or not finding["reproduction"].strip():
                errors.append(f"{scope}: missing reproduction")
        else:
            reward = finding.get("observed_reward")
            if not (reward in ("0", "1") or type(reward) is int or type(reward) is float and math.isfinite(reward)):
                errors.append(f"{scope}: invalid or missing observed_reward")
            elif contract in JOURNAL_CONTRACTS and not _matches_run(finding, scope, records, contract):
                errors.append(f"{scope}: no matching reproduction record for scope, command and reward")
            if contract in JOURNAL_CONTRACTS:
                fields = [("observation", 20)]
                if contract == "s9r-v6":
                    fields.append(("command", 1))
                for field, minimum in fields:
                    value = finding.get(field)
                    if not isinstance(value, str) or len(value.strip()) < minimum:
                        errors.append(f"{scope}: missing or short {field}")
            else:
                from .legacy_probe_findings import legacy_execution_errors
                errors.extend(legacy_execution_errors(scope, finding, contract))
        if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
            errors.append(f"{scope}: missing reason")
    return errors
