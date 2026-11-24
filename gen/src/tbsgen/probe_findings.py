"""Shared by the review-task verifier and the host when ingesting a verdict."""

import math
import re


_NOT_RUN = re.compile(r"\b(?:not\s+run|didn['’]t\s+run|could(?:\s+not|n['’]t)\s+run|unable\s+to\s+run|skipped)\b", re.I)


def has_evidence(ev) -> bool:
    if isinstance(ev, str):
        return len(ev.strip()) >= 8
    if isinstance(ev, list):
        return any(has_evidence(x) for x in ev)
    if isinstance(ev, dict):
        return any(has_evidence(v) for v in ev.values())
    return False


def finding_errors(verdict, suspected_scopes):
    findings = verdict.get("probe_findings", {})
    if not isinstance(findings, dict):
        return ["probe_findings must be an object keyed by scope ID"]
    issues = verdict.get("issues")
    blocking_ids = set()
    for issue in issues if isinstance(issues, list) else ():
        if (isinstance(issue, dict) and issue.get("severity") == "blocking"
                and isinstance(issue.get("id"), str) and has_evidence(issue.get("evidence"))):
            blocking_ids.add(issue["id"])
    errors = []
    for scope in dict.fromkeys([*suspected_scopes, *findings]):
        finding = findings.get(scope)
        if not isinstance(finding, dict):
            errors.append(f"missing probe finding: {scope}")
            continue
        reward = finding.get("observed_reward")
        if not (reward in ("0", "1") or type(reward) is int or type(reward) is float and math.isfinite(reward)):
            errors.append(f"{scope}: invalid or missing observed_reward")
        for field, minimum in (("command", 1), ("observation", 20)):
            value = finding.get(field)
            if not isinstance(value, str) or len(value.strip()) < minimum or _NOT_RUN.search(value):
                errors.append(f"{scope}: expected {field} from a completed run")
        if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
            errors.append(f"{scope}: missing reason")
        if finding.get("classification") == "grader_gap":
            issue = finding.get("issue_id")
            if not isinstance(issue, str) or issue not in blocking_ids:
                errors.append(f"{scope}: grader_gap requires an evidenced blocking issue")
        elif finding.get("classification") != "probe_artefact":
            errors.append(f"{scope}: expected grader_gap or probe_artefact")
    return errors
