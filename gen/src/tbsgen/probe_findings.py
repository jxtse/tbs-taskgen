"""Shared by the review-task verifier and the host when ingesting a verdict."""

import math
import re
import shlex


CURRENT_CONTRACT = "s9r-v5"
_NEVER_EXECUTED = re.compile(r"\b(?:(?:never|(?:did\s+)?not|didn['’]t)\s+(?:run|executed?)|"
                             r"(?:was|were)\s+never\s+(?:run|executed))\b", re.I)
_NOT_RUN = re.compile(r"\b(?:(?:did\s+)?not\s+run|didn['’]t\s+run|was\s+not\s+executed|"
                      r"could(?:\s+not|n['’]t)\s+run|unable\s+to\s+run|skipped\s+running|no\s+run)\b", re.I)
_V3_NOT_RUN = re.compile(r"\b(?:not\s+run|didn['’]t\s+run|could(?:\s+not|n['’]t)\s+run|"
                         r"unable\s+to\s+run|skipped)\b", re.I)


def _looks_like_command(value):
    if len(value.strip()) < 4:
        return False
    try:
        tokens = shlex.split(value)
    except ValueError:
        return False
    if not tokens:
        return False
    # Paths, conventional executable names, and custom tools invoked with options.
    return any("/" in t and t.rsplit("/", 1)[-1] for t in tokens) or bool(
        re.fullmatch(r"(?:python[0-9.]*|pytest|bash|sh|env|make|ctest|just|uv|node|npm|cargo|go|ruby|Rscript|"
                     r"echo|cat|grep|awk|sed|diff|cmp|test|true|false|"
                     r"[a-zA-Z][\w.-]*[-_.][\w.-]+)", tokens[0])
        or re.fullmatch(r"[a-zA-Z][\w.-]*", tokens[0]) and any(t.startswith("-") for t in tokens[1:]))


def has_evidence(ev) -> bool:
    if isinstance(ev, str):
        return len(ev.strip()) >= 8
    if isinstance(ev, list):
        return any(has_evidence(x) for x in ev)
    if isinstance(ev, dict):
        return any(has_evidence(v) for v in ev.values())
    return False


def finding_errors(verdict, suspected_scopes, contract=CURRENT_CONTRACT):
    if contract == "s9r-v1":
        return []
    if contract not in ("s9r-v2", "s9r-v3", "s9r-v4", CURRENT_CONTRACT):
        return ["unknown review contract"]
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
    scopes = suspected_scopes if contract == "s9r-v2" else [*suspected_scopes, *findings]
    for scope in dict.fromkeys(scopes):
        finding = findings.get(scope)
        if not isinstance(finding, dict):
            errors.append(f"missing probe finding: {scope}")
            continue
        if contract == "s9r-v2":
            if not isinstance(finding.get("reproduction"), str) or not finding["reproduction"].strip():
                errors.append(f"{scope}: missing reproduction")
        else:
            reward = finding.get("observed_reward")
            if not (reward in ("0", "1") or type(reward) is int or type(reward) is float and math.isfinite(reward)):
                errors.append(f"{scope}: invalid or missing observed_reward")
            negation = _V3_NOT_RUN if contract == "s9r-v3" else _NOT_RUN
            for field, minimum in (("command", 1), ("observation", 20)):
                value = finding.get(field)
                if (not isinstance(value, str) or len(value.strip()) < minimum or negation.search(value)
                        or contract == CURRENT_CONTRACT and _NEVER_EXECUTED.search(value)
                        or field == "command" and contract != "s9r-v3" and not _looks_like_command(value)):
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
