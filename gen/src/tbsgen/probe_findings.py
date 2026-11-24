"""Shared by the review-task verifier and the host when ingesting a verdict."""


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
    for scope in suspected_scopes:
        finding = findings.get(scope)
        if not isinstance(finding, dict):
            errors.append(f"missing probe finding: {scope}")
            continue
        for key in ("reproduction", "reason"):
            if not isinstance(finding.get(key), str) or not finding[key].strip():
                errors.append(f"{scope}: missing {key}")
        if finding.get("classification") == "grader_gap":
            issue = finding.get("issue_id")
            if not isinstance(issue, str) or issue not in blocking_ids:
                errors.append(f"{scope}: grader_gap requires an evidenced blocking issue")
        elif finding.get("classification") != "probe_artefact":
            errors.append(f"{scope}: expected grader_gap or probe_artefact")
    return errors
