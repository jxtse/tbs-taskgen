"""Historical S9R execution checks, used on the host for pre-v6 review jobs."""
import re
import shlex


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


def legacy_execution_errors(scope, finding, contract):
    errors = []
    negation = _V3_NOT_RUN if contract == "s9r-v3" else _NOT_RUN
    for field, minimum in (("command", 1), ("observation", 20)):
        value = finding.get(field)
        if (not isinstance(value, str) or len(value.strip()) < minimum or negation.search(value)
                or contract == "s9r-v5" and _NEVER_EXECUTED.search(value)
                or field == "command" and contract != "s9r-v3" and not _looks_like_command(value)):
            errors.append(f"{scope}: expected {field} from a completed run")
    return errors
