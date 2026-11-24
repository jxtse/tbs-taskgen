"""Compare reconstructed requests with a local tree of private capture files."""

import json

from .claude_wire import Session, is_boundary, request
from .core import lines, loads
from .profile import capture_profile


def difference(actual, expected, path="$"):
    if type(actual) is not type(expected):
        return f"{path} ({type(actual).__name__} vs {type(expected).__name__})"
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return f"{path} keys differ: {sorted(actual.keys() ^ expected.keys())}"
        for k in expected:
            if diff := difference(actual[k], expected[k], f"{path}.{k}"):
                return diff
    elif isinstance(actual, list):
        for i, (a, b) in enumerate(zip(actual, expected)):
            if diff := difference(a, b, f"{path}[{i}]"):
                return diff
        if len(actual) != len(expected):
            return f"{path} length {len(actual)} vs {len(expected)}"
    elif actual != expected:
        return f"{path} value differs"
    return None


def normalized(messages, proxy=False):
    for i, m in enumerate(messages):
        c = m["content"]
        if not isinstance(c, list):
            continue
        for b in c:
            if "signature" in b and not b["signature"]:
                del b["signature"]
            if proxy:
                b.pop("cache_control", None)
        if proxy:
            if i == 0:
                c[:] = [b for b in c if not b.get("text", "").startswith("<system-reminder>")]
            if len(c) == 1 and c[0]["type"] == "text":
                m["content"] = c[0]["text"]
    return messages


def post_leaf(session, summary):
    for end in session.windows():
        path = session.chain(end)
        for i, r in enumerate(path):
            if not is_boundary(r):
                continue
            c = session.summary(r)["message"]["content"]
            text = c if isinstance(c, str) else "".join(b.get("text", "") for b in c)
            if text.strip() == summary.strip():
                for j in range(i + 1, len(path)):
                    if path[j]["type"] == "assistant":
                        return path[j - 1]["uuid"]
                return end
    raise ValueError("capture summary not found on active branch")


def compare(path, profile, post):
    raw = loads(path.read_bytes())
    gold = raw.get("body", raw)
    agent = next((p for p in path.parents if p.name == "agent"), None)
    directory = (agent or path.parent) / "sessions"
    candidates = list(directory.glob("projects/*/*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"expected one native session, found {len(candidates)}")
    records = list(lines(candidates[0]))
    versions = {r["version"] for r in records if r.get("version")}
    if versions != {profile["version"]}:
        raise ValueError("session version missing or different from profile")
    session = Session(records)
    if post:
        expected = normalized(gold["messages"], proxy=True)
        summary = expected[0]["content"]
        leaf = post_leaf(session, summary if isinstance(summary, str) else summary[0]["text"])
        got = request(session, leaf, profile, {}, cache=False)
        return difference(normalized(got["messages"]), expected, "$.messages")
    sentinel = session.resume("ZZZ_RESUME_SENTINEL")
    leaf = session.after_attachments(sentinel) if sentinel else session.leaf
    models = [r["message"].get("model") for r in records if r["type"] == "assistant"
              and r["message"].get("model") not in (None, "<synthetic>")]
    # Request controls of the reference capture protocol, independent of the gold body.
    options = dict(model=models[-1], stream=True, max_tokens=128000, thinking={"type": "adaptive"},
                   output_config={"effort": "max"},
                   context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
    got = request(session, leaf, profile, options)
    return difference(got, {k: v for k, v in gold.items() if k != "metadata"})


def wire_check(root, profile=None):
    files = [sorted(root.rglob(pattern)) for pattern in ("first_request_raw.json", "post_compaction_request_*.json")]
    if profile is None and files[0]:
        version, data = capture_profile(files[0][0])
        profile = dict(version=version, **{k: loads(v) for k, v in data.items()})
    report = {}
    for post, (group, paths) in enumerate(zip(("first_request", "post_compaction"), files)):
        count = 0
        first = None
        for path in paths:
            try:
                if not profile:
                    raise ValueError("missing profile and no first capture")
                diff = compare(path, profile, post)
            except (OSError, ValueError, KeyError, IndexError, TypeError, StopIteration) as exc:
                diff = f"{type(exc).__name__}: {exc}"
            if diff is None:
                count += 1
            elif first is None:
                first = {"capture": str(path.relative_to(root)), "difference": diff}
        report[group] = dict(total=len(paths), exact=count, mismatch=len(paths) - count, first_difference=first)
    print(json.dumps(report, indent=2))
    return int(not any(files) or any(v["mismatch"] for v in report.values()))
