"""S8 local repair + preservation ("保工") review.

Repair input: failing gate log + depgraph + current files.  Output: only changed files.
Preservation review (mechanical, no model; cheap-model second opinion 待定):
  * no new files, no deleted files; only files that already exist may change
  * generator / test file count unchanged
  * test_outputs.py: number of check() calls does not drop; every threshold literal stays identical
    (thresholds are owned by S6B; a repair may not loosen them)
  * depgraph unchanged (the repair prompt does not receive write access to _spec/)
  * instruction.md: no new mention of derive/internal/hidden_truth secret names
Rejected repairs are recorded and count as a used round.
"""
from __future__ import annotations

import re
from pathlib import Path

CHECK_RE = re.compile(r"\bcheck\s*\(")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _check_calls(src: str) -> int:
    return len(CHECK_RE.findall(src))


def _threshold_literals(src: str) -> list[str]:
    """3rd positional argument literals of check(name, value, THRESHOLD, direction)"""
    out = []
    for m in re.finditer(r"\bcheck\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", src):
        parts = [p.strip() for p in re.split(r",(?![^()]*\))", m.group(1))]
        if len(parts) >= 3:
            out.append(parts[2])
    return sorted(out)


def review(task: Path, patch: dict[str, str], depgraph: dict) -> dict:
    reasons = []
    existing = {str(p.relative_to(task)) for p in task.rglob("*") if p.is_file()}
    new = sorted(set(patch) - existing)
    if new:
        reasons.append(f"new files: {new}")
    for rel, body in patch.items():
        if rel.startswith("_spec") or ".." in rel or rel.startswith("/"):
            reasons.append(f"forbidden path {rel}")
    t = "tests/test_outputs.py"
    if t in patch and (task / t).exists():
        old = (task / t).read_text()
        if _check_calls(patch[t]) < _check_calls(old):
            reasons.append("fewer check() calls")
        if _threshold_literals(patch[t]) != _threshold_literals(old):
            reasons.append("threshold literals changed")
    if "instruction.md" in patch:
        secrets = [n for node in depgraph.get("nodes", []) if node.get("provision") in
                   ("derive", "internal", "hidden_truth") for n in node.get("secret_names", [])]
        old = (task / "instruction.md").read_text() if (task / "instruction.md").exists() else ""
        for s in secrets:
            pat = rf"\b{re.escape(s)}\b"
            if re.search(pat, patch["instruction.md"], re.I) and not re.search(pat, old, re.I):
                reasons.append(f"instruction now mentions secret {s}")
    return dict(ok=not reasons, reasons=reasons, files=sorted(patch), method="mechanical-v1 (cheap review 待定)")


def apply(task: Path, patch: dict[str, str]):
    for rel, body in patch.items():
        p = task / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def files_for_repair(task: Path, gate: dict, max_chars: int = 60000) -> str:
    """Files relevant to failing gates (tests + solution + generator by default), trimmed."""
    want = ["instruction.md", "tests/test_outputs.py", "tests/measure.py", "solution/solve.py",
            "generator/generate.py"] + sorted(str(p.relative_to(task)) for p in (task / "baselines").glob("*.py"))
    out, n = [], 0
    for rel in want:
        p = task / rel
        if p.exists():
            body = p.read_text()
            out.append(f"=== FILE: {rel} ===\n{body}")
            n += len(body)
            if n > max_chars:
                break
    return "\n".join(out)
