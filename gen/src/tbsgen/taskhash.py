"""Review launcher preflight. Run on the build host with the exact rewritten Harbor config."""
import hashlib
import json
import sys
from pathlib import Path

REVIEW_HASH_ENV = "TBSGEN_REVIEW_TASK_SHA"


def pkg_sha(pkg: Path, *, exclude=()) -> str:
    """Content hash of a packaged task (paths + bytes)."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(pkg).rglob("*") if p.is_file()):
        if f.relative_to(pkg).as_posix() in exclude:
            continue
        h.update(str(f.relative_to(pkg)).encode() + b"\0")
        with f.open("rb") as source:
            hashlib.file_digest(source, lambda: h)
        h.update(b"\0")
    return h.hexdigest()[:16]


def verified_review_hash(config: Path) -> str:
    conf = json.loads(config.read_text())
    task, = conf["tasks"]
    agent, = conf["agents"]
    path = Path(task["path"])
    if not path.is_dir():
        raise ValueError(f"missing review task: {path}")
    actual = pkg_sha(path)
    if actual != agent.get("env", {}).get(REVIEW_HASH_ENV):
        raise ValueError(f"review task hash mismatch: {path}")
    return actual


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 taskhash.py WORKER_JOB_CONFIG")
    try:
        print(verified_review_hash(Path(sys.argv[1])))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        sys.exit(str(exc))
