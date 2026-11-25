#!/usr/bin/env python3
"""Installed as tbs-repro in the review image."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid


def record(scope, command, logs):
    path = logs / "review/repro.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_file = logs / "verifier/reward.txt"
    digest = hashlib.sha256()
    with path.open("a+b") as journal:
        # Graders share a single reward file. Only a fresh reward belongs to this run.
        fcntl.flock(journal, fcntl.LOCK_EX)
        reward_file.unlink(missing_ok=True)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                digest.update(chunk)
                sys.stderr.buffer.write(chunk)
            code = proc.wait()
        try:
            reward = float(reward_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            reward = None
        row = dict(repro_id=uuid.uuid4().hex, scope_id=scope, command=command, exit_code=code,
                   reward=reward if reward is not None and math.isfinite(reward) else None,
                   output_sha256=digest.hexdigest())
        journal.seek(0, 2)
        prefix = b""
        if journal.tell():
            journal.seek(-1, 2)
            if journal.read(1) != b"\n":
                prefix = b"\n"
        journal.write(prefix + json.dumps(row).encode() + b"\n")
    return row


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit("usage: tbs-repro <scope_id> -- <command...>")
    result = record(sys.argv[1], sys.argv[3:], Path("/logs"))
    print(json.dumps(result))
    sys.exit(result["exit_code"] if result["exit_code"] >= 0 else 128 - result["exit_code"])
