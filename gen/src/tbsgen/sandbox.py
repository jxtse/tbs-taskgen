"""Sandbox runner: run a command in a resource-capped, network-less Docker container (same flags as the old
remote/validate.py drun).  Docker only: generated task code hard-codes /app and /tests paths.

    run(cfg, mounts=[(host, container, "ro"|"rw")], cmd="...", timeout=300, name="tbsgen-...")
    -> {"rc", "out", "err", "sec"}
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path


def sh(cmd: list[str], timeout: int, env=None, cwd=None) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return dict(rc=p.returncode, out=p.stdout[-4000:], err=p.stderr[-4000:], sec=round(time.time() - t0, 2))
    except subprocess.TimeoutExpired:
        return dict(rc=-9, out="", err=f"TIMEOUT after {timeout}s", sec=float(timeout))


def docker_available() -> bool:
    return shutil.which("docker") is not None and sh(["docker", "info", "--format", "{{.ServerVersion}}"], 20)["rc"] == 0


def image_exists(image: str) -> bool:
    return sh(["docker", "image", "inspect", image], 30)["rc"] == 0


def run(cfg: dict, mounts: list[tuple], cmd: str, timeout: int, name: str = "", root: bool = False) -> dict:
    sb = cfg["sandbox"]
    name = re.sub(r"[^a-zA-Z0-9_.-]", "-", (name or "tbsgen") + "-" + uuid.uuid4().hex[:6])[:120]
    if sb["backend"] != "docker":
        raise ValueError("only the docker sandbox backend is supported (task code hard-codes /app, /tests paths)")
    args = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", str(sb["cpus"]),
            "--memory", sb["memory"], "--memory-swap", sb["memory"], "--pids-limit", str(sb["pids"]),
            "--user", "0:0" if root else f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=2",
            "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MPLCONFIGDIR=/tmp"]
    for src, dst, mode in mounts:
        args += ["-v", f"{Path(src).resolve()}:{dst}:{mode}"]
    args += [sb["image"], "bash", "-lc", cmd]
    r = sh(args, timeout + 60)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return r
