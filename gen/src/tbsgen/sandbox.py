"""Sandbox runner: run a command in a resource-capped, network-less Docker container.
Docker only: generated task code hard-codes /app and /tests paths.

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
    if sb["backend"] == "ssh":
        return run_remote(cfg, mounts, cmd, timeout, name, root)
    if sb["backend"] != "docker":
        raise ValueError("sandbox.backend must be docker or ssh (task code hard-codes /app, /tests paths)")
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


# ------------------------------------------------------------------------------------------- remote (ssh) backend
# sandbox.backend = "ssh": the same network-less, resource-capped `docker run`, on sandbox.host.  Code-grounded tasks
# port the authors' code into the reference / generator; that code runs only on the build host, never here.
# Each mount is copied to a per-call directory on the host (rsync), rw mounts are copied back after the run.
def _ssh(host: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", host]


def run_remote(cfg: dict, mounts: list[tuple], cmd: str, timeout: int, name: str, root: bool) -> dict:
    import shlex
    sb = cfg["sandbox"]
    host, base = sb.get("host"), sb.get("remote_dir")
    if not host or not base:
        raise ValueError("sandbox.backend=ssh needs sandbox.host and sandbox.remote_dir")
    rd = f"{base.rstrip('/')}/{name}"
    t0 = time.time()
    r = sh(_ssh(host) + [f"mkdir -p {shlex.quote(rd)}"], 60)
    if r["rc"] != 0:
        return dict(rc=-1, out="", err="ssh mkdir failed: " + r["err"], sec=round(time.time() - t0, 2))
    margs = []
    for i, (src, dst, mode) in enumerate(mounts):
        src = Path(src).resolve()
        sub = f"{rd}/m{i}"
        if src.is_dir():
            r = sh(["rsync", "-a", "--delete", "-e", " ".join(_ssh(host)[:-1]), f"{src}/", f"{host}:{sub}/"], 600)
        else:
            r = sh(["rsync", "-a", "-e", " ".join(_ssh(host)[:-1]), str(src), f"{host}:{sub}"], 600)
        if r["rc"] != 0:
            sh(_ssh(host) + [f"rm -rf {shlex.quote(rd)}"], 120)
            return dict(rc=-1, out="", err=f"rsync to {host} failed: {r['err'][-300:]}", sec=round(time.time() - t0, 2))
        margs += ["-v", f"{sub}:{dst}:{mode}"]
    user = "0:0" if root else "$(id -u):$(id -g)"
    dargs = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", str(sb["cpus"]),
             "--memory", sb["memory"], "--memory-swap", sb["memory"], "--pids-limit", str(sb["pids"]),
             "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MPLCONFIGDIR=/tmp"]
    line = (" ".join(shlex.quote(a) for a in dargs) + f" --user {user} " +
            " ".join(shlex.quote(a) for a in margs) + f" {shlex.quote(sb['image'])} bash -lc {shlex.quote(cmd)}")
    # rw mounts are written by container users other than the ssh user: make them writable for everyone
    prep = " ".join(f"chmod -R a+rwX {shlex.quote(f'{rd}/m{i}')};" for i, (_, _, m) in enumerate(mounts) if m == "rw")
    env = f"DOCKER_CONFIG={shlex.quote(sb['docker_config'])} " if sb.get("docker_config") else ""
    r = sh(_ssh(host) + [f"{prep} {env}timeout -k 30 {int(timeout) + 30} {line}"], timeout + 120)
    sh(_ssh(host) + [f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1; "
                     f"docker run --rm -v {shlex.quote(rd)}:/x {shlex.quote(sb['image'])} chmod -R a+rwX /x "
                     ">/dev/null 2>&1 || true"], 120)
    for i, (src, dst, mode) in enumerate(mounts):
        if mode == "rw" and Path(src).is_dir():
            b = sh(["rsync", "-a", "--delete", "-e", " ".join(_ssh(host)[:-1]), f"{host}:{rd}/m{i}/", f"{Path(src)}/"],
                   600)
            if b["rc"] != 0:
                r = dict(r, rc=r["rc"] if r["rc"] else -1, err=(r["err"] + f"\nrsync back failed: {b['err'][-300:]}"))
    sh(_ssh(host) + [f"rm -rf {shlex.quote(rd)}"], 120)
    r["sec"] = round(time.time() - t0, 2)
    r["remote"] = host
    return r
