"""Local Harbor oracle / nop check of a packaged task (Docker on this machine; no model, no remote).

    tbsgen harbor-local --run DIR --item ID [--harbor BIN]

Copies items/<id>/pkg to <run>/harbor-local/tasks/<id> with the "local_base" Dockerfile (FROM the sandbox
image, so no pip download), runs `harbor run -a oracle` and `-a nop`, and records rewards in
items/<id>/harbor_local.json.  Expected: oracle reward 1, nop reward 0.  The task's network policy is kept
(oracle and nop need no network; the agent phase allowlist is enforced by Harbor's Docker egress sidecar).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .package import dockerfile


def find_harbor(cfg: dict) -> str | None:
    root = cfg["calib"].get("harbor_root")
    cand = [os.environ.get("TBS_HARBOR_BIN"), shutil.which("harbor"),
            str(Path(root) / ".venv" / "bin" / "harbor") if root else None]
    return next((c for c in cand if c and Path(c).exists()), None)


def _reward(job_dir: Path) -> dict:
    out = {}
    for t in sorted(job_dir.iterdir()) if job_dir.exists() else []:
        f = t / "result.json"
        if t.is_dir() and f.exists():
            r = json.loads(f.read_text())
            rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            exc = (r.get("exception_info") or {}).get("exception_type")
            out = dict(trial=t.name, reward=rew, exception=exc)
    return out


def check(cfg: dict, run_root: Path, item_dir: Path, harbor: str | None = None, timeout: int = 3600) -> dict:
    harbor = harbor or find_harbor(cfg)
    if not harbor:
        return dict(ok=False, error="harbor binary not found (set TBS_HARBOR_BIN)")
    base = run_root / "harbor-local"
    tdir = base / "tasks" / item_dir.name
    if tdir.exists():
        shutil.rmtree(tdir)
    shutil.copytree(item_dir / "pkg", tdir)
    (tdir / "environment" / "Dockerfile").write_text(dockerfile(cfg, "local_base"))
    res = dict(item=item_dir.name, task=str(tdir), ts=time.time())
    for agent in ("oracle", "nop"):
        job = f"{item_dir.name}-{agent}-{time.strftime('%Y%m%d%H%M%S')}"
        t0 = time.time()
        env = {k: v for k, v in os.environ.items()     # a foreign DOCKER_CONFIG hides the compose plugin
               if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "DOCKER_CONFIG")}
        p = subprocess.run([harbor, "run", "-p", str(tdir), "-a", agent, "-o", str(base / "jobs"), "--job-name", job,
                            "-n", "1", "-q", "--yes"], capture_output=True, text=True, timeout=timeout, env=env)
        (base / "logs").mkdir(parents=True, exist_ok=True)
        (base / "logs" / f"{job}.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
        res[agent] = dict(job=job, rc=p.returncode, sec=round(time.time() - t0, 1), **_reward(base / "jobs" / job))
    res["ok"] = res["oracle"].get("reward") == 1.0 and res["nop"].get("reward") == 0.0
    (item_dir / "harbor_local.json").write_text(json.dumps(res, indent=1))
    return res
