"""PKG: build the Harbor task directory items/<id>/pkg/ from the working tree + seed-0 generation.

Standard Harbor task layout, plus:
  * tests/hidden copied only into tests/ (never into environment/)
  * provenance.json carries spec hash, gate summary, thresholds, ledger cost of the item
  * Dockerfile: "selfcontained" (pip install of the pinned stack, portable to remote sandboxes / self-hosted Docker servers) or
    "local_base" (FROM tbsgen-base:py311-sci-v2, only for local Docker checks); either way the package index is
    removed after the build (OFFLINE_INDEX)
  * [environment] network policy from package.network (default allowlist; see prerelease.py)
  * validate_pkg(): structural check that Harbor can load the task (task.toml parses, required files)
"""
from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

from .runner import TEST_SH, metrics_helper_text, timing_budget

TASK_TOML = """schema_version = "1.0"

[task]
name = "tbsgen-v2/{task_id}"
description = {desc}
keywords = {keywords}

[metadata]
domain = {domain}
field = {field}
form = {form}
expert_time_estimate_hours = {expert_h}
generator = "tbsgen v2"

[verifier]
timeout_sec = {verifier_sec}
{verifier_network}
[agent]
timeout_sec = {agent_sec}
{agent_network}
[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 4096
gpus = 0
{env_network}
"""

# Optional (package.network.disable_index): after the build, no package index -- pip / uv / conda refuse to
# download.  Off by default: agents whose setup pip-installs their own runtime (e.g. a model proxy) would break,
# and the agent-phase network policy is the barrier that holds even against an agent that overrides these.
OFFLINE_INDEX = """# no package index at run time: dependencies were installed above
RUN rm -f /etc/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf /root/.condarc /etc/conda/.condarc \\
        /root/.config/uv/uv.toml /etc/uv/uv.toml
ENV PIP_NO_INDEX=1 PIP_INDEX_URL=http://127.0.0.1:9/no-index PIP_EXTRA_INDEX_URL= PIP_FIND_LINKS= \\
    UV_OFFLINE=1 UV_NO_INDEX=1 CONDA_OFFLINE=true PIP_DISABLE_PIP_VERSION_CHECK=1
"""

DOCKERFILE_SELF = """FROM python:3.11-slim
RUN python -m pip install --no-cache-dir numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 \\
        statsmodels==0.14.4 pytest==8.3.3
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
{offline}WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
"""

DOCKERFILE_LOCAL = """FROM {image}
{offline}WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
"""

NETWORK_MODES = ("agent-allowlist", "no-network", "public")


def network_policy(cfg: dict) -> dict:
    """task.toml network lines per table; see prerelease.py.  agent-allowlist: the environment starts public (the
    agent CLI installs during agent setup), the agent phase is an allowlist (`package.network.agent_hosts` plus
    the model endpoint the launcher adds per run), the verifier phase is offline."""
    net = cfg["package"].get("network") or {}
    mode = net.get("mode", "agent-allowlist")
    if mode not in NETWORK_MODES:
        raise ValueError(f"package.network.mode must be one of {NETWORK_MODES}, not {mode!r}")
    if mode == "agent-allowlist":
        hosts = json.dumps([str(h) for h in net.get("agent_hosts") or []])
        return dict(env_network='network_mode = "public"          # agent setup only; see [agent] / [verifier]',
                    agent_network=("# no internet while the agent works, except these hosts and the model endpoint the\n"
                                   "# launcher adds per run (harbor run --allow-agent-host HOST / agents[].extra_allowed_hosts)\n"
                                   f'network_mode = "allowlist"\nallowed_hosts = {hosts}\n'),
                    verifier_network='network_mode = "no-network"\n')
    return dict(env_network=f'network_mode = "{mode}"', agent_network="", verifier_network="")


def dockerfile(cfg: dict, mode: str | None = None) -> str:
    off = OFFLINE_INDEX if (cfg["package"].get("network") or {}).get("disable_index") else ""
    if (mode or cfg["package"]["dockerfile_mode"]) == "selfcontained":
        return DOCKERFILE_SELF.format(offline=off)
    return DOCKERFILE_LOCAL.format(image=cfg["sandbox"]["image"], offline=off)



SOLVE_SH = """#!/bin/bash
set -euo pipefail
python /solution/solve.py
"""

REQUIRED = ["task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh", "tests/test_outputs.py",
            "tests/_metrics.py", "solution/solve.sh", "solution/solve.py"]


def _q(s) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def package(cfg: dict, item_dir: Path, gen0: Path, spec: dict, provenance: dict) -> Path:
    task = item_dir / "task"
    out = item_dir / "pkg"
    if out.exists():
        shutil.rmtree(out)
    (out / "environment").mkdir(parents=True)
    shutil.copytree(gen0 / "environment" / "data", out / "environment" / "data")
    (out / "tests").mkdir()
    shutil.copy(task / "tests" / "test_outputs.py", out / "tests" / "test_outputs.py")
    if (task / "tests" / "measure.py").exists():       # test_outputs.py imports compute() from it
        shutil.copy(task / "tests" / "measure.py", out / "tests" / "measure.py")
    shutil.copytree(gen0 / "tests" / "truth", out / "tests" / "truth")
    if any((gen0 / "tests" / "hidden").rglob("*")):
        shutil.copytree(gen0 / "tests" / "hidden", out / "tests" / "hidden")
    (out / "tests" / "_metrics.py").write_text(metrics_helper_text(cfg, spec))
    (out / "tests" / "test.sh").write_text(TEST_SH)
    (out / "solution").mkdir()
    shutil.copy(task / "solution" / "solve.py", out / "solution" / "solve.py")
    (out / "solution" / "solve.sh").write_text(SOLVE_SH)
    (out / "instruction.md").write_text((task / "instruction.md").read_text())
    pk = cfg["package"]
    (out / "environment" / "Dockerfile").write_text(dockerfile(cfg))
    (out / "task.toml").write_text(TASK_TOML.format(
        task_id=item_dir.name, desc=_q(spec.get("title", item_dir.name)),
        keywords=json.dumps(spec.get("competencies", [])[:8], ensure_ascii=False),
        domain=_q(spec.get("domain", "")), field=_q(spec.get("field", "")), form=_q(spec.get("form", "")),
        expert_h=float(spec.get("expert_time_hours", 2.0)), verifier_sec=float(pk["verifier_timeout"]) + timing_budget(cfg, spec),
        agent_sec=float(pk["agent_hours"] * 3600), **network_policy(cfg)))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1, ensure_ascii=False, default=str))
    for p in (out / "tests" / "test.sh", out / "solution" / "solve.sh"):
        p.chmod(0o755)
    return out


def validate_pkg(pkg: Path) -> dict:
    missing = [r for r in REQUIRED if not (pkg / r).exists()]
    errs = []
    try:
        t = tomllib.loads((pkg / "task.toml").read_text())
        for sec in ("task", "verifier", "agent", "environment"):
            if sec not in t:
                errs.append(f"task.toml missing [{sec}]")
        if (t.get("environment") or {}).get("network_mode") not in NETWORK_MODES:
            errs.append("task.toml [environment] needs network_mode")
    except Exception as e:  # noqa: BLE001
        errs.append(f"task.toml: {e}")
    leaked = [str(p.relative_to(pkg)) for p in (pkg / "environment").rglob("*")
              if p.is_file() and any(s in p.parts for s in ("tests", "truth", "generator", "solution"))]
    if leaked:
        errs.append(f"answer-bearing files under environment/: {leaked[:5]}")
    for gen_only in ("generator", "baselines", "_spec"):
        if (pkg / gen_only).exists():
            errs.append(f"{gen_only}/ must not be packaged")
    return dict(ok=not missing and not errs, missing=missing, errors=errs)
