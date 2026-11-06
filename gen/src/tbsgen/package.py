"""PKG: build the Harbor task directory items/<id>/pkg/ from the working tree + seed-0 generation.

Standard Harbor task layout, plus:
  * tests/hidden copied only into tests/ (never into environment/)
  * provenance.json carries spec hash, gate summary, thresholds, ledger cost of the item
  * Dockerfile: "selfcontained" (pip install of the pinned stack, portable to remote sandboxes / self-hosted Docker servers) or
    "local_base" (FROM tbsgen-base:py311-sci-v2, only for local Docker checks)
  * validate_pkg(): structural check that Harbor can load the task (task.toml parses, required files)
"""
from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

from .runner import METRICS_HELPER

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

[agent]
timeout_sec = {agent_sec}

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 4096
gpus = 0
network_mode = "public"
"""

DOCKERFILE_SELF = """FROM python:3.11-slim
RUN python -m pip install --no-cache-dir numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 \\
        statsmodels==0.14.4 pytest==8.3.3
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
"""

DOCKERFILE_LOCAL = """FROM {image}
WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
"""

TEST_SH = """#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
cd /tests
if python -m pytest -q -p no:cacheprovider /tests/test_outputs.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

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
    shutil.copy(METRICS_HELPER, out / "tests" / "_metrics.py")
    (out / "tests" / "test.sh").write_text(TEST_SH)
    (out / "solution").mkdir()
    shutil.copy(task / "solution" / "solve.py", out / "solution" / "solve.py")
    (out / "solution" / "solve.sh").write_text(SOLVE_SH)
    (out / "instruction.md").write_text((task / "instruction.md").read_text())
    pk = cfg["package"]
    df = DOCKERFILE_SELF if pk["dockerfile_mode"] == "selfcontained" else \
        DOCKERFILE_LOCAL.format(image=cfg["sandbox"]["image"])
    (out / "environment" / "Dockerfile").write_text(df)
    (out / "task.toml").write_text(TASK_TOML.format(
        task_id=item_dir.name, desc=_q(spec.get("title", item_dir.name)),
        keywords=json.dumps(spec.get("competencies", [])[:8], ensure_ascii=False),
        domain=_q(spec.get("domain", "")), field=_q(spec.get("field", "")), form=_q(spec.get("form", "")),
        expert_h=float(spec.get("expert_time_hours", 2.0)), verifier_sec=float(pk["verifier_timeout"]),
        agent_sec=float(pk["agent_hours"] * 3600)))
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
