"""Running a task tree in the sandbox: generate (seed), solve (ref / mid / naive / exploit / nop), measure, test.

Task working tree (items/<id>/task/):
    generator/generate.py  solution/solve.py  baselines/{mid,naive,exploit_*}.py
    tests/measure.py  tests/test_outputs.py  tests/_metrics.py (injected)  instruction.md
Each run writes items/<id>/runs/<tag>/... .  Container layout matches Harbor: /app/data (ro), /app/output,
/tests (truth + hidden copied in only for the verifier step), /logs/verifier.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import sandbox

METRICS_HELPER = Path(__file__).with_name("templates_metrics.py")


# Verifier steps run as root on container-local copies of /tests and /app/output, as under Harbor, so that
# _metrics.run_solution really drops the agent program to `nobody` and the chmod lock on /tests/truth is enforced
# (bind mounts from the host would not enforce it).  Before v2.1 the local runner used the host uid: the lock was
# skipped and a truth-reading solution_cli.py went unnoticed until a Harbor run (e2e-iter1).
# /app/data is present too: under Harbor the verifier runs in the agent's container (e2e-iter1: a measure.py that
# reads /app/data/config.json got the missing-output sentinel locally after v2.1 dropped that mount)
VERIFIER_PRE = ("mkdir -p /app && cp -a /src/tests /tests && cp -a /src/output /app/output && "
                "cp -a /src/data /app/data && chown -R -h root:root /tests /app/output /app/data && "
                "chmod 700 /src && ")   # staging mounts stay in the container: make them root-only (review #2)


def tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def generate(cfg: dict, task: Path, seed: int, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(task / "generator", out / "generator")
    for d in ("environment/data", "tests/truth", "tests/hidden"):
        (out / d).mkdir(parents=True)
    r = sandbox.run(cfg, [(out, "/task", "rw")], f"cd /task && python generator/generate.py {seed}",
                    cfg["gen"]["gen_timeout"], f"tbsgen-gen-{task.parent.name}-{seed}")
    r["hash_env"] = tree_hash(out / "environment") if r["rc"] == 0 else None
    r["hash_tests"] = tree_hash(out / "tests") if r["rc"] == 0 else None
    r["public_mb"] = round(sum(p.stat().st_size for p in (out / "environment").rglob("*") if p.is_file()) / 1e6, 3)
    return r


# Non-root processes left by the agent phase are killed before pytest.  The reward is written last, by
# _metrics.finalize_reward: it kills every non-root process (the agent program ran as `nobody`), makes
# /logs/verifier unwritable for `nobody`, removes any reward file written before and creates reward.txt
# (O_EXCL|O_NOFOLLOW).  Before v2.5 a detached process left by solution_cli.py could rewrite
# reward.txt after pytest (S9R review of e2e-prod-20261109).
TEST_SH = """#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
cd /tests
python /tests/_metrics.py --reap >/dev/null 2>&1
rm -rf /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/metrics.jsonl
python -m pytest -q -p no:cacheprovider /tests/test_outputs.py
rc=$?
if ! python /tests/_metrics.py --finalize-reward "$rc"; then
  pkill -KILL -u nobody 2>/dev/null
  rm -f /logs/verifier/reward.txt
  echo 0 > /logs/verifier/reward.txt
fi
"""


REPEATS_LINE = "TIMING_REPEATS = 3  # tbsgen:timing_repeats"


def timing_repeats(cfg: dict, spec: dict | None) -> int:
    """Timed runs per hidden set in _metrics.run_solution (wall time = median): thresholds.timing_repeats for a task
    with a resource metric (spec.is_resource_metric), 1 otherwise (nothing else depends on the repeats)."""
    from .spec import is_resource_metric
    n = max(1, int(cfg["thresholds"].get("timing_repeats", 3)))
    if spec is not None and not any(is_resource_metric(m) for m in spec.get("metrics", [])):
        return 1
    return n


def timing_budget(cfg: dict, spec: dict | None) -> float:
    """Extra verifier seconds for the timed repeats (0 when the task has no resource metric).  package() adds it to
    the task's verifier timeout and the local verifier gets it too, so the scored runs keep their full timeout."""
    return float(cfg["thresholds"].get("timing_budget_sec", 300)) if timing_repeats(cfg, spec) > 1 else 0.0


def resource_gating(cfg: dict, spec: dict | None) -> bool:
    """Whether resource metrics (wall time / memory) decide a run for this task (spec.resource_metric_gating,
    default false: recorded but never deciding — only speed-contribution papers gate on them)."""
    return bool((spec or {}).get("resource_metric_gating"))


def metrics_helper_text(cfg: dict, spec: dict | None, item_dir: Path | None = None) -> str:
    """tests/_metrics.py as copied into a task: the template with TIMING_REPEATS, the machine-speed reference
    (thresholds.speed_bench_sec of this item, when thresholds exist) and the resource-metric gating flag set for
    this task."""
    src = METRICS_HELPER.read_text()
    if REPEATS_LINE not in src:
        raise RuntimeError("templates_metrics.py: TIMING_REPEATS marker line missing")
    tb = "TIMING_BUDGET_SEC = 300.0  # tbsgen:timing_budget"
    if tb not in src:
        raise RuntimeError("templates_metrics.py: TIMING_BUDGET_SEC marker line missing")
    sb = "SPEED_BASE = None  # tbsgen:speed_base"
    rg = "RESOURCE_GATING = False  # tbsgen:resource_gating"
    if sb not in src or rg not in src:
        raise RuntimeError("templates_metrics.py: SPEED_BASE / RESOURCE_GATING marker lines missing")
    base = None
    if item_dir is not None:
        try:
            th = json.loads((item_dir / "thresholds.json").read_text())
        except (OSError, ValueError):
            th = {}
        b = (th or {}).get("speed_bench_sec")
        base = float(b) if isinstance(b, (int, float)) and b > 0 else None
    src = src.replace(sb, f"SPEED_BASE = {base!r}  # tbsgen:speed_base")
    src = src.replace(rg, f"RESOURCE_GATING = {resource_gating(cfg, spec)!s}  # tbsgen:resource_gating")
    src = src.replace(tb, f"TIMING_BUDGET_SEC = {timing_budget(cfg, spec)!r}  # tbsgen:timing_budget")
    return src.replace(REPEATS_LINE, f"TIMING_REPEATS = {timing_repeats(cfg, spec)}  # tbsgen:timing_repeats")


def task_spec(task: Path) -> dict | None:
    p = task.parent / "_spec" / "spec.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def _verifier_tests(cfg: dict, task: Path, gen: Path, dst: Path, test_file: str | None):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(task / "tests", dst, ignore=shutil.ignore_patterns("truth", "hidden", "__pycache__"))
    shutil.copytree(gen / "tests" / "truth", dst / "truth")
    shutil.copytree(gen / "tests" / "hidden", dst / "hidden")
    (dst / "_metrics.py").write_text(metrics_helper_text(cfg, task_spec(task), task.parent))
    (dst / "test.sh").write_text(TEST_SH)       # the packaged verifier entry point, used as is by test()


def solve(cfg: dict, task: Path, gen: Path, prog: str | None, work: Path, tag: str) -> dict:
    """prog: path relative to task (solution/solve.py, baselines/mid.py, ...) or None for nop."""
    outd = work / f"out-{tag}"
    if outd.exists():
        shutil.rmtree(outd)
    outd.mkdir(parents=True)
    outd.chmod(0o777)
    if prog is None:
        return dict(rc=0, out="", err="", sec=0.0, nop=True, outdir=str(outd))
    p = task / prog
    timeout = cfg["gen"]["solve_timeout"] if prog.startswith("solution") else cfg["gen"]["baseline_timeout"]
    r = sandbox.run(cfg, [(gen / "environment" / "data", "/app/data", "ro"), (outd, "/app/output", "rw"),
                          (p.parent, "/sol", "ro")], f"cd /app && python /sol/{p.name}", timeout,
                    f"tbsgen-solve-{tag}")
    r["outdir"] = str(outd)
    return r


def measure(cfg: dict, task: Path, gen: Path, outd: Path, work: Path, tag: str) -> dict:
    """Run tests/measure.py (no thresholds) -> {metric: value}"""
    tests = work / f"mtests-{tag}"
    _verifier_tests(cfg, task, gen, tests, None)
    logs = work / f"mlogs-{tag}"
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o777)
    r = sandbox.run(cfg, [(outd, "/src/output", "ro"), (tests, "/src/tests", "ro"), (logs, "/logs/verifier", "rw"),
                          (gen / "environment" / "data", "/src/data", "ro")],
                    VERIFIER_PRE + "cd /tests && python /tests/measure.py",
                    int(cfg["gen"]["test_timeout"] + timing_budget(cfg, task_spec(task))),
                    f"tbsgen-measure-{tag}", root=True)
    vals = {}
    for line in reversed(r["out"].strip().splitlines()):
        try:
            vals = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    r["values"] = {k: float(v) for k, v in vals.items() if isinstance(v, (int, float))} if isinstance(vals, dict) else {}
    return r


def speed_bench(cfg: dict, task: Path, gen: Path, work: Path, seed) -> float | None:
    """The machine-speed reference measured inside the task's own sandbox image (the same environment the
    thresholds are measured and applied in): median fixed-computation time from tests/_metrics.py."""
    tests = work / f"mtests-bench{seed}"
    _verifier_tests(cfg, task, gen, tests, None)
    logs = work / f"mlogs-bench{seed}"
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o777)
    r = sandbox.run(cfg, [(tests, "/src/tests", "ro"), (logs, "/logs/verifier", "rw"),
                          (gen / "environment" / "data", "/src/data", "ro")],
                    VERIFIER_PRE + "python /tests/_metrics.py --speed-bench", 300,
                    f"tbsgen-bench-{seed}", root=True)
    try:
        d = json.loads(r["out"].strip().splitlines()[-1])
        return float(d.get("speed_bench_sec")) if d.get("speed_bench_sec") else None
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        return None


def test(cfg: dict, task: Path, gen: Path, outd: Path, work: Path, tag: str) -> dict:
    """The packaged verifier: tests/test.sh (pytest, then _metrics.finalize_reward).  passed = reward.txt reads 1
    after a settle delay (gen.verifier_settle_sec) during which the container keeps running, as under Harbor where
    reward.txt is read after test.sh returns: a process the agent program left behind gets the time to act."""
    tests = work / f"tests-{tag}"
    _verifier_tests(cfg, task, gen, tests, "test_outputs.py")
    logs = work / f"logs-{tag}"
    if logs.exists():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)
    logs.chmod(0o777)
    settle = float(cfg["gen"].get("verifier_settle_sec", 1.0))
    r = sandbox.run(cfg, [(outd, "/src/output", "ro"), (tests, "/src/tests", "ro"), (logs, "/logs/verifier", "rw"),
                          (gen / "environment" / "data", "/src/data", "ro")],
                    VERIFIER_PRE + f"cd /tests && bash /tests/test.sh; rc=$?; sleep {settle:g}; exit $rc",
                    int(cfg["gen"]["test_timeout"] + timing_budget(cfg, task_spec(task))), f"tbsgen-test-{tag}", root=True)
    mf = logs / "metrics.jsonl"
    metrics = []
    if mf.exists():
        for x in mf.read_text().splitlines():
            try:
                metrics.append(json.loads(x)) if x.strip() else None
            except json.JSONDecodeError:
                continue
    rf = logs / "reward.txt"
    reward = rf.read_text().strip() if rf.exists() and not rf.is_symlink() else None
    acc = [m for m in metrics if not m.get("aux") and "passed" in m and m.get("gated", True)]
    r.update(passed=reward == "1", reward=reward, metrics=metrics, n_checks=len(acc),
             n_failed=sum(not m["passed"] for m in acc), failed=[m["name"] for m in acc if not m["passed"]])
    return r


TRUTH_PROBE = r'''import json, os, sys
# tbsgen truth-isolation probe: an adversarial solution_cli.py.  The trusted side writes the exact paths to try
# into probe_targets.json next to this file (verifier-only files, their /src copies, and the same files reached
# through an agent-planted symlink /app/output/leak -> /tests/truth).  Every path is opened directly; nothing
# relies on listing directories (a 0711 directory cannot be listed but its files may still be readable).
here = os.path.dirname(os.path.abspath(__file__))
targets = json.load(open(os.path.join(here, "probe_targets.json")))
hits = []
for p in targets:
    try:
        with open(p, "rb") as f:
            f.read(16)
        hits.append(p)
    except Exception:
        pass
sys.stderr.write("TBSGEN_TRUTH_PROBE attempted=%d readable=%d %s\\n" % (len(targets), len(hits), " ".join(hits[:5])))
sys.exit(3)
'''


def truth_probe(cfg: dict, task: Path, gen: Path, ref_out: Path, work: Path, tag: str) -> dict:
    """PROG: run the verifier on the reference's public outputs with solution_cli.py replaced by TRUTH_PROBE.
    ok = the verifier executed the probe through _metrics.run_solution, the probe attempted every target path
    and could read none of them.  Targets: each file under tests/ except hidden inputs (at /tests/... and at the
    /src/tests/... staging copy) plus each truth file through a pre-planted symlink /app/output/leak."""
    outd = work / f"out-{tag}"
    if outd.exists():
        shutil.rmtree(outd)
    shutil.copytree(ref_out, outd, symlinks=True)
    tests = work / f"tests-{tag}"
    _verifier_tests(cfg, task, gen, tests, "test_outputs.py")
    rel = sorted(str(p.relative_to(tests)) for p in tests.rglob("*")
                 if p.is_file() and not str(p.relative_to(tests)).startswith("hidden"))
    truth_rel = sorted(str(p.relative_to(tests / "truth")) for p in (tests / "truth").rglob("*") if p.is_file()) \
        if (tests / "truth").exists() else []
    targets = [f"/tests/{r}" for r in rel] + [f"/src/tests/{r}" for r in rel] + \
        [f"/app/output/leak/{r}" for r in truth_rel]
    (outd / "solution_cli.py").write_text(TRUTH_PROBE)
    (outd / "probe_targets.json").write_text(json.dumps(targets))
    leak = outd / "leak"
    if leak.exists() or leak.is_symlink():
        leak.unlink()
    leak.symlink_to("/tests/truth")
    t = test(cfg, task, gen, outd, work, tag)
    runs = [m for m in t["metrics"] if m.get("name") == "_solution_cli_run"]
    want = f"TBSGEN_TRUTH_PROBE attempted={len(targets)} readable=0"
    bad = [m.get("stderr_tail", "") for m in runs if want not in m.get("stderr_tail", "")]
    return dict(ok=bool(runs) and bool(truth_rel) and not bad, n_runs=len(runs), via_run_solution=bool(runs),
                n_targets=len(targets), readable=[x[-400:] for x in bad][:3], passed=t["passed"])