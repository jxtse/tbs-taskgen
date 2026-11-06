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
                "cp -a /src/data /app/data && chown -R root:root /tests /app/output /app/data && ")


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


def _verifier_tests(task: Path, gen: Path, dst: Path, test_file: str | None):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(task / "tests", dst, ignore=shutil.ignore_patterns("truth", "hidden", "__pycache__"))
    shutil.copytree(gen / "tests" / "truth", dst / "truth")
    shutil.copytree(gen / "tests" / "hidden", dst / "hidden")
    shutil.copy(METRICS_HELPER, dst / "_metrics.py")


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
    _verifier_tests(task, gen, tests, None)
    logs = work / f"mlogs-{tag}"
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o777)
    r = sandbox.run(cfg, [(outd, "/src/output", "ro"), (tests, "/src/tests", "ro"), (logs, "/logs/verifier", "rw"),
                          (gen / "environment" / "data", "/src/data", "ro")],
                    VERIFIER_PRE + "cd /tests && python /tests/measure.py", cfg["gen"]["test_timeout"],
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


def test(cfg: dict, task: Path, gen: Path, outd: Path, work: Path, tag: str) -> dict:
    tests = work / f"tests-{tag}"
    _verifier_tests(task, gen, tests, "test_outputs.py")
    logs = work / f"logs-{tag}"
    if logs.exists():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)
    logs.chmod(0o777)
    r = sandbox.run(cfg, [(outd, "/src/output", "ro"), (tests, "/src/tests", "ro"), (logs, "/logs/verifier", "rw"),
                          (gen / "environment" / "data", "/src/data", "ro")],
                    VERIFIER_PRE + "cd /tests && python -m pytest -q -p no:cacheprovider /tests/test_outputs.py",
                    cfg["gen"]["test_timeout"], f"tbsgen-test-{tag}", root=True)
    mf = logs / "metrics.jsonl"
    metrics = [json.loads(x) for x in mf.read_text().splitlines() if x.strip()] if mf.exists() else []
    acc = [m for m in metrics if not m.get("aux")]
    r.update(passed=r["rc"] == 0, metrics=metrics, n_checks=len(acc), n_failed=sum(not m["passed"] for m in acc),
             failed=[m["name"] for m in acc if not m["passed"]])
    return r


TRUTH_PROBE = r'''import os, sys
# tbsgen truth-isolation probe: an adversarial solution_cli.py that tries to read verifier-only files.
hits = []
for dp, dn, fn in os.walk("/tests"):
    if "/hidden" in dp:
        continue
    for f in fn:
        p = os.path.join(dp, f)
        try:
            open(p, "rb").read(16)
            hits.append(p)
        except Exception:
            pass
sys.stderr.write("TBSGEN_TRUTH_READABLE " + str(len(hits)) + " " + " ".join(hits[:5]) + "\\n")
sys.exit(3)
'''


def truth_probe(cfg: dict, task: Path, gen: Path, ref_out: Path, work: Path, tag: str) -> dict:
    """PROG: run the verifier on the reference's public outputs with solution_cli.py replaced by TRUTH_PROBE.
    ok = the verifier executed the program through _metrics.run_solution (an aux record exists) and the program
    could not read any verifier-only file under /tests (other than the hidden inputs)."""
    outd = work / f"out-{tag}"
    if outd.exists():
        shutil.rmtree(outd)
    shutil.copytree(ref_out, outd)
    (outd / "solution_cli.py").write_text(TRUTH_PROBE)
    t = test(cfg, task, gen, outd, work, tag)
    runs = [m for m in t["metrics"] if m.get("name") == "_solution_cli_run"]
    readable = [m.get("stderr_tail", "") for m in runs if "TBSGEN_TRUTH_READABLE 0" not in m.get("stderr_tail", "")]
    return dict(ok=bool(runs) and not readable, n_runs=len(runs), via_run_solution=bool(runs),
                readable=[x[-300:] for x in readable][:3], passed=t["passed"])
