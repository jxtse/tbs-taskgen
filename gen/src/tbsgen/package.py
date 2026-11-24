"""PKG: build the Harbor task directory items/<id>/pkg/ from the working tree + first configured seed's generation.

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

import ast
import json
import shlex
import shutil
import tomllib
from pathlib import Path

from . import sandbox
from .spec import valid_pins
from .state import read_json
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
cpus = {cpus}
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
        statsmodels==0.14.4 pytest==8.3.3{extra}
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
    return dict(env_network=f'network_mode = "{mode}"', agent_network=f'network_mode = "{mode}"\n',
                verifier_network='network_mode = "no-network"\n')


def dockerfile(cfg: dict, mode: str | None = None, extra_pip=()) -> str:
    if not valid_pins(list(extra_pip)):
        raise ValueError("invalid extra_pip")
    off = OFFLINE_INDEX if (cfg["package"].get("network") or {}).get("disable_index") else ""
    if (mode or cfg["package"]["dockerfile_mode"]) == "selfcontained":
        return DOCKERFILE_SELF.format(offline=off, extra="".join(" " + pin for pin in extra_pip))
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
    (out / "tests" / "_metrics.py").write_text(metrics_helper_text(cfg, spec, item_dir))
    (out / "tests" / "test.sh").write_text(TEST_SH)
    (out / "solution").mkdir()
    shutil.copy(task / "solution" / "solve.py", out / "solution" / "solve.py")
    (out / "solution" / "solve.sh").write_text(SOLVE_SH)
    (out / "instruction.md").write_text(instruction_text(cfg, spec, (task / "instruction.md").read_text()))
    pk = cfg["package"]
    (out / "environment" / "Dockerfile").write_text(dockerfile(cfg, extra_pip=spec.get("extra_pip", [])))
    (out / "task.toml").write_text(TASK_TOML.format(
        task_id=item_dir.name, desc=_q(spec.get("title", item_dir.name)),
        keywords=json.dumps(spec.get("competencies", [])[:8], ensure_ascii=False),
        domain=_q(spec.get("domain", "")), field=_q(spec.get("field", "")), form=_q(spec.get("form", "")),
        expert_h=float(spec.get("expert_time_hours", 2.0)), verifier_sec=float(pk["verifier_timeout"]) + timing_budget(cfg, spec),
        agent_sec=float(pk["agent_hours"] * 3600), cpus=TASK_CPUS, **network_policy(cfg)))
    provenance = dict(provenance, extra_pip=spec.get("extra_pip", []),
                      verifier_overhead_sec=pk.get("verifier_overhead_sec", 120))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1, ensure_ascii=False, default=str))
    for p in (out / "tests" / "test.sh", out / "solution" / "solve.sh"):
        p.chmod(0o755)
    return out


def _helper_const(src: str, name: str):
    """Value of a pipeline-written `NAME = <literal>  # tbsgen:...` line of tests/_metrics.py (None if absent)."""
    import ast
    import re
    m = re.search(rf"^{name} = (.+?)  # tbsgen:", src, re.M)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None


def verifier_time_errors(pkg: Path, overhead: float) -> list[str]:
    """The agent program's verifier time must fit the task's verifier timeout: the scored runs' cap
    (_metrics.SCORED_BUDGET_SEC) + the timed repeats' budget (TIMING_BUDGET_SEC) + `overhead` seconds for the rest
    of the verifier <= [verifier].timeout_sec.  Otherwise Harbor kills the verifier mid-run (no reward) instead of
    the helper failing the program like a timeout (review of e2e-batch2: run_solution(1800) under 1200 s)."""
    try:
        t = tomllib.loads((pkg / "task.toml").read_text())
        vt = float((t.get("verifier") or {})["timeout_sec"])
        src = (pkg / "tests" / "_metrics.py").read_text()
    except Exception as e:  # noqa: BLE001
        return [f"verifier time: {e}"]
    import math
    if not math.isfinite(overhead) or overhead < 0:
        return [f"verifier time: package.verifier_overhead_sec must be finite and >= 0 (got {overhead})"]
    if not math.isfinite(vt):
        return [f"verifier time: [verifier].timeout_sec must be finite (got {vt})"]
    sc, tb = _helper_const(src, "SCORED_BUDGET_SEC"), _helper_const(src, "TIMING_BUDGET_SEC")
    if not isinstance(sc, (int, float)) or isinstance(sc, bool) or not math.isfinite(sc) or sc <= 0:
        return ["verifier time: tests/_metrics.py has no SCORED_BUDGET_SEC cap (old helper template)"]
    if not isinstance(tb, (int, float)) or isinstance(tb, bool) or not math.isfinite(tb) or tb < 0:
        return ["verifier time: tests/_metrics.py has no TIMING_BUDGET_SEC"]
    if sc + tb + overhead > vt + 1e-6:
        return [f"verifier time: scored {sc:.0f}s + timing repeats {tb:.0f}s + overhead {overhead:.0f}s "
                f"> verifier timeout {vt:.0f}s"]
    return []


TASK_CPUS = 2          # [environment].cpus of every packaged task (also stated in the instruction's time note)
TIME_NOTE_MARK = "<!-- tbsgen:verifier-time -->"
TIME_NOTE_END = "<!-- /tbsgen:verifier-time -->"


def strip_time_note(text: str) -> str:
    import re as _re
    return _re.sub(r"\n*" + _re.escape(TIME_NOTE_MARK) + r".*?" + _re.escape(TIME_NOTE_END) + r"\n?", "\n", text,
                   flags=_re.S)


def instruction_text(cfg: dict, spec: dict, text: str) -> str:
    """The packaged statement.  PROG: add the verifier's time limit for the agent program between two markers (the
    first, scored run of solution_cli.py on each hidden input set together get runner.scored_budget() seconds on
    TASK_CPUS cores; timed repeats have their own budget; a run cut off scores as a failed run) -- e2e-batch2
    review: an unstated 780 s cap is a missing condition.  Only the marked block is replaced (text after it is
    kept), so it is idempotent and safe if the marked statement was copied back into task/."""
    from .runner import scored_budget
    text = strip_time_note(text).rstrip() + "\n"
    if spec.get("form") != "PROG":
        return text
    sb = f"{scored_budget(cfg):g}"
    return (text + "\n" + TIME_NOTE_MARK + "\n## Verifier time limit\n"
            f"The verifier runs `solution_cli.py` once on each hidden input set, on {TASK_CPUS} CPU cores. These scored "
            f"runs together get at most {sb} seconds of wall time (timed repeats for runtime metrics, if any, have "
            "their own allowance). A run still going when the time is used up is stopped and scored as a failed run "
            "(missing or partial outputs). Make sure your program finishes well within this limit.\n" + TIME_NOTE_END
            + "\n")


def run_solution_timeouts(src: str, budget: float, *, warnings: list[str] | None = None) -> list[str]:
    """Resolve simple numeric bindings. Dynamic timeouts are diagnostics, not budget violations."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"cannot check run_solution timeouts: {e.msg} at line {e.lineno}"]
    names = {"run_solution"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname for a in node.names if a.name == "run_solution" and a.asname)
    errors = []
    # Module constants are taken in source order: a module-level call sees the value bound above it; a call inside a
    # function (run later, at pytest time) sees the final module-level value.
    def divide(left, op, right):
        if left is None or right in (None, 0):
            return None
        try:
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Div):
                return left / right
        except ArithmeticError:
            pass
        return None

    def numeric(node, bindings):
        if isinstance(node, ast.Name):
            return bindings.get(node.id)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp):
            return divide(numeric(node.left, bindings), node.op, numeric(node.right, bindings))
        return None

    def bind(node, bindings):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = numeric(node.value, bindings)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(target, ast.Name):
                    bindings[target.id] = value
                else:
                    for name in local_names(target):
                        bindings[name] = None
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            bindings[name] = divide(bindings.get(name), node.op, numeric(node.value, bindings))
        else:
            for name in local_names(node):
                bindings[name] = None

    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

    def local_names(node):
        if isinstance(node, scopes):
            if hasattr(node, "name"):
                yield node.name
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id
        for child in ast.iter_child_nodes(node):
            yield from local_names(child)

    final = {}
    for node in tree.body:
        bind(node, final)

    def visit(node, visible, deferred, class_parent):
        if isinstance(node, scopes):
            body = node.body if isinstance(node.body, list) else [node.body]
            # Bases, decorators and default arguments execute in the defining scope.
            for child in ast.iter_child_nodes(node):
                if child not in body:
                    visit(child, visible, deferred, class_parent)
            if isinstance(node, ast.ClassDef):
                # A class body's namespace is private to that body. Its nested classes start from
                # the same enclosing function/module bindings, at this definition point.
                class_scope = dict(class_parent)
                for stmt in body:
                    visit(stmt, class_scope, deferred, class_parent)
                    bind(stmt, class_scope)
            else:
                local = {n.arg for n in ast.walk(node.args) if isinstance(n, ast.arg)}
                local.update(name for stmt in body for name in local_names(stmt))
                current = {k: v for k, v in deferred.items() if k not in local}
                later = dict(current)
                for stmt in body:
                    bind(stmt, later)
                for stmt in body:
                    visit(stmt, current, later, current)
                    bind(stmt, current)
            return
        if isinstance(node, ast.If):
            visit(node.test, visible, deferred, class_parent)
            for branch in (node.body, node.orelse):
                bindings = dict(visible)
                for stmt in branch:
                    visit(stmt, bindings, deferred, bindings if class_parent is visible else class_parent)
                    bind(stmt, bindings)
            return
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name in names or getattr(node.func, "attr", None) == "run_solution":
                arg = next((k.value for k in node.keywords if k.arg == "timeout_sec"), None)
                if arg is None and node.args:
                    arg = node.args[0]
                value = numeric(arg, visible)
                if value is None and arg is not None:
                    diagnostics = warnings if warnings is not None else errors
                    diagnostics.append(f"line {node.lineno}: run_solution timeout unknown")
                elif value is not None and value >= budget:
                    errors.append(f"line {node.lineno}: run_solution timeout {value:g}s >= verifier budget {budget:g}s")
        if isinstance(node, ast.stmt) and any(isinstance(c, ast.stmt) for c in ast.iter_child_nodes(node)):
            # Branch/loop execution is unknown. Never reuse a binding it might overwrite.
            visible = dict(visible)
            for name in local_names(node):
                visible[name] = None
        for child in ast.iter_child_nodes(node):
            visit(child, visible, deferred, class_parent)

    sofar = {}
    for node in tree.body:
        visit(node, sofar, final, sofar)
        bind(node, sofar)
    return errors


def network_errors(t: dict, mode: str) -> tuple[list, list]:
    errors, warnings = [], []
    if t.get("verifier", {}).get("network_mode") != "no-network":
        errors.append("[verifier].network_mode must explicitly be no-network")
    agent = t.get("agent", {}).get("network_mode")
    if agent == "public" and mode == "public":
        warnings.append("agent network is public for a comparison run")
    elif agent not in ("allowlist", "no-network"):
        errors.append("[agent].network_mode must explicitly be allowlist or no-network")
    return errors, warnings


def sandbox_pin_errors(cfg: dict, pins: list) -> list[str]:
    if not pins:
        return []
    if not valid_pins(pins):
        return ["extra_pip_unsupported: invalid package pins"]
    # Both local and SSH runners use the configured image, offline. Query installed metadata only.
    code = ("from importlib.metadata import version\n"
            f"for pin in {pins!r}:\n"
            "    name, wanted = pin.split('==')\n"
            "    if version(name) != wanted: raise SystemExit(pin)\n")
    try:
        result = sandbox.run(cfg, [], "python -c " + shlex.quote(code), 30, "tbsgen-check-pins")
    except (OSError, ValueError) as e:
        return [f"extra_pip_unsupported: cannot verify {pins} in the configured sandbox: {e}"]
    if result["rc"]:
        return [f"extra_pip_unsupported: {pins} in {cfg['sandbox']['image']}: {result['err'][-300:]}"]
    return []


def validate_pkg(pkg: Path, cfg: dict | None = None) -> dict:
    provenance = read_json(pkg / "provenance.json", {})
    missing = [r for r in REQUIRED if not (pkg / r).exists()]
    errs, warnings = [], []
    try:
        t = tomllib.loads((pkg / "task.toml").read_text())
        for sec in ("task", "verifier", "agent", "environment"):
            if sec not in t:
                errs.append(f"task.toml missing [{sec}]")
        if (t.get("environment") or {}).get("network_mode") not in NETWORK_MODES:
            errs.append("task.toml [environment] needs network_mode")
        policy = ((cfg or {}).get("package", {}).get("network") or {}).get("mode", "agent-allowlist")
        network, warnings = network_errors(t, policy)
        errs.extend(network)
        overhead = float((cfg or {}).get("package", {}).get("verifier_overhead_sec", provenance.get("verifier_overhead_sec", 120)))
        budget = float(t["verifier"]["timeout_sec"]) - overhead
        for name in ("measure.py", "test_outputs.py"):
            path = pkg / "tests" / name
            if path.is_file():
                unknown = []
                errs.extend(f"{name}: {e}" for e in run_solution_timeouts(path.read_text(), budget, warnings=unknown))
                warnings.extend(f"{name}: {w}" for w in unknown)
    except Exception as e:  # noqa: BLE001
        errs.append(f"task.toml: {e}")
    if cfg is not None and not missing:
        pins = provenance.get("extra_pip", [])
        errs.extend(sandbox_pin_errors(cfg, pins))
        errs += verifier_time_errors(pkg, float((cfg.get("package") or {}).get("verifier_overhead_sec", 120)))
    leaked = [str(p.relative_to(pkg)) for p in (pkg / "environment").rglob("*")
              if p.is_file() and any(s in p.parts for s in ("tests", "truth", "generator", "solution"))]
    if leaked:
        errs.append(f"answer-bearing files under environment/: {leaked[:5]}")
    for gen_only in ("generator", "baselines", "_spec"):
        if (pkg / gen_only).exists():
            errs.append(f"{gen_only}/ must not be packaged")
    return dict(ok=not missing and not errs, missing=missing, errors=errs, warnings=warnings)
