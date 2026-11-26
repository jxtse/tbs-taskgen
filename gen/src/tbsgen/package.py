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
import math
import re
import shlex
import shutil
import tempfile
import tomllib
from itertools import chain
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


def package(cfg: dict, item_dir: Path, gen0: Path, spec: dict, provenance: dict,
            *, out: Path | None = None) -> Path:
    from . import taskimage
    from .state import read_json
    state = read_json(item_dir / "state.json", {})
    if taskimage.is_task(state) or taskimage.is_task(spec):
        receipt = read_json(item_dir / "S6I_image.json", {})
        if not receipt.get("ok") or any(not state.get(k) or state[k] != receipt.get(k)
                                        for k in ("image_digest", "image_tag", "image_manifest")):
            raise taskimage.ImageUnavailable("package requires a matching verified S6I receipt")
        alias = taskimage.harbor_tag(item_dir, state["image_digest"])
        recipe = taskimage.data_recipe(alias)
        provenance = dict(provenance, base=receipt["base"], **{k: state[k] for k in
                          ("image_kind", "image_tag", "image_digest", "image_manifest", "inventory_hash")})
    else:
        recipe = dockerfile(cfg, extra_pip=spec.get("extra_pip", []))
    task = item_dir / "task"
    out = out or item_dir / "pkg"
    if out.exists():
        shutil.rmtree(out)
    (out / "environment").mkdir(parents=True)
    shutil.copytree(gen0 / "environment" / "data", out / "environment" / "data")
    (out / "tests").mkdir()
    for path in _authored_test_files(task / "tests"):
        shutil.copy(path, out / "tests" / path.name)
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
    (out / "environment/Dockerfile").write_text(recipe)
    (out / "task.toml").write_text(TASK_TOML.format(
        task_id=item_dir.name, desc=_q(spec.get("title", item_dir.name)),
        keywords=json.dumps(spec.get("competencies", [])[:8], ensure_ascii=False),
        domain=_q(spec.get("domain", "")), field=_q(spec.get("field", "")), form=_q(spec.get("form", "")),
        expert_h=float(spec.get("expert_time_hours", 2.0)), verifier_sec=source_verifier_timeout(cfg, spec, task),
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


def _authored_test_files(tests: Path):
    yield tests / "test_outputs.py"
    if (tests / "measure.py").exists():
        yield tests / "measure.py"


def _timeout_calls(src: str) -> tuple[list, list[str]]:
    """Static call sites and numeric bindings, without importing package code."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [], [f"cannot check run_solution timeouts: {e.msg} at line {e.lineno}"]
    calls = []
    # Alias identity travels through the same lexical bindings as numeric constants.
    solver, possible_solver = object(), object()

    def merged(left, right):
        if left == right:
            return left
        return possible_solver if left in (solver, possible_solver) or right in (solver, possible_solver) else None

    called = {getattr(n.func, "id", getattr(n.func, "attr", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    shared = {name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names}
    # Module constants are taken in source order: a module-level call sees the value bound above it; a call inside a
    # function (run later, at pytest time) sees the final module-level value.
    def arithmetic(left, op, right):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        try:
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
        except ArithmeticError:
            pass
        return None

    def numeric(node, bindings):
        if isinstance(node, ast.Name):
            value = bindings.get(node.id)
            return value if node.id not in shared and isinstance(value, (int, float)) else None
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                return float(node.value)
            except OverflowError:
                return None
        if isinstance(node, ast.BinOp):
            return arithmetic(numeric(node.left, bindings), node.op, numeric(node.right, bindings))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = numeric(node.operand, bindings)
            return -value if value is not None and isinstance(node.op, ast.USub) else value
        return None

    def binding(value, bindings):
        if isinstance(value, ast.Name):
            target = bindings.get(value.id)
            return target if target in (solver, possible_solver) else numeric(value, bindings)
        if isinstance(value, ast.Attribute) and value.attr == "run_solution":
            return solver
        if isinstance(value, ast.IfExp):
            return merged(binding(value.body, bindings), binding(value.orelse, bindings))
        if isinstance(value, ast.NamedExpr):
            return binding(value.value, bindings)
        return numeric(value, bindings)

    def expression_writes(node):
        # Include expression assignments that can run within this statement, excluding nested bodies.
        if isinstance(node, ast.NamedExpr):
            yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.stmt) and not (isinstance(node, ast.Lambda) and child is node.body):
                yield from expression_writes(child)

    def invalidate_walrus(node, bindings):
        for write in expression_writes(node):
            name = write.target.id
            candidate = binding(write.value, bindings)
            # Expression evaluation can be conditional. Ignore old numeric constants everywhere in the statement.
            bindings[name] = possible_solver if any(x in (solver, possible_solver)
                                                    for x in (bindings.get(name), candidate)) else None

    def assignment_pairs(target, value):
        """Pair literal unpacking positions; keep opaque unpacked callables uncertain."""
        if isinstance(target, ast.Name):
            yield target.id, value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if (isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts)
                    and not any(isinstance(x, ast.Starred) for x in (*target.elts, *value.elts))):
                for dest, src in zip(target.elts, value.elts):
                    yield from assignment_pairs(dest, src)
            else:
                for name in local_names(target):
                    yield name, possible_solver

    def invalidate_block(node, bindings):
        # Loops and exceptional control flow can leave any of their assignments visible.
        # Keep possible aliases (including chains) when forgetting a precise binding.
        names = set(local_names(node))
        aliases = set()
        pending = [node]
        while pending:
            child = pending.pop()
            if isinstance(child, scopes):
                continue
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    aliases.update(assignment_pairs(target, child.value))
            if isinstance(child, ast.ImportFrom):
                for part in child.names:
                    if part.name == "run_solution":
                        bindings[part.asname or part.name] = possible_solver
            pending.extend(ast.iter_child_nodes(child))
        while aliases:
            found = {(name, value) for name, value in aliases
                     if value is possible_solver or binding(value, bindings) in (solver, possible_solver)}
            if not found:
                break
            bindings.update({name: possible_solver for name, _ in found})
            aliases -= found
        for name in names:
            if name == "*":
                for key in bindings:
                    bindings[key] = merged(bindings[key], None)
            else:
                bindings[name] = merged(bindings.get(name), None)

    def bind(node, bindings):
        invalidate_walrus(node, bindings)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            # Evaluate every RHS against the old bindings before updating the LHS (including swaps).
            resolved = {name: possible_solver if value is possible_solver else binding(value, bindings)
                        for target in targets for name, value in assignment_pairs(target, node.value)}
            bindings.update(resolved)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            bindings[name] = arithmetic(bindings.get(name), node.op, numeric(node.value, bindings))
        elif isinstance(node, ast.ImportFrom):
            for part in node.names:
                if part.name == "*":
                    invalidate_block(node, bindings)
                else:
                    bindings[part.asname or part.name] = solver if part.name == "run_solution" else None
        elif isinstance(node, ast.If):
            arms = []
            for branch in (node.body, node.orelse):
                values = dict(bindings)
                for stmt in branch:
                    bind(stmt, values)
                arms.append(values)
            for name in set(arms[0]) | set(arms[1]):
                bindings[name] = merged(arms[0].get(name), arms[1].get(name))
        elif isinstance(node, ast.Try) and not node.handlers:
            # Execution reaches the following statement only after normal completion.
            for stmt in (*node.body, *node.finalbody):
                bind(stmt, bindings)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.Match, ast.With, ast.AsyncWith)):
            invalidate_block(node, bindings)
        else:
            for name in local_names(node):
                bindings[name] = merged(bindings.get(name), None) if not isinstance(node, scopes) else None

    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

    def local_names(node):
        if isinstance(node, scopes):
            if hasattr(node, "name"):
                yield node.name
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            yield node.id
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield from (part.asname or part.name.split(".")[0] for part in node.names)
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            yield node.name
        if isinstance(node, ast.MatchMapping) and node.rest:
            yield node.rest
        for child in ast.iter_child_nodes(node):
            yield from local_names(child)

    def deferred_bindings(body, current):
        # Pytest test functions execute after module import. An explicitly invoked function or
        # closure may have run earlier, while its enclosing scope held a different value.
        final = dict(current)
        stable = dict(final)
        for stmt in body:
            before = dict(final)
            bind(stmt, final)
            for key, value in final.items():
                stable[key] = merged(stable.get(key), value) if key in before else value
        return final, stable

    def visit_branch(statements, initial, deferred, class_parent, nested_class):
        bindings = dict(initial)
        for stmt in statements:
            visit(stmt, bindings, deferred, class_parent if nested_class else bindings)
            bind(stmt, bindings)

    def visit(node, visible, deferred, class_parent):
        if isinstance(node, (ast.stmt, ast.Lambda)) and any(expression_writes(node)):
            original = visible
            visible = dict(visible)
            invalidate_walrus(node, visible)
            if class_parent is original:
                class_parent = visible
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
                explicit = isinstance(node, ast.Lambda) or node.name in called
                later_values = deferred[1] if explicit else deferred[0]
                current = {k: v for k, v in later_values.items() if k not in local}
                params = [*node.args.posonlyargs, *node.args.args]
                defaults = list(zip(reversed(params), reversed(node.args.defaults)))
                defaults.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))
                for param, default in defaults:
                    if binding(default, visible) in (solver, possible_solver):
                        current[param.arg] = possible_solver  # callers can supply another callable
                later = deferred_bindings(body, current)
                for stmt in body:
                    visit(stmt, current, later, current)
                    bind(stmt, current)
            return
        if isinstance(node, ast.If):
            after_test = visible
            visit(node.test, after_test, deferred, class_parent)
            for branch in (node.body, node.orelse):
                visit_branch(branch, after_test, deferred, class_parent, class_parent is not visible)
            return
        if isinstance(node, (ast.Try, ast.TryStar)):
            visit_branch(node.body, visible, deferred, class_parent, class_parent is not visible)
            normal = dict(visible)
            for stmt in node.body:
                bind(stmt, normal)
            visit_branch(node.orelse, normal, deferred, class_parent, class_parent is not visible)
            # Handlers and finally may run after a partially completed body. Forget the
            # bindings it may have changed, then visit each suite in its own source order.
            uncertain = dict(visible)
            may_have_exited = False
            for stmt in (*node.body, *node.orelse):
                # Leading assignments of literals to names complete before any later
                # exception can enter finally. After a possible exception, join conservatively.
                if (not may_have_exited and isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant)
                        and all(isinstance(target, ast.Name) for target in stmt.targets)):
                    bind(stmt, uncertain)
                else:
                    may_have_exited = True
                    invalidate_block(stmt, uncertain)
            for handler in node.handlers:
                local = dict(uncertain)
                if handler.type:
                    visit(handler.type, local, deferred, class_parent)
                if handler.name:
                    local[handler.name] = None
                visit_branch(handler.body, local, deferred, class_parent, class_parent is not visible)
            for handler in node.handlers:
                invalidate_block(handler, uncertain)
            visit_branch(node.finalbody, uncertain, deferred, class_parent, class_parent is not visible)
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # An iteration may inherit values from earlier iterations. Within the body,
            # unconditional assignments establish new bindings in statement order.
            entering = dict(visible)
            invalidate_block(node, entering)
            if isinstance(node, ast.While):
                visit(node.test, entering, deferred, class_parent)
            else:
                visit(node.iter, visible, deferred, class_parent)
                visit(node.target, entering, deferred, class_parent)
            for branch in (node.body, node.orelse):
                visit_branch(branch, entering, deferred, class_parent, class_parent is not visible)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            current = dict(visible)
            for generator in node.generators:
                visit(generator.iter, current, deferred, class_parent)
                current.update(dict.fromkeys(local_names(generator.target)))
                for condition in generator.ifs:
                    visit(condition, current, deferred, class_parent)
            elements = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            for element in elements:
                visit(element, current, deferred, class_parent)
            return
        if isinstance(node, ast.Call):
            target = binding(node.func, visible)
            if target in (solver, possible_solver):
                arg = next((k.value for k in node.keywords if k.arg == "timeout_sec"), None)
                if arg is None and node.args:
                    arg = node.args[0]
                if isinstance(arg, ast.Starred) and isinstance(arg.value, (ast.List, ast.Tuple)):
                    arg = next(iter(arg.value.elts), None)
                if arg is None:
                    for kw in node.keywords:
                        if kw.arg is None and isinstance(kw.value, ast.Dict):
                            for key, val in reversed(list(zip(kw.value.keys, kw.value.values))):
                                if not isinstance(key, ast.Constant):
                                    break    # a later dynamic key or **mapping may override the literal
                                if key.value == "timeout_sec":
                                    arg = val
                                    break
                            if arg is not None:
                                break
                value = numeric(arg, visible) if target is solver else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    value = None
                calls.append((node.lineno, value))
        if isinstance(node, ast.stmt) and any(isinstance(c, ast.stmt) for c in ast.iter_child_nodes(node)):
            # Branch/loop execution is unknown. Never reuse a binding it might overwrite.
            visible = dict(visible)
            invalidate_block(node, visible)
        for child in ast.iter_child_nodes(node):
            visit(child, visible, deferred, class_parent)

    sofar = {"run_solution": solver}
    final = deferred_bindings(tree.body, sofar)
    for node in tree.body:
        visit(node, sofar, final, sofar)
        bind(node, sofar)
    return calls, []


def _timeout_diagnostics(calls, budget, errors, unknown):
    for line, value in calls:
        if value is None:
            unknown.append(f"line {line}: run_solution timeout unknown")
        elif value >= budget:
            errors.append(f"line {line}: run_solution timeout {value:g}s >= verifier budget {budget:g}s")


def run_solution_timeouts(src: str, budget: float, *, warnings: list[str] | None = None) -> list[str]:
    """Per-call diagnostics for callers inspecting one source file."""
    calls, errors = _timeout_calls(src)
    _timeout_diagnostics(calls, budget, errors, warnings if warnings is not None else errors)
    return errors


def source_verifier_timeout(cfg: dict, spec: dict, task: Path) -> float:
    pk = cfg["package"]
    default = float(pk["verifier_timeout"]) + timing_budget(cfg, spec)
    state = read_json(task.parent / "state.json", {})
    approved = (state.get("verifier_timeout_pending") or {}).get("new") or state.get("approved_verifier_timeout")
    if approved is None:
        # Migration for a successful fix made before its approval was retained in state.
        approved = (read_json(task.parent / "PKG_check.json", {}).get("verifier_timeout_fix") or {}).get("new")
    if approved is None or approved > pk.get("max_verifier_timeout_s", 7200):
        return default
    source = task / "task.toml"
    try:
        value = tomllib.loads(source.read_text())["verifier"]["timeout_sec"]
        if type(value) in (int, float) and value == approved and math.isfinite(value):
            return max(default, float(value))
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        # A legacy source without a usable override still builds with the configured timeout.
        # An attempted repair must successfully update the source TOML before the item advances.
        pass
    return default


def verifier_timeout_check(tests: Path, current: float, overhead: float, *, packaged_helper: str | None = None) -> dict:
    """Gate and repair use the same call sites, sum and reserved verifier overhead."""
    total = 0.0
    budget = current - overhead
    errors, unknown, invalid = [], [], []
    paths = tests.glob("*.py") if packaged_helper is None else _authored_test_files(tests)
    sources = ((p.name, p.read_text()) for p in sorted(paths) if p.is_file())
    if packaged_helper is not None:
        sources = chain(sources, [("_metrics.py", packaged_helper)])
    for name, source in sources:
        calls, parse_errors = _timeout_calls(source)
        problems = []
        diagnostics = []
        # Invalid Python is a grader error. It cannot be repaired by increasing the timeout.
        invalid.extend(f"{name}: {p}" for p in parse_errors)
        _timeout_diagnostics(calls, budget, problems, diagnostics)
        errors.extend(f"{name}: {p}" for p in problems)
        unknown.extend(f"{name}: {p}" for p in diagnostics)
        total += sum(value for _, value in calls if value is not None)
    if total and total >= budget and not errors:
        errors.append(f"run_solution timeouts sum to {total:g}s >= verifier budget {budget:g}s")
    # The gate is strict at equality. Leave at least one second when a large configured
    # overhead would otherwise put the repaired package exactly on the gate boundary.
    required = None
    if math.isfinite(total + overhead) and overhead >= 0:
        required = max(math.ceil(total + max(overhead, 300)), math.floor(total + overhead) + 1)
    return dict(current=current, sum_sec=total, required=required, errors=errors, unknown=unknown, invalid=invalid)


def set_verifier_timeout(path: Path, seconds: float):
    """Update the one generated TOML field while preserving all unrelated bytes."""
    original = path.read_bytes() if path.exists() else b""
    before = tomllib.loads(original.decode())
    header = re.search(rb"(?m)^[ \t]*\[verifier\][ \t]*(?:#[^\r\n]*)?\r?$", original)
    value = str(float(seconds)).encode()
    if header:
        end = re.search(rb"(?m)^\s*\[", original[header.end():])
        stop = header.end() + end.start() if end else len(original)
        section = original[header.end():stop]
        key = re.search(rb"(?m)^([ \t]*timeout_sec[ \t]*=[ \t]*)([^#\s]+)", section)
        if key:
            section = section[:key.start(2)] + value + section[key.end(2):]
        else:
            section = b"\ntimeout_sec = " + value + section
        updated = original[:header.end()] + section + original[stop:]
    else:
        newline = b"\n" if original and not original.endswith(b"\n") else b""
        updated = original + newline + b"[verifier]\ntimeout_sec = " + value + b"\n"
    if not isinstance(before.setdefault("verifier", {}), dict):
        raise ValueError("source verifier is not a TOML table")
    before["verifier"]["timeout_sec"] = float(seconds)
    if tomllib.loads(updated.decode()) != before:
        raise ValueError("could not safely update [verifier].timeout_sec")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(updated)
            handle.flush()
            if path.exists():
                shutil.copymode(path, tmp)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


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


def sandbox_pin_errors(cfg: dict, pins: list, image: str | None = None) -> list[str]:
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
        kwargs = {"image": image} if image else {}
        result = sandbox.run(cfg, [], "python -c " + shlex.quote(code), 30, "tbsgen-check-pins", **kwargs)
    except (OSError, ValueError) as e:
        return [f"extra_pip_unsupported: cannot verify {pins} in the configured sandbox: {e}"]
    if result["rc"]:
        return [f"extra_pip_unsupported: {pins} in {image or cfg['sandbox']['image']}: {result['err'][-300:]}"]
    return []


def validate_pkg(pkg: Path, cfg: dict | None = None) -> dict:
    provenance = read_json(pkg / "provenance.json", {})
    missing = [r for r in REQUIRED if not (pkg / r).exists()]
    errs, warnings = [], []
    check = {}
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
        check = verifier_timeout_check(pkg / "tests", float(t["verifier"]["timeout_sec"]), overhead)
        errs.extend(check["errors"])
        errs.extend(check["unknown"])
        errs.extend(check["invalid"])
    except Exception as e:  # noqa: BLE001
        errs.append(f"task.toml: {e}")
    if cfg is not None and not missing:
        pins = provenance.get("extra_pip", [])
        errs.extend(sandbox_pin_errors(cfg, pins, image=provenance.get("image_digest")))
        errs += verifier_time_errors(pkg, float((cfg.get("package") or {}).get("verifier_overhead_sec", 120)))
    leaked = [str(p.relative_to(pkg)) for p in (pkg / "environment").rglob("*")
              if p.is_file() and any(s in p.parts for s in ("tests", "truth", "generator", "solution"))]
    if leaked:
        errs.append(f"answer-bearing files under environment/: {leaked[:5]}")
    for gen_only in ("generator", "baselines", "_spec"):
        if (pkg / gen_only).exists():
            errs.append(f"{gen_only}/ must not be packaged")
    return dict(ok=not missing and not errs, missing=missing, errors=errs, warnings=warnings, run_solution_timeouts=check)
