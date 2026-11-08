"""Pre-release checks of a packaged task: network policy of the agent phase, and ready-made implementations.

Network (`package.network`): an agent solving a packaged task must not reach the open internet, or it can
`pip install` the paper's own library or fetch a finished implementation from GitHub.  Harbor (0.23+) supports
per-task network policies in task.toml:

    [environment] network_mode = "public" | "no-network" | "allowlist", allowed_hosts = [...]

and run-time host merges (`harbor run --allow-agent-host HOST`, `agent.extra_allowed_hosts` in a job config)
that apply to the agent phase only.  `allowlist` is enforced by Harbor's Docker environment (an egress sidecar,
needs a Linux kernel with nftables fib) and by its E2B-based sandboxes; a provider that cannot enforce it
refuses to start the trial instead of running open.  Installed agents (Claude Code, Codex) set up during agent
setup, before the agent phase, under the [environment] baseline: an install step that downloads needs its hosts
in the baseline allowlist (`package.network.install_hosts`), while the model endpoint is added per run by the
launcher (`--allow-agent-host`; calib.agent_allowed_hosts).

Modes written by `package` (`package.network.mode`):
  "agent-allowlist" (default)  [environment] network_mode = "public" (agent setup installs the agent CLI),
                               [agent] network_mode = "allowlist", allowed_hosts = package.network.agent_hosts
                               (plus the model endpoint the launcher adds per run: calib.agent_allowed_hosts ->
                               agents[].extra_allowed_hosts), [verifier] network_mode = "no-network".  Changing the
                               policy between phases needs a provider with dynamic network policy: Harbor's Docker
                               environment (egress sidecar; Linux kernel with nftables fib) and its E2B-based
                               sandboxes have it; a provider without it refuses the trial instead of running open.
  "no-network"                 offline throughout (oracle / nop checks; an agent that calls a model cannot run).
  "public"                     the old behaviour; comparison runs only.
With `package.network.disable_index`, the image additionally has its package-index configuration removed and
`PIP_NO_INDEX=1` etc. set after the build (off by default: agent setups that pip-install their own runtime would
break; it is a speed bump, not a barrier -- the agent can unset it).

Ready-made implementations (`readymade_check`): rule check, blocking at S7.  Signals of the paper's own code:
code_links / data_links in the retrieval record (repository names), distribution names declared in the item's
environment (requirements files, `pip install` lines in a Dockerfile under the task), and the method names in
the S5 spec.  A hit is (a) a dependency of the task image or (b) an import in the reference solution or a
baseline that matches one of those signals.  Known companion repositories are written to the task metadata
(provenance `companion_code`) for calibration and human review.
"""
from __future__ import annotations

import ast
import contextlib
import re
from pathlib import Path

# the pinned stack of the sandbox / task image: never a signal of a ready-made implementation
BASE_STACK = {"numpy", "scipy", "pandas", "scikit-learn", "sklearn", "statsmodels", "pytest", "matplotlib"}
STDLIB_HINT = {"os", "sys", "re", "json", "csv", "math", "pathlib", "itertools", "functools", "collections",
               "random", "time", "typing", "dataclasses", "argparse", "subprocess", "shutil", "glob", "io",
               "statistics", "hashlib", "warnings", "heapq", "bisect", "copy", "pickle", "struct", "string",
               "textwrap", "enum", "abc", "contextlib", "operator", "decimal", "fractions", "numbers", "_metrics",
               "measure"}
GENERIC = {"python", "code", "src", "lib", "main", "master", "data", "dataset", "datasets", "paper", "project",
           "model", "models", "tools", "toolbox", "utils", "repo", "release", "releases", "supplementary", "the",
           "and", "for", "with", "method", "methods", "package", "software", "analysis", "github", "gitlab"}
CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "sourceforge.net", "pypi.org",
              "cran.r-project.org", "huggingface.co")

PIP_LINE = re.compile(r"\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+install\b(.*)", re.I)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


def _tokens(name: str) -> set[str]:
    n = _norm(name)
    out = {n, n.replace("-", ""), n.replace("-", "_")}
    if n.startswith(("py", "python-")) and len(n) > 4:
        out.add(n.removeprefix("python-").removeprefix("py"))
    if n.endswith(("-py", "py")) and len(n) > 4:
        out.add(n.removesuffix("-py").removesuffix("py"))
    return {t.strip("-_") for t in out if len(t.strip("-_")) >= 3 and t.strip("-_") not in GENERIC}


def repo_names(links: list[str]) -> list[dict]:
    """Repository / package names from code / data links: github.com/<owner>/<repo>, pypi.org/project/<name>, ..."""
    out = []
    for u in links or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not re.match(r"[a-z][a-z0-9+.-]*://", u, re.I):          # retrieval writes github.com/owner/repo too
            u = "https://" + u.removeprefix("//")
        m = re.match(r"https?://(?:www\.)?([^/]+)/+([^/?#]+)(?:/+([^/?#]+))?", u)
        if not m:
            continue
        host, a, b = m.group(1).lower(), m.group(2), m.group(3)
        if not any(host == h or host.endswith("." + h) for h in CODE_HOSTS):
            continue
        if host.endswith("pypi.org") and a == "project" and b:
            out.append(dict(url=u, host=host, name=b, kind="pypi"))
        elif host.endswith("cran.r-project.org") and b:
            out.append(dict(url=u, host=host, name=b, kind="cran"))
        elif b:
            out.append(dict(url=u, host=host, owner=a, name=re.sub(r"(\.git)?(@.*)?$", "", b), kind="repo"))
    return out


def signals(candidate: dict, spec: dict | None) -> dict:
    """Names that identify the paper's own implementation."""
    links = list(candidate.get("code_links") or candidate.get("code_urls") or []) + \
        list(candidate.get("data_links") or candidate.get("data_urls") or [])
    repos = repo_names(links)
    names: dict[str, str] = {}
    for r in repos:
        for t in _tokens(r["name"]):
            names.setdefault(t, f"{r['kind']} {r.get('owner', '') + '/' if r.get('owner') else ''}{r['name']}")
    for m in (spec or {}).get("method_names") or []:
        for t in _tokens(str(m)):
            names.setdefault(t, f"method name {m}")
    # software names given in the candidate record (retrieval: `libs`, `software`, `tools`)
    for k in ("libs", "software", "tools", "packages"):
        for m in candidate.get(k) or []:
            for t in _tokens(str(m)):
                names.setdefault(t, f"{k} {m}")
    return dict(repos=repos, names=names)


_REQ_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


def _spec_name(tok: str) -> list[str]:
    """Distribution name(s) of one requirement token: `name[extra]>=1`, `"name==1"`, `name @ url`, a VCS / archive
    URL (`git+https://github.com/o/repo.git@v1#egg=name`), or a local path (its last component)."""
    tok = tok.strip().strip("'\"")
    if not tok or tok.startswith("-"):
        return []
    out = []
    if (m := re.search(r"[#&]egg=([A-Za-z0-9._-]+)", tok)):
        out.append(m.group(1))
    url = re.sub(r"^[a-z]+\+", "", tok)
    if re.match(r"(https?|ssh|file)://", url) or re.match(r"[\w.-]+\.[a-z]{2,}/", url):
        out += [r["name"] for r in repo_names([url])]
        if not out:
            out.append(re.split(r"[/#?]", url.rstrip("/"))[-1].split("-")[0])
        return out
    if tok.startswith((".", "/", "~")):
        return [Path(tok.rstrip("/")).name]
    if (m := _REQ_NAME.match(tok)):
        out.append(m.group(0))
    return out


def _req_names(text: str) -> list[str]:
    """Distribution names in a requirements file (incl. `-e <url or path>` lines)."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not line or line.startswith("#"):
            continue
        if (m := re.match(r"(-e|--editable)\s+(.+)", line)):
            out += _spec_name(m.group(2))
        elif not line.startswith("-"):
            out += _spec_name(line.split(";")[0].split(" @ ")[-1] if " @ " in line else line.split(";")[0])
            if " @ " in line:
                out += _spec_name(line.split(" @ ")[0])
    return out


def environment_packages(task: Path) -> list[dict]:
    """Packages the task image installs beyond the pinned stack: requirements*.txt / pyproject / environment.yml
    and pip lines of any Dockerfile under environment/ (task source tree or packaged task)."""
    out = []
    env = task / "environment"
    for p in sorted(env.rglob("*")) if env.exists() else []:
        if not p.is_file() or p.stat().st_size > 1_000_000:
            continue
        rel = str(p.relative_to(task))
        name = p.name.lower()
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if name.startswith("requirements") and name.endswith(".txt"):
            out += [dict(name=n, file=rel) for n in _req_names(txt)]
        elif name in ("environment.yml", "environment.yaml"):
            out += [dict(name=n, file=rel) for n in re.findall(r"^\s*-\s*(?:pip:\s*)?([A-Za-z0-9][\w.-]*)", txt, re.M)]
        elif name == "dockerfile" or name.endswith(".dockerfile"):
            joined = re.sub(r"\\[ \t]*\r?\n", " ", txt)          # Dockerfile line continuations
            for cmd in re.split(r"&&|;|\|\||\n", joined):
                m = PIP_LINE.search(cmd)
                if not m:
                    continue
                words = m.group(1).split()
                skip_next = False
                for w in words:
                    if skip_next:
                        skip_next = False
                        continue
                    if w in ("-r", "--requirement", "-c", "--constraint", "-i", "--index-url", "--extra-index-url",
                             "-f", "--find-links", "-t", "--target", "--prefix", "--root"):
                        skip_next = True
                        continue
                    if w in ("-e", "--editable"):
                        continue
                    out += [dict(name=n, file=rel) for n in _spec_name(w)]
        elif name == "pyproject.toml":
            import tomllib
            with contextlib.suppress(ValueError):
                t = tomllib.loads(txt)
                deps = list((t.get("project") or {}).get("dependencies") or [])
                for v in ((t.get("project") or {}).get("optional-dependencies") or {}).values():
                    deps += list(v)
                for v in (t.get("dependency-groups") or {}).values():
                    deps += [d for d in v if isinstance(d, str)]
                out += [dict(name=n, file=rel) for d in deps for n in _req_names(str(d))]
    seen, uniq = set(), []
    for x in out:
        k = (_norm(x["name"]), x["file"])
        if k not in seen and k[0] not in BASE_STACK:
            seen.add(k)
            uniq.append(x)
    return uniq


def imports(py: Path) -> set[str]:
    try:
        tree = ast.parse(py.read_text(errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
            out.add(str(node.args[0].value).split(".")[0])
    return out


def _match(name: str, sig: dict[str, str]) -> str | None:
    for t in _tokens(name):
        if t in sig:
            return sig[t]
    return None


def readymade_check(task: Path, candidate: dict, spec: dict | None, public_env: Path | None = None) -> dict:
    """Rule check: does the task image install, or does the reference / a baseline import, the paper's own
    implementation?  Any third-party import of the reference or baselines outside the pinned stack is reported
    too (the sandbox image has only that stack, so such an import is either unused or a dependency to review)."""
    sg = signals(candidate, spec)
    findings = []
    for pkg in environment_packages(task):
        if (why := _match(pkg["name"], sg["names"])):
            findings.append(dict(kind="dependency", name=pkg["name"], file=pkg["file"], matches=why))
    for rel in ["solution/solve.py", *sorted(str(p.relative_to(task)) for p in (task / "baselines").glob("*.py"))]:
        p = task / rel
        if not p.exists():
            continue
        for mod in sorted(imports(p) - STDLIB_HINT):
            if _norm(mod) in BASE_STACK:
                continue
            # a local module of that name is a vendored copy of the implementation: just as bad
            why = _match(mod, sg["names"])
            if why:
                findings.append(dict(kind="import", name=mod, file=rel, matches=why))
    # code shipped to the agent under environment/ whose name matches the paper's implementation: the task tree and
    # the generator's public output (`public_env`, e.g. runs/s7-N/detA/environment at S7)
    shipped = [task / "environment"] + ([public_env] if public_env else [])
    for env, p in [(e, p) for e in shipped if e.exists() for p in sorted(e.rglob("*"))]:
        if p.is_file() and p.suffix.lower() in (".py", ".r", ".jl", ".m", ".c", ".cpp", ".f90", ".ipynb", ".whl",
                                                ".tar", ".gz", ".zip"):
            for part in p.relative_to(env).parts:
                name = re.sub(r"\.(py|r|jl|m|c|cpp|f90|ipynb|whl|tar|gz|zip)$", "", part, flags=re.I)
                if (why := _match(name.split("-")[0], sg["names"]) or _match(name, sg["names"])):
                    findings.append(dict(kind="public_code", name=part, file=str(p.relative_to(env.parent)),
                                         matches=why))
                    break
    return dict(ok=not findings, findings=findings, companion_code=[r["url"] for r in sg["repos"]],
                signals=sorted(sg["names"]))
