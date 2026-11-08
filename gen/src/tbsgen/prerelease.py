"""Pre-release checks of a packaged task: network policy of the agent phase, and ready-made implementations.

Network (`package.network`): an agent solving a packaged task must not reach the open internet, or it can
`pip install` the paper's own library or fetch a finished implementation from GitHub.  Harbor (checked against
0.23.0) supports per-phase network policies in task.toml -- [environment] network_mode = "public" | "no-network" |
"allowlist" with allowed_hosts is the baseline, [agent] / [verifier] override it for those phases -- and run-time
host merges for the agent phase only (`harbor run --allow-agent-host HOST`, agents[].extra_allowed_hosts in a job
config).  Installed agents (Claude Code, Codex) are set up under the baseline before the agent-phase policy is
applied; the model endpoint is added per run by the launcher (calib.agent_allowed_hosts).

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

Scope (design contract): this is a NAME-BASED rule check.  It covers these naming forms only:
  - requirement specifiers `name[extra]==v`, `name @ url`, `-e <url|path>`, `#egg=name`, VCS URLs
    (`git+https://host/owner/repo.git@ref`), direct archive URLs / files (`decay-fit-1.2.0.tar.gz`,
    `x-1.0-py3-none-any.whl#sha256=...`: version and fragment removed), local paths (last component);
  - code-host links: github / bitbucket / codeberg / huggingface `owner/repo`, GitLab `group/sub/.../repo` (last
    project segment), PyPI `/project/<name>` and `/simple/<name>`, CRAN `?package=<name>` / `/package=<name>` /
    `/web/packages/<name>` / `/src/contrib/<name>_<ver>.tar.gz`;
  - files shipped under environment/ whose path component (minus version / branch suffix) or compiled-module stem
    (`decayfit.cpython-311-*.so`, `libdecayfit.so`, `decayfit.pyc`) matches a signal; name variants: case,
    `-`/`_`/`.`, `py`/`python-` prefix, `-py`/`py` suffix.
It does NOT promise to find an implementation that was renamed, vendored under another name, obfuscated, or
fetched at run time by other means; calibration (a strong agent's score) and human review cover those.
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
    stems = {n}
    if n.startswith(("py", "python-")) and len(n) > 4:
        stems.add(n.removeprefix("python-").removeprefix("py").strip("-"))
    if n.endswith(("-py", "py")) and len(n) > 4:
        stems.add(n.removesuffix("-py").removesuffix("py").strip("-"))
    out = set()
    for s in stems:                       # every variant in every separator form: py_decay_fit -> decayfit
        out |= {s, s.replace("-", ""), s.replace("-", "_")}
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
        m = re.match(r"https?://(?:[^@/]*@)?(?:www\.)?([^/:?#]+)(?::\d+)?(/[^?#]*)?", u)
        if not m:
            continue
        host = m.group(1).lower()
        parts = [p for p in (m.group(2) or "").split("/") if p]
        if not any(host == h or host.endswith("." + h) for h in CODE_HOSTS):
            continue
        if host.endswith("cran.r-project.org"):
            # /package=<name>, /web/packages/<name>/..., /src/contrib/<name>_<ver>.tar.gz
            q = re.search(r"[/?&]package=([A-Za-z0-9.]+)", u)
            name = q.group(1) if q else None
            if not name and len(parts) >= 3 and parts[:2] == ["web", "packages"]:
                name = parts[2]
            if not name and parts[:2] == ["src", "contrib"] and len(parts) >= 3:
                name = parts[-1].split("_")[0]
            if name:
                out.append(dict(url=u, host=host, name=name, kind="cran"))
            continue
        if host.endswith("pypi.org"):
            if len(parts) >= 2 and parts[0] in ("project", "simple"):
                out.append(dict(url=u, host=host, name=parts[1], kind="pypi"))
            continue
        if host.endswith("gitlab.com") and "-" in parts:      # gitlab.com/group/sub/.../repo/-/tree/main
            parts = parts[:parts.index("-")]
        elif not host.endswith("gitlab.com"):
            parts = parts[:2]                                   # github / bitbucket / ...: owner/repo/...
        if len(parts) >= 2:                                     # gitlab: the last segment is the project
            name = re.sub(r"(\.git)?(@.*)?$", "", parts[-1])
            out.append(dict(url=u, host=host, owner="/".join(parts[:-1]), name=name, kind="repo"))
    return out


_ARCHIVE = re.compile(r"\.(tar\.gz|tar\.bz2|tar\.xz|tgz|zip|whl|egg|tar)$", re.I)


def _archive_dist(fname: str) -> str:
    """Distribution name of a source / wheel archive file name: `decay-fit-1.2.0.tar.gz` -> `decay-fit`,
    `decay_fit-1.2-py3-none-any.whl` -> `decay_fit`, `DecayFit-main.zip` -> `DecayFit`."""
    stem = _ARCHIVE.sub("", fname)
    return re.split(r"[-_]v?\d|-(?:main|master)$", stem, maxsplit=1)[0] or stem


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
            # direct archive / file URL: last path segment without query / fragment (#sha256=...) and version
            path = re.split(r"[#?]", url, maxsplit=1)[0].rstrip("/")
            out.append(_archive_dist(path.rsplit("/", 1)[-1]))
        return out
    if tok.startswith((".", "/", "~")) or "/" in tok:      # a local path (a distribution name has no "/")
        return [_archive_dist(Path(re.split(r"[#?]", tok)[0].rstrip("/")).name)]
    if _ARCHIVE.search(re.split(r"[#?]", tok)[0]):
        return [_archive_dist(re.split(r"[#?]", tok)[0])]
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


# source / package / compiled files (compound archive suffixes first)
_CODE_FILE = re.compile(r"\.(tar\.gz|tar\.bz2|tar\.xz|tgz|zip|whl|tar|gz|egg|py|pyx|r|jl|m|c|cc|cpp|h|f90|f|ipynb"
                        r"|so|pyd|pyc|pyo|dylib|dll|a|o|jar|mex\w*)$", re.I)


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
        if p.is_file() and _CODE_FILE.search(p.name):
            for part in p.relative_to(env).parts:
                stem = _CODE_FILE.sub("", part)
                # DecayFit-1.2.0 / decayfit_v2 / decayfit-main: also the part before a version / branch suffix
                base = re.split(r"[-_.]v?\d|[-_](?:main|master|src|dist)$", stem, maxsplit=1)[0]
                # compiled modules: decayfit.cpython-311-x86_64-linux-gnu.so, decayfit.abi3.so, libdecayfit.so
                first = re.sub(r"^lib(?=\w{3})", "", stem.split(".")[0])
                if (why := _match(stem, sg["names"]) or _match(base, sg["names"]) or _match(first, sg["names"])):
                    findings.append(dict(kind="public_code", name=part, file=str(p.relative_to(env.parent)),
                                         matches=why))
                    break
    return dict(ok=not findings, findings=findings, companion_code=[r["url"] for r in sg["repos"]],
                signals=sorted(sg["names"]))
