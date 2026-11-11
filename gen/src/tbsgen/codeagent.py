"""Code-grounded stages that run on the build host (gu26): S3c (fetch the authors' code), S4c (runnability gate) and
S5G (task authoring by a Claude Code agent).

Where things run
  * Third-party code is never executed on the orchestrating machine.  S3c clones on the build host
    (`code.host`, over ssh) into `<code.remote_root>/<run_tag>/<item>/repo` (internal run directory; never in a
    public repository).  S4c / S5G are Harbor jobs on the build host (same launcher route as the S9 teacher:
    `code.agent_launcher` in `code.agent_harbor_root`, environment `code.agent_environment`, the model endpoint
    as the agent's only allowed host besides the install-time network).
  * The task's own image (pkg/) never contains the authors' repository or package: S5G writes task files only
    (generator / solution / tests / baselines / instruction); `_spec/author_code.json` records what the reference
    and the generator use from the authors' code, and S7's ready-made check is extended with the repository,
    package and main function names (prerelease.readymade_check(author_names=...)).

S3c   `S3c_code.json`: repo, commit SHA, size, languages, licence (recorded only), LFS skipped, size cap
      (`code.max_repo_mb`, default 500).  Zenodo / CRAN / PyPI archives: downloaded and unpacked the same way,
      the archive's sha256 is the pin.  Drops: `code_fetch_failed`, `code_too_large`.
S4c   A Claude Code agent installs the environment in a Docker image on the build host (network allowed while
      installing, within `code.runnability_timeout_s`), runs the authors' example / test / main entry once and
      writes /app/out/runnability.json + run.log + HOWTO.md ("how to run the authors' code").  Drops:
      `code_not_runnable`, `code_needs_gpu`, `code_needs_commercial_license` (MATLAB tried under Octave first;
      Mathematica, commercial solvers -> drop), `code_runnability_agent_failed`.
S5G   A Claude Code agent with the paper, the repository, HOWTO.md and the harness contract (S5_SPEC / S6A_GEN /
      METRICS_API) writes exactly the files S5 + S6A write (_spec/spec.json and task/{generator,solution,tests,
      baselines,instruction.md}) and may run them in its container until reference and baselines separate.
      Then the unchanged S6A_run -> S6B -> S6C -> S7 -> S8 -> PKG -> S9 -> S9R -> S10 take over.
      `code.authoring = "single_call"` keeps the old S5 + S6A path (fallback).

Budget: an agent job reserves `code.agent_est_usd` in the ledger (stage S4c / S5G, the teacher model) and is
settled with Claude Code's reported cost (or token usage x models.prices).  Per paper the job is capped in time
(`code.authoring_timeout_s`, default 7200) and, through Claude Code `--max-budget-usd`, in estimated spend
(`code.authoring_max_usd`, default 20).
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from .state import read_json, write_json

# ----------------------------------------------------------------------------------------------- remote shell


def _c(cfg: dict) -> dict:
    return cfg.get("code") or {}


def ssh_run(cfg: dict, cmd: str, timeout: float = 600, stdin: bytes | None = None) -> dict:
    """Run a shell command on the build host.  -> {rc, out, err}.  host=None/"local" is for tests only (it never
    executes third-party code: the commands are fixed git / curl / tar / du calls)."""
    host = _c(cfg).get("host")
    args = ["bash", "-lc", cmd] if host in (None, "", "local") else \
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", host, cmd]
    try:
        p = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
        return dict(rc=p.returncode, out=p.stdout.decode("utf8", "replace")[-20000:],
                    err=p.stderr.decode("utf8", "replace")[-4000:])
    except subprocess.TimeoutExpired:
        return dict(rc=-9, out="", err=f"timeout after {timeout}s")
    except OSError as e:
        return dict(rc=-1, out="", err=f"{type(e).__name__}: {e}")


def remote_item_dir(cfg: dict, item_id: str) -> str:
    root = _c(cfg).get("remote_root")
    if not root:
        raise ValueError("code.remote_root is not set (build-host directory for the authors' code)")
    tag = _c(cfg).get("run_tag") or "run"
    return f"{root.rstrip('/')}/{tag}/{item_id}"


# ----------------------------------------------------------------------------------------------- S3c
FETCH_SH = r"""
set -uo pipefail
D={dir}; URL={url}; KIND={kind}; MAXMB={maxmb}; REF={ref}
mkdir -p "$D"; cd "$D"
rm -rf repo.tmp; mkdir -p dl
emit() {{ echo "TBSGEN_RESULT $1"; }}
if [ "$KIND" = git ]; then
  export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
  if ! timeout {tmo} git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=120 clone --quiet --depth 1 \
       --no-recurse-submodules ${{REF:+--branch "$REF"}} "$URL" repo.tmp 2> dl/clone.err; then
    emit "{{\"ok\": false, \"why\": \"clone_failed\", \"err\": $(tail -c 400 dl/clone.err | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}}"; exit 0
  fi
  SHA=$(git -C repo.tmp rev-parse HEAD); PIN="$SHA"
else
  if ! timeout {tmo} curl -sSfL --max-filesize $((MAXMB*1024*1024)) -o dl/archive "$URL" 2> dl/curl.err; then
    emit "{{\"ok\": false, \"why\": \"download_failed\", \"err\": $(tail -c 400 dl/curl.err | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}}"; exit 0
  fi
  PIN=sha256:$(sha256sum dl/archive | cut -d' ' -f1); SHA=""
  mkdir -p repo.tmp
  (cd repo.tmp && (unzip -q ../dl/archive 2>/dev/null || tar xf ../dl/archive 2>/dev/null)) || {{
    emit '{{"ok": false, "why": "unpack_failed"}}'; exit 0; }}
  # a single top-level directory is the project root
  n=$(ls -A repo.tmp | wc -l); if [ "$n" = 1 ] && [ -d "repo.tmp/$(ls -A repo.tmp)" ]; then
    mv "repo.tmp/$(ls -A repo.tmp)" repo.tmp2 && rmdir repo.tmp && mv repo.tmp2 repo.tmp; fi
fi
MB=$(python3 -c 'import os,sys;print(int(sum(os.path.getsize(os.path.join(d,f)) for d,ds,fs in os.walk("repo.tmp") if not (ds.remove(".git") if ".git" in ds else None) for f in fs if not os.path.islink(os.path.join(d,f)))/1048576+0.999))')
if [ "$MB" -gt "$MAXMB" ]; then rm -rf repo.tmp; emit "{{\"ok\": false, \"why\": \"too_large\", \"mb\": $MB}}"; exit 0; fi
rm -rf repo; mv repo.tmp repo
LFS=$(grep -l "filter=lfs" repo/.gitattributes 2>/dev/null | wc -l)
python3 - "$PIN" "$SHA" "$MB" "$LFS" <<'EOF'
import json, os, sys, collections
pin, sha, mb, lfs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
EXT = {{".py": "Python", ".ipynb": "Python", ".r": "R", ".jl": "Julia", ".m": "MATLAB", ".c": "C", ".h": "C",
       ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".f90": "Fortran", ".f": "Fortran", ".rs": "Rust", ".go": "Go",
       ".java": "Java", ".wl": "Mathematica", ".nb": "Mathematica", ".cu": "CUDA", ".pyx": "Cython", ".stan": "Stan"}}
langs = collections.Counter(); files = 0; deps = []; lic = None
DEP = {{"requirements.txt", "environment.yml", "environment.yaml", "setup.py", "pyproject.toml", "setup.cfg",
       "DESCRIPTION", "Project.toml", "Cargo.toml", "CMakeLists.txt", "Makefile", "Dockerfile", "renv.lock"}}
for dp, dn, fn in os.walk("repo"):
    dn[:] = [d for d in dn if d not in (".git", "node_modules")]
    for f in fn:
        files += 1
        p = os.path.join(dp, f)
        e = os.path.splitext(f)[1].lower()
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if e in EXT:
            langs[EXT[e]] += sz
        if f in DEP or f.startswith("requirements"):
            deps.append(os.path.relpath(p, "repo"))
        if dp == "repo" and f.upper().startswith(("LICENSE", "LICENCE", "COPYING")) and lic is None:
            lic = open(p, errors="ignore").read(300).strip().splitlines()[0][:120] if sz else None
print("TBSGEN_RESULT " + json.dumps(dict(ok=True, pin=pin, sha=sha or None, size_mb=mb, lfs_skipped=bool(lfs),
      files=files, languages={{k: round(v / 1024) for k, v in langs.most_common(6)}}, dep_files=sorted(deps)[:20],
      license_file=lic)))
EOF
"""


def fetch_spec(primary: dict) -> dict:
    """What S3c fetches for the S3b primary record: a git URL (GitHub / GitLab / Bitbucket / Bioconductor) or an
    archive URL (Zenodo software record, CRAN / PyPI source distribution)."""
    h = primary.get("host")
    if h in ("github", "gitlab", "bitbucket"):
        return dict(kind="git", url=(primary.get("url") or "").rstrip("/") + ".git", ref=None)
    if h == "bioconductor" and primary.get("git"):
        return dict(kind="git", url=primary["git"], ref=None)
    if h == "zenodo":
        rec = str(primary.get("key", "")).split(":")[-1]
        files = [f for f in primary.get("files") or [] if re.search(r"\.(zip|tar\.gz|tgz|tar)$", f, re.I)]
        if not files:
            return dict(kind="none", why="zenodo record without an archive")
        return dict(kind="archive", url=f"https://zenodo.org/records/{rec}/files/{files[0]}?download=1", ref=None)
    if h == "cran":
        return dict(kind="archive", url="", ref=None, cran=str(primary.get("key", "")).split(":")[-1])
    if h == "pypi":
        return dict(kind="pypi", name=str(primary.get("key", "")).split(":")[-1])
    return dict(kind="none", why=f"host {h} not fetchable")


def s3c_fetch(cfg: dict, item_id: str, primary: dict, runner=None) -> dict:
    run = runner or (lambda cmd, timeout: ssh_run(cfg, cmd, timeout=timeout))
    spec = fetch_spec(primary)
    if spec["kind"] == "pypi":                          # resolve the sdist URL on the build host side
        st, d = _pypi_sdist(spec["name"], cfg)
        spec = dict(kind="archive", url=d, ref=None) if st else dict(kind="none", why=d)
    if spec.get("cran"):
        tb = _cran_tarball(str(spec["cran"]), cfg)
        spec = dict(kind="archive", url=f"https://cran.r-project.org/src/contrib/{tb}", ref=None) if tb else \
            dict(kind="none", why="CRAN version unknown")
    if spec["kind"] == "none":
        return dict(ok=False, why="unsupported", detail=spec.get("why"))
    d = remote_item_dir(cfg, item_id)
    mx = int(_c(cfg).get("max_repo_mb", 500))
    tmo = int(_c(cfg).get("fetch_timeout_s", 900))
    sh = FETCH_SH.format(dir=shlex.quote(d), url=shlex.quote(str(spec["url"])), kind=spec["kind"], maxmb=mx,
                         ref=shlex.quote(spec.get("ref") or ""), tmo=tmo)
    r = run(sh, tmo + 120)
    res = None
    for line in reversed((r.get("out") or "").splitlines()):
        if line.startswith("TBSGEN_RESULT "):
            try:
                res = json.loads(line[len("TBSGEN_RESULT "):])
            except ValueError:
                pass
            break
    if res is None:
        return dict(ok=False, why="fetch_error", rc=r.get("rc"), err=(r.get("err") or "")[-400:], transient=True)
    return dict(res, source=spec, remote_dir=d, host=_c(cfg).get("host"))


def _pypi_sdist(name: str, cfg: dict) -> tuple[bool, str]:
    from .codegate import http_json
    st, d = http_json(f"https://pypi.org/pypi/{name}/json", cfg)
    if st != 200 or not isinstance(d, dict):
        return False, f"pypi {st}"
    for u in d.get("urls") or []:
        if u.get("packagetype") == "sdist":
            return True, u["url"]
    return False, "no sdist"


def _cran_tarball(name: str, cfg: dict) -> str | None:
    from .codegate import http_json
    st, d = http_json(f"https://crandb.r-pkg.org/{name}", cfg)
    if st == 200 and isinstance(d, dict) and d.get("Version"):
        return f"{name}_{d['Version']}.tar.gz"
    return None


# ----------------------------------------------------------------------------------------------- agent jobs
AGENT_TASK_TOML = """schema_version = "1.0"
[task]
name = "tbsgen-{kind}/{item}"
description = "tbsgen {kind} agent job for {item}"
[metadata]
generator = "tbsgen code-grounded {kind}"
[verifier]
timeout_sec = 300.0
network_mode = "no-network"
[agent]
timeout_sec = {timeout:.1f}
# installs need package indexes / code hosts while the agent works (build host only; the packaged task the solver
# sees keeps its own offline policy).  The model endpoint is added per run (agents[].extra_allowed_hosts).
network_mode = "allowlist"
allowed_hosts = {hosts}
[environment]
build_timeout_sec = 1800.0
cpus = {cpus}
memory_mb = {mem}
storage_mb = 20480
gpus = 0
network_mode = "public"
"""

AGENT_DOCKERFILE = """FROM {image}
# toolchains are installed by the agent while it works (allowlisted package hosts), not at build time
WORKDIR /app
COPY work /work
RUN mkdir -p /app/out /logs/verifier && chmod -R a+rwX /work /app
"""

VERIFY_SH = """#!/bin/bash
# structure check of the agent's deliverable; reward 1 = the expected files exist
mkdir -p /logs/verifier
cp -r /app/out /logs/verifier/out 2>/dev/null || true
python3 /tests/check.py > /logs/verifier/check.txt 2>&1 && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
"""

CHECK_PY = """import json, os, sys
need = {need}
miss = [p for p in need if not os.path.exists(os.path.join("/app/out", p))]
print("missing:", miss)
sys.exit(1 if miss else 0)
"""


def build_agent_task(cfg: dict, kind: str, item_id: str, work: Path, out: Path, instruction: str, need: list[str],
                     timeout_s: float, image: str | None = None) -> Path:
    """A Harbor task whose environment has /work (inputs: paper, repo, contract) and whose agent writes /app/out."""
    if out.exists():
        shutil.rmtree(out)
    (out / "environment").mkdir(parents=True)
    shutil.copytree(work, out / "environment" / "work", symlinks=False)
    (out / "environment" / "Dockerfile").write_text(AGENT_DOCKERFILE.format(
        image=image or _c(cfg).get("agent_base_image", "python:3.11-slim")))
    (out / "tests").mkdir()
    (out / "tests" / "test.sh").write_text(VERIFY_SH)
    (out / "tests" / "test.sh").chmod(0o755)
    (out / "tests" / "check.py").write_text(CHECK_PY.format(need=json.dumps(need)))
    (out / "solution").mkdir()
    (out / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    hint = str(_c(cfg).get("agent_hints") or "").strip()
    (out / "instruction.md").write_text(instruction + (f"\n\n## Notes for this build host\n{hint}\n" if hint else ""))
    hosts = [str(h) for h in _c(cfg).get("install_hosts") or []]
    (out / "task.toml").write_text(AGENT_TASK_TOML.format(kind=kind, item=item_id, timeout=float(timeout_s),
                                                          hosts=json.dumps(hosts),
                                                          cpus=int(_c(cfg).get("agent_cpus", 4)),
                                                          mem=int(_c(cfg).get("agent_memory_mb", 16384))))
    return out


def agent_job_config(cfg: dict, task: Path, job_name: str, jobs_dir: Path, max_usd: float | None) -> dict:
    c = cfg["calib"]
    from .calib import AGENT_ENV, _ref, role
    kw = {"version": c["cc_version"], "reasoning_effort": _c(cfg).get("agent_effort", "high")}
    if max_usd:
        kw["max_budget_usd"] = f"{float(max_usd):.2f}"
    agent = {**_ref(_c(cfg).get("agent") or role(c, "teacher", "agent"), "name", c["harness"]),
             "model_name": _c(cfg).get("agent_model") or c["teacher_model"], "kwargs": kw, "env": dict(AGENT_ENV)}
    hosts = [str(h) for h in (_c(cfg).get("agent_allowed_hosts") or role(c, "teacher", "agent_allowed_hosts") or [])]
    if hosts:
        agent["extra_allowed_hosts"] = hosts
    env = _c(cfg).get("agent_environment") or role(c, "teacher", "environment")
    return {"job_name": job_name, "jobs_dir": str(jobs_dir), "n_attempts": 1, "n_concurrent_trials": 1,
            "timeout_multiplier": 1.0, "agent_timeout_multiplier": 1.0,
            "retry": {"max_retries": 1, "include_exceptions": ["AgentSetupTimeoutError", "EnvironmentStartTimeoutError"]},
            "environment": {**_ref(env, "type", "docker"), "delete": True},
            "verifier": {"disable": False}, "agents": [agent], "tasks": [{"path": str(task.resolve())}]}


def launch_agent(cfg: dict, it, kind: str, conf: dict, ledger, est_usd: float) -> dict:
    """intent -> reserved -> launched, persisted in <item>/<kind>/run.json (one job per stage attempt)."""
    d = it.p(kind)
    d.mkdir(parents=True, exist_ok=True)
    recp = d / "run.json"
    rec = read_json(recp, None) or {}
    if rec.get("job_name") != conf["job_name"]:
        rec = dict(job_name=conf["job_name"], state="intent", ts=time.time(), kind=kind,
                   job_dir=str(Path(conf["jobs_dir"]) / conf["job_name"]), config=str((d / "job.json").resolve()))
        (d / "job.json").write_text(json.dumps(conf, indent=1))
        write_json(recp, rec)
    if rec["state"] == "intent":
        model = conf["agents"][0]["model_name"]
        rec["ledger_call"] = ledger.reserve(model, kind, it.item_id, True, est_usd=float(est_usd)) \
            if ledger is not None else None
        rec["model"] = model
        rec["state"] = "reserved"
        write_json(recp, rec)
    if rec["state"] == "reserved":
        if Path(rec["job_dir"], "config.json").exists():
            rec["state"] = "launched"
            write_json(recp, rec)
            return rec
        root = _c(cfg).get("agent_harbor_root") or cfg["calib"].get("teacher_harbor_root")
        launcher = _c(cfg).get("agent_launcher") or cfg["calib"].get("teacher_launcher")
        if not (root and launcher):
            rec.update(state="launch_failed", error="code.agent_harbor_root / agent_launcher not set")
            if ledger is not None and rec.get("ledger_call"):
                ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=rec["error"])
            write_json(recp, rec)
            return rec
        log = d / "launch.log"
        inner = (f"cd {shlex.quote(root)} && env -u PYTHONPATH -u VIRTUAL_ENV ./{launcher} run "
                 f"{shlex.quote(rec['config'])} > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(log))}.exit")
        rec.update(launch_attempted=time.time(), log=str(log), tmux=conf["job_name"][-60:])
        write_json(recp, rec)
        r = subprocess.run([cfg["calib"].get("tmux_bin", "tmux"), "new-session", "-d", "-s", rec["tmux"], "bash", "-lc",
                            inner], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            rec.update(state="launch_failed", error=f"tmux rc={r.returncode}: {r.stderr[-300:]}")
            if ledger is not None and rec.get("ledger_call"):
                ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=rec["error"])
        else:
            rec["state"] = "launched"
        write_json(recp, rec)
    return rec


def poll_agent(cfg: dict, it, kind: str, ledger) -> dict | None:
    """None while running; else the finished record with trial dir, cost, outcome (settled once)."""
    recp = it.p(kind, "run.json")
    rec = read_json(recp, None)
    if not rec or rec.get("state") not in ("launched",):
        return rec
    jd = Path(rec["job_dir"])
    res = read_json(jd / "result.json", {}) or {}
    exitf = Path(str(rec.get("log", "")) + ".exit")
    if not res.get("finished_at"):
        if exitf.exists() and not (jd / "config.json").exists():
            rec.update(state="finished", outcome="launcher_failed",
                       error=Path(rec["log"]).read_text(errors="replace")[-400:] if Path(rec["log"]).exists() else "")
            if ledger is not None and rec.get("ledger_call"):
                ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=rec["error"][:300])
            write_json(recp, rec)
            return rec
        if exitf.exists() and time.time() - exitf.stat().st_mtime > 300:
            rec.update(state="finished", outcome="no_result")
            if ledger is not None and rec.get("ledger_call"):
                ledger.settle(rec["ledger_call"], None, status="error", note="job ended without result.json")
            write_json(recp, rec)
            return rec
        return None
    from .calib import trial_cost
    trials = [t for t in jd.iterdir() if t.is_dir() and (t / "result.json").exists()]
    trial = trials[0] if trials else None
    cost = trial_cost(trial, rec.get("model"), cfg["models"].get("prices") or {}) if trial else \
        dict(cost_usd=None, cost_kind="unknown", usage=None)
    tr = read_json(trial / "result.json", {}) if trial else {}
    exc = ((tr or {}).get("exception_info") or {}).get("exception_type")
    rec.update(state="finished", trial=str(trial) if trial else None, cost_usd=cost["cost_usd"],
               cost_kind=cost["cost_kind"], usage=cost.get("usage"), exception=exc,
               reward=(((tr or {}).get("verifier_result") or {}).get("rewards") or {}).get("reward"),
               minutes=_minutes(tr))
    if ledger is not None and rec.get("ledger_call"):
        ledger.settle(rec["ledger_call"], cost["cost_usd"], status="ok" if not exc else "error",
                      kind=cost["cost_kind"], usage=cost.get("usage"), note=f"{kind} agent {rec['job_name']}")
    write_json(recp, rec)
    return rec


def _minutes(tr: dict) -> float | None:
    from datetime import datetime
    a = (tr or {}).get("agent_execution") or {}
    try:
        return round((datetime.fromisoformat(a["finished_at"].replace("Z", "+00:00")) -
                      datetime.fromisoformat(a["started_at"].replace("Z", "+00:00"))).total_seconds() / 60, 1)
    except (KeyError, ValueError, AttributeError):
        return None


def deliverable(rec: dict) -> Path | None:
    """The agent's /app/out as copied by the verifier (logs/verifier/out)."""
    if not rec or not rec.get("trial"):
        return None
    p = Path(rec["trial"]) / "verifier" / "out"
    return p if p.exists() else None


# ----------------------------------------------------------------------------------------------- inputs
def stage_inputs(cfg: dict, it, work: Path, with_contract: bool) -> dict:
    """/work for an agent job: paper.txt, candidate.json, code.json, the authors' repository (synced from the
    build host directory by the launcher: we write a pointer here and the job copies it at build time) and the
    harness contract."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copy(it.p("fulltext.txt"), work / "paper.txt")
    write_json(work / "candidate.json", {k: v for k, v in (read_json(it.p("S1_candidate.json"), {}) or {}).items()
                                         if k in ("title", "doi", "arxiv", "year", "venue", "abstract")})
    s3c = read_json(it.p("S3c_code.json"), {}) or {}
    write_json(work / "code.json", {k: s3c.get(k) for k in ("source", "pin", "sha", "size_mb", "languages",
                                                             "dep_files", "license_file", "lfs_skipped")})
    # the authors' repository is NOT copied here (it never leaves the build host): the agent launcher copies
    # <remote_dir>/repo into environment/work/repo of the synced task on the build host before the job starts
    (work / ".tbsgen-remote-repo").write_text(str(s3c.get("remote_dir", "")) + "/repo\n")
    # a re-authoring job (S6B sent the task back) sees why the previous attempt failed
    fb = {n: read_json(it.p(n), None) for n in ("S6A_measure.json", "S6B_decisions.json", "S6_failure.json")}
    if any(v is not None for v in fb.values()) and with_contract:
        write_json(work / "previous_attempt.json", {k: v for k, v in fb.items() if v is not None})
    if (it.p("S4c") / "HOWTO.md").exists():
        shutil.copy(it.p("S4c") / "HOWTO.md", work / "HOWTO.md")
        if (it.p("S4c") / "runnability.json").exists():
            shutil.copy(it.p("S4c") / "runnability.json", work / "runnability.json")
    if with_contract:
        from .prompts import METRICS_API, S5_SPEC, S6A_GEN
        (work / "contract").mkdir()
        (work / "contract" / "S5_SPEC.txt").write_text(S5_SPEC)
        (work / "contract" / "S6A_GEN.txt").write_text(S6A_GEN)
        (work / "contract" / "METRICS_API.txt").write_text(METRICS_API)
        from .runner import METRICS_HELPER
        shutil.copy(METRICS_HELPER, work / "contract" / "_metrics_template.py")
    return dict(work=str(work))


def sha_tree(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


def env_ok(cfg: dict, agent: bool = False) -> str | None:
    """Why S3c (agent=False) / the agent stages (agent=True) cannot run with this config (None = fine)."""
    c = _c(cfg)
    if not c.get("host") or not c.get("remote_root"):
        return "code.host / code.remote_root not set"
    if agent and not (c.get("agent_harbor_root") or cfg["calib"].get("teacher_harbor_root")):
        return "code.agent_harbor_root not set"
    return None


