"""Code-grounded stages that run on the build host: S3c (fetch the authors' code), S4c (runnability gate) and
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
import os
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


def run_id(cfg: dict) -> str:
    """A persistent unique id of this run directory (run_id.txt in the run root, created once)."""
    root = Path(str((cfg.get("run") or {}).get("root") or "."))
    f = root / "code_run_id.txt"
    if not f.exists():
        import uuid
        root.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(f"{root.name[:40]}-{uuid.uuid4().hex[:8]}\n")
        try:
            os.link(tmp, f)                 # atomic create-if-absent
        except FileExistsError:
            pass
        tmp.unlink()
    return f.read_text().strip()


def remote_item_dir(cfg: dict, item_id: str) -> str:
    root = _c(cfg).get("remote_root")
    if not root or not str(root).startswith("/"):
        raise ValueError("code.remote_root must be an absolute build-host directory for the authors' code")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", item_id):
        raise ValueError(f"unsafe item id {item_id!r}")
    tag = _c(cfg).get("run_tag") or run_id(cfg)
    return f"{root.rstrip('/')}/{tag}/{item_id}"


# ----------------------------------------------------------------------------------------------- S3c
FETCH_SH = r"""
set -uo pipefail
ROOT={root}; D={dir}; URL={url}; KIND={kind}; MAXMB={maxmb}; REF={ref}; VIA={kind}
emit() {{ echo "TBSGEN_RESULT $1"; }}
case "$D" in "$ROOT"/*) ;; *) emit '{{"ok": false, "why": "bad_dir"}}'; exit 0 ;; esac
mkdir -p "$D" && cd "$D" || {{ emit '{{"ok": false, "why": "fetch_error", "err": "cannot create item dir"}}'; exit 0; }}
[ "$(pwd -P)" = "$(cd "$D" && pwd -P)" ] || exit 1
W=$(mktemp -d "$D/fetch.XXXXXX") || {{ emit '{{"ok": false, "why": "fetch_error", "err": "mktemp"}}'; exit 0; }}
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/dl"
if [ "$KIND" = git ]; then
  export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
  # a 30 s reachability probe first: a host that hangs (e.g. github.com from the build host) must not eat the full
  # clone timeout before the codeload fallback below
  if ! timeout 30 git ls-remote -q "$URL" HEAD > /dev/null 2> "$W/dl/clone.err" || \
     ! timeout {tmo} git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=120 clone --quiet --depth 1 \
       --no-recurse-submodules ${{REF:+--branch "$REF"}} "$URL" "$W/repo.tmp" 2> "$W/dl/clone.err"; then
    rm -rf "$W/repo.tmp"
    # GitHub fallback (github.com unreachable from the build host while api / codeload work): resolve the default
    # branch head SHA through the API, download that exact commit's tarball from codeload (still pinned to the SHA)
    GH=$(echo "$URL" | sed -nE 's#^https://github\.com/([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)\.git$#\1#p')
    SHA=""
    if [ -n "$GH" ]; then
      SHA=$(timeout 60 curl -sSf -H "Accept: application/vnd.github.sha" "https://api.github.com/repos/$GH/commits/${{REF:-HEAD}}" 2>/dev/null | grep -E '^[0-9a-f]{{40}}$' || true)
    fi
    if [ -z "$SHA" ] || ! timeout {tmo} curl -sSfL --max-filesize $((MAXMB*1024*1024)) -o "$W/dl/archive.tgz" \
         "https://codeload.github.com/$GH/tar.gz/$SHA" 2>> "$W/dl/clone.err"; then
      emit "{{\"ok\": false, \"why\": \"clone_failed\", \"err\": $(tail -c 400 "$W/dl/clone.err" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}}"; exit 0
    fi
    mkdir -p "$W/repo.tmp" && tar xzf "$W/dl/archive.tgz" -C "$W/repo.tmp" --strip-components=1 --no-same-owner 2>/dev/null || {{
      emit '{{"ok": false, "why": "unpack_failed"}}'; exit 0; }}
    find "$W/repo.tmp" -type l -delete
    VIA=codeload
  else
    SHA=$(git -C "$W/repo.tmp" rev-parse HEAD)
  fi
  PIN="$SHA"
else
  if ! timeout {tmo} curl -sSfL --max-filesize $((MAXMB*1024*1024)) -o "$W/dl/archive" "$URL" 2> "$W/dl/curl.err"; then
    emit "{{\"ok\": false, \"why\": \"download_failed\", \"err\": $(tail -c 400 "$W/dl/curl.err" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}}"; exit 0
  fi
  PIN=sha256:$(sha256sum "$W/dl/archive" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$W/dl/archive" | cut -d' ' -f1); SHA=""
  mkdir -p "$W/repo.tmp"
  (cd "$W/repo.tmp" && (unzip -q ../dl/archive 2>/dev/null || tar xf ../dl/archive 2>/dev/null)) || {{
    emit '{{"ok": false, "why": "unpack_failed"}}'; exit 0; }}
  n=$(ls -A "$W/repo.tmp" | wc -l | tr -d ' ')
  if [ "$n" = 1 ] && [ -d "$W/repo.tmp/$(ls -A "$W/repo.tmp")" ]; then
    TOP="$W/repo.tmp/$(ls -A "$W/repo.tmp")"
    [ -L "$TOP" ] && {{ emit '{{"ok": false, "why": "unpack_failed", "err": "top-level link"}}'; exit 0; }}
    mv "$TOP" "$W/repo.tmp2" && rmdir "$W/repo.tmp" && mv "$W/repo.tmp2" "$W/repo.tmp"; fi
  # archives may carry links: drop every link (a link can point anywhere on the build host)
  find "$W/repo.tmp" -type l -delete
fi
cd "$W/repo.tmp/.." || exit 1
find "$W/repo.tmp" -type l -delete                 # links can point anywhere on the build host (both branches)
MB=$(python3 -c 'import os,sys;print(int(sum(os.path.getsize(os.path.join(d,f)) for d,ds,fs in os.walk("repo.tmp") if not (ds.remove(".git") if ".git" in ds else None) for f in fs if not os.path.islink(os.path.join(d,f)))/1048576+0.999))')
if [ "$MB" -gt "$MAXMB" ]; then rm -rf "$W/repo.tmp"; emit "{{\"ok\": false, \"why\": \"too_large\", \"mb\": $MB}}"; exit 0; fi
# immutable per pin: <item>/<pin>/repo is never replaced (another run / fetch of the same pin reuses it)
PD="$D/pins/$(echo "$PIN" | tr ':' '-' | cut -c1-72)"
mkdir -p "$D/pins"
if [ ! -d "$PD/repo" ]; then
  # atomic publish: a complete <pin>/ appears with one rename; a concurrent fetch of the same pin keeps the first
  mkdir -p "$W/pub" && mv "$W/repo.tmp" "$W/pub/repo" && echo "$PIN" > "$W/pub/PIN" && \
    mv -T "$W/pub" "$PD" 2>/dev/null || mv "$W/pub" "$PD" 2>/dev/null || true
fi
[ -d "$PD/repo" ] && [ "$(cat "$PD/PIN" 2>/dev/null)" = "$PIN" ] || {{ emit '{{"ok": false, "why": "fetch_error", "err": "pin publish failed"}}'; exit 0; }}
cd "$PD" || exit 1
LFS=$(grep -l "filter=lfs" repo/.gitattributes 2>/dev/null | wc -l | tr -d ' ')
python3 - "$PIN" "$SHA" "$MB" "$LFS" "$PD" "$VIA" <<'EOF'
import json, os, sys, collections
pin, sha, mb, lfs, pd, via = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
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
        if os.path.islink(p):
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if e in EXT:
            langs[EXT[e]] += sz
        if f in DEP or f.startswith("requirements"):
            deps.append(os.path.relpath(p, "repo"))
        if dp == "repo" and f.upper().startswith(("LICENSE", "LICENCE", "COPYING")) and lic is None \
                and os.path.isfile(p) and not os.path.islink(p) and sz:      # never follow a link out of the tree
            lines = open(p, errors="ignore").read(300).strip().splitlines()
            lic = lines[0][:120] if lines else None
print("TBSGEN_RESULT " + json.dumps(dict(ok=True, pin=pin, sha=sha or None, size_mb=mb, lfs_skipped=bool(lfs),
      remote_dir=pd, via=via or "git",
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
    sh = FETCH_SH.format(root=shlex.quote(str(_c(cfg)["remote_root"]).rstrip("/")), dir=shlex.quote(d),
                         url=shlex.quote(str(spec["url"])), kind=spec["kind"], maxmb=mx,
                         ref=shlex.quote(spec.get("ref") or ""), tmo=tmo)
    # probe (30 s) + clone (tmo) + API (60 s) + codeload download (tmo) + unpack / size / publish
    r = run(sh, 2 * tmo + 300)
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
    res.setdefault("remote_dir", d)
    return dict(res, source=spec, host=_c(cfg).get("host"))


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
{repo_check}
RUN mkdir -p /app/out /logs/verifier && chmod -R a+rwX /work /app
"""

RUN_WRAPPER = """#!/bin/bash
# usage: /work/tbsgen-run.sh <command...>   runs the command, records a receipt in /app/out/runs/ (cmd, rc, seconds)
# and the log in /app/out/runs/<n>.log.  S4c accepts "runnable" only with a receipt of rc 0.
mkdir -p /app/out/runs
n=$(ls /app/out/runs/*.json 2>/dev/null | wc -l)
t0=$(date +%s)
"$@" > /app/out/runs/$n.log 2>&1
rc=$?
python3 - "$n" "$rc" "$(( $(date +%s) - t0 ))" "$*" <<'EOF'
import json, sys
n, rc, sec, cmd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump(dict(cmd=cmd, rc=rc, seconds=sec, log=f"runs/{n}.log"), open(f"/app/out/runs/{n}.json", "w"))
EOF
tail -n 40 /app/out/runs/$n.log
exit $rc
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
    marker = read_json(work / ".tbsgen-remote-repo", None)
    # the image does not build unless the launcher injected the pinned authors' repo (no model spend otherwise)
    check = (f"RUN test -d /work/repo && test -n \"$(ls -A /work/repo)\" && "
             f"test \"$(cat /work/repo.PIN)\" = {json.dumps(str(marker.get('pin')))} || "
             f"(echo 'authors repo not injected or pin mismatch' >&2; exit 1)") if marker else ""
    (out / "environment" / "Dockerfile").write_text(AGENT_DOCKERFILE.format(
        image=image or _c(cfg).get("agent_base_image", "python:3.11-slim"), repo_check=check))
    # S4c run receipts: the agent runs the authors' example through this wrapper (command, exit code, log)
    (out / "environment" / "work" / "tbsgen-run.sh").write_text(RUN_WRAPPER)
    (out / "environment" / "work" / "tbsgen-run.sh").chmod(0o755)
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


def launch_agent(cfg: dict, it, kind: str, conf: dict, ledger, est_usd: float, extra: dict | None = None) -> dict:
    """intent -> reserved -> launched, persisted in <item>/<kind>/run.json (one job per stage attempt).  extra:
    fields stored with the intent (e.g. budget, harden_round) so a resumed job keeps them."""
    d = it.p(kind)
    d.mkdir(parents=True, exist_ok=True)
    recp = d / "run.json"
    rec = read_json(recp, None) or {}
    if rec.get("job_name") != conf["job_name"]:
        rec = dict(job_name=conf["job_name"], state="intent", ts=time.time(), kind=kind,
                   job_dir=str(Path(conf["jobs_dir"]) / conf["job_name"]), config=str((d / "job.json").resolve()),
                   **(extra or {}))
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
        # the previous job of this stage used the same log: its exit file would make poll_agent see this job as
        # a launcher that already failed (before Harbor writes config.json)
        for stale in (log, Path(str(log) + ".exit")):
            if stale.exists():
                stale.replace(d / f"{stale.name}.{rec.get('job_name', 'prev')}.prev")
        inner = (f"cd {shlex.quote(root)} && env -u PYTHONPATH -u VIRTUAL_ENV ./{launcher} run "
                 f"{shlex.quote(rec['config'])} > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(log))}.exit")
        rec.update(launch_attempted=time.time(), log=str(log), tmux=conf["job_name"][-60:])
        write_json(recp, rec)
        r = subprocess.run([cfg["calib"].get("tmux_bin", "tmux"), "new-session", "-d", "-s", rec["tmux"], "bash", "-lc",
                            inner], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            # tmux could not create the session: the job never started (nothing spent)
            rec.update(state="launch_failed", not_started=True, error=f"tmux rc={r.returncode}: {r.stderr[-300:]}")
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
            rec.update(state="finished", outcome="launcher_failed", not_started=True,
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
    rec.update(state="finished", finished_ts=time.time(), trial=str(trial) if trial else None, cost_usd=cost["cost_usd"],
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
    """The agent's /app/out as copied by the verifier (logs/verifier/out); None if any component below the trial
    directory is a link."""
    if not rec or not rec.get("trial"):
        return None
    t = Path(rec["trial"])
    p = t / "verifier" / "out"
    return p if p.is_dir() and _inside(t, p) else None


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
    # the authors' repository is NOT copied here (it never leaves the build host).  Launcher contract: when the
    # synced task has environment/work/.tbsgen-remote-repo (JSON {repo_dir, pin}), the launcher copies repo_dir
    # into environment/work/repo ON the build host after checking <repo_dir>/../PIN == pin, and refuses otherwise
    # (reference implementation: scripts/build-host-launch.sh).
    write_json(work / ".tbsgen-remote-repo", dict(repo_dir=str(s3c.get("remote_dir", "")) + "/repo",
                                                  pin=s3c.get("pin")))
    # a re-authoring job (S6B sent the task back) sees why the previous attempt failed
    fb = {n: read_json(it.p(n), None) for n in ("S6A_measure.json", "S6B_decisions.json", "S6_failure.json")}
    if it.state.get("harden_active") and not it.state.get("harden_built"):
        fb = {n: v for n, v in fb.items() if n == "S6_failure.json"}   # the old version's are in previous_version/
    if any(v is not None for v in fb.values()) and with_contract:
        write_json(work / "previous_attempt.json", {k: v for k, v in fb.items() if v is not None})
    if (it.p("S4c") / "HOWTO.md").exists():
        shutil.copy(it.p("S4c") / "HOWTO.md", work / "HOWTO.md")
        if (it.p("S4c") / "runnability.json").exists():
            shutil.copy(it.p("S4c") / "runnability.json", work / "runnability.json")
    if with_contract:
        _stage_difficulty(cfg, it, work)
    if with_contract:
        from .prompts import METRICS_API, S5_SPEC, S6A_GEN
        (work / "contract").mkdir()
        (work / "contract" / "S5_SPEC.txt").write_text(S5_SPEC)
        (work / "contract" / "S6A_GEN.txt").write_text(S6A_GEN)
        (work / "contract" / "METRICS_API.txt").write_text(METRICS_API)
        from .runner import METRICS_HELPER
        shutil.copy(METRICS_HELPER, work / "contract" / "_metrics_template.py")
    return dict(work=str(work))


def _stage_difficulty(cfg: dict, it, work: Path) -> None:
    """S5G inputs of the difficulty targeting (difficulty.py): the weakness profile (/work/weakness_profile.md),
    and in a hardening round the previous version (/work/previous_version/: its task files, spec, difficulty
    design, thresholds, measurements, authoring report) plus the student's passing-run summary
    (/work/student_feedback.json).  Nothing from the verifier beyond per-metric aggregates is staged."""
    from . import difficulty
    prof = difficulty.load_profile(cfg)
    if prof:
        (work / "weakness_profile.md").write_text(prof["text"])
    if not it.state.get("harden_active"):
        return
    k = int(it.state.get("harden_round", 0))
    rd = it.p("harden", f"round-{k}")
    prev = rd / "prev"
    if not (prev / "DONE").exists():
        return
    dst = work / "previous_version"
    dst.mkdir()
    if (prev / "task").is_dir():
        safe_copytree(prev / "task", dst / "task", root=prev)
    for rel in ("_spec/spec.json", "_spec/difficulty_design.json", "thresholds.json", "S6A_measure.json",
                "S5G/report.md"):
        if (prev / rel).is_file():
            (dst / Path(rel).name).write_bytes((prev / rel).read_bytes())
    for f in ("feedback.json",):
        if (rd / f).is_file():
            shutil.copy(rd / f, work / "student_feedback.json")
    earlier = []
    for j in range(1, k):
        fb = read_json(it.p("harden", f"round-{j}", "feedback.json"), None)
        if fb:
            earlier.append(dict(round=j, **{x: fb.get(x) for x in ("minutes", "metric_margins")}))
    if earlier:
        write_json(work / "earlier_rounds.json", earlier)


def sha_tree(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


def env_ok(cfg: dict, agent: bool = False) -> str | None:
    """Why S3c (agent=False) / the agent stages (agent=True) cannot run with this config (None = fine).  A
    code-grounded run must execute every generator / reference / baseline on the build host (the reference may
    port the authors' code): sandbox.backend must be "ssh"."""
    c = _c(cfg)
    if not c.get("host") or not c.get("remote_root"):
        return "code.host / code.remote_root not set"
    if (cfg.get("sandbox") or {}).get("backend") != "ssh":
        return "sandbox.backend must be \"ssh\" (build host) for code-grounded runs"
    if agent and not (c.get("agent_harbor_root") or cfg["calib"].get("teacher_harbor_root")):
        return "code.agent_harbor_root not set"
    return None




def _inside(root: Path, p: Path) -> bool:
    """No component of p below root (nor root itself) is a symlink, and p resolves inside root."""
    root = Path(root)
    if root.is_symlink():
        return False
    cur = root
    for part in Path(p).relative_to(root).parts:
        cur = cur / part
        if cur.is_symlink():
            return False
    try:
        return Path(p).resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def safe_copytree(src: Path, dst: Path, root: Path | None = None) -> list[str]:
    """Copy an agent deliverable tree, refusing symlinks (at any level, from the trusted `root` down) and special
    files.  Returns the refused relative paths; nothing outside the tree is ever read."""
    refused = []
    src = Path(src)
    if not _inside(Path(root or src), src) or not src.is_dir():
        return ["."]
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            refused.append(str(rel))
            continue
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst / rel, follow_symlinks=False)
    return refused


def safe_read(p: Path, limit: int = 2_000_000, root: Path | None = None) -> str | None:
    """Read a deliverable file only if no path component from the trusted root down is a link."""
    p = Path(p)
    if not _inside(Path(root or p.parent), p) or not p.is_file() or p.stat().st_size > limit:
        return None
    return p.read_text(errors="replace")
