"""S9 difficulty calibration: sequential GLM-5.3 (Claude Code 2.1.280) runs + Opus teacher branch.
Only scheduling + result collection.  Submitting is gated by cfg.calib.allow_submit (default False): the
pipeline writes a Harbor job config and a single-use manual submission command and the item waits in "S9/waiting".

Decision rule:
    GLM#1 fail  -> teacher#1; pass -> "hard" (keep); fail -> teacher#2 (teacher_runs_max) ; fail ->
                   "frontier" (calib.frontier on and every decisive teacher run a non-timeout fail: a task no model
                   solved; released only after S9R confirms it is correct, see review.py) else "parked:teacher_failed"
    GLM#1 pass  -> (quick_pass_minutes set and duration below it -> "easy") else GLM#2
    GLM#2 fail  -> "unstable" (keep: GLM 1/2);  GLM#2 pass -> "easy" (S9b: with calib.variant_depth > 0 the item
                   gets a harder variant round and is recalibrated instead of being finished, see stages_b.s9)
Infra outcomes (API quota/auth/stream errors with reward 0, sandbox start/setup timeouts, cancelled) are not
attempts; they are retried up to calib.infra_retries, then the item is parked "infra".
A trial with reward 1 counts as a pass even if Claude Code later raised an API error (this happens: the task
can be solved before the error).

Every collected run also gets a result type (RESULT_TYPES): pass / fail_delivered / fail_no_output /
timeout_with_output / timeout_no_output / infra.  The label rule above uses outcome only, except that
calib.timeout_no_output (default "fail" = unchanged) can make a student run that timed out without writing its
deliverable non-decisive ("retry": rerun the student up to calib.timeout_no_output_retries times) or park the item
("park").
"""
from __future__ import annotations

import json
import os
import re
import shlex
import time
from collections.abc import Callable
import uuid
from pathlib import Path

from . import taskimage
from .state import item_lock, read_json, write_json

INFRA_EXC = {"NetworkConnectionError", "EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "CancelledError",
             "UnknownApiError", "RateLimitError", "AgentInstallError"}
AGENT_ENV = {
    "API_FORCE_IDLE_TIMEOUT": "0", "API_TIMEOUT_MS": "3600000", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_EAGER_FLUSH": "1", "CLAUDE_ENABLE_STREAM_WATCHDOG": "0", "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "3600000",
    "IS_SANDBOX": "1", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000", "ENABLE_TOOL_SEARCH": "true"}


DEFAULT_ENTRY = "/app/output/solution_cli.py"        # _metrics.ENTRY default (TBSGEN_ENTRY)
RESULT_TYPES = ("pass", "fail_delivered", "fail_no_output", "timeout_with_output", "timeout_no_output", "infra")
TIMEOUT_EXC = {"AgentTimeoutError"}
TNO_POLICIES = ("fail", "retry", "park")


def role(c: dict, who: str, key: str):
    """Per-role calib setting: the teacher's `teacher_<key>` when set (not None), else the shared `<key>`.  Lets the
    teacher run with its own launcher root / environment / allowed hosts / prepared tasks (e.g. a different model
    route) while the student keeps the shared ones."""
    if who != "glm":
        v = c.get(f"teacher_{key}")
        if v is not None:
            return v
    return c.get(key)


def decisive_runs(runs: list[dict], cfg: dict) -> list[dict]:
    """The runs that count as attempts (infra runs and retried student timeouts without output removed).  Raises
    _Park(<label>) when the timeout_no_output policy parks the item."""
    c = cfg["calib"]
    pol = c.get("timeout_no_output") or "fail"
    if pol not in TNO_POLICIES:
        raise ValueError(f"calib.timeout_no_output must be one of {TNO_POLICIES}, got {pol!r}")
    real, skipped = [], 0
    for r in runs:
        if r["outcome"] == "infra":
            continue
        if r["who"] == "glm" and r["outcome"] == "fail" and r.get("result_type") == "timeout_no_output":
            if pol == "park":
                raise _Park("parked:timeout_no_output")
            if pol == "retry" and skipped < int(c.get("timeout_no_output_retries", 1)):
                skipped += 1                # not decisive: the student is run again
                continue
        real.append(r)
    return real


class _Park(Exception):
    pass


def is_timeout(r: dict) -> bool:
    """A run that ended on the agent time limit (a frontier label needs real failures, not slow ones)."""
    return str(r.get("result_type") or "").startswith("timeout") or \
        (r.get("trial") or {}).get("exception") in TIMEOUT_EXC


FRONTIER_CHECKS = ("a_consistency", "b_missing_conditions", "e_shortcuts_leaks", "j_genuine_difficulty")


def frontier_evidence(runs: list[dict], cfg: dict) -> dict:
    """Release record of a frontier task: decisive / infra / timeout counts of the teacher runs of this epoch."""
    tea = [r for r in runs if r.get("who") == "teacher" and r.get("outcome") is not None and
           not r.get("preflight_refused") and not r.get("obsolete_probe_job")]
    dec = [r for r in tea if r["outcome"] != "infra"]
    return dict(teacher_runs=len(tea), teacher_decisive=len(dec), teacher_infra=len(tea) - len(dec),
                teacher_timeouts=sum(1 for r in dec if is_timeout(r)),
                teacher_result_types=[r.get("result_type") for r in dec])


def next_action(runs: list[dict], cfg: dict) -> tuple[str, str | None]:
    """runs: [{"who": "glm"|"teacher", "outcome": "pass"|"fail"|"infra", "minutes": float,
    "result_type": optional}] in order.
    Returns ("run", "glm"|"teacher") or ("label", "hard"|"unstable"|"easy"|"frontier"|"parked:<why>").
    (The hardening loop -- difficulty.should_harden -- is checked by S9 before this rule.)"""
    c = cfg["calib"]
    try:
        real = decisive_runs(runs, cfg)
    except _Park as e:
        return "label", str(e)
    trailing_infra = 0
    for r in reversed(runs):
        if r.get("preflight_refused") or r.get("obsolete_probe_job"):
            continue                    # no attempt was made
        if r["outcome"] != "infra":
            break
        trailing_infra += 1
    if trailing_infra > c["infra_retries"]:
        return "label", "parked:infra"
    glm = [r for r in real if r["who"] == "glm"]
    tea = [r for r in real if r["who"] == "teacher"]
    if not glm:
        return "run", "glm"
    if glm[0]["outcome"] == "fail":
        if any(t["outcome"] == "pass" for t in tea):
            return "label", "hard"
        if len(tea) < c["teacher_runs_max"]:
            if not c.get("teacher_available", True):
                return "label", "parked:teacher_unavailable"
            return "run", "teacher"
        if c.get("frontier", True) and tea and not any(is_timeout(t) for t in tea):
            return "label", "frontier"
        return "label", "parked:teacher_failed"
    # GLM#1 passed
    q = c.get("quick_pass_minutes")
    if len(glm) == 1:
        if q and glm[0].get("minutes") is not None and glm[0]["minutes"] < q:
            return "label", "easy"
        return "run", "glm"
    return "label", "unstable" if glm[1]["outcome"] == "fail" else "easy"


def prepared_dir(cfg: dict, who: str = "glm") -> str | None:
    """calib.prepared_tasks_dir (teacher: calib.teacher_prepared_tasks_dir if set; "" = use pkg/ directly)."""
    return role(cfg["calib"], who, "prepared_tasks_dir") or None


def task_path(cfg: dict, item_dir: Path, who: str = "glm") -> Path:
    """Task folder the job runs: the sandbox-prepared copy (docker_image pointing at a registered template) when
    calib.prepared_tasks_dir has one for this item, else the item's own pkg/ (local Docker backends)."""
    d = prepared_dir(cfg, who)
    if d and (Path(d) / item_dir.name / "task.toml").exists():
        return Path(d) / item_dir.name
    return item_dir / "pkg"


def pkg_sha(pkg: Path, *, exclude=()) -> str:
    """Content hash of a packaged task (paths + bytes): a prepared copy made from another pkg/ is stale."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p for p in Path(pkg).rglob("*") if p.is_file()):
        if f.relative_to(pkg).as_posix() in exclude:
            continue
        h.update(str(f.relative_to(pkg)).encode() + b"\0")
        with f.open("rb") as source:
            hashlib.file_digest(source, lambda: h)
        h.update(b"\0")
    return h.hexdigest()[:16]


def _prepare_receipt(path: Path) -> dict:
    try:
        prev = read_json(path, {})
    except (OSError, ValueError):
        return {}
    return prev if isinstance(prev, dict) else {}


def _prepared_matches(receipt: dict, out: Path) -> bool:
    return (receipt.get("ok") and receipt.get("out") == str(out.resolve())
            and (out / "task.toml").is_file() and receipt.get("prepared_sha") == pkg_sha(out))


def ensure_prepared(cfg: dict, item_dir: Path, who: str = "glm", force: bool = False) -> dict:
    """Environment-preparation hook (calib.prepare_command): build the run-ready copy of pkg/ that a remote
    backend needs (image + registered template, task.toml pointing at it) at <prepared_tasks_dir>/<item_id>.

    The command is a template run with bash in calib.prepare_cwd (default: the role's harbor_root); placeholders
    {item_id} {pkg} {out} {who} are shell-quoted, and the same values are in the environment as TBSGEN_ITEM_ID /
    TBSGEN_PKG / TBSGEN_PREPARED_DIR / TBSGEN_WHO.  It must leave <out>/task.toml.  The outcome is recorded in
    calib/prepare.json with source and prepared hashes and the resolved output path. A successful copy is reused
    while all three match. A failed preparation for the same source and destination is not retried automatically (force=True, i.e. `tbsgen calib prepare`, retries)."""
    import subprocess
    c = cfg["calib"]
    cmd, d = c.get("prepare_command"), prepared_dir(cfg, who)
    if not d:
        return dict(ok=True, skipped="no prepared_tasks_dir")
    if not cmd:
        epoch = read_json(item_dir / "state.json", {}).get("calib_epoch", 0)
        why = stale_prepared(cfg, item_dir, who, epoch)
        return dict(ok=not why and (Path(d) / item_dir.name / "task.toml").is_file(),
                    skipped="no prepare_command", error=why)
    out = (Path(d) / item_dir.name).resolve()          # the hook runs in another cwd
    pkg = item_dir / "pkg"
    sha = pkg_sha(pkg)
    recp = item_dir / "calib" / f"prepare-{who}.json" if who != "glm" else item_dir / "calib" / "prepare.json"
    prev = _prepare_receipt(recp)
    image = taskimage.package_identity(item_dir)
    if prev.get("pkg_sha") == sha and prev.get("out") == str(out) and not force:
        if _prepared_matches(prev, out) and (not image or _push_matches(item_dir, prev)):
            return dict(prev, reused=True)
        if not prev.get("ok"):
            return dict(prev, reused=True)          # same pkg failed before: needs `tbsgen calib prepare`
    vals = dict(item_id=item_dir.name, pkg=str(pkg.resolve()), out=str(out), who=who)
    line = cmd.format(**{k: shlex.quote(v) for k, v in vals.items()})
    cwd = c.get("prepare_cwd") or role(c, who, "harbor_root") or str(item_dir)
    log = item_dir / "calib" / ("prepare.log" if who == "glm" else f"prepare-{who}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TBSGEN_ITEM_ID=vals["item_id"], TBSGEN_PKG=vals["pkg"], TBSGEN_PREPARED_DIR=vals["out"],
               TBSGEN_WHO=who)
    for key in ("TBSGEN_IMAGE_OCI", "TBSGEN_IMAGE_TAG", "TBSGEN_IMAGE_DIGEST"):
        env.pop(key, None)
    t0 = time.time()
    try:
        if image:
            env["TBSGEN_PKG_SHA"] = sha
            env.update(taskimage.publish(cfg, item_dir))
        with log.open("w") as fh:
            r = subprocess.run(["bash", "-c", line], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=float(c.get("prepare_timeout_s") or 3600))
        rc, err = r.returncode, None
    except Exception as e:  # noqa: BLE001  (timeout, missing cwd ...)
        rc, err = None, f"{type(e).__name__}: {e}"
    ok = rc == 0 and (out / "task.toml").exists()
    if not ok and err is None:
        tail = log.read_text(errors="replace")[-300:] if log.exists() else ""
        err = f"exit {rc}" + ("" if (out / "task.toml").exists() else f"; no {out}/task.toml") + f": {tail}"
    res = dict(ok=ok, pkg_sha=sha, out=str(out), rc=rc, error=err, sec=round(time.time() - t0, 1), ts=time.time(),
               log=str(log))
    if ok:
        res["prepared_sha"] = pkg_sha(out)
    if ok and image:
        res["published_digest"] = env.get("TBSGEN_IMAGE_DIGEST")
        if not _push_matches(item_dir, res):
            res.update(ok=False, error="prepare_command did not record the verified OCI digest for this package")
    write_json(recp, res)
    return res


def _ref(import_path: str | None, builtin_key: str, builtin: str) -> dict:
    """Harbor agent/environment reference: custom import path if configured, else the built-in by name/type."""
    return {"import_path": import_path} if import_path else {builtin_key: builtin}


def job_config(cfg: dict, pkg: Path, who: str, job_name: str) -> dict:
    c = cfg["calib"]
    if who == "glm":
        agent = {**_ref(c.get("student_agent"), "name", c["harness"]), "model_name": c["student_model"],
                 "kwargs": {"version": c["cc_version"], "reasoning_effort": "max"}, "env": dict(AGENT_ENV)}
    else:
        agent = {**_ref(c.get("teacher_agent"), "name", c["harness"]), "model_name": c["teacher_model"],
                 "kwargs": {"version": c["cc_version"], "reasoning_effort": "max", "thinking_display": "summarized"},
                 "env": dict(AGENT_ENV)}
    hosts = [str(h) for h in role(c, who, "agent_allowed_hosts") or []]
    if hosts:
        agent["extra_allowed_hosts"] = hosts            # the model endpoint during the agent phase only
    return {"job_name": job_name, "jobs_dir": c["jobs_dir"], "n_attempts": 1, "n_concurrent_trials": 1,
            "timeout_multiplier": 1.0, "agent_timeout_multiplier": 1.0,
            "retry": {"max_retries": 2, "include_exceptions": ["AgentSetupTimeoutError", "EnvironmentStartTimeoutError",
                                                              "NetworkConnectionError"]},
            "environment": {**_ref(role(c, who, "environment"), "type", "docker"), "delete": True},
            "verifier": {"disable": False},
            "agents": [agent], "tasks": [{"path": str(pkg.resolve())}]}


def agent_network(task: Path) -> tuple[str, list[str]]:
    """Network mode + hosts of the agent phase as Harbor resolves them from the task's own task.toml: [agent]
    overrides [environment]; a missing task.toml or table means "public"."""
    import tomllib
    try:
        t = tomllib.loads((task / "task.toml").read_text())
    except (OSError, ValueError):
        return "public", []
    env, ag = t.get("environment") or {}, t.get("agent") or {}
    src = ag if ag.get("network_mode") else env
    return str(src.get("network_mode") or "public"), [str(h) for h in src.get("allowed_hosts") or []]


def network_problem(cfg: dict, task: Path, conf: dict) -> str | None:
    """Why an agent that calls a model could not run this job: the task's agent phase is offline, or an allowlist
    that names no host while the job config adds none (the model endpoint would be unreachable)."""
    mode, hosts = agent_network(task)
    extra = [h for a in conf.get("agents") or [] for h in a.get("extra_allowed_hosts") or []]
    if mode == "no-network":
        return f"{task}/task.toml: the agent phase is offline (no-network); an agent that calls a model cannot run"
    if mode == "allowlist" and not hosts and not extra:
        return (f"{task}/task.toml: the agent phase is an allowlist without hosts: set calib.agent_allowed_hosts to "
                "the model endpoint host(s), or add --allow-agent-host to the command")
    return None


def job_name_for(item_dir: Path, who: str, n: int, epoch: int = 0) -> str:
    """Stable per (item, who, attempt, calibration epoch): a re-run after a crash finds the same job instead of
    starting another.  epoch > 0 = recalibration after an S9R review revision (the archived epoch's job dirs keep
    their names)."""
    return f"tbsgen-{item_dir.name}-{who}{n}" + (f"-e{epoch}" if epoch else "")


def manual_command(item_dir, job, attempt, config_path=None):
    argv = ["tbsgen", "calib", "launch", "--item", item_dir.name, "--job", job, "--attempt", str(attempt),
            "--run", str(item_dir.resolve().parent.parent)]
    if config_path:
        argv.extend(["--config", config_path])
    return shlex.join(argv)


def migrate_manual_commands(store, config_path, *, deadline=None):
    """Upgrade pending commands on startup, including budget-paused items that never reach a worker."""
    def legacy(rec):
        return (rec.get("state") in ("intent", "reserved") and rec.get("outcome") is None
                and rec.get("command") and not rec["command"].startswith("tbsgen calib launch "))

    for directory in ("calib", "review"):
        for path in (store.root / "items").glob(f"*/{directory}/runs.json"):
            if not any(legacy(rec) for rec in read_json(path, [])):
                continue
            it = store.item(path.parent.parent.name)
            with item_lock(it, deadline=deadline):
                # Another worker may have settled the intent or appended a run since the scan.
                runs = read_json(path, [])
                pending = [rec for rec in runs if legacy(rec)]
                for rec in pending:
                    if config_path and not rec.get("config_path"):
                        rec["config_path"] = config_path
                    rec["legacy_command"] = rec["command"]
                    rec["command"] = manual_command(it.dir, rec["job_name"], rec.get("attempt", rec["n"]),
                                                    rec.get("config_path"))
                if pending:
                    write_json(path, runs)


def new_permit(run_dir):
    permit = run_dir.resolve() / f".launch-permit-{uuid.uuid4().hex}"
    permit.touch(exist_ok=False)
    return str(permit)


def prepare(cfg: dict, item_dir: Path, who: str, n: int, epoch: int = 0) -> dict:
    """Intent record for attempt n (no side effects outside item_dir).  Persist it BEFORE reserve/launch."""
    c = cfg["calib"]
    job_name = job_name_for(item_dir, who, n, epoch)
    conf = job_config(cfg, task_path(cfg, item_dir, who), who, job_name)
    p = item_dir / "calib" / f"{job_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(conf, indent=1))
    p = p.resolve()
    launcher = c.get("student_launcher") if who == "glm" else c.get("teacher_launcher")
    model = c["student_model"] if who == "glm" else c["teacher_model"]
    root = role(c, who, "harbor_root")
    command = manual_command(item_dir, job_name, n, cfg.get("config_path"))
    rec = dict(who=who, n=n, item=item_dir.name, model=model, job_name=job_name, config=str(p),
               command=command, state="intent", submitted=False, remote_route=is_remote_launcher(c, launcher),
               ts=time.time(), job_dir=str(Path(c["jobs_dir"]) / job_name), task=conf["tasks"][0]["path"],
               outcome=None, epoch=epoch)
    if cfg.get("config_path"):
        rec["config_path"] = cfg["config_path"]
    if _unprepared(cfg, who, rec["task"]):
        # remote sandbox backends cannot build a Dockerfile: a missing prepared template means the run would die
        # at environment start (e2e-iter1: FileUploadException, no model use) -> do not launch
        rec["blocked"] = "no prepared sandbox template for this item"
    why = network_problem(cfg, Path(rec["task"]), conf)
    if why:
        # an automatic launch would only burn a sandbox; a printed command is for a human, who can add the host
        rec["blocked" if c.get("allow_submit") else "warning"] = why
    if c.get("allow_submit") and not (launcher and root):
        rec["blocked"] = (f"calib.allow_submit needs calib.harbor_root and calib.{'student' if who == 'glm' else 'teacher'}"
                          "_launcher (or set allow_submit=false and run the printed command yourself)")
    return rec


def _unprepared(cfg: dict, who: str, task: str) -> bool:
    d = prepared_dir(cfg, who)
    return bool(d) and not Path(task).resolve().is_relative_to(Path(d).resolve())


def _push_matches(item_dir, prepared):
    push = read_json(item_dir / "calib/image-push.json", {})
    return bool(prepared.get("published_digest") and push.get("digest") == prepared["published_digest"]
                and push.get("pkg_sha") == prepared.get("pkg_sha"))


def stale_prepared(cfg: dict, item_dir: Path, who: str, epoch: int) -> str | None:
    """After a review revision (epoch > 0) a prepared copy must be one made from the current pkg/ (recorded by
    ensure_prepared with the pkg hash); a copy prepared by hand before the revision would calibrate the old task."""
    directory = prepared_dir(cfg, who)
    if not directory:
        return None
    out = (Path(directory) / item_dir.name).resolve()
    image = taskimage.package_identity(item_dir)
    recp = item_dir / "calib" / ("prepare.json" if who == "glm" else f"prepare-{who}.json")
    prev = _prepare_receipt(recp)
    if (not epoch and not image and not cfg["calib"].get("prepare_command") and not recp.exists()
            and not read_json(item_dir / "state.json", {}).get("require_prepared_receipt")):
        return None
    if image and not _push_matches(item_dir, prev):
        return "prepared task has no matching verified OCI digest; run calib prepare"
    if prev.get("pkg_sha") == pkg_sha(item_dir / "pkg") and _prepared_matches(prev, out):
        return None
    return (f"the prepared {who} copy may predate the current package (no calib/{recp.name} for the current pkg/): "
            "set calib.prepare_command so S9 can rebuild it, or retry the configured hook with `tbsgen calib prepare --item ID`")


def bind_launch_task(cfg, item_dir, rec, conf):
    """Bind an unsent attempt to the verified prepared path, including after a destination change."""
    current = task_path(cfg, item_dir, rec["who"]).resolve()
    if _unprepared(cfg, rec["who"], str(current)):
        return "no prepared sandbox template for this item"
    why = stale_prepared(cfg, item_dir, rec["who"], int(rec.get("epoch") or 0))
    if why:
        return why
    tasks = conf.get("tasks") or []
    # A crash can leave the config pointing at the verified destination before runs.json is saved.
    if len(tasks) != 1 or tasks[0].get("path") not in (rec["task"], str(current)):
        return "job config names a different task; prepare the current task before launch"
    if tasks[0]["path"] != str(current):
        tasks[0]["path"] = str(current)
        write_json(Path(rec["config"]), conf)
    rec["task"] = str(current)
    return None


def remote_alias(job_dir: Path, *, if_missing=None) -> dict:
    """Marker a remote launcher leaves in the local job dir (tbsgen-remote.json: host, local_task, remote_task,
    remote_job_dir): the job ran on another host from a synced copy of the task, so Harbor's config.json names the
    remote path.  ingest() accepts that path only when the marker maps it to the run's own local task."""
    try:
        m = json.loads((Path(job_dir) / "tbsgen-remote.json").read_text())
    except FileNotFoundError:
        return if_missing or {}
    except (OSError, ValueError):
        return {}
    return m if isinstance(m, dict) else {}


def task_matches(job_dir: Path, job_task: str | None, want_task: str | None, *, marker=None) -> bool:
    if job_task == want_task:
        return True
    m = remote_alias(job_dir) if marker is None else marker
    return bool(m) and m.get("remote_task") == job_task and m.get("local_task") == want_task


def is_remote_launcher(settings: dict, launcher: str | None) -> bool:
    """Classify the effective launcher before submission. An explicit list replaces the default."""
    if not launcher:
        return False
    name = Path(launcher).name
    names = settings.get("remote_launchers")
    return name != "harbor" if names is None else any(name == Path(entry).name for entry in names)


def marker_paths(rec):
    if rec.get("job_dir"):
        yield Path(rec["job_dir"]) / "tbsgen-remote.json"
    for directory in run_marker_dirs(rec):
        yield directory / "tbsgen-remote.json"
        if rec.get("job_name"):
            yield directory / (Path(rec["job_name"]).name + ".remote.json")


def run_marker_dirs(rec):
    if rec.get("run_dir"):
        yield Path(rec["run_dir"])
    if rec.get("config"):
        yield Path(rec["config"]).parent


def archive_run_marker(rec):
    """Keep the resolved attempt's evidence when the stage directory is reused for another job."""
    for directory in run_marker_dirs(rec):
        marker = directory / "tbsgen-remote.json"
        if marker.is_file() and directory != Path(rec["job_dir"]):
            marker.replace(directory / (Path(rec["job_name"]).name + ".remote.json"))


def remember_remote_marker(rec: dict) -> bool:
    """Keep the launcher's local identity mapping across replacement of the synced directory."""
    if rec.get("remote_marker"):
        return False
    changed = False
    for path in marker_paths(rec):
        marker = _load(path)
        if isinstance(marker, dict) and marker != rec.get("remote_marker"):
            rec.update(remote_route=True, remote_marker=marker)
            changed = True
            if marker:
                break
    return changed


def has_remote_marker(rec: dict) -> bool:
    return any(path.is_file() for path in marker_paths(rec))


def remote_job(rec: dict, ledger=None) -> bool:
    """A remote launcher is configured on intents too. Only protect a possible submission."""
    if rec.get("state") == "remote_unknown" or rec.get("remote_marker") is not None or has_remote_marker(rec):
        return True
    if not rec.get("remote_route"):
        return False
    if (rec.get("launch_attempted") or rec.get("state") == "launched"
            or rec.get("job_dir") and Path(rec["job_dir"]).is_dir()
            or rec.get("log") and Path(rec["log"]).is_file()):
        return True
    stage = rec.get("ledger_stage", "S9")
    return ledger is not None and ledger.job_was_sent(
        rec["item"], stage, reservation_key(rec, rec["item"], stage), rec.get("ledger_call"))


def job_started(rec: dict) -> bool:
    """Local evidence of a possible launch. A route alone is also present on a fresh intent."""
    jd = Path(rec["job_dir"])
    return (rec.get("state") == "remote_unknown" or has_remote_marker(rec)
            or jd.is_dir()
            or bool(rec.get("log") and Path(rec["log"]).exists()))


def pending_launcher(rec: dict, ledger=None) -> bool:
    return rec.get("outcome") is None and (rec.get("state", "launched") == "launched" or remote_job(rec, ledger))


def check_job_identity(jd, rec, *, review=False, marker=None, partial=False):
    conf = read_json(jd / "config.json", {})
    agents = conf.get("agents") or [{}]
    tasks = conf.get("tasks") or [{}]
    ident = dict(job_name=conf.get("job_name"), model=agents[0].get("model_name"), task=tasks[0].get("path"))
    want = dict(job_name=rec["job_name"],
                model=rec.get("model", None if review else ident["model"]),
                task=rec.get("task", None if review else ident["task"]))
    if marker is None:
        marker = remote_alias(jd, if_missing=rec.get("remote_marker"))
    if task_matches(jd, ident["task"], want["task"], marker=marker):
        ident["task"] = want["task"]                  # a remote launcher's synced copy of the run's own task
    if any(got != want[key] for key, got in ident.items() if not partial or got is not None):
        raise ValueError(f"job identity mismatch: job {ident} vs pending run {want}")


def restore_remote_marker(jd, rec, remote_task, *, review, partial):
    path = jd / "tbsgen-remote.json"
    remember_remote_marker(rec)
    local = remote_alias(jd)
    marker = local or rec.get("remote_marker")
    if remote_task:
        if marker and marker.get("remote_task") != remote_task:
            raise ValueError("--remote-task conflicts with the recorded marker")
        marker = marker or dict(local_task=rec["task"], remote_task=remote_task)
    if marker is not None and (not path.exists() or marker != local):
        if (jd / "config.json").exists():
            check_job_identity(jd, rec, review=review, marker=marker, partial=partial)
        write_json(path, marker)
        rec["remote_marker"] = marker


def reservation_key(rec, item, stage):
    if stage in ("S4c", "S5G"):
        return rec.get("reservation_key") or json.dumps([item, stage, rec["job_name"]])
    attempt = rec.get("attempt", rec.get("n", 1))
    return rec.get("reservation_key") or json.dumps([item, stage, rec["job_name"], attempt])


def bind_started_call(cfg, rec, ledger, cost=None):
    stage = rec.get("ledger_stage", "S9")
    rec["ledger_call"] = ledger.adopt_started_job(
        rec["model"], stage, rec["item"], rec["job_name"], reservation_key(rec, rec["item"], stage),
        float(rec.get("est_usd", cfg["calib"]["est_usd_per_run"])), cost, rec.get("ledger_call"))


def adopt_started(cfg, rec, ledger, save):
    """Recover an existing Harbor job with its original identity and without a new submission."""
    if remote_job(rec, ledger):
        reconcile_remote(cfg, rec, ledger, save)
        return rec
    if ledger is not None:
        stage = rec.get("ledger_stage", "S9")
        jd = Path(rec["job_dir"])
        cost = None
        if read_json(jd / "result.json", {}).get("finished_at"):
            check_job_identity(jd, rec, review=stage == "S9R")
            cost = job_cost(collect(jd, rec["model"], cfg["models"].get("prices", {})))
        if not rec.get("ledger_call"):
            bind_started_call(cfg, rec, ledger, cost)
        elif cost is not None:
            usd, kind, usage = cost
            ledger.settle(rec["ledger_call"], usd, kind=kind, usage=usage)
    rec.update(state="launched", submitted=True, recovered=True)
    save(rec)
    return rec


def launch_deadline(c, attempted):
    return attempted + float(c.get("launch_grace_s", 60))


def permit_command(rec: dict, run_dir: Path, command: str) -> str:
    """Give this send its own single-use name, including when the same job is retried."""
    permit = rec.get("launch_permit") or new_permit(run_dir)
    rec["launch_permit"] = permit
    claim = shlex.join(["python3", "-c", "import os,sys; os.rename(sys.argv[1], sys.argv[2])",
                        str(permit), str(permit) + ".claimed"])
    return f"{claim} || exit 97\n{command}"


def reconcile_launch(c, rec, exists, now):
    """Race the launcher for its permit. A repeated recovery observes the same winner."""
    if remote_job(rec):
        return True
    permit = rec.get("launch_permit")
    if permit:
        try:
            os.rename(permit, permit + ".revoked")
            return False
        except FileNotFoundError:
            if Path(permit + ".claimed").is_file():
                if exists():
                    return True
                alive = session_alive(c, rec)
                return True if exists() or remote_job(rec) else alive
            if Path(permit + ".revoked").is_file():
                return False
    # Older launches had no fence. Keep their session/job checks and grace period.
    if exists():
        return True
    attempt = rec.get("launch_attempted")
    if attempt is None:
        return False
    after = rec.get("confirm_launch_after", launch_deadline(c, attempt))
    alive = session_alive(c, rec)
    if alive or exists():  # the client may have created the job during has-session
        return True
    return None if now < after else alive


def launch_evidence(c: dict, rec: dict) -> bool | None:
    """Revoke an outstanding permit or recognize its claim, falling back to legacy launch evidence."""
    if not rec.get("launch_attempted") and rec.get("state") in ("intent", "reserved"):
        return job_started(rec)
    return reconcile_launch(c, rec, lambda: job_started(rec), time.time())


def _fail_launch(rec, ledger, save, error):
    if ledger is not None:
        stage = rec.get("ledger_stage", "S9")
        rec["ledger_call"] = ledger.cancel_unstarted_job(
            rec["item"], stage, reservation_key(rec, rec["item"], stage), rec.get("ledger_call"), "not_launched", error)
    rec.pop("confirm_launch_after", None)
    rec.update(state="launch_failed", outcome="infra", result_type="infra", cost_usd=0.0,
               cost_kind="not_launched", error=error)
    save(rec)
    return rec


def bind_reservation(rec, ledger):
    """Recover the ledger key even if the process died before saving the reservation id."""
    stage = rec.get("ledger_stage", "S9")
    attempt = rec.get("attempt", rec.get("n", 1))
    key = rec.get("reservation_key") or json.dumps([rec["item"], stage, rec["job_name"], attempt])
    if not rec.get("ledger_call") and not rec.get("reservation_key") and "ts" in rec:
        ledger.adopt_legacy_agent_call(key, rec["item"], stage, rec["model"], rec["ts"])
    rec["reservation_key"] = key


def launch(cfg: dict, rec: dict, ledger, save, *, started=None) -> dict:
    """intent -> reserved -> launched.  `save(rec)` persists the run record; it is called after every step so a
    crash at any point is recoverable. Resume submits an unsent intent, or reconciles its launch permit."""
    c = cfg["calib"]

    launcher = rec.get("launcher_script") or c.get("student_launcher" if rec["who"] == "glm" else "teacher_launcher")
    rec.setdefault("remote_route", is_remote_launcher(c, launcher))

    def reserve_job(send=False):
        if ledger is None:
            return None
        bind_reservation(rec, ledger)
        stage = rec.get("ledger_stage", "S9")
        return ledger.reserve_job(rec, save, rec["model"], stage, rec["item"], float(c["est_usd_per_run"]),
                                  send=send)

    if remote_job(rec, ledger):
        return adopt_started(cfg, rec, ledger, save)
    if rec["state"] not in ("intent", "reserved", "cancelled_unstarted"):
        return rec
    if started is None:
        started = launch_evidence(c, rec)
    if started is None:
        return rec
    if started:
        return adopt_started(cfg, rec, ledger, save)
    if rec.get("launch_attempted"):
        return _fail_launch(rec, ledger, save, rec.get("launch_uncertain") or "launch did not start")
    # Preserve the persisted job identity, model and route. Only an unstarted calibration
    # job may switch to the run-ready copy of this same item.
    try:
        conf = json.loads(Path(rec["config"]).read_text())
    except (OSError, ValueError) as e:
        rec["blocked"] = f"job config {rec['config']} unreadable ({type(e).__name__}); run prepare again"
        save(rec)
        return rec
    if rec["who"] != "review":
        why = bind_launch_task(cfg, Path(rec["config"]).parent.parent, rec, conf)
        if why:
            rec["blocked"] = why
            save(rec)
            return rec
    if rec["state"] in ("intent", "cancelled_unstarted"):
        hosts = [str(h) for h in role(c, rec["who"], "agent_allowed_hosts") or []]
        for a in conf.get("agents") or []:
            if hosts:
                a["extra_allowed_hosts"] = hosts
            else:
                a.pop("extra_allowed_hosts", None)
        tmp = Path(rec["config"] + ".tmp")
        tmp.write_text(json.dumps(conf, indent=1))
        os.replace(tmp, rec["config"])                     # atomic: an interrupted write leaves the old config
        why = network_problem(cfg, Path(rec["task"]), conf)
        if why:
            rec["blocked"] = why
            save(rec)
            return rec
        rec.pop("blocked", None)
        rec["ledger_call"] = reserve_job()
        rec["state"] = "reserved"
        save(rec)
    # Reserved attempts may predate a config change. Check their own config before sending.
    why = network_problem(cfg, Path(rec["task"]), conf)
    if why:
        return _fail_launch(rec, ledger, save, f"not launched: {why}")
    return _start_launcher(cfg, rec, ledger, save, reserve_job)


def _start_launcher(cfg, rec, ledger, save, reserve_job):
    import subprocess

    c = cfg["calib"]
    item_dir = Path(rec["config"]).parent.parent
    log = item_dir / "calib" / (rec["job_name"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    launcher = rec.get("launcher_script") or c.get("student_launcher" if rec["who"] == "glm" else "teacher_launcher")
    root = rec.get("launcher_root") or role(c, rec["who"], "harbor_root")
    if not (launcher and root):
        err = "no launcher / harbor_root for this role"
        if remote_job(rec):
            reconcile_remote(cfg, rec, ledger, save)
            return rec
        if ledger is not None and rec.get("ledger_call"):
            ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=f"launch failed: {err}")
        rec.update(state="launch_failed", outcome="infra", result_type="infra", cost_usd=0.0, cost_kind="not_launched", error=err)
        save(rec)
        return rec
    # the launcher's exit status is written next to the log, so a launcher that dies at once is visible
    inner = (f"cd {shlex.quote(root)} && env -u PYTHONPATH -u VIRTUAL_ENV ./{launcher} run "
             f"{shlex.quote(rec['config'])} > {shlex.quote(str(log))} 2>&1; "
             f"echo $? > {shlex.quote(str(log) + '.exit')}")
    inner = permit_command(rec, Path(rec["config"]).parent, inner)
    rec.update(log=str(log), tmux=rec.get("tmux") or rec["job_name"][-60:], launch_attempted=time.time(),
               launcher_script=launcher, launcher_root=root, remote_route=is_remote_launcher(c, launcher))
    if Path(str(log) + ".exit").exists():
        Path(str(log) + ".exit").unlink()                  # a previous attempt's exit status
    save(rec)
    call = reserve_job(send=True)
    if call != rec.get("ledger_call"):  # an unsent reservation rolled into a new budget day
        rec["ledger_call"] = call
    save(rec)
    try:
        r = subprocess.run([c.get("tmux_bin", "tmux"), "new-session", "-d", "-s", rec["tmux"],
                            "bash", "-lc", inner], capture_output=True, text=True, timeout=60)
        err = None if r.returncode == 0 else f"tmux rc={r.returncode}: {r.stderr[-300:]}"
    except (subprocess.TimeoutExpired, OSError) as exc:
        err = str(exc)
    if err:
        if remote_job(rec):
            reconcile_remote(cfg, rec, ledger, save)
            return rec
        started = launch_evidence(c, rec)
        if started is False:
            return _fail_launch(rec, ledger, save, err)
        rec.update(launch_uncertain=err[:300])
        if started is None:
            save(rec)
            return rec
    remember_remote_marker(rec)
    rec.pop("confirm_launch_after", None)
    rec.update(state="launched", submitted=True, tmux=rec["job_name"][-60:])
    save(rec)
    return rec


# the build-host runner writes "relay preflight http=<code>" and exits 3 before starting Harbor when the model relay
# is down; the launcher copies that log tail and "<job>: harbor exit 3" into the job's own launcher log
_PREFLIGHT_FAIL = re.compile(r"relay preflight http=(?!200\b)\d+\s*$", re.M)


def relay_preflight_refused(rec: dict) -> bool:
    """Explicit evidence that this job never reached Harbor because the model relay refused the preflight: the
    job's OWN launcher log (not a shared one), its exit status 3, the non-200 preflight line and "harbor exit 3".
    Such a job ran no model, so it is neither an attempt nor an infrastructure failure of the task: the stage waits
    calib.preflight_retry_s and launches again, without using up a retry (an outage only delays the work)."""
    if remote_job(rec):
        return False
    log = Path(str(rec.get("log", "")))
    if not rec.get("job_name") or not log.name.startswith(str(rec["job_name"])) or not log.exists():
        return False
    if rec.get("job_dir") and (Path(rec["job_dir"]) / "config.json").exists():
        return False                    # Harbor created the job: it may have run, whatever the launcher log says
    try:
        code = Path(str(log) + ".exit").read_text().strip()
        text = log.read_text(errors="replace")
    except OSError:
        return False
    return code == "3" and bool(_PREFLIGHT_FAIL.search(text)) and "harbor exit 3" in text


def preflight_wait(cfg: dict, rec: dict | None) -> str | None:
    """While the last job was refused by the relay preflight less than calib.preflight_retry_s ago: the reason to
    wait (None = launch now)."""
    if not rec or not rec.get("preflight_refused"):
        return None
    left = float(rec.get("refused_ts") or 0) + float(cfg["calib"].get("preflight_retry_s", 600)) - time.time()
    if left <= 0:
        return None
    return f"relay_down: preflight refused, retry after {time.strftime('%H:%M', time.localtime(time.time() + left))}"


DELIVERY_METRICS = "verifier/metrics.jsonl"


def _trial_reward(result):
    return ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")


def _s9_requires_metrics(result):
    return not result.get("exception_info") or _trial_reward(result) is not None


def _s9_complete(trial, result, contract):
    return not _s9_requires_metrics(result) or (trial / DELIVERY_METRICS).is_file()


def _s9r_complete(trial, result, contract):
    from .review import CURRENT_CONTRACT, JOURNAL_CONTRACTS, REPRO_JOURNAL, VERIFIED_VERDICT

    return (trial / VERIFIED_VERDICT).is_file() and (
        (contract or CURRENT_CONTRACT) not in JOURNAL_CONTRACTS or (trial / REPRO_JOURNAL).is_file())


CompletenessRule = Callable[[Path, dict, str | None], bool]
_COMPLETENESS_RULES: dict[str, CompletenessRule] = {"S9": _s9_complete, "S9R": _s9r_complete}


def register_completeness(stage: str, fn: CompletenessRule) -> None:
    """Register fn(trial_dir, result_json, contract) for this stage before reconciling jobs."""
    _COMPLETENESS_RULES[stage] = fn


def remote_trial_set(jd: Path, stage: str, *, contract=None) -> tuple[list[Path], bool]:
    """Unregistered stages require only a readable trial result. All stages check the configured trial count."""
    rule = _COMPLETENESS_RULES.get(stage)
    return configured_trial_set(
        jd, lambda trial, result: rule is None or rule(trial, result, contract))


def configured_trial_set(jd, has_outputs):
    """Match readable trial results and their stage outputs to the configured number of trials."""
    trials = sorted(p for p in jd.glob("*") if p.is_dir())
    complete = [p for p in trials if isinstance(result := _load(p / "result.json"), dict)
                and has_outputs(p, result)]
    config = _load(jd / "config.json")
    if not isinstance(config, dict):
        return complete, False
    tasks, agents = config.get("tasks"), config.get("agents")
    attempts = config.get("n_attempts", 1)
    if (not isinstance(tasks, list) or not tasks or not isinstance(agents, list) or not agents
            or type(attempts) is not int or attempts < 1):
        return complete, False
    expected = len(tasks) * len(agents) * attempts
    return complete, len(complete) >= expected and len(complete) == len(trials)


def complete_remote_trials(jd: Path, stage: str, *, contract=None) -> bool:
    _, complete = remote_trial_set(jd, stage, contract=contract)
    return complete


def _remote_results_ready(cfg, rec, final=False):
    stage = rec.get("ledger_stage")
    if stage in ("S4c", "S5G"):
        from .codeagent import complete_remote_agent
        return complete_remote_agent(cfg, rec, final=final)
    from .review import job_contract
    return complete_remote_trials(Path(rec["job_dir"]), stage or "S9", contract=job_contract(rec))


def reconcile_remote(cfg, rec, ledger, save):
    """Observe the original submission locally. Only the operator can clear remote_unknown."""
    jd = Path(rec["job_dir"])
    state = rec.get("state")
    remembered = remember_remote_marker(rec)
    if state == "remote_unknown":
        action = "remote_unknown"
    else:
        rec.setdefault("tmux", rec["job_name"][-60:])
        exit_file = Path(rec["log"] + ".exit") if rec.get("log") else None
        alive = False
        if not (exit_file and exit_file.exists()):
            alive = session_alive(cfg["calib"], rec)
            if exit_file and exit_file.exists():  # final sync may have ended during the session probe
                alive = False
        summary = _load(jd / "result.json")
        ended = alive is False or isinstance(summary, dict) and summary.get("finished_at")
        # The launcher writes its exit status after the final sync; only then is a trial without outputs final.
        final_sync = bool(exit_file and exit_file.exists())
        if ended and _remote_results_ready(cfg, rec, final=final_sync):
            action = "awaiting_ingest"
        else:
            action = "waiting" if alive is True else "remote_unknown"
    remembered = remembered or remember_remote_marker(rec)
    new_state = "remote_unknown" if action == "remote_unknown" else "launched"
    if state != new_state or not rec.get("ledger_call"):
        if ledger is not None:
            bind_started_call(cfg, rec, ledger)
        rec.update(state=new_state, submitted=True)
        save(rec)
    elif remembered:
        save(rec)
    return action


def launcher_failed(rec: dict, c: dict | None = None) -> str | None:
    """Launcher exited without creating a Harbor job (bad cwd, missing script, key fetch failure ...), or its
    tmux session vanished without an exit status and without a job."""
    if remote_job(rec):
        return None  # reconcile_remote owns the remote launcher's exit protocol
    if rec.get("state") != "launched":
        return None
    ex = Path(rec.get("log", "") + ".exit")
    if ex.exists():
        if not remote_job(rec) and not (Path(rec["job_dir"]) / "result.json").exists():
            tail = Path(rec["log"]).read_text(errors="replace")[-300:] if Path(rec["log"]).exists() else ""
            return f"launcher exit {ex.read_text().strip()} without a job: {tail}"
    elif c is not None and not job_started(rec):
        if rec.get("launch_permit"):
            failed = revoke_lost_session(c, rec)
        elif rec.get("launch_attempted"):
            failed = launch_evidence(c, rec) is False
        else:
            failed = session_alive(c, rec) is False
        if failed and not ex.exists() and not job_started(rec):
            return "launcher session gone without an exit status and without a job"
    return None


def revoke_lost_session(c, rec):
    """After tmux acknowledged, allow the live session to claim. Fence it before refunding a lost session."""
    permit = rec["launch_permit"]
    if Path(permit + ".revoked").is_file():
        return True
    if session_alive(c, rec) is not False:
        return False
    if Path(permit + ".claimed").is_file():
        return not job_started(rec)
    return launch_evidence(c, rec) is False


def session_alive(c: dict, rec: dict) -> bool | None:
    """Probe the manual wrapper or tmux session. None means its status cannot be determined."""
    import subprocess
    if pid := rec.get("manual_launcher_pid"):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return None
        return True
    name = rec.get("tmux")
    if not name:
        return None
    try:
        r = subprocess.run([c.get("tmux_bin", "tmux"), "has-session", "-t", f"={name}"], capture_output=True,
                           text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode == 0:
        return True
    err = (r.stderr or "").lower()
    if "can't find session" in err or "no server running" in err:
        return False
    if "error connecting" in err and "no such file or directory" in err:
        return False                                    # no tmux server socket at all
    return None                                         # permission errors etc.: cannot tell


def launcher_died(rec: dict, c: dict | None = None) -> bool:
    """The job of a launched run will never finish: Harbor created it (result.json) but never wrote finished_at,
    and either the launcher exited (its .exit file, written after Harbor returns) or its tmux session is gone
    without an .exit (killed / reboot).  .exit and finished_at are re-read after the session check, so a job
    ending meanwhile is not taken for a dead one."""
    if rec.get("state") != "launched" or remote_job(rec):
        return False
    jr = Path(rec["job_dir"]) / "result.json"
    ex = Path(rec.get("log", "") + ".exit")
    if ex.exists():
        if not jr.exists():
            return False                                # launcher_failed() covers an exit without a job
        # Harbor writes finished_at before it returns and the .exit is written after: read finished_at only
        # after seeing .exit
        return not (_load(jr) or {}).get("finished_at")
    if c is None or not job_started(rec) or session_alive(c, rec) is not False:
        return False
    # session gone without an exit status (killed / reboot), job started (log/config) or not
    return not ex.exists() and not (_load(jr) or {}).get("finished_at") and launcher_failed(rec, c) is None


def actual_spend(ledger) -> float:
    """Settled spend the run-total gate compares with: reported costs and token-based estimates.  Runs whose cost
    is unknown (cost_kind "reservation": the reserved estimate stands in) are not counted as spent here; they
    still count at their reservation in Ledger.reserve()'s admission check."""
    with ledger._db() as db:
        return db.execute("SELECT COALESCE(SUM(usd),0) FROM calls WHERE real=1 AND status!='reserved' "
                          "AND COALESCE(cost_kind,'') != 'reservation'").fetchone()[0]


# ------------------------------------------------------------------ trial cost / tokens / result type
_TOK = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")


def trial_usage(trial: Path, r: dict | None = None) -> tuple[dict | None, str | None]:
    """Token usage of a trial's agent ({input_tokens (uncached), cache_read_input_tokens,
    cache_creation_input_tokens, output_tokens}) and where it came from, or (None, None).  Sources in order:
    Harbor's agent_result token counts (n_input_tokens includes cache reads), the ATIF trajectory's final_metrics,
    the Claude Code stream-json log (agent/claude-code.txt; per message id the largest usage)."""
    r = r if r is not None else _load(trial / "result.json")
    traj = (_load(trial / "agent" / "trajectory.json") or {}).get("final_metrics") or {}
    cw = int((traj.get("extra") or {}).get("total_cache_creation_input_tokens") or 0)
    ar = (r or {}).get("agent_result") or {}
    if ar.get("n_input_tokens") or ar.get("n_output_tokens"):
        n_in, n_c = int(ar.get("n_input_tokens") or 0), int(ar.get("n_cache_tokens") or 0)
        return dict(input_tokens=max(n_in - n_c - cw, 0), cache_read_input_tokens=n_c,
                    cache_creation_input_tokens=cw, output_tokens=int(ar.get("n_output_tokens") or 0)), "agent_result"
    if traj.get("total_prompt_tokens") or traj.get("total_completion_tokens"):
        n_in, n_c = int(traj.get("total_prompt_tokens") or 0), int(traj.get("total_cached_tokens") or 0)
        return dict(input_tokens=max(n_in - n_c - cw, 0), cache_read_input_tokens=n_c,
                    cache_creation_input_tokens=cw, output_tokens=int(traj.get("total_completion_tokens") or 0)), \
            "trajectory"
    f = trial / "agent" / "claude-code.txt"
    if f.exists():
        per: dict = {}
        with f.open(errors="replace") as fh:
            for line in fh:
                if '"usage"' not in line or '"assistant"' not in line:
                    continue
                try:
                    m = (json.loads(line).get("message") or {})
                except json.JSONDecodeError:
                    continue
                u = m.get("usage") or {}
                k = m.get("id") or id(m)
                old = per.get(k, {})
                per[k] = {t: max(int(old.get(t, 0)), int(u.get(t) or 0)) for t in _TOK}
        if per:
            return {t: sum(v[t] for v in per.values()) for t in _TOK}, "stream_json"
    return None, None


def trial_cost(trial: Path, model: str | None, prices: dict, r: dict | None = None) -> dict:
    """{cost_usd, cost_kind, cost_source, usage}.  cost_kind: "reported" (Harbor cost_usd), "tokens" (token usage
    x models.prices[model]: an estimate from real usage), "unknown" (no cost and no usable usage/price)."""
    r = r if r is not None else _load(trial / "result.json") or {}
    usage, src = trial_usage(trial, r)
    cost = ((r or {}).get("agent_result") or {}).get("cost_usd")
    if cost is not None:
        return dict(cost_usd=float(cost), cost_kind="reported", cost_source="harbor", usage=usage)
    from .budget import price_usd
    if usage and model and (prices.get(model) or prices.get(model.split(":")[0])):
        return dict(cost_usd=round(price_usd(prices, model, usage), 4), cost_kind="tokens", cost_source=src,
                    usage=usage)
    return dict(cost_usd=None, cost_kind="unknown", cost_source=src,
                usage=usage, note=None if usage else "no token usage in the trial",
                **({"price_missing": model} if usage else {}))


def delivered(trial: Path, *, required=False) -> bool | None:
    """Did the agent leave its deliverable?  From the verifier's metrics.jsonl (tbsgen's _metrics.run_solution
    records `_solution_cli_run`): the entry file missing ("can't open file" / "No such file") = False; any other
    run = True; no record = None (unknown)."""
    try:
        lines = (trial / DELIVERY_METRICS).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        if required:
            raise
        return None
    seen = None
    for line in lines:
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(o, dict) or o.get("name") != "_solution_cli_run" or o.get("lock_failed"):
            continue
        if "entry_exists" in o:                         # recorded by newer _metrics.run_solution
            seen = (seen or False) or bool(o["entry_exists"])
            continue
        err = str(o.get("stderr_tail") or "")
        # older records: only the interpreter failing to open the entry file itself (a helper the deliverable
        # runs that is missing does not count)
        entry = str(o.get("entry") or DEFAULT_ENTRY)
        missing = o.get("rc") not in (0, None) and f"can't open file '{entry}'" in err
        seen = (seen or False) or not missing
    return seen


def result_type(outcome: str, exc: str | None, deliv: bool | None) -> str:
    """pass / infra as the outcome; a failed run splits by timeout and deliverable (unknown counts as delivered:
    only a confirmed missing deliverable is "no_output")."""
    if outcome in ("pass", "infra"):
        return outcome
    if exc in TIMEOUT_EXC:
        return "timeout_no_output" if deliv is False else "timeout_with_output"
    return "fail_no_output" if deliv is False else "fail_delivered"


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _api_status(trial: Path) -> tuple[int | None, str | None]:
    f = trial / "agent" / "claude-code.txt"
    if not f.exists():
        return None, None
    for line in reversed(f.read_text(errors="replace").splitlines()):
        if '"type":"result"' in line:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            return o.get("api_error_status"), (o.get("result") or "")[:200]
    return None, None


def classify_trial(trial: Path, model: str | None = None, prices: dict | None = None, *, require_metrics=False) -> dict:
    r = json.loads((trial / "result.json").read_text())
    ei = r.get("exception_info") or {}
    exc = ei.get("exception_type")
    reward = _trial_reward(r)
    status, msg = _api_status(trial)
    minutes = None
    try:
        from datetime import datetime
        a = (r.get("agent_execution") or {})
        if a.get("started_at") and a.get("finished_at"):
            minutes = (datetime.fromisoformat(a["finished_at"].replace("Z", "+00:00")) -
                       datetime.fromisoformat(a["started_at"].replace("Z", "+00:00"))).total_seconds() / 60
    except Exception:  # noqa: BLE001
        pass
    if reward is not None and float(reward) >= 1.0:
        outcome = "pass"
    elif exc in INFRA_EXC or (status in (401, 402, 403, 429, 529)):
        outcome = "infra"
    elif reward is None:
        outcome = "infra"
    else:
        outcome = "fail"
    quota = bool(msg and re.search(r"额度|quota|429", msg))
    deliv = delivered(trial, required=require_metrics and _s9_requires_metrics(r))
    c = trial_cost(trial, model, prices or {}, r)
    return dict(trial=trial.name, outcome=outcome, result_type=result_type(outcome, exc, deliv), delivered=deliv,
                reward=reward, exception=exc, api_status=status, api_msg=msg, quota=quota, minutes=minutes,
                cost_usd=c["cost_usd"], cost_kind=c["cost_kind"], cost_source=c["cost_source"], usage=c["usage"])


def collect(job_dir: Path, model: str | None = None, prices: dict | None = None, *,
            trial_dirs=None, require_metrics=False) -> list[dict]:
    if trial_dirs is None:
        trial_dirs = sorted(p.parent for p in Path(job_dir).glob("*/result.json"))
    return [classify_trial(t, model, prices, require_metrics=require_metrics) for t in trial_dirs]


def job_cost(trials: list[dict]) -> tuple[float | None, str, dict | None]:
    """Cost of a whole Harbor job (all its trials, retries included): (usd, kind, usage).  Any trial with an
    unknown cost makes the job's cost unknown (None, "reservation"); otherwise "reported" if every trial was
    reported by Harbor, else "tokens"."""
    tot = {k: 0 for k in _TOK}
    for t in trials:
        for k in _TOK:
            tot[k] += int((t.get("usage") or {}).get(k) or 0)
    usage = tot if any(t.get("usage") for t in trials) else None
    if not trials or any(t.get("cost_usd") is None for t in trials):
        return None, "reservation", usage
    kind = "reported" if all(t.get("cost_kind") == "reported" for t in trials) else "tokens"
    return round(sum(float(t["cost_usd"]) for t in trials), 4), kind, usage


def run_summary(runs: list[dict]) -> list[dict]:
    """Per-run calibration record for the release manifest."""
    keys = ("who", "n", "model", "job_name", "outcome", "result_type", "minutes", "cost_usd", "cost_kind")
    out = []
    for r in runs:
        if r.get("obsolete_probe_job"):
            continue
        t = r.get("trial") or {}
        d = {k: r.get(k, t.get(k)) for k in keys}
        if d["result_type"] is None and r.get("outcome") == "infra":
            d["result_type"] = "infra"
        d["reward"] = t.get("reward")
        d["exception"] = t.get("exception")
        out.append(d)
    return out
