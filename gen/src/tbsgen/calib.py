"""S9 difficulty calibration: sequential GLM-5.3 (Claude Code 2.1.280) runs + Opus teacher branch.
Only scheduling + result collection.  Submitting is gated by cfg.calib.allow_submit (default False): the
pipeline writes a Harbor job config + the exact launcher command and the item waits in "S9/waiting".

Decision rule:
    GLM#1 fail  -> teacher#1; pass -> "hard" (keep); fail -> teacher#2 (teacher_runs_max) ; fail -> "parked"
                   (possible defect or too hard; human spot check)
    GLM#1 pass  -> (quick_pass_minutes set and duration below it -> "easy") else GLM#2
    GLM#2 fail  -> "unstable" (keep: GLM 1/2);  GLM#2 pass -> "easy" (S9b variants disabled in MVP)
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
import time
from pathlib import Path

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


def next_action(runs: list[dict], cfg: dict) -> tuple[str, str | None]:
    """runs: [{"who": "glm"|"teacher", "outcome": "pass"|"fail"|"infra", "minutes": float,
    "result_type": optional}] in order.
    Returns ("run", "glm"|"teacher") or ("label", "hard"|"unstable"|"easy"|"parked:<why>")."""
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
                return "label", "parked:timeout_no_output"
            if pol == "retry" and skipped < int(c.get("timeout_no_output_retries", 1)):
                skipped += 1                # not decisive: the student is run again
                continue
        real.append(r)
    trailing_infra = 0
    for r in reversed(runs):
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


def pkg_sha(pkg: Path) -> str:
    """Content hash of a packaged task (paths + bytes): a prepared copy made from another pkg/ is stale."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p for p in Path(pkg).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(pkg)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def ensure_prepared(cfg: dict, item_dir: Path, who: str = "glm", force: bool = False) -> dict:
    """Environment-preparation hook (calib.prepare_command): build the run-ready copy of pkg/ that a remote
    backend needs (image + registered template, task.toml pointing at it) at <prepared_tasks_dir>/<item_id>.

    The command is a template run with bash in calib.prepare_cwd (default: the role's harbor_root); placeholders
    {item_id} {pkg} {out} {who} are shell-quoted, and the same values are in the environment as TBSGEN_ITEM_ID /
    TBSGEN_PKG / TBSGEN_PREPARED_DIR / TBSGEN_WHO.  It must leave <out>/task.toml.  The outcome is recorded in
    calib/prepare.json with the pkg/ hash: a copy made from the same pkg/ is reused, a failure for the same pkg/
    is not retried automatically (force=True, i.e. `tbsgen calib prepare`, retries)."""
    import shlex
    import subprocess
    c = cfg["calib"]
    cmd, d = c.get("prepare_command"), prepared_dir(cfg, who)
    if not cmd or not d:
        return dict(ok=bool(not d or (Path(d) / item_dir.name / "task.toml").exists()), skipped="no prepare_command")
    out = Path(d).resolve() / item_dir.name          # the hook runs in another cwd
    pkg = item_dir / "pkg"
    sha = pkg_sha(pkg)
    recp = item_dir / "calib" / f"prepare-{who}.json" if who != "glm" else item_dir / "calib" / "prepare.json"
    prev = {}
    try:
        prev = json.loads(recp.read_text())
    except (OSError, ValueError):
        pass
    if prev.get("pkg_sha") == sha and not force:
        if prev.get("ok") and (out / "task.toml").exists():
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
    t0 = time.time()
    try:
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
    recp.write_text(json.dumps(res, indent=1))
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
    command = (f"cd {root} && ./{launcher} run {p}" if launcher and root else f"harbor run --config {p} --yes")
    rec = dict(who=who, n=n, item=item_dir.name, model=model, job_name=job_name, config=str(p),
               command=command, state="intent", submitted=False,
               ts=time.time(), job_dir=str(Path(c["jobs_dir"]) / job_name), task=conf["tasks"][0]["path"],
               outcome=None, epoch=epoch)
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


def stale_prepared(cfg: dict, item_dir: Path, who: str, epoch: int) -> str | None:
    """After a review revision (epoch > 0) a prepared copy must be one made from the current pkg/ (recorded by
    ensure_prepared with the pkg hash); a copy prepared by hand before the revision would calibrate the old task."""
    if not epoch or not prepared_dir(cfg, who):
        return None
    recp = item_dir / "calib" / ("prepare.json" if who == "glm" else f"prepare-{who}.json")
    try:
        prev = json.loads(recp.read_text())
    except (OSError, ValueError):
        prev = {}
    if prev.get("ok") and prev.get("pkg_sha") == pkg_sha(item_dir / "pkg"):
        return None
    return (f"the prepared {who} copy may predate the review revision (no calib/{recp.name} for the current pkg/): "
            "set calib.prepare_command (S9 re-prepares) or prepare it and run `tbsgen calib prepare --item ID`")


def remote_alias(job_dir: Path) -> dict:
    """Marker a remote launcher leaves in the local job dir (tbsgen-remote.json: host, local_task, remote_task,
    remote_job_dir): the job ran on another host from a synced copy of the task, so Harbor's config.json names the
    remote path.  ingest() accepts that path only when the marker maps it to the run's own local task."""
    try:
        m = json.loads((Path(job_dir) / "tbsgen-remote.json").read_text())
    except (OSError, ValueError):
        return {}
    return m if isinstance(m, dict) else {}


def task_matches(job_dir: Path, job_task: str | None, want_task: str | None) -> bool:
    if job_task == want_task:
        return True
    m = remote_alias(job_dir)
    return bool(m) and m.get("remote_task") == job_task and m.get("local_task") == want_task


def job_started(rec: dict) -> bool:
    """Harbor created the job dir (or the launcher wrote its start marker): the job exists, never relaunch."""
    jd = Path(rec["job_dir"])
    return (jd / "config.json").exists() or (jd / "result.json").exists() or Path(rec.get("log", "/x")).exists()


def launch(cfg: dict, rec: dict, ledger, save) -> dict:
    """intent -> reserved -> launched.  `save(rec)` persists the run record; it is called after every step so a
    crash at any point is recoverable: a reserved-but-not-started job is launched once on resume, a started job
    is never launched again.  Any launch exception settles the reservation at 0 and marks the run failed."""
    import shlex
    import subprocess
    c = cfg["calib"]
    if rec["state"] == "intent" and job_started(rec):
        # its printed command was already run (by hand): take the job over as it is (task, config, identity
        # unchanged); no preparation, path rewrite or pre-submit check applies to a job that exists
        rec["ledger_call"] = ledger.reserve(rec["model"], rec.get("ledger_stage", "S9"), rec["item"], True,
                                            est_usd=float(c["est_usd_per_run"])) if ledger is not None else None
        rec.update(state="launched", submitted=True, recovered=True)
        save(rec)
        return rec
    if rec["state"] == "intent":
        # an intent has no side effect outside item_dir.  It may have been written while submit was off, before
        # calib.agent_allowed_hosts was set: update only that field of its job config (model, job dir and task stay
        # those of the record, which ingest() checks the result against) and re-check it
        try:
            conf = json.loads(Path(rec["config"]).read_text())
        except (OSError, ValueError) as e:
            # never rebuild from the current settings: model / job dir could differ from the record
            rec["blocked"] = f"job config {rec['config']} unreadable ({type(e).__name__}); run prepare again"
            save(rec)
            return rec
        hosts = [str(h) for h in role(c, rec["who"], "agent_allowed_hosts") or []]
        # an intent written before its sandbox copy existed points at pkg/: switch it to the prepared copy (same
        # task, run-ready); the record's task changes with it, so ingest() still checks the job against it
        item_dir = Path(rec["config"]).parent.parent
        cur = task_path(cfg, item_dir, rec["who"])
        if Path(rec["task"]) == (item_dir / "pkg").resolve() and cur != item_dir / "pkg":
            for t in conf.get("tasks") or []:
                t["path"] = str(cur.resolve())
            rec["task"] = str(cur.resolve())
        if _unprepared(cfg, rec["who"], rec["task"]):
            rec["blocked"] = "no prepared sandbox template for this item"
            save(rec)
            return rec
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
        rec["ledger_call"] = ledger.reserve(rec["model"], rec.get("ledger_stage", "S9"), rec["item"], True,
                                            est_usd=float(c["est_usd_per_run"])) if ledger is not None else None
        rec["state"] = "reserved"
        save(rec)
    if rec["state"] != "reserved":
        return rec
    if not job_started(rec):
        # the record may predate a config change (written while submit was off): re-check the job's own config
        try:
            conf = json.loads(Path(rec["config"]).read_text())
        except (OSError, ValueError):
            conf = {}
        why = network_problem(cfg, Path(rec["task"]), conf)
        if why:
            if ledger is not None and rec.get("ledger_call"):
                ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=f"not launched: {why}"[:500])
            rec.update(state="launch_failed", outcome="infra", result_type="infra", cost_usd=0.0, cost_kind="not_launched", error=f"not launched: {why}")
            save(rec)
            return rec
    if job_started(rec):                                   # crashed after launch, before recording it
        rec.update(state="launched", submitted=True, recovered=True)
        save(rec)
        return rec
    if rec.get("launch_attempted"):
        # a launch was attempted before a crash and no job dir exists yet: the launcher may still be starting it
        alive = session_alive(c, rec)
        if alive or alive is None:
            rec.update(state="launched", submitted=True, recovered=True)
            save(rec)
            return rec
        # session gone and no job: the launcher exited without one (launcher_failed() in s9 settles it) or never
        # ran; either way nothing is running, so it is safe to launch again below
    item_dir = Path(rec["config"]).parent.parent
    log = item_dir / "calib" / f"{rec['job_name']}.log"
    launcher = c["student_launcher"] if rec["who"] == "glm" else c["teacher_launcher"]
    root = role(c, rec["who"], "harbor_root")
    if not (launcher and root):
        err = "no launcher / harbor_root for this role"
        if ledger is not None and rec.get("ledger_call"):
            ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=f"launch failed: {err}")
        rec.update(state="launch_failed", outcome="infra", result_type="infra", cost_usd=0.0, cost_kind="not_launched", error=err)
        save(rec)
        return rec
    # the launcher's exit status is written next to the log, so a launcher that dies at once is visible
    inner = (f"cd {shlex.quote(root)} && env -u PYTHONPATH -u VIRTUAL_ENV ./{launcher} run "
             f"{shlex.quote(rec['config'])} > {shlex.quote(str(log))} 2>&1; "
             f"echo $? > {shlex.quote(str(log) + '.exit')}")
    rec.update(log=str(log), tmux=rec["job_name"][-60:], launch_attempted=time.time())
    if Path(str(log) + ".exit").exists():
        Path(str(log) + ".exit").unlink()                  # a previous attempt's exit status
    save(rec)                                              # persisted before the side effect
    try:
        r = subprocess.run([c.get("tmux_bin", "tmux"), "new-session", "-d", "-s", rec["job_name"][-60:],
                            "bash", "-lc", inner], capture_output=True, text=True, timeout=60)
        err = None if r.returncode == 0 else f"tmux rc={r.returncode}: {r.stderr[-300:]}"
        uncertain = False
    except Exception as e:  # noqa: BLE001  (missing tmux, timeout, ...)
        err = f"{type(e).__name__}: {e}"
        uncertain = isinstance(e, subprocess.TimeoutExpired)    # tmux may have started the session anyway
    if err:
        alive = session_alive(c, rec)
        if job_started(rec) or alive or (uncertain and alive is None):
            # the launch may have happened: keep the reservation, never start another attempt; watch / s9
            # settle it from the job or the session
            rec.update(state="launched", submitted=True, launch_uncertain=err[:300])
            save(rec)
            return rec
        if ledger is not None and rec.get("ledger_call"):
            ledger.settle(rec["ledger_call"], 0.0, status="error", kind="not_launched", note=f"launch failed: {err}"[:500])
        rec.update(state="launch_failed", outcome="infra", result_type="infra", cost_usd=0.0, cost_kind="not_launched", error=err)
        save(rec)
        return rec
    rec.update(state="launched", submitted=True, tmux=rec["job_name"][-60:])
    save(rec)
    return rec


def launcher_failed(rec: dict, c: dict | None = None) -> str | None:
    """Launcher exited without creating a Harbor job (bad cwd, missing script, key fetch failure ...), or its
    tmux session vanished without an exit status and without a job."""
    ex = Path(rec.get("log", "") + ".exit")
    if rec.get("state") == "launched" and c is not None and not ex.exists() and not job_started(rec) and \
            session_alive(c, rec) is False and not ex.exists() and not job_started(rec):   # re-read after the check
        return "launcher session gone without an exit status and without a job"
    if rec.get("state") == "launched" and ex.exists() and not (Path(rec["job_dir"]) / "result.json").exists():
        tail = Path(rec["log"]).read_text(errors="replace")[-300:] if Path(rec["log"]).exists() else ""
        return f"launcher exit {ex.read_text().strip()} without a job: {tail}"
    return None


def session_alive(c: dict, rec: dict) -> bool | None:
    """Is the tmux session this run was launched in still there?  None = cannot tell (no tmux name recorded,
    e.g. a job started by hand, or tmux unusable).  The launchers run Harbor in the foreground, so the session
    lives exactly as long as the job's Harbor process."""
    import subprocess
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
    if rec.get("state") != "launched":
        return False
    jr = Path(rec["job_dir"]) / "result.json"
    ex = Path(rec.get("log", "") + ".exit")
    if ex.exists():
        if not jr.exists():
            return False                                # launcher_failed() covers an exit without a job
        # Harbor writes finished_at before it returns and the .exit is written after: read finished_at only
        # after seeing .exit
        return not (_load(jr) or {}).get("finished_at")
    if c is None or session_alive(c, rec) is not False:
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


def delivered(trial: Path) -> bool | None:
    """Did the agent leave its deliverable?  From the verifier's metrics.jsonl (tbsgen's _metrics.run_solution
    records `_solution_cli_run`): the entry file missing ("can't open file" / "No such file") = False; any other
    run = True; no record = None (unknown)."""
    f = trial / "verifier" / "metrics.jsonl"
    if not f.exists():
        return None
    seen = None
    for line in f.read_text(errors="replace").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("name") != "_solution_cli_run" or o.get("lock_failed"):
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


def classify_trial(trial: Path, model: str | None = None, prices: dict | None = None) -> dict:
    r = json.loads((trial / "result.json").read_text())
    ei = r.get("exception_info") or {}
    exc = ei.get("exception_type")
    reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
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
    deliv = delivered(trial)
    c = trial_cost(trial, model, prices or {}, r)
    return dict(trial=trial.name, outcome=outcome, result_type=result_type(outcome, exc, deliv), delivered=deliv,
                reward=reward, exception=exc, api_status=status, api_msg=msg, quota=quota, minutes=minutes,
                cost_usd=c["cost_usd"], cost_kind=c["cost_kind"], cost_source=c["cost_source"], usage=c["usage"])


def collect(job_dir: Path, model: str | None = None, prices: dict | None = None) -> list[dict]:
    return [classify_trial(t, model, prices) for t in sorted(Path(job_dir).iterdir()) if (t / "result.json").exists()]


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
        t = r.get("trial") or {}
        d = {k: r.get(k, t.get(k)) for k in keys}
        if d["result_type"] is None and r.get("outcome") == "infra":
            d["result_type"] = "infra"
        d["reward"] = t.get("reward")
        d["exception"] = t.get("exception")
        out.append(d)
    return out
