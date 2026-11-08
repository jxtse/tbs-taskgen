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
"""
from __future__ import annotations

import json
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


def next_action(runs: list[dict], cfg: dict) -> tuple[str, str | None]:
    """runs: [{"who": "glm"|"teacher", "outcome": "pass"|"fail"|"infra", "minutes": float}] in order.
    Returns ("run", "glm"|"teacher") or ("label", "hard"|"unstable"|"easy"|"parked:<why>")."""
    c = cfg["calib"]
    infra = [r for r in runs if r["outcome"] == "infra"]
    real = [r for r in runs if r["outcome"] != "infra"]
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


def task_path(cfg: dict, item_dir: Path) -> Path:
    """Task folder the job runs: the sandbox-prepared copy (docker_image pointing at a registered template) when
    calib.prepared_tasks_dir has one for this item, else the item's own pkg/ (local Docker backends)."""
    d = cfg["calib"].get("prepared_tasks_dir")
    if d and (Path(d) / item_dir.name / "task.toml").exists():
        return Path(d) / item_dir.name
    return item_dir / "pkg"


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
    hosts = [str(h) for h in c.get("agent_allowed_hosts") or []]
    if hosts:
        agent["extra_allowed_hosts"] = hosts            # the model endpoint during the agent phase only
    return {"job_name": job_name, "jobs_dir": c["jobs_dir"], "n_attempts": 1, "n_concurrent_trials": 1,
            "timeout_multiplier": 1.0, "agent_timeout_multiplier": 1.0,
            "retry": {"max_retries": 2, "include_exceptions": ["AgentSetupTimeoutError", "EnvironmentStartTimeoutError",
                                                              "NetworkConnectionError"]},
            "environment": {**_ref(c.get("environment"), "type", "docker"), "delete": True},
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


def job_name_for(item_dir: Path, who: str, n: int) -> str:
    """Stable per (item, who, attempt): a re-run after a crash finds the same job instead of starting another."""
    return f"tbsgen-{item_dir.name}-{who}{n}"


def prepare(cfg: dict, item_dir: Path, who: str, n: int) -> dict:
    """Intent record for attempt n (no side effects outside item_dir).  Persist it BEFORE reserve/launch."""
    c = cfg["calib"]
    job_name = job_name_for(item_dir, who, n)
    conf = job_config(cfg, task_path(cfg, item_dir), who, job_name)
    p = item_dir / "calib" / f"{job_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(conf, indent=1))
    p = p.resolve()
    launcher = c.get("student_launcher") if who == "glm" else c.get("teacher_launcher")
    model = c["student_model"] if who == "glm" else c["teacher_model"]
    command = (f"cd {c['harbor_root']} && ./{launcher} run {p}" if launcher and c.get("harbor_root")
               else f"harbor run --config {p} --yes")
    rec = dict(who=who, n=n, item=item_dir.name, model=model, job_name=job_name, config=str(p),
               command=command, state="intent", submitted=False,
               ts=time.time(), job_dir=str(Path(c["jobs_dir"]) / job_name), task=conf["tasks"][0]["path"],
               outcome=None)
    if c.get("prepared_tasks_dir") and not rec["task"].startswith(str(Path(c["prepared_tasks_dir"]))):
        # remote sandbox backends cannot build a Dockerfile: a missing prepared template means the run would die
        # at environment start (e2e-iter1: FileUploadException, no model use) -> do not launch
        rec["blocked"] = "no prepared sandbox template for this item"
    why = network_problem(cfg, Path(rec["task"]), conf)
    if why:
        # an automatic launch would only burn a sandbox; a printed command is for a human, who can add the host
        rec["blocked" if c.get("allow_submit") else "warning"] = why
    if c.get("allow_submit") and not (launcher and c.get("harbor_root")):
        rec["blocked"] = (f"calib.allow_submit needs calib.harbor_root and calib.{'student' if who == 'glm' else 'teacher'}"
                          "_launcher (or set allow_submit=false and run the printed command yourself)")
    return rec


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
    if rec["state"] == "intent":
        # an intent has no side effect outside item_dir: rewrite its job config from the current settings (it may
        # have been written while submit was off, before calib.agent_allowed_hosts was set) and re-check it
        conf = job_config(cfg, Path(rec["task"]), rec["who"], rec["job_name"])
        Path(rec["config"]).write_text(json.dumps(conf, indent=1))
        why = network_problem(cfg, Path(rec["task"]), conf)
        if why:
            rec["blocked"] = why
            save(rec)
            return rec
        rec.pop("blocked", None)
        rec["ledger_call"] = ledger.reserve(rec["model"], "S9", rec["item"], True,
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
                ledger.settle(rec["ledger_call"], 0.0, status="error", note=f"not launched: {why}"[:500])
            rec.update(state="launch_failed", outcome="infra", error=f"not launched: {why}")
            save(rec)
            return rec
    if job_started(rec):                                   # crashed after launch, before recording it
        rec.update(state="launched", submitted=True, recovered=True)
        save(rec)
        return rec
    item_dir = Path(rec["config"]).parent.parent
    log = item_dir / "calib" / f"{rec['job_name']}.log"
    launcher = c["student_launcher"] if rec["who"] == "glm" else c["teacher_launcher"]
    # the launcher's exit status is written next to the log, so a launcher that dies at once is visible
    inner = (f"cd {shlex.quote(c['harbor_root'])} && env -u PYTHONPATH -u VIRTUAL_ENV ./{launcher} run "
             f"{shlex.quote(rec['config'])} > {shlex.quote(str(log))} 2>&1; "
             f"echo $? > {shlex.quote(str(log) + '.exit')}")
    rec["log"] = str(log)
    try:
        r = subprocess.run([c.get("tmux_bin", "tmux"), "new-session", "-d", "-s", rec["job_name"][-60:],
                            "bash", "-lc", inner], capture_output=True, text=True, timeout=60)
        err = None if r.returncode == 0 else f"tmux rc={r.returncode}: {r.stderr[-300:]}"
    except Exception as e:  # noqa: BLE001  (missing tmux, timeout, ...)
        err = f"{type(e).__name__}: {e}"
    if err:
        if ledger is not None and rec.get("ledger_call"):
            ledger.settle(rec["ledger_call"], 0.0, status="error", note=f"launch failed: {err}"[:500])
        rec.update(state="launch_failed", outcome="infra", error=err)
        save(rec)
        return rec
    rec.update(state="launched", submitted=True, tmux=rec["job_name"][-60:])
    save(rec)
    return rec


def launcher_failed(rec: dict) -> str | None:
    """Launcher exited without creating a Harbor job (bad cwd, missing script, key fetch failure ...)."""
    ex = Path(rec.get("log", "") + ".exit")
    if rec.get("state") == "launched" and ex.exists() and not (Path(rec["job_dir"]) / "result.json").exists():
        tail = Path(rec["log"]).read_text(errors="replace")[-300:] if Path(rec["log"]).exists() else ""
        return f"launcher exit {ex.read_text().strip()} without a job: {tail}"
    return None


def actual_spend(ledger) -> float:
    with ledger._db() as db:
        return db.execute("SELECT COALESCE(SUM(usd),0) FROM calls WHERE real=1 AND status!='reserved'").fetchone()[0]


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


def classify_trial(trial: Path) -> dict:
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
    cost = (r.get("agent_result") or {}).get("cost_usd")
    return dict(trial=trial.name, outcome=outcome, reward=reward, exception=exc, api_status=status,
                api_msg=msg, quota=quota, minutes=minutes, cost_usd=cost)


def collect(job_dir: Path) -> list[dict]:
    return [classify_trial(t) for t in sorted(Path(job_dir).iterdir()) if (t / "result.json").exists()]
