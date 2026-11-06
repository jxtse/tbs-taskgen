"""S9 difficulty calibration: sequential GLM-5.3 (Claude Code 2.1.280) runs + Opus teacher branch.
Only scheduling + result collection.  Submitting is gated by cfg.calib.allow_submit (default False): the
pipeline writes a Harbor job config + the exact launcher command and the item waits in "S9/waiting".

Decision rule (decision 2026-11-05, DESIGN §3.5 order):
    GLM#1 fail  -> teacher#1; pass -> "hard" (keep); fail -> teacher#2 (teacher_runs_max) ; fail -> "parked"
                   (possible defect or too hard; human spot check)
    GLM#1 pass  -> (quick_pass_minutes set and duration below it -> "easy") else GLM#2
    GLM#2 fail  -> "unstable" (keep: GLM 1/2);  GLM#2 pass -> "easy" (S9b variants disabled in MVP)
Infra outcomes (API quota/auth/stream errors with reward 0, sandbox start/setup timeouts, cancelled) are not
attempts; they are retried up to calib.infra_retries, then the item is parked "infra".
A trial with reward 1 counts as a pass even if Claude Code later raised an API error (on 2026-11-02, 9 of the
104 UnknownApiError teacher trials had reward 1).
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


def job_config(cfg: dict, pkg: Path, who: str, job_name: str) -> dict:
    c = cfg["calib"]
    if who == "glm":
        agent = {"import_path": c["student_agent"], "model_name": c["student_model"],
                 "kwargs": {"version": c["cc_version"], "reasoning_effort": "max"}, "env": dict(AGENT_ENV)}
    else:
        agent = {"import_path": c["teacher_agent"], "model_name": c["teacher_model"],
                 "kwargs": {"version": c["cc_version"], "reasoning_effort": "max", "thinking_display": "summarized"},
                 "env": dict(AGENT_ENV)}
    return {"job_name": job_name, "jobs_dir": c["jobs_dir"], "n_attempts": 1, "n_concurrent_trials": 1,
            "timeout_multiplier": 1.0, "agent_timeout_multiplier": 1.0,
            "retry": {"max_retries": 2, "include_exceptions": ["AgentSetupTimeoutError", "EnvironmentStartTimeoutError",
                                                              "NetworkConnectionError"]},
            "environment": {"import_path": c["environment"], "delete": True}, "verifier": {"disable": False},
            "agents": [agent], "tasks": [{"path": str(pkg.resolve())}]}


def submit(cfg: dict, item_dir: Path, who: str, n: int, ledger=None) -> dict:
    """Write the job config + launcher command.  With calib.allow_submit=True: reserve the run's estimated cost in
    the ledger (refused -> BudgetStop, nothing launched) and start the launcher detached in tmux."""
    c = cfg["calib"]
    job_name = f"tbsgen-{item_dir.name}-{who}{n}-{time.strftime('%Y%m%d%H%M%S')}"
    conf = job_config(cfg, task_path(cfg, item_dir), who, job_name)
    p = item_dir / "calib" / f"{job_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(conf, indent=1))
    p = p.resolve()
    launcher = c["student_launcher"] if who == "glm" else c["teacher_launcher"]
    cmd = f"cd {c['harbor_root']} && ./{launcher} run {p}"
    rec = dict(who=who, n=n, job_name=job_name, config=str(p), command=cmd, submitted=False, ts=time.time(),
               job_dir=str(Path(c["jobs_dir"]) / job_name), task=conf["tasks"][0]["path"])
    if not c.get("allow_submit"):
        return rec
    import shlex
    import subprocess
    model = c["student_model"] if who == "glm" else c["teacher_model"]
    if ledger is not None:
        rec["ledger_call"] = ledger.reserve(model, "S9", item_dir.name, True, est_usd=float(c["est_usd_per_run"]))
    log = item_dir / "calib" / f"{job_name}.log"
    # `env` must wrap the launcher, not `cd` (env would exec /usr/bin/cd in a child and the launcher would start
    # from the wrong directory -- first real canary launch failed this way)
    inner = (f"cd {shlex.quote(c['harbor_root'])} && env -u PYTHONPATH -u VIRTUAL_ENV ./{launcher} run "
             f"{shlex.quote(str(p))} > {shlex.quote(str(log))} 2>&1")
    r = subprocess.run(["tmux", "new-session", "-d", "-s", job_name[-60:], "bash", "-lc", inner],
                       capture_output=True, text=True)
    if r.returncode != 0:
        if ledger is not None:
            ledger.finish(rec["ledger_call"], error_class="launch", error=r.stderr[-300:])
        raise RuntimeError(f"tmux launch failed: {r.stderr[-300:]}")
    rec.update(submitted=True, tmux=job_name[-60:], log=str(log), model=model)
    return rec


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
