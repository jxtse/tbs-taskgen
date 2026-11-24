from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

from .core import digest, file_hash, lines, read, tree_hash


def session_info(trial, config):
    agent = config.get("agent") or {}
    sig = (str(agent.get("import_path", "")) + " " + str(agent.get("name", ""))).lower()
    patterns = {
        "claude-code": "agent/sessions/projects/*/*.jsonl",
        "codex": "agent/sessions/????/??/??/rollout-*.jsonl",
        "pi": "agent/pi/sessions/*.jsonl",
    }
    choices = {k: sorted(trial.glob(v)) for k, v in patterns.items()}
    custom = (agent.get("kwargs") or {}).get("session_path") or agent.get("session_path")
    if custom and (trial / custom).is_file():
        choices["pi"] = [trial / custom]
    kind = next(
        (
            k
            for k, pattern in (("claude-code", "claude"), ("codex", "codex"), ("pi", r"\bpi|[_.]pi"))
            if re.search(pattern, sig)
        ),
        None,
    )
    if kind is None:
        found = [k for k, v in choices.items() if v]
        kind = found[0] if len(found) == 1 else "unknown"
    return kind, choices.get(kind, [])


def released_task(ctx, run, tid, row):
    if Path(tid).name != tid or tid in {".", "..", ""}:
        raise ValueError("invalid item_id")
    item = run / "items" / tid
    pkg = item / "pkg"
    if not pkg.is_dir():
        pkg = run / row["pkg"]
    state = read(item / "state.json", {})
    spec = read(item / "_spec/spec.json", {})
    prov = read(pkg / "provenance.json", {})
    candidate = read(item / "S1_candidate.json", {})
    review = row.get("review") or {}
    if isinstance(review, str):
        review = {"verdict": review}
    local = read(item / "calib/runs.json", [])
    local_by_key = {(r.get("job_name"), r.get("n")): r for r in local}
    cal = []
    for attempt in row.get("calib_runs", local):
        detail = local_by_key.get((attempt.get("job_name"), attempt.get("n")), {})
        merged = {k: detail[k] for k in ("trial", "delivered", "api_status") if k in detail}
        cal.append({**merged, **attempt})
    network = prov.get("network") or {}
    thresholds = read(item / "thresholds.json", prov.get("thresholds") or {})
    task = dict(
        task_id=tid,
        pkg=str(pkg),
        pkg_sha256=tree_hash(pkg) if pkg.is_dir() else None,
        label=row.get("label"),
        released=state.get("stage") == "S10" and state.get("status") == "done",
        verdict=review.get("verdict"),
        label_trustworthy=all(
            v is not False
            for v in (
                row.get("label_trustworthy"),
                review.get("label_trustworthy"),
                state.get("label_trustworthy"),
            )
        ),
        domain=spec.get("domain") or row.get("domain", ""),
        field=spec.get("field") or row.get("field", ""),
        title=row.get("title") or prov.get("title") or candidate.get("title", ""),
        doi=row.get("doi") or prov.get("doi") or candidate.get("doi", ""),
        metrics=spec.get("metrics", prov.get("thresholds", {})),
        calib_runs=cal,
        thresholds=thresholds.get("thresholds", thresholds),
        prompt_versions=[spec.get("prompt_version"), prov.get("prompt_version")],
        assumptions=spec.get("generator_assumptions", []),
        offline=row.get("offline", not network.get("agent_hosts")),
    )
    task["source_hashes"] = {
        str(pkg / name): ctx.content_hash(pkg / name)
        for name in ("instruction.md", "tests/test_outputs.py", "tests/measure.py", "tests/test.sh")
    }
    return task


def harbor_trial(ctx, trial, data, tasks, by_job):
    config = read(trial / "config.json", data.get("config") or {})
    job = trial.parent.name
    possible = [data.get("task_name"), data.get("task_id"), (config.get("task") or {}).get("path")]
    ids = set()
    for value in possible:
        if isinstance(value, dict):
            value = value.get("path")
        if value:
            ids.update(set(Path(value).parts) & tasks.keys())
    matches = by_job.get(job, [])
    if not ids and len({t for t, _ in matches}) == 1:
        ids.add(matches[0][0])
    tid = next(iter(ids)) if len(ids) == 1 else None
    matching = [r for t, r in matches if t == tid]
    calibration = next((r for r in matching if r.get("trial") == trial.name), {})
    if not calibration and len(matching) == 1:
        calibration = matching[0]
    kind, sessions = session_info(trial, config)
    agent = config.get("agent") or {}
    info = data.get("agent_info") or {}
    reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exc = (data.get("exception_info") or {}).get("exception_type") or calibration.get("exception")
    result_type = calibration.get("result_type") or data.get("result_type", "")
    timeout = bool(re.search("timeout", str(exc) + str(result_type), re.I))
    hint = bool(
        calibration.get("hint")
        or (trial / "agent/hint-meta.json").exists()
        or data.get("hint_run")
        or (config.get("metadata") or {}).get("hint_run")
        or (trial / "agent/resume-proxy").exists()
        or (agent.get("kwargs") or {}).get("hint")
    )
    captures = sorted(
        p for p in (trial / "agent").rglob("calls.jsonl") if (p.parent / "manifest.json").is_file()
    )
    identity = str(data.get("id") or digest(str(trial)))
    minutes = calibration.get("minutes")
    execution = data.get("agent_execution") or data
    if minutes is None and execution.get("finished_at") and execution.get("started_at"):
        minutes = (
            datetime.fromisoformat(execution["finished_at"]) - datetime.fromisoformat(execution["started_at"])
        ).total_seconds() / 60
    submitted = data.get("delivered", calibration.get("delivered"))
    metrics = trial / "verifier/metrics.jsonl"
    if metrics.is_file():
        entries = [
            e["entry_exists"]
            for e in lines(metrics)
            if e.get("name") == "_solution_cli_run" and not e.get("lock_failed") and "entry_exists" in e
        ]
        if entries:
            submitted = any(entries)
    response_models = set()
    api_status = calibration.get("api_status")
    for log in sorted((trial / "agent/model-proxy").glob("requests-*.jsonl")):
        for entry in lines(log):
            response_models.update(v for v in (entry.get("response_models") or []) if isinstance(v, str))
            if entry.get("response_model"):
                response_models.add(entry["response_model"])
            if isinstance(entry.get("status"), int):
                api_status = entry["status"]
    record = dict(
        response_models=sorted(response_models),
        api_status=api_status,
        trial_id=identity,
        trial_name=trial.name,
        calib_n=calibration.get("n"),
        task_id=tid,
        job_name=job,
        harness=kind,
        who=calibration.get("who"),
        model=agent.get("model_name")
        or calibration.get("model")
        or (info.get("model_info") or {}).get("name"),
        reported_model=(info.get("model_info") or {}).get("name"),
        agent_version=info.get("version"),
        reward=reward,
        exception=exc,
        timed_out=timeout,
        minutes=minutes,
        result_type=result_type,
        outcome=calibration.get("outcome"),
        submitted=submitted,
        cost_usd=(data.get("agent_result") or {}).get("cost_usd", calibration.get("cost_usd")),
        verifier=str(trial / "verifier"),
        sessions=list(map(str, sessions)),
        hint=hint,
        capture=str(captures[0]) if len(captures) == 1 else None,
        trial=str(trial),
        clean_end=not exc and not timeout and bool(data.get("finished_at")),
    )
    record["source_hashes"] = {}
    for path in [
        *sessions,
        *captures,
        *(p.parent / "manifest.json" for p in captures),
        *(f for p in captures for f in (p.parent.parent / "blobs").glob("*")),
    ]:
        if path.is_file():
            record["source_hashes"][str(path)] = ctx.content_hash(path)
    return record


def discover(ctx, runs, jobs):
    """Inventory only release inputs and Harbor trial artifacts; no recursive run-wide scan."""
    tasks, trials, evidence = {}, {}, []
    roots = {ctx.path(p) for p in jobs}
    for run in sorted({ctx.path(p) for p in runs}):
        manifest = run / "release/manifest.jsonl"
        evidence.append((str(manifest), file_hash(manifest)))
        # Later manifest lines supersede earlier releases of the same item.
        releases = {r["item_id"]: r for r in lines(manifest)}
        for tid, row in releases.items():
            task = released_task(ctx, run, tid, row)
            if tid not in tasks:
                tasks[tid] = task
                continue
            prior = tasks[tid]
            if any(task[k] != prior[k] for k in task if k not in {"pkg", "source_hashes", "calib_runs"}):
                raise ValueError(f"conflicting releases: {tid}")
            known = {digest(r) for r in prior["calib_runs"]}
            prior["calib_runs"].extend(r for r in task["calib_runs"] if digest(r) not in known)
        if (run / "calib-jobs").is_dir():
            roots.update(p for p in (run / "calib-jobs").iterdir() if p.is_dir())
    by_job = defaultdict(list)
    for t in tasks.values():
        for r in t["calib_runs"]:
            by_job[r.get("job_name")].append((t["task_id"], r))
    for root in sorted(roots):
        if not root.is_dir():
            raise ValueError(f"missing job directory {root}")
        for result in sorted(
            {*root.glob("*/result.json"), *root.glob("*/*/result.json"), root / "result.json"}
        ):
            if not result.is_file():
                continue
            trial = result.parent
            data = read(result)
            if not any(k in data for k in ("task_name", "task_id", "verifier_result", "exception_info")):
                continue  # Harbor job summary
            record = harbor_trial(ctx, trial, data, tasks, by_job)
            identity = record["trial_id"]
            if identity in trials and trials[identity] != record:
                raise ValueError(f"conflicting trial id {identity}")
            trials[identity] = record
    return list(tasks.values()), list(trials.values()), sorted(set(evidence))


def ingest(ctx, runs, jobs):
    if not runs:
        raise ValueError("at least one run is required")
    tasks, trials, evidence = discover(ctx, runs, jobs)

    def build():
        ctx.emit("tasks", tasks)
        ctx.emit("trials", trials)
        ctx.emit(
            "ingest.ledger",
            [
                *(dict(task_id=t["task_id"], reason="release_manifest") for t in tasks),
                *(
                    dict(trial_id=t["trial_id"], reason="matched" if t["task_id"] else "unknown_task")
                    for t in trials
                ),
            ],
        )
        return {"count": len(tasks), "trials": len(trials)}

    return ctx.stage("ingest", [tasks, trials, evidence], build)
