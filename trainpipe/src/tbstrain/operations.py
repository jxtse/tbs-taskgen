"""Operator-facing Harbor configuration generation and MinT submission."""

import os
import shlex
import subprocess
import time

from .core import atomic, digest, file_hash, loads
from .export import validate


def rollout(ctx):
    cfg = ctx.cfg["rollout"]

    def build():
        ids = {r["task_id"] for r in ctx.rows("selection")}
        tasks = [{"path": t["pkg"]} for t in ctx.rows("tasks") if t["task_id"] in ids]
        harness = cfg["harness"]
        if harness not in {"claude-code", "pi", "codex"} or cfg["attempts"] < 1 or cfg["concurrency"] < 1:
            raise ValueError("invalid rollout settings")
        agent = {**cfg["agents"][harness], "model_name": cfg["model"]}
        job = dict(
            job_name=cfg["job_name"],
            jobs_dir=str(ctx.path(cfg["jobs_dir"])),
            tasks=tasks,
            agents=[agent],
            environment=cfg["environment"],
            n_attempts=cfg["attempts"],
            n_concurrent_trials=cfg["concurrency"],
        )
        directory = ctx.work / "rollout"
        directory.mkdir()
        atomic(directory / "job.json", job)
        launch = directory / "launch.sh"
        launch.write_text(
            '#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nexec '
            + shlex.quote(cfg.get("harbor_cli", "harbor"))
            + " run -c job.json\n",
            "utf-8",
        )
        launch.chmod(0o755)
        ctx.emit("rollout.ledger", (dict(task_id=tid, reason="configured") for tid in sorted(ids)))
        return {"count": len(tasks), "attempts_per_task": cfg["attempts"]}

    return ctx.stage("rollout", [ctx.dependency("select"), cfg], build)


def ignored_fields(value):
    found = set()
    if isinstance(value, dict):
        for key, entry in value.items():
            if key in ("fields_ignored", "preflight.fields_ignored") and isinstance(entry, list):
                found.update(str(v) for v in entry)
            if isinstance(entry, str) and "preflight.fields_ignored" in entry and "loss_mask" in str(value):
                found.add("loss_mask")
            found.update(ignored_fields(entry))
    if isinstance(value, list):
        for entry in value:
            found.update(ignored_fields(entry))
    return found


def submit(ctx, confirm, allow_warn):
    cfg = ctx.cfg["submit"]
    dataset = ctx.out / "train.jsonl"
    checked = validate(ctx, dataset)
    if not checked["valid"]:
        raise ValueError("dataset failed local validation")
    binary = os.environ.get(cfg.get("cli_env", "MINT_RECURSIVE_CLI")) or cfg.get("cli", "mint-recursive")
    if "/" in binary:
        binary = str(ctx.path(binary))
    hyper = ctx.path(cfg["config_file"])
    note = cfg.get("note", "Scientific task trajectories")
    if len(note) > 80:
        raise ValueError("upload note exceeds 80 characters")
    dataset_hash = file_hash(dataset)
    config_hash = digest([file_hash(hyper), cfg["base_model"]])
    keys = [
        ctx.dependency("export"),
        checked["key"],
        dataset_hash,
        config_hash,
        cfg,
        binary,
        confirm,
        allow_warn,
    ]
    receipts = ctx.out / "submit-responses" / digest(keys)

    def build():
        receipts.mkdir(parents=True, exist_ok=True)
        log = []

        def call(label, *args):
            command = [binary, *args, "--json", "--timeout", str(cfg.get("timeout", "30m"))]
            p = subprocess.run(command, capture_output=True, text=True, check=False)
            receipt = receipts / f"{time.time_ns()}-{label}.json"
            try:
                response = loads(p.stdout)
            except ValueError:
                response = {"stdout": p.stdout, "stderr": p.stderr, "returncode": p.returncode}
                atomic(receipt, response)
                raise ValueError(f"{label} returned invalid JSON, see {receipt}") from None
            atomic(receipt, response)
            log.append(dict(step=label, response=str(receipt), reason="received", returncode=p.returncode))
            ctx.emit("submit.ledger", log)
            # Keep the decision log and every receipt even if a gate refuses the next command.
            atomic(receipts / "decisions.json", log)
            if p.returncode:
                raise ValueError(f"{label} failed with exit {p.returncode}, see {receipt}")
            return response

        uploaded = call(
            "upload",
            "dataset",
            "upload",
            str(dataset),
            "--type",
            "sft",
            "--note",
            note,
            "--idempotency-key",
            dataset_hash,
        )
        version = uploaded["dataset_version"]
        report = call("report", "dataset", "report", version)
        if (
            report.get("dataset_version", version) != version
            or str(report["status"]).lower() != "available"
            or int(report["fatal_count"]) != 0
            or (not allow_warn and int(report["warn_count"]) != 0)
        ):
            raise ValueError(
                f"dataset report refused training: status={report['status']}, "
                f"fatal_count={report['fatal_count']}, warn_count={report['warn_count']}"
            )
        args = [
            "train",
            "start",
            "--dataset",
            version,
            "--base-model",
            cfg["base_model"],
            "--config-file",
            str(hyper),
        ]
        dry = call("dry-run", *args, "--dry-run")
        fields = ignored_fields(dry)
        execution = None
        if confirm:
            started = call("start", *args, "--idempotency-key", digest([version, config_hash]))
            execution = started["execution_id"]
            fields.update(ignored_fields(call("show", "train", "show", execution)))
            fields.update(ignored_fields(call("events", "train", "events", execution)))
        return {
            "count": len(log),
            "dataset_version": version,
            "execution_id": execution,
            "fields_ignored": sorted(fields),
            "loss_mask_ignored_observed": "loss_mask" in fields,
        }

    return ctx.stage("submit", keys, build)
