import argparse
from pathlib import Path
import sys

from .core import Context, atomic, read
from .export import export, validate
from .ingest import ingest
from .judge import judge
from .operations import rollout, submit
from .screen import screen
from .profile import configured_profiles, extract
from .wire_check import wire_check
from .select import select
from .tasks import difficulty, filter_tasks


def parser():
    root = argparse.ArgumentParser(prog="tbstrain")
    root.add_argument("--config", type=Path)
    sub = root.add_subparsers(dest="command", required=True)
    command = sub.add_parser("wire-profile").add_subparsers(dest="operation", required=True)
    profile = command.add_parser("extract")
    profile.add_argument("--from", dest="captures", type=Path, nargs="+", action="extend", required=True)
    profile.add_argument("--out", type=Path, required=True)
    profile.add_argument("--version")
    verify = sub.add_parser("wire-check")
    verify.add_argument("--captures", type=Path, required=True)
    verify.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    for name in (
        "run",
        "ingest",
        "difficulty",
        "filter-tasks",
        "screen",
        "judge",
        "select",
        "export",
        "validate",
        "submit",
        "rollout",
    ):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        if name == "ingest":
            p.add_argument("--run", nargs="+", action="extend", default=[])
            p.add_argument("--job", nargs="+", action="extend", default=[])
        elif name == "judge":
            p.add_argument("--kind", choices=["task", "trajectory"], required=True)
        elif name == "validate":
            p.add_argument("file", type=Path)
            p.add_argument("--reference", type=Path)
            p.add_argument("--allow-masked-assistant", action="store_true")
        elif name == "export":
            p.add_argument("--target", choices=["mint-sft"], default="mint-sft")
        elif name == "submit":
            p.add_argument("--confirm", action="store_true")
            p.add_argument("--allow-warn", action="store_true")
    return root


def pipeline_actions(ctx, ingest_overrides):
    return {
        "ingest": lambda: ingest(ctx, ingest_overrides),
        "difficulty": lambda: difficulty(ctx),
        "filter-tasks": lambda: filter_tasks(ctx),
        "screen": lambda: screen(ctx),
        "judge-task": lambda: judge(ctx, "task"),
        "judge-trajectory": lambda: judge(ctx, "trajectory"),
        "select": lambda: select(ctx),
        "export": lambda: export(ctx),
    }


def execute(args, transport=None):
    ctx = Context(args.config)
    if args.command == "wire-profile":
        manifest = extract(args.captures, args.out, args.version)
        print(f"extracted Claude Code {manifest['version']} profile to {args.out}")
        return 0
    if args.command == "wire-check":
        profile = configured_profiles(ctx).get("claude-code") if args.config else None
        return wire_check(args.captures, profile)
    ctx.transport = transport
    overrides = {
        key: [str(Path(p).absolute()) for p in values]
        for key, values in (("runs", getattr(args, "run", [])), ("jobs", getattr(args, "job", [])))
        if values
    }
    actions = {
        **pipeline_actions(ctx, overrides),
        "validate": lambda: validate(
            ctx,
            getattr(args, "file", ctx.out / "train.jsonl"),
            getattr(args, "allow_masked_assistant", False),
            getattr(args, "reference", None),
        ),
        "submit": lambda: submit(ctx, args.confirm, args.allow_warn),
        "rollout": lambda: rollout(ctx),
    }
    stage = args.command if args.command != "judge" else "judge-" + args.kind
    pipeline = (
        "ingest",
        "difficulty",
        "filter-tasks",
        "screen",
        "judge-task",
        "judge-trajectory",
        "select",
        "export",
        "validate",
    )
    stages = pipeline if stage == "run" else (stage,)
    with ctx.lock():
        for name in stages:
            try:
                report = actions[name]()
            except Exception as exc:
                # Some input failures precede hashing (e.g. a missing manifest).
                path = ctx.out / "reports" / (name + ".json")
                previous = read(path, {})
                if previous.get("error") != str(exc):
                    atomic(path, {"stage": name, "error": str(exc)})
                raise
            if report.get("valid") is False:
                return 1
    return 0


def main(argv=None, *, transport=None):
    args = parser().parse_args(argv)
    try:
        return execute(args, transport)
    except (OSError, ValueError, KeyError, RuntimeError, ExceptionGroup) as exc:
        print(f"tbstrain: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
