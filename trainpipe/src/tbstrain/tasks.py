from collections import defaultdict
from pathlib import Path
import re
import unicodedata

from .core import file_hash


def valid(r):
    kind = str(r.get("result_type") or "").lower()
    exc = str(r.get("exception") or "").lower()
    if (
        r.get("outcome") == "infra"
        or "infra" in kind
        or any(s in exc for s in ("setup", "environment", "connection", "cancelled", "installerror"))
    ):
        return False
    if (
        "api" in exc
        or "api" in kind
        or "ratelimit" in exc
        or r.get("api_status") in (401, 402, 403, 429, 529)
    ) and r.get("delivered", r.get("submitted")) is not True:
        return False
    timeout = r.get("timed_out") or "timeout" in kind or "timeout" in exc
    if timeout and (kind == "timeout_no_output" or r.get("delivered", r.get("submitted")) is False):
        return False
    if timeout and kind != "timeout_with_output" and r.get("delivered", r.get("submitted")) is not True:
        return False
    return type(r.get("reward")) in (int, float)


def difficulty_row(task, trials, cfg):
    cal = task["calib_runs"]
    roles = {"student": {"student", "glm"}, "teacher": {"teacher", "opus"}}
    models = {}
    for role, aliases in roles.items():
        explicit = cfg.get(role + "_model")
        inferred = {r.get("model") for r in cal if r.get("who") in aliases and r.get("model")}
        if len(inferred) > 1 and not explicit:
            raise ValueError(f"set {role}_model for mixed calibration models")
        models[role] = explicit or next(iter(inferred), None)
    # A partially recovered multi-attempt job replaces only the corresponding summaries.
    remaining = list(cal)
    for trial in trials:
        candidates = [
            r
            for r in remaining
            if r.get("job_name")
            and r["job_name"] == trial["job_name"]
            and (not r.get("trial") or r["trial"] == trial.get("trial_name"))
            and (trial.get("calib_n") is None or r.get("n") == trial["calib_n"])
        ]
        if candidates:
            match = next((r for r in candidates if r.get("reward") == trial.get("reward")), candidates[0])
            remaining.remove(match)
    attempts = remaining + trials
    scored = [r for r in attempts if valid(r)]
    counts = {}
    for role, model in models.items():
        subset = [r for r in scored if (r.get("model") == model if model else r.get("who") in roles[role])]
        counts[role + "_attempts"] = len(subset)
        counts[role + "_passes"] = sum(r["reward"] == 1 for r in subset)
    n = counts["student_attempts"]
    rate = counts["student_passes"] / n if n else None
    passed = sum(r["reward"] == 1 for r in scored)
    lo, hi = cfg.get("hard_max", 1 / 3), cfg.get("medium_max", 2 / 3)
    if not 0 <= lo < hi <= 1:
        raise ValueError("invalid tier boundaries")
    tier = "unclassified"
    if not passed:
        tier = "unsolved"
    elif rate is not None:
        if rate <= lo:
            if counts["teacher_passes"]:
                tier = "hard"
        else:
            tier = "medium" if rate <= hi else "easy"
    return dict(
        task_id=task["task_id"],
        tier=tier,
        rate=rate,
        passes=passed,
        **counts,
        teacher_passed=bool(counts["teacher_passes"]),
        models=models,
        attempts=len(attempts),
        valid=len(scored),
        invalid=len(attempts) - len(scored),
    )


def difficulty(ctx):
    cfg = ctx.cfg.get("difficulty", {})

    def build():
        groups = defaultdict(list)
        for r in ctx.rows("trials"):
            if r["current_package"]:
                groups[r["task_id"]].append(r)
        rows = [difficulty_row(t, groups[t["task_id"]], cfg) for t in ctx.rows("tasks")]
        ctx.emit("difficulty", rows)
        ctx.emit("difficulty.ledger", (dict(**r, reason=r["tier"]) for r in rows))
        return {"count": len(rows)}

    return ctx.stage("difficulty", [cfg, ctx.dependency("ingest")], build)


def tokens(title):
    return set(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", title or "").casefold()))


def doi_key(doi):
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", (doi or "").strip().casefold())


def filter_tasks(ctx):
    from importlib import import_module

    try:
        rules = import_module("tbsgen.statement")
        requires_assumptions = import_module("tbsgen.spec")._requires_assumptions
    except ModuleNotFoundError as exc:
        if exc.name not in {"tbsgen", "tbsgen.statement"}:
            raise
        rules = None
    ctx.check_sources("tasks")
    cfg = ctx.cfg.get("filter", {})
    holdout = cfg.get("exclude", [])
    excluded_dois = {doi_key(v) for v in holdout}
    excluded_titles = [tokens(v) for v in holdout]

    def near(a, b):
        return bool(a and b) and len(a & b) / len(a | b) >= 0.8

    def build():
        counts = {r["task_id"]: r for r in ctx.rows("difficulty")}
        rows = []
        kept = []
        for task in sorted(ctx.rows("tasks"), key=lambda t: (-counts[t["task_id"]]["passes"], t["task_id"])):
            why = []
            tid = task["task_id"]
            title, doi = tokens(task["title"]), doi_key(task["doi"])
            if not task["released"]:
                why.append("unreleased")
            if task["verdict"] not in cfg.get("review_allow", ["pass"]):
                why.append("review_verdict")
            if not task["label_trustworthy"]:
                why.append("untrustworthy")
            if (
                tid in holdout
                or (doi and doi in excluded_dois)
                or any(near(title, s) for s in excluded_titles)
            ):
                why.append("holdout")
            statement = Path(task["pkg"]) / "instruction.md"
            if not statement.is_file() or not statement.read_text("utf-8").strip():
                why.append("missing_statement")
            elif rules:
                content = statement.read_text("utf-8")
                if (
                    rules.statement_clean(content)
                    or rules.threshold_format(content, task["thresholds"])
                    or rules.assumptions_stated(
                        content,
                        task["assumptions"],
                        any(requires_assumptions(v) for v in task["prompt_versions"]),
                    )
                ):
                    why.append("statement_hygiene")
            for other_id, other_title, other_doi in kept:
                if (doi and doi == other_doi) or near(title, other_title):
                    why.append("duplicate:" + other_id)
                    break
            if not why:
                kept.append((tid, title, doi))
            rows.append(
                dict(
                    task_id=tid,
                    keep=not why,
                    reasons=why or ["passed"],
                    hygiene="checked" if rules else "skipped: tbsgen.statement unavailable",
                )
            )
        ctx.emit("filtered", rows)
        ctx.emit("filter-tasks.ledger", rows)
        return {"count": len(kept), "evaluated": len(rows)}

    return ctx.stage(
        "filter-tasks",
        [ctx.dependency("difficulty"), cfg, file_hash(Path(rules.__file__)) if rules else None],
        build,
    )
