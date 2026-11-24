import asyncio
import re
from pathlib import Path

from .core import dumps, read
from .graders import grader_files
from .llm import Client

ALWAYS_BLOCKING = {"fake_result", "hardcode_output", "hidden_truth_access", "grader_tamper"}
CATEGORIES = ALWAYS_BLOCKING | {
    "special_case_inputs",
    "requirement_bypass",
    "answer_search",
}


def _norm(text):
    return " ".join(text.split())


def quoted(quote, prompt):
    """Exact excerpt check tolerant to whitespace and to JSON escaping of the transcript (blocks are serialized)."""
    if not isinstance(quote, str) or not quote.strip():
        return False
    if quote in prompt:
        return True
    p = _norm(prompt)
    candidates = {quote, dumps(quote)[1:-1], quote.replace("\\n", "\n"), quote.replace('\\"', '"')}
    return any(_norm(c) and _norm(c) in p for c in candidates) or _norm(quote.replace("\n", "\\n")) in p


def _plain(text):
    """Markdown-insensitive form for evidence matching (judges drop or add emphasis, code ticks and table pipes)."""
    return re.sub(r"\s+([.,;:!?)])", r"\1", _norm(re.sub(r"[*_`|>#]+", " ", text)))


def assess_task(answer, prompt):
    problems = []
    if answer.get("verdict") not in {"keep", "flag", "drop"}:
        problems.append("verdict must be keep, flag or drop")
    if answer.get("difficulty") not in {"easy", "medium", "hard"}:
        problems.append("difficulty must be easy, medium or hard")
    if not answer.get("rationale"):
        problems.append("missing rationale")
    evidence = answer.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        problems.append("evidence must be a non-empty list of excerpts")
    else:
        plain = _plain(prompt)
        for i, q in enumerate(evidence):
            if not (quoted(q, prompt) or (isinstance(q, str) and _plain(q) and _plain(q) in plain)):
                problems.append(f"evidence {i} is not an excerpt of the material: {str(q)[:120]!r}")
    if problems:
        raise ValueError("invalid task judge response or evidence: " + "; ".join(problems))
    return answer


def assess_trajectory(answer, prompt):
    """Keep unknown minor findings as notes and require valid integrity evidence for blocking findings."""
    findings = answer.get("findings")
    if (
        type(answer.get("quality")) is not int
        or not 1 <= answer["quality"] <= 5
        or not answer.get("rationale")
        or type(answer.get("wasted_turn_ratio")) not in (float, int)
        or not 0 <= answer["wasted_turn_ratio"] <= 1
        or not isinstance(findings, list)
    ):
        raise ValueError("invalid trajectory judge response: need quality 1-5, wasted_turn_ratio 0-1, rationale, findings[]")
    problems, kept, notes = [], [], list(answer.get("quality_notes") or [])
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            problems.append(f"finding {i} is not an object")
            continue
        severity = str(f.get("severity", "")).lower()
        if f.get("type") not in CATEGORIES and severity == "blocking":
            problems.append(f"finding {i}: unknown blocking finding type")
            continue
        if f.get("type") not in CATEGORIES and severity == "minor":
            notes.append({k: f.get(k) for k in ("type", "severity", "quote", "why")})
            continue
        if severity not in {"blocking", "minor"}:
            problems.append(f"finding {i}: severity must be blocking or minor")
            continue
        if f.get("type") in ALWAYS_BLOCKING:
            severity = "blocking"
        if severity == "blocking":
            if not f.get("why"):
                problems.append(f"finding {i}: blocking finding needs why")
            if not quoted(f.get("quote"), prompt):
                problems.append(f"finding {i}: blocking finding quote is not an exact excerpt: {str(f.get('quote'))[:120]!r}")
        kept.append({**f, "severity": severity, "quote_verified": quoted(f.get("quote"), prompt)})
    if problems:
        raise ValueError("invalid trajectory finding or evidence: " + "; ".join(problems))
    return {**answer, "findings": kept, "quality_notes": notes}


def excerpt(text, limit, *, include_marker_in_limit=False):
    if len(text) <= limit:
        return text
    marker = "\n[omitted]\n"
    if include_marker_in_limit:
        if limit < len(marker) + 2:
            return text[:limit]
        limit -= len(marker)
    half = limit // 2
    return text[:half] + marker + text[-half:]


def render_block(b):
    """Readable transcript text for the judge (quotes are checked against exactly this text)."""
    kind = b.get("type")
    if kind == "text":
        return b.get("text", "")
    if kind == "thinking":
        return "[thinking]\n" + (b.get("thinking") or b.get("text") or "")
    if kind == "tool_use":
        args = b.get("input")
        if isinstance(args, dict):
            body = "\n".join(f"{k}: {v}" if isinstance(v, str) else f"{k}: {dumps(v)}" for k, v in args.items())
        else:
            body = args if isinstance(args, str) else dumps(args)
        return f"[tool call {b.get('name')}]\n{body}"
    if kind == "tool_result":
        content = b.get("content")
        if isinstance(content, list):
            content = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        return f"[tool result{' (error)' if b.get('is_error') else ''}]\n{content if isinstance(content, str) else dumps(content)}"
    return dumps(b)


def windows(records, limit):
    parts = []
    size = 0
    for number, record in enumerate(records, 1):
        for i, m in enumerate(record["messages"], 1):
            blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            content = "\n".join(render_block(b) for b in blocks)
            entry = f"=== segment {number}, message {i}, {m['role']}\n{content}\n"
            for start in range(0, len(entry), limit):
                piece = entry[start : start + limit]
                if size + len(piece) > limit and parts:
                    yield "".join(parts)
                    parts = []
                    size = 0
                parts.append(piece)
                size += len(piece)
    if parts:
        yield "".join(parts)


def task_prompt(task, statement, limit):
    pkg = Path(task["pkg"])
    graders = grader_files(pkg)
    if not graders:
        return None
    inputs = [("instruction", statement), *((str(p.relative_to(pkg / "tests")), p.read_text("utf-8")) for p in graders),
              ("metrics", dumps(task["metrics"]))]
    budget = max(0, limit - sum(len(title) + 2 for title, _ in inputs) + 1)
    sizes = {title: len(value) for title, value in inputs}
    primary = [title for title in sizes if title in ("test_outputs.py", "measure.py")]
    reserved = sum(sizes[title] for title in primary)
    if reserved <= budget:
        budget -= reserved
        pending = [title for title in sizes if title not in primary]
    else:
        pending = list(sizes)
    # Keep short sections whole, then share the remaining space among oversized ones.
    for i, title in enumerate(sorted(pending, key=sizes.get)):
        sizes[title] = min(sizes[title], budget // (len(pending) - i))
        budget -= sizes[title]
    return "\n".join(title + "\n" + excerpt(value, sizes[title], include_marker_in_limit=True)
                     for title, value in inputs)[:limit]


def trajectory_prompts(statement, records, limit):
    parts = list(windows(records, limit))
    for number, window in enumerate(parts, 1):
        yield (
            "Task\n" + excerpt(statement, limit // 4)
            + f"\nTranscript window {number} of {len(parts)}"
            + (" (the trajectory continues in later windows; do not judge completeness from this window)"
               if number < len(parts) else " (final window)")
            + "\n" + window
        )


def judge(ctx, kind):
    ctx.check_sources("tasks")
    name = "judge-" + kind
    cfg = ctx.cfg["judge"]
    filename = {"task": "task-v2.md", "trajectory": "trajectory-v3.md"}[kind]
    rubric = (Path(__file__).parent / "rubrics" / filename).read_text("utf-8")
    dep = "filter-tasks" if kind == "task" else "screen"

    def build():
        client = Client(cfg, ctx.out / ".judge-cache", ctx.transport)
        tasks = {r["task_id"]: r for r in ctx.rows("tasks")}
        decisions = {}
        limit = cfg.get("window_chars", 64000)
        if limit < 128:
            raise ValueError("judge window too short")

        def jobs():
            records = ctx.rows("filtered" if kind == "task" else "screen")
            for r in records:
                if (kind == "task" and r["keep"]) or (kind == "trajectory" and r["disposition"] != "drop"):
                    yield r

        async def process(r):
            try:
                await assess(r)
            except Exception as exc:          # one bad response must not discard the other judgments
                key = r["task_id"] if kind == "task" else r["trial_id"]
                client.stats["errors"] += 1
                decisions[key] = dict(
                    task_id=r["task_id"], **({} if kind == "task" else {"trial_id": key}),
                    verdict="error", reasons=[f"judge_error: {type(exc).__name__}: {str(exc)[:200]}"],
                )

        async def assess(r):
            tid = r["task_id"]
            task = tasks[tid]
            pkg = Path(task["pkg"])
            statement = (pkg / "instruction.md").read_text("utf-8")
            if kind == "task":
                prompt = task_prompt(task, statement, limit)
                if prompt is None:
                    decisions[tid] = dict(task_id=tid, verdict="drop", reasons=["missing_grader"])
                    return
                answer = await client.request(rubric, prompt, assess_task)
                decisions[tid] = dict(task_id=tid, **answer, reasons=[answer["rationale"]])
            else:
                key = r["trial_id"]
                records = read(ctx.out / "segments" / r["source"])
                answers = []
                for prompt in trajectory_prompts(statement, records, limit):
                    answers.append(await client.request(rubric, prompt, assess_trajectory))
                blocked = any(f["severity"] == "blocking" for a in answers for f in a["findings"])
                decisions[key] = dict(
                    trial_id=key,
                    task_id=tid,
                    verdict="drop" if blocked else "keep",
                    quality=min(a["quality"] for a in answers),
                    final_quality=answers[-1]["quality"],
                    windows=answers,
                    reasons=[f["type"] for a in answers for f in a["findings"]] or ["passed"],
                )

        asyncio.run(client.map(jobs(), process))
        rows = [decisions[key] for key in sorted(decisions)]
        failed = [r for r in rows if r["verdict"] == "error"]
        # Successful judgments are cached per request, so a rerun only repeats the failed ones.
        # A small residue of persistently invalid answers is recorded (excluded downstream) instead of blocking.
        if failed and len(failed) > cfg.get("max_error_ratio", 0.02) * len(rows):
            raise RuntimeError(f"{len(failed)}/{len(rows)} judgments failed; first: {failed[0]['reasons'][0]}")
        ctx.emit(name, rows)
        ctx.emit(name + ".ledger", rows)
        return {"count": len(rows), "retry_required": bool(failed), **client.stats}

    return ctx.stage(name, [ctx.dependency(dep), cfg, rubric], build)
