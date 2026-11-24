import asyncio
from pathlib import Path

from .core import dumps, read
from .llm import Client

CATEGORIES = {
    "hardcode_output",
    "special_case_inputs",
    "hidden_truth_access",
    "grader_tamper",
    "fake_result",
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


def assess_task(answer, prompt):
    if (
        answer.get("verdict") not in {"keep", "flag", "drop"}
        or answer.get("difficulty") not in {"easy", "medium", "hard"}
        or not answer.get("rationale")
        or not isinstance(answer.get("evidence"), list)
        or not answer["evidence"]
        or not all(quoted(q, prompt) for q in answer["evidence"])
    ):
        raise ValueError("invalid task judge response or evidence")
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
        if severity == "blocking":
            if not f.get("why"):
                problems.append(f"finding {i}: blocking finding needs why")
            if not quoted(f.get("quote"), prompt):
                problems.append(f"finding {i}: blocking finding quote is not an exact excerpt: {str(f.get('quote'))[:120]!r}")
        kept.append({**f, "severity": severity, "quote_verified": quoted(f.get("quote"), prompt)})
    if problems:
        raise ValueError("invalid trajectory finding or evidence: " + "; ".join(problems))
    return {**answer, "findings": kept, "quality_notes": notes}


def excerpt(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[omitted]\n" + text[-half:]


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


def judge(ctx, kind):
    ctx.check_sources("tasks")
    name = "judge-" + kind
    cfg = ctx.cfg["judge"]
    rubric = (Path(__file__).parent / "rubrics" / (kind + "-v1.md")).read_text("utf-8")
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
                files = [pkg / "tests/test_outputs.py", pkg / "tests/measure.py", pkg / "tests/test.sh"]
                graders = [p for p in files if p.is_file()]
                if not graders:
                    decisions[tid] = dict(task_id=tid, verdict="drop", reasons=["missing_grader"])
                    return
                inputs = [
                    ("instruction", statement),
                    *((p.name, p.read_text("utf-8")) for p in graders),
                    ("metrics", dumps(task["metrics"])),
                ]
                prompt = "\n".join(
                    title + "\n" + excerpt(value, limit // len(inputs)) for title, value in inputs
                )
                answer = await client.request(rubric, prompt, assess_task)
                decisions[tid] = dict(task_id=tid, **answer, reasons=[answer["rationale"]])
            else:
                key = r["trial_id"]
                records = read(ctx.out / "segments" / r["source"])
                answers = []
                parts = list(windows(records, limit))
                for number, w in enumerate(parts, 1):
                    prompt = (
                        "Task\n" + excerpt(statement, limit // 4)
                        + f"\nTranscript window {number} of {len(parts)}"
                        + (" (the trajectory continues in later windows; do not judge completeness from this window)"
                           if number < len(parts) else " (final window)")
                        + "\n" + w
                    )
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
