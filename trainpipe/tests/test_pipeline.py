from urllib.error import HTTPError

import pytest

from tbstrain.cli import main
from tbstrain.core import lines, read, write_lines
from tbstrain.tasks import difficulty_row
from tbstrain.select import mix
from tbstrain.llm import Client
from tbstrain.screen import inspect_segment
from tbstrain.wire import render
from tbstrain.native import codex, message, pi, segment

from conftest import codex_events, pi_events


def test_run_resume_and_artifacts(completed, judge_stub):
    cfg, out = completed
    assert {r["task_id"]: r["tier"] for r in lines(out / "difficulty.jsonl")} == {
        x: x for x in ("hard", "medium", "easy")
    }
    screens = {r["harness"]: r for r in lines(out / "screen.jsonl")}
    assert screens["claude-code"]["disposition"] == "strip"
    assert screens["claude-code"]["stripped"] == ["query"]
    assert screens["pi"]["cut_events"] == 2
    rows = list(lines(out / "train.jsonl"))
    assert len(rows) == 4
    first = next(r for r in rows if r["meta"]["compaction"])
    assert first["messages"][-1]["content"] == "<summary>\nTwo samples are available.\n</summary>"
    assert [m["tool_call_id"] for m in first["messages"] if m["role"] == "tool"] == ["sample-1", "sample-2"]
    assert [m["role"] for m in first["messages"][-3:]] == ["system", "user", "assistant"]
    assert any(m.get("reasoning_content") == "Compare the two synthetic samples." for m in first["messages"])
    text = (out / "train.jsonl").read_text()
    for forbidden in ["Replay should disappear", "/tests", "unfinished follow-up"]:
        assert forbidden not in text
    assert all(r["loss_mask"] == [int(m["role"] == "assistant") for m in r["messages"]] for r in rows)
    calls = len(judge_stub.calls)
    target = (out / "train.jsonl").resolve()
    assert main(["--config", str(cfg), "run"], transport=judge_stub) == 0
    assert len(judge_stub.calls) == calls and calls > 0
    assert target == (out / "train.jsonl").resolve()
    assert main(["rollout", "--config", str(cfg)]) == 0
    job = read(out / "rollout/job.json")
    assert (
        len(job["tasks"]) == 3 and job["n_attempts"] == 2 and job["agents"][0]["kwargs"] == {"effort": "high"}
    )
    assert not (cfg.parent / "more-jobs").exists()


def test_new_native_content_invalidates_stage(completed, judge_stub):
    cfg, out = completed
    old = (out / "screen.jsonl").resolve()
    path = next((cfg.parent / "run").rglob("rollout-*.jsonl"))
    events = codex_events()
    events[1]["payload"]["model"] = "other-model"
    write_lines(path, events)
    assert main(["run", "--config", str(cfg)], transport=judge_stub) == 0
    assert old != (out / "screen.jsonl").resolve()
    row = next(r for r in lines(out / "screen.jsonl") if r["harness"] == "codex")
    assert "model_not_trainable" in row["flags"] and row["disposition"] == "drop"


@pytest.mark.parametrize(
    "changes,tier,n,bad",
    [
        ([], "hard", 3, 0),
        ([dict(who="student", model="s", reward=1, timed_out=True, submitted=True)], "medium", 4, 0),
        ([dict(who="student", model="s", reward=1, outcome="infra")], "hard", 3, 1),
        ([dict(who="student", model="s", reward=0, exception="APIError", submitted=False)], "hard", 3, 1),
        ([dict(who="student", model="s", reward=0, result_type="timeout_no_output")], "hard", 3, 1),
    ],
)
def test_valid_attempts(changes, tier, n, bad):
    cal = [dict(who="student", model="s", reward=v) for v in [0, 1, 0]] + [
        dict(who="teacher", model="t", reward=1)
    ]
    r = difficulty_row(dict(task_id="toy", calib_runs=cal + changes), [], {})
    assert (r["tier"], r["student_attempts"], r["invalid"]) == (tier, n, bad)


def test_duplicate_and_holdout(workspace, judge_stub):
    manifest = workspace.parent / "run/release/manifest.jsonl"
    rows = list(lines(manifest))
    rows[0]["doi"] = "https://doi.org/10.example/toy"
    rows[1]["doi"] = "10.example/toy"
    rows[2]["title"] = rows[0]["title"]
    write_lines(manifest, rows)
    for stage in ("ingest", "difficulty", "filter-tasks"):
        assert main([stage, "--config", str(workspace)]) == 0
    filtered = list(lines(workspace.parent / "out/filtered.jsonl"))
    assert [r["task_id"] for r in filtered if r["keep"]] == ["easy"]
    assert all("duplicate:easy" in r["reasons"] for r in filtered if not r["keep"])
    with workspace.open("a") as f:
        f.write('\n[filter]\nexclude=["10.example/toy","hard"]\n')
    assert main(["filter-tasks", "--config", str(workspace)]) == 0
    assert all("holdout" in r["reasons"] for r in lines(workspace.parent / "out/filtered.jsonl"))


@pytest.mark.parametrize(
    "tool,args,output,error,expected",
    [
        ("read", {"path": "/work/data"}, "sample", False, "keep"),
        ("Bash", {"command": "find / -name truth"}, "No results", False, "strip"),
        ("Read", {"file_path": "/tests/grader"}, "some value", False, "drop"),
        ("read", {"path": "/work/file"}, "/logs/verifier/score", False, "drop"),
        ("WebFetch", {"url": "https://science.example"}, "article", False, "drop"),
        ("exec_command", {"cmd": "curl https://science.example"}, "", False, "drop"),
        ("Bash", {"command": "pip install toy"}, "Network disabled", True, "strip"),
    ],
)
def test_screen_calls(tool, args, output, error, expected):
    record = segment(
        [
            message("user", "Task"),
            message("assistant", [dict(type="tool_use", id="x", name=tool, input=args)]),
            message("user", [dict(type="tool_result", tool_use_id="x", content=output, is_error=error)]),
            message("assistant", "Done"),
        ]
    )
    flags, fatal, strip = inspect_segment(record, True)
    assert ("drop" if fatal else "strip" if strip else "keep") == expected
    if expected == "strip":
        assert len(render(record, {})["messages"]) == 2


def test_shares():
    shares = {"hard": 0.45, "medium": 0.40, "easy": 0.15}
    from collections import Counter

    result = mix({k: [k] * n for k, n in zip(shares, [9, 8, 50])}, shares)
    counts = Counter(result)
    assert all(abs(counts[k] / len(result) - v) < 1 / len(result) for k, v in shares.items())


def test_judge_retry_cache(tmp_path):
    import asyncio

    calls = []

    async def fake(*args):
        calls.append(args)
        if len(calls) == 1:
            raise HTTPError("https://judge.example", 429, "busy", {}, None)
        return {"content": [{"type": "text", "text": '{"ok":true}'}], "usage": {"input_tokens": 10}}

    cfg = {
        "concurrency": 128,
        "model": "m",
        "base_url": "https://judge.example",
        "backoff": 0,
        "input_tokens_per_million": 1,
    }
    for _ in range(2):
        client = Client(cfg, tmp_path, fake)

        async def one(_):
            assert await client.request("rubric-v1", "prompt", lambda a, p: a) == {"ok": True}

        asyncio.run(client.map(range(128), one))
    assert len(calls) == 2 and client.stats["cache_hits"] == 128


def test_codex_missing_compaction_evidence():
    events = codex_events() + [{"type": "compacted", "payload": {"message": "Summary"}}]
    with pytest.raises(ValueError, match="without_captured_calls"):
        codex(events, {}, {})


def test_pi_reject_interior_partial():
    events = pi_events()
    events[3]["event"]["message"]["stopReason"] = "aborted"
    events.append(pi_events()[3])
    with pytest.raises(ValueError, match="interior"):
        pi(events, {}, {})


def test_default_selection_uses_easy_cap(workspace, judge_stub):
    workspace.write_text(workspace.read_text().replace('mode = "shares"\n', ''))
    assert main(["run", "--config", str(workspace)], transport=judge_stub) == 0
    assert sorted(r["tier"] for r in lines(workspace.parent / "out/selection.jsonl")) == ["hard", "medium"]


@pytest.mark.parametrize("mode", ["drop_row", "drop_trajectory", "mask_after"])
def test_export_image_policy(workspace, judge_stub, mode):
    from conftest import block, cc_message, claude_events

    # Keep the other tiers selectable even if this trajectory is entirely dropped.
    config = workspace.read_text().replace('mode = "shares"', 'mode = "easy_cap"')
    if mode != "drop_row":
        config = config.replace("[export]", f'[export]\nimages="{mode}"')
    workspace.write_text(config)
    path = next((workspace.parent / "run/calib-jobs/hard-0").rglob("tiny.jsonl"))
    events = claude_events()
    events[3]["message"]["content"][0]["content"] = [block("tile"), {"type": "image", "data": "toy-bytes"}]
    follow = cc_message("assistant", [block("Describe the tile.")], uuid="after-image",
                        parentUuid=events[3]["uuid"], version="1")
    events[4]["parentUuid"] = follow["uuid"]
    events.insert(4, follow)
    write_lines(path, events)

    assert main(["run", "--config", str(workspace)], transport=judge_stub) == 0
    out = workspace.parent / "out"
    rows = {r["meta"]["segment"]: r for r in lines(out / "train.jsonl") if r["meta"]["harness"] == "claude-code"}
    dropped = {r["segment"] for r in lines(out / "export.ledger.jsonl") if r["reason"] == "image_content"}
    if mode == "drop_row":
        assert set(rows) == {1} and dropped == {0}
    elif mode == "drop_trajectory":
        assert not rows and dropped == {0, 1}
    else:
        assert set(rows) == {0, 1} and not dropped
        assert [v for m, v in zip(rows[0]["messages"], rows[0]["loss_mask"]) if m["role"] == "assistant"] == [1, 0, 0]
        assert rows[1]["loss_mask"][-1] == 1
    counts = dict(rows=1, dropped_rows=len(dropped), dropped_trajectories=int(mode == "drop_trajectory"),
                  masked_turns=2 if mode == "mask_after" else 0)
    assert read(out / "reports/export.json")["image_content"] == read(out / "reports/select.json")["image_content"] == counts
    assert "toy-bytes" not in (out / "train.sources.jsonl").read_text()


@pytest.mark.parametrize("harness", ["claude", "codex", "pi"])
@pytest.mark.parametrize("role", ["user", "tool"])
def test_native_image_context(harness, role, tmp_path):
    from conftest import block, cc_message, link_events, toy_profile
    from tbstrain.native import claude
    from tbstrain.wire import image_dependents

    image = dict(type="image", mimeType="image/png", data="dG95")
    if harness == "claude":
        events = link_events([
            cc_message("user", [image] if role == "user" else "Question"),
            cc_message("assistant", [dict(type="tool_use", id="a", name="sample", input={})]),
            cc_message("user", [dict(type="tool_result", tool_use_id="a", content=[image] if role == "tool" else "ok")]),
            cc_message("assistant", [block("Done.")]),
        ])
        records, _, _ = claude(events, {"profile": toy_profile()}, dict(trial=str(tmp_path), agent_version="1"))
    elif harness == "codex":
        events = codex_events()
        content = [dict(type="input_image", image_url="data:image/png;base64,dG95")]
        if role == "user":
            events[2]["payload"]["content"] = content
        else:
            events[5]["payload"]["output"] = content
        records, _, _ = codex(events, {}, {})
    else:
        events = [
            dict(type="message", message=dict(role="user", content=[image] if role == "user" else "Question")),
            dict(type="message", message=dict(role="assistant", stopReason="toolUse",
                 content=[dict(type="toolCall", id="a", name="sample", arguments={})])),
            dict(type="message", message=dict(role="toolResult", toolCallId="a", content=[image] if role == "tool" else "ok")),
            dict(type="message", message=dict(role="assistant", content=[block("Done.")], stopReason="stop")),
        ]
        records, _, _ = pi(events, {"profile": toy_profile()}, {})
    messages = records[0]["messages"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert list(image_dependents(messages)) == assistants[(0 if role == "user" else 1):]
    assert "dG95" not in str(records)
