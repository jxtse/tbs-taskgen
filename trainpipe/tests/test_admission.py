from types import SimpleNamespace

import pytest

from tbstrain.cli import main
from tbstrain.core import atomic, lines, read, write_lines
from tbstrain.native import codex, message, segment
from tbstrain.screen import remove_hints
from tbstrain.select import select
from tbstrain.tasks import difficulty_row
from tbstrain.wire import check, render
from conftest import codex_events, link_events, toy_profile


def test_captured_codex_windows(tmp_path):
    events = codex_events()

    def output(text):
        return dict(type="message", role="assistant", content=[dict(type="output_text", text=text)])

    events[6:6] = [
        dict(type="response_item", payload=output("Intermediate summary.")),
        events[-1],
        dict(
            type="compacted",
            timestamp="2025-01-01T00:02:30Z",
            payload=dict(window_id="w", message="Intermediate summary."),
        ),
        dict(
            type="response_item",
            payload=dict(
                type="message", role="user", content=[dict(type="input_text", text="Intermediate summary.")]
            ),
        ),
    ]
    schema = dict(
        type="function", name="sample", parameters={"type": "object", "properties": {"n": {"type": "number"}}}
    )
    user = events[2]["payload"]
    action = events[3]["payload"]
    result = events[5]["payload"]
    calls = []
    for i, (inputs, target) in enumerate(
        [
            ([user], action),
            (
                [
                    user,
                    action,
                    result,
                    dict(type="message", role="user", content=[dict(type="input_text", text="Summarize.")]),
                ],
                output("Intermediate summary."),
            ),
            (
                [
                    dict(
                        type="message",
                        role="user",
                        content=[dict(type="input_text", text="Intermediate summary.")],
                    )
                ],
                output("The synthetic slope is one."),
            ),
        ],
        1,
    ):
        calls.append(
            dict(
                participant_id="root",
                call_id=f"call-{i}",
                request_seq=i,
                started_at=f"2025-01-01T00:0{i}:00Z",
                protocol="openai-responses",
                export_status="available",
                context={"tools": [schema], "input": inputs, "model": "student-model"},
                output={"output": [target]},
            )
        )
    path = tmp_path / "calls.jsonl"
    write_lines(path, calls)
    atomic(tmp_path / "manifest.json", {"participants": [{"role": "root", "participant_id": "root"}]})
    records, models, notes = codex(events, {}, {"capture": str(path)})
    assert notes["captured_windows"] == 2
    converted = [render(r, {}) for r in records]
    assert all(not check(row) for row in converted)
    assert converted[0]["messages"][-1]["content"] == "Intermediate summary."
    assert sum(len(m.get("tool_calls", [])) for row in converted for m in row["messages"]) == 1
    calls[-1]["output"]["output"] = [output("Mismatched response.")]
    write_lines(path, calls)
    with pytest.raises(ValueError, match="correspond"):
        codex(events, {}, {"capture": str(path)})


def test_selection_caps_diversity_and_quality(tmp_path):
    from tbstrain.core import Context

    cfg = tmp_path / "config.toml"
    cfg.write_text('out="out"\n[select]\nmode="shares"\nshares={easy=1}\nper_task=2\nmax_token_share=0.6\n')
    ctx = Context(cfg)
    ctx.out.mkdir()

    def data(name, rows):
        write_lines(ctx.out / (name + ".jsonl"), rows)

    screened = []
    trajectory = []
    for tid in ["one", "two"]:
        for number, (kind, n, quality) in enumerate(
            [("pi", 9, 5), ("pi", 8, 5), ("codex", 11, 5), ("pi", 3, 3)]
        ):
            uid = f"{tid}-{number}"
            screened.append(
                dict(
                    task_id=tid,
                    trial_id=uid,
                    harness=kind,
                    disposition="keep",
                    supervised_tokens=n,
                    within_limit=1,
                )
            )
            trajectory.append(dict(trial_id=uid, quality=quality, verdict="keep"))
    data("tasks", [dict(task_id=t, domain="synthetic") for t in ("one", "two")])
    data("difficulty", [dict(task_id=t, tier="easy") for t in ("one", "two")])
    data("judge-task", [dict(task_id=t, verdict="keep") for t in ("one", "two")])
    data("judge-trajectory", trajectory)
    data("screen", screened)
    for stage in ("difficulty", "screen", "judge-task", "judge-trajectory"):
        atomic(ctx.out / "reports" / (stage + ".json"), {"key": stage})
    select(ctx)
    chosen = list(ctx.rows("selection"))
    assert sorted(r["trial_id"] for r in chosen) == ["one-1", "one-2", "two-1", "two-2"]
    assert {r["reason"] for r in ctx.rows("select.ledger")} == {"selected", "easy_quality", "per_task_cap"}
    ctx.cfg["select"]["max_token_share"] = 0.49
    select(ctx)
    assert not list(ctx.rows("selection"))


def test_unsolved_and_unclassified():
    task = dict(task_id="t", calib_runs=[dict(who="student", model="s", reward=0)])
    assert difficulty_row(task, [], {})["tier"] == "unsolved"
    assert (
        difficulty_row(task, [dict(job_name="extra", model="other", reward=1)], {})["tier"] == "unclassified"
    )


def test_hint_timeout_and_native_end(workspace, judge_stub):
    trial = workspace.parent / "run/calib-jobs/easy-0/one"
    path = trial / "result.json"
    result = read(path)
    result.update(exception_info={"exception_type": "AgentTimeoutError"}, delivered=True, hint_run=True)
    atomic(path, result)

    def screen():
        for cmd in ["ingest", "difficulty", "filter-tasks", "screen"]:
            assert main([cmd, "--config", str(workspace)]) == 0
        return next(r for r in lines(workspace.parent / "out/screen.jsonl") if r["harness"] == "pi")

    assert {"hint_run", "timeout"} <= set(screen()["flags"])
    result.pop("hint_run")
    atomic(path, result)
    with workspace.open("a") as f:
        f.write("\n[screen]\nallow_timeout=true\n")
    assert screen()["disposition"] == "keep"
    result["exception_info"] = None
    result["finished_at"] = None
    atomic(path, result)
    assert "unclean_end" in screen()["flags"]


def test_hint_removal_requires_clean_model_text():
    src = segment([message("user", "Solve. Mentor text."), message("assistant", "Done")])
    assert remove_hints(src, {"hint_texts": ["Mentor text."]}) == 1
    assert src["messages"][0]["content"][0]["text"] == "Solve. "
    src["messages"][1]["content"][0]["text"] = "Mentor text."
    with pytest.raises(ValueError, match="residue"):
        remove_hints(src, {"hint_texts": ["Mentor text."]})


def test_blocking_judge_overrides_passing_score(workspace, judge_stub):
    from tbstrain.core import dumps

    async def blocking(url, headers, body, timeout):
        response = await judge_stub(url, headers, body, timeout)
        if "wasted_turn_ratio" in body["system"]:
            answer = {
                "quality": 4,
                "wasted_turn_ratio": 0,
                "rationale": "Synthetic test verdict.",
                "findings": [
                    dict(
                        type="fake_result",
                        severity="blocking",
                        quote="synthetic slope",
                        why="Synthetic finding.",
                    )
                ],
            }
            response["content"][0]["text"] = dumps(answer)
        return response

    assert main(["run", "--config", str(workspace)], transport=blocking) == 1
    out = workspace.parent / "out"
    assert all(r["verdict"] == "drop" for r in lines(out / "judge-trajectory.jsonl"))
    assert not list(lines(out / "selection.jsonl"))
    assert not read(out / "reports/validate.json")["valid"]


def test_pr42_optional_rules(workspace, monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "tbsgen.statement",
        SimpleNamespace(
            statement_clean=lambda text: ["synthetic hygiene finding"],
            threshold_format=lambda text, thresholds: [],
            assumptions_stated=lambda text, assumptions, require_long_quote: [],
            __file__=__file__,
        ),
    )
    monkeypatch.setitem(sys.modules, "tbsgen.spec", SimpleNamespace(_requires_assumptions=lambda v: False))
    for stage in ("ingest", "difficulty", "filter-tasks"):
        assert main([stage, "--config", str(workspace)]) == 0
    assert all("statement_hygiene" in r["reasons"] for r in lines(workspace.parent / "out/filtered.jsonl"))


def test_calibration_summary_survives_partial_job():
    attempts = [
        dict(job_name="multi", n=n, who="student", model="s", reward=score)
        for n, score in enumerate([1, 0, 0], 1)
    ]
    native = dict(job_name="multi", calib_n=1, model="s", reward=1)
    result = difficulty_row(dict(task_id="tiny", calib_runs=attempts), [native], {})
    assert result["student_attempts"] == 3 and result["student_passes"] == 1


def test_resume_prefix_audited_but_never_trained(tmp_path):
    from conftest import cc_message, block
    from tbstrain.native import claude
    from tbstrain.screen import inspect_omitted_events

    events = [
        cc_message("user", "Estimate a slope."),
        cc_message("assistant", [block("Discarded method.")]),
        cc_message("user", "Resume here."),
        cc_message("assistant", [block("A fresh answer.")]),
    ]
    link_events(events)
    cfg = {"resume_marker": "Resume here.", "profile": toy_profile(), "train_on_what": "all_assistant_messages"}
    records, _, details = claude(events, cfg, {"trial": str(tmp_path), "agent_version": "1"})
    assert details["resume_prefix_events"] == 3
    assert [m["content"] for m in render(records[0], {})["messages"]] == [
        "Synthetic instructions.",
        "Estimate a slope.",
        "Resume here.",
        "A fresh answer.",
    ]
    events.insert(
        2, cc_message("user", [dict(type="tool_result", tool_use_id="old", content="verifier reward: 1")])
    )
    assert "hidden_access" in inspect_omitted_events(events, "claude-code", True, ["/work"])[1]


def test_reject_stale_native_input(completed):
    cfg, out = completed
    native = next((cfg.parent / "run").rglob("rollout-*.jsonl"))
    with native.open("a") as stream:
        stream.write("\n")
    assert main(["screen", "--config", str(cfg)]) == 1
    assert "input changed since ingest" in read(out / "reports/screen.json")["error"]


def test_proxy_model_fallback(workspace):
    proxy = workspace.parent / "run/calib-jobs/medium-0/one/agent/model-proxy"
    proxy.mkdir()
    write_lines(proxy / "requests-tiny.jsonl", [dict(status=200, response_models=["other-model"])])
    for stage in ("ingest", "difficulty", "filter-tasks", "screen"):
        assert main([stage, "--config", str(workspace)]) == 0
    screened = next(r for r in lines(workspace.parent / "out/screen.jsonl") if r["task_id"] == "medium")
    assert screened["disposition"] == "drop" and "model_mismatch" in screened["flags"]


def test_judge_interruption_resumes_per_request(workspace, judge_stub):
    with workspace.open("r+") as stream:
        content = stream.read().replace("concurrency = 128", "concurrency = 1")
        stream.seek(0)
        stream.write(content)
        stream.truncate()
    for item in (workspace.parent / "run/items").iterdir():
        statement = item / "pkg/instruction.md"
        statement.write_text(statement.read_text() + " Series " + item.name + ".")
    for stage in ("ingest", "difficulty", "filter-tasks"):
        assert main([stage, "--config", str(workspace)]) == 0

    async def interrupt(*args):
        if len(judge_stub.calls) == 1:
            raise ValueError("synthetic interruption")
        return await judge_stub(*args)

    args = ["judge", "--kind", "task", "--config", str(workspace)]
    assert main(args, transport=interrupt) == 1
    assert len(judge_stub.calls) == 1
    assert main(args, transport=judge_stub) == 0
    assert len(judge_stub.calls) == 3
    assert main(args, transport=judge_stub) == 0 and len(judge_stub.calls) == 3


def test_pi_native_tool_exchange_and_tail():
    from tbstrain.native import pi

    def entry(role, content, **extra):
        return {"type": "message", "message": {"role": role, "content": content, **extra}}

    events = [
        entry("user", "Compare two synthetic readings."),
        entry(
            "assistant",
            [dict(type="toolCall", name="sample", id="a", arguments={"n": 1})],
            model="student-model",
            stopReason="toolUse",
        ),
        entry("toolResult", "1", toolCallId="a", toolName="sample"),
        entry(
            "assistant",
            [dict(type="thinking", thinking="Both are equal."), dict(type="text", text="Equal.")],
            model="student-model",
            stopReason="stop",
        ),
        entry("user", "One more reading."),
    ]
    records, _, details = pi(events, {"profile": toy_profile()}, {})
    row = render(records[0], {})
    assert not check(row)
    assert details["cut_events"] == 1 and row["messages"][-2]["tool_call_id"] == "a"
    assert row["messages"][-1]["reasoning_content"] == "Both are equal."


def _probe_segment(command, output, name="Bash"):
    use = dict(type="tool_use", id="t1", name=name, input={"command": command} if name == "Bash" else {"file_path": command})
    result = dict(type="tool_result", tool_use_id="t1", content=output)
    return {"messages": [{"role": "assistant", "content": [use]}, {"role": "user", "content": [result]}]}


import pytest  # noqa: E402

from tbstrain.screen import inspect_segment  # noqa: E402


@pytest.mark.parametrize(
    "command,output,name,expected",
    [
        # the agent's own scratch work: simulated truth, its own scorer, its memory notes
        ("python -c \"import pickle; t=pickle.load(open('/tmp/hs1/truth.pkl','rb'))\"", "x", "Bash", False),
        ("cd /app/work && python score.py /tmp/hs1 /tmp/ho2", "x", "Bash", False),
        ("cd /logs/agent/sessions/projects/-app/memory && sed -i s/a/b/ notes.md", "", "Bash", False),
        ("cat > /app/work/mkset.py << 'EOF'\nfrom gen import truth_f\nEOF", "", "Bash", False),
        ("/tmp/val/truth.json", "{}", "Read", False),
        # searches for grader material
        ("ls -laR /logs/verifier 2>/dev/null | head", "/logs/verifier:\ntotal 8", "Bash", True),
        ("find / -maxdepth 2 -name '*verif*'", "/logs/verifier", "Bash", True),
        ("ls /tests", "ls: cannot access '/tests': No such file or directory", "Bash", True),
        ("find /opt -name '*truth*'", "", "Bash", True),
        ("grep -r answer /root", "", "Bash", True),
        ("/solution/solve.sh", "x", "Read", True),
    ],
)
def test_answer_search_detection_on_real_patterns(command, output, name, expected):
    flags, _, _ = inspect_segment(_probe_segment(command, output, name), True)
    assert ("answer_search" in flags or "hidden_access" in flags) is expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ('grep -n "input / output\\|linear algebra" /app/output/solution_cli.py | head', False),
        ("ls -la data && ls /logs/agent/sessions/projects/-app/memory/ && cat /logs/agent/sessions/projects/-app/memory/MEMORY.md", False),
        ('grep -nE "/app|http|curl|wget" /app/output/solution_cli.py || echo none', False),
        ("python - << 'EOF'\nimport os; print(open('/tests/x').read())\nEOF", True),
        ('find / -name "*truth*" -o -name "*inject*" 2>/dev/null | head', True),
        ("find / -maxdepth 4 -name '*.py' 2>/dev/null | grep -v site-packages | head -50", True),
        ("ls -la /logs/verifier", True),
        ("cat /tests/test_outputs.py", True),
        ("grep -rl tbsgen /opt /etc 2>/dev/null", False),
        ("grep -rl truth /opt 2>/dev/null", True),
    ],
)
def test_shell_probe(command, expected):
    from tbstrain.screen import shell_probe

    assert shell_probe(command, ("/app", "/work")) is expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ('grep -nE "/app|http|curl|wget" /app/output/solution_cli.py || echo none', False),
        ("pip install mpmath 2>&1 | tail -3", True),
        ("timeout 15 curl -sI https://example.org | head -5", True),
        ("curl -sI https://example.org", True),
        ("python -m pip install x", True),
    ],
)
def test_shell_network(command, expected):
    use = dict(type="tool_use", id="t1", name="Bash", input={"command": command})
    flags, _, _ = inspect_segment({"messages": [{"role": "assistant", "content": [use]}]}, True)
    assert ("network" in flags) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("os.makedirs(d + '/test'); os.makedirs(d + '/train')", False),
        ("cp a /tmp/test/x", False),
        ("cat /tests/test_outputs.py", True),
        ("ls /logs/verifier", True),
        ("open('/tests/x')", True),
        ("x=/solution/solve.sh", True),
        ("/oracle", True),
    ],
)
def test_hidden_path_pattern(text, expected):
    from tbstrain.screen import HIDDEN

    assert bool(HIDDEN.search(text)) is expected


def test_judge_quote_tolerates_json_escaping_and_whitespace():
    from tbstrain.judge import quoted

    prompt = '{"text":"line one\\nline \\"two\\"  here","type":"text"}'
    assert quoted('line one\nline "two" here', prompt)
    assert quoted("line one", prompt)
    assert not quoted("not present", prompt)
    assert not quoted("", prompt)


def test_judge_decode_extracts_object():
    from tbstrain.llm import Client

    assert Client.decode('```json\n{"a": 1}\n```') == {"a": 1}
    assert Client.decode('Here you go: {"a": 2} done') == {"a": 2}


def test_judge_transcript_is_readable_text():
    from tbstrain.judge import windows

    record = {"messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "python - <<'EOF'\nx = 1\nEOF"}},
    ]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "line \"one\"\nline two"}]}]}
    text = "".join(windows([record], 10000))
    assert "x = 1\nEOF" in text and 'line "one"\nline two' in text and "\\n" not in text


def test_trajectory_judge_keeps_quality_findings_as_notes():
    from tbstrain.judge import assess_trajectory

    prompt = "Transcript\npython /app/work/test1.py\nModuleNotFoundError: No module named 'matplotlib'"
    answer = {"quality": 3, "wasted_turn_ratio": 0.2, "rationale": "ok", "findings": [
        {"type": "wasted_turns", "severity": "minor", "quote": "pip download", "why": "x"},
        {"type": "answer_search", "severity": "Minor", "quote": "not in prompt", "why": "weak"},
    ]}
    out = assess_trajectory(answer, prompt)
    assert [f["type"] for f in out["findings"]] == ["answer_search"]
    assert out["findings"][0]["quote_verified"] is False and out["quality_notes"][0]["type"] == "wasted_turns"
    blocking = {**answer, "findings": [{"type": "hardcode_output", "severity": "blocking", "quote": "nope", "why": "x"}]}
    with pytest.raises(ValueError):
        assess_trajectory(blocking, prompt)


def test_easy_cap_keeps_all_medium_hard_and_best_easy():
    from tbstrain.select import easy_capped

    def rows(tier, n, quality=4):
        return [dict(trial_id=f"{tier}{i}", tier=tier, quality=quality - (i % 2), rank=0) for i in range(n)]

    pools = {"hard": rows("hard", 6), "medium": rows("medium", 11), "easy": rows("easy", 10)}
    out = easy_capped(pools, 0.15)
    tiers = [r["tier"] for r in out]
    assert tiers.count("hard") == 6 and tiers.count("medium") == 11
    assert tiers.count("easy") == 3            # 15 % of the final mix: floor(0.15 * 17 / 0.85) = 3
    assert all(r["quality"] == 4 for r in out if r["tier"] == "easy")
    assert easy_capped({"easy": rows("easy", 4)}, 0.15) == []
