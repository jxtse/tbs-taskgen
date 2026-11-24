from types import SimpleNamespace

import pytest

from tbstrain.cli import main
from tbstrain.core import atomic, lines, read, write_lines
from tbstrain.native import codex, message, segment
from tbstrain.screen import inspect_segment
from tbstrain.select import select
from tbstrain.tasks import difficulty_row
from tbstrain.wire import check, render
from conftest import codex_events, link_events, toy_profile


@pytest.mark.parametrize("summary,expected", [
    ("Intermediate summary.", "Intermediate summary."),
    ("Summarize.\nIntermediate summary.", "Intermediate summary."),
    ("Summarize.", None),
])
def test_captured_codex_windows(tmp_path, summary, expected):
    events = codex_events()

    def output(text):
        return dict(type="message", role="assistant", content=[dict(type="output_text", text=text)])

    events[6:6] = [
        dict(type="response_item", payload=output(summary)),
        events[-1],
        dict(
            type="compacted",
            timestamp="2025-01-01T00:02:30Z",
            payload=dict(window_id="w", message=summary),
        ),
        dict(
            type="response_item",
            payload=dict(
                type="message", role="user", content=[dict(type="input_text", text=summary)]
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
                output(summary),
            ),
            (
                [
                    dict(
                        type="message",
                        role="user",
                        content=[dict(type="input_text", text=summary)],
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
    if expected:
        assert converted[0]["messages"][-1]["content"] == expected
        assert converted[0]["meta"]["summary_model"] == "student-model"
    else:
        assert not any(r["meta"].get("compaction") for r in converted)
    assert sum(len(m.get("tool_calls", [])) for row in converted for m in row["messages"]) == 1
    calls[-1]["output"]["output"] = [output("Mismatched response.")]
    write_lines(path, calls)
    with pytest.raises(ValueError, match="correspond"):
        codex(events, {}, {"capture": str(path)})


def test_selection_caps_diversity_and_quality(tmp_path, monkeypatch):
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
    monkeypatch.setattr(ctx, "dependencies", lambda *stages: list(stages))
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
    workspace.write_text(workspace.read_text().replace("[screen]", "[screen]\nallow_timeout=true"))
    assert screen()["disposition"] == "keep"
    result["exception_info"] = None
    result["finished_at"] = None
    atomic(path, result)
    assert "unclean_end" in screen()["flags"]


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


def test_nested_trial_with_extra_rollouts_counts_once(workspace):
    run = workspace.parent / "run"
    manifest = run / "release/manifest.jsonl"
    hard = next(row for row in lines(manifest) if row["item_id"] == "hard")
    hard["calib_runs"] = hard["calib_runs"][:2]
    hard["calib_runs"][0].update(reward=0, outcome="fail")  # native verifier supersedes this summary
    # The first trial name is in the release. Recover the second from local runs.json.
    del hard["calib_runs"][1]["trial"]
    write_lines(manifest, [hard])
    source = run / "calib-jobs/hard-0/one"
    config = read(source / "config.json")
    for job, name, reward in [(run / "calib-jobs/hard-1", "one", 0),
                              (workspace.parent / "extras", "two", 1),
                              (workspace.parent / "extras", "three", 1)]:
        atomic(job / name / "config.json", config)
        atomic(job / name / "result.json", dict(id=job.name + name, task_name="hard",
               verifier_result={"rewards": {"reward": reward}}))
    assert main(["ingest", "--config", str(workspace), "--job", str(workspace.parent / "extras")]) == 0
    assert main(["difficulty", "--config", str(workspace)]) == 0
    row, = lines(workspace.parent / "out/difficulty.jsonl")
    assert (row["attempts"], row["student_passes"], row["rate"], row["tier"]) == (4, 3, 0.75, "easy")


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
    assert screened["disposition"] == "drop" and "model_not_trainable" in screened["flags"]


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


@pytest.mark.parametrize(
    "command,output,name,expected",
    [
        # Invented commands from a toy mosaic task, never executed.
        ("python -c \"from pathlib import Path; print(Path('/tmp/toy-mosaic/truth.tsv').stat())\"", "sample", "Bash", False),
        ("cd /app/mosaic && python paint_tiles.py /tmp/toy-palette", "painted", "Bash", False),
        ("cd /logs/agent/toy-mosaic && sed -i s/blue/green/ palette.txt", "", "Bash", False),
        ("cat > /app/mosaic/make_tile.py <<'TILE'\nfrom toy_palette import reference_tile\nTILE", "", "Bash", False),
        ("/tmp/toy-mosaic/reference.tsv", "blue", "Read", False),
        ("ls -R /logs/verifier/tile-scores | head -n 7", "tile-score.tsv", "Bash", True),
        ("find / -maxdepth 3 -name mosaic-reference.tsv", "mosaic-reference.tsv", "Bash", True),
        ("ls /tests/toy-mosaic", "No files found", "Bash", True),
        ("find /opt/toy-mosaic -iname '*reference*.tsv'", "", "Bash", True),
        ("grep -ri expected_tile /root/toy-palette", "", "Bash", True),
        ("/solution/toy-mosaic/palette.tsv", "blue", "Read", True),
    ],
)
def test_synthetic_mosaic_probe_detection(command, output, name, expected):
    flags, _, _ = inspect_segment(_probe_segment(command, output, name), True)
    assert ("answer_search" in flags or "hidden_access" in flags) is expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ('grep -n "tile / palette" /app/toy-mosaic/palette.txt | head -n 4', False),
        ("ls /app/toy-mosaic && ls /logs/agent/toy-mosaic && cat /logs/agent/toy-mosaic/notes", False),
        ('rg "wget|curl" /app/toy-mosaic/guide.txt || echo blue', False),
        ("python - <<'TILE'\nprint(open('/tests/mosaic/reference.tsv').readline())\nTILE", True),
        ('find / -iname "*palette-truth*" -print | head -n 7', True),
        ("find / -maxdepth 3 -iname '*.tile' | grep mosaic | head -n 6", True),
        ("ls -l /logs/verifier/toy-mosaic", True),
        ("cat /tests/toy-mosaic/reference.tsv", True),
        ("grep -ril mosaic /opt/palette", False),
        ("grep -ril truth_tile /opt/palette", True),
    ],
)
def test_shell_probe(command, expected):
    from tbstrain.screen import shell_probe

    assert shell_probe(command, ("/app", "/work")) is expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ('rg "wget|curl" /app/toy-mosaic/guide.txt || echo blue', False),
        ("pip install toy-mosaic-kit 2>&1 | tail -n 7", True),
        ("timeout 9 curl -I https://tiles.example/palette | head -n 2", True),
        ("curl -s https://tiles.example/swatches", True),
        ("python -m pip install toy-palette-kit", True),
    ],
)
def test_shell_network(command, expected):
    use = dict(type="tool_use", id="t1", name="Bash", input={"command": command})
    flags, _, _ = inspect_segment({"messages": [{"role": "assistant", "content": [use]}]}, True)
    assert ("network" in flags) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("for suffix in ['/tiles', '/test']: make_folder(palette_dir + suffix)", False),
        ("cp swatch.tsv /tmp/toy-mosaic/test/tile.tsv", False),
        ("cat /tests/mosaic/palette.tsv", True),
        ("ls /logs/verifier/toy-mosaic", True),
        ("open('/tests/mosaic/shapes.tsv')", True),
        ("palette=/solution/mosaic/tile.tsv", True),
        ("/oracle/mosaic", True),
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

    prompt = "Transcript\npython /app/toy-mosaic/palette.py\nMissing synthetic tile"
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


def run_screen(cfg):
    for stage in ("ingest", "difficulty", "filter-tasks", "screen"):
        assert main([stage, "--config", str(cfg)]) == 0
    return {r["harness"]: r for r in lines(cfg.parent / "out/screen.jsonl")}


def test_train_models_required(workspace):
    workspace.write_text(workspace.read_text().replace('train_models = ["student-model"]', ""))
    for stage in ("ingest", "difficulty", "filter-tasks"):
        assert main([stage, "--config", str(workspace)]) == 0
    assert main(["screen", "--config", str(workspace)]) == 1
    assert "train_models" in read(workspace.parent / "out/reports/screen.json")["error"]


def test_teacher_requires_explicit_distillation(workspace):
    from conftest import claude_events

    root = workspace.parent / "run/calib-jobs"
    trial = root / "hard-0/one"
    trial.parent.rename(root / "hard-teacher")
    trial = root / "hard-teacher/one"
    events = claude_events()
    for e in events:
        if e["type"] == "assistant":
            e["message"]["model"] = "teacher-model"
    write_lines(next(trial.rglob("tiny.jsonl")), events)
    config = read(trial / "config.json")
    config["agent"]["model_name"] = "teacher-model"
    atomic(trial / "config.json", config)
    result = read(trial / "result.json")
    result["agent_info"]["model_info"]["name"] = "teacher-model"
    atomic(trial / "result.json", result)
    row = run_screen(workspace)["claude-code"]
    assert row["disposition"] == "drop" and "model_not_trainable" in row["reasons"]
    hard = next(r for r in lines(workspace.parent / "out/difficulty.jsonl") if r["task_id"] == "hard")
    assert hard["tier"] == "hard" and hard["teacher_passes"] == 1
    workspace.write_text(workspace.read_text().replace('["student-model"]', '["student-model", "teacher-model"]'))
    assert run_screen(workspace)["claude-code"]["disposition"] != "drop"
    events[-1]["message"].pop("model")
    write_lines(next(trial.rglob("tiny.jsonl")), events)
    assert "model_not_trainable" in run_screen(workspace)["claude-code"]["reasons"]


@pytest.mark.parametrize("marker", ["block", "run"])
def test_claude_hint_excludes_trajectory(workspace, marker):
    path = next((workspace.parent / "run/calib-jobs/hard-0").rglob("tiny.jsonl"))
    events = list(lines(path))
    if marker == "block":
        events[-1]["message"]["content"][0]["hint"] = True
    else:
        events[0]["hint_run"] = True
    write_lines(path, events)
    row = run_screen(workspace)["claude-code"]
    assert row["disposition"] == "drop" and "hint_run" in row["reasons"]
    workspace.write_text(workspace.read_text().replace("[screen]", '[screen]\nhint_mode="strip"'))
    assert main(["screen", "--config", str(workspace)]) == 1


def test_codex_result_splits_turn_without_token_count():
    events = codex_events()
    del events[4]
    events[4]["payload"]["output"] = "violet"
    events[5]["payload"]["content"][0]["text"] = "The sample is violet."
    record, = codex(events, {}, {})[0]
    row = render(record, {})
    assert check(row) == []
    call, result, claim = row["messages"][-3:]
    assert call["tool_calls"][0]["id"] == result["tool_call_id"]
    assert result["content"] == "violet" and claim["content"] == "The sample is violet."
    assert not call["content"] and "tool_calls" not in claim
    call["content"] = claim["content"]
    del row["messages"][-2:]
    del row["loss_mask"][-2:]
    assert "tool_response_missing" in check(row)


@pytest.mark.parametrize("returned", [["other"], ["beta", "alpha"], ["alpha"]])
def test_raw_final_tool_ids_checked_before_trim(returned):
    ids = ["alpha"] if returned == ["other"] else ["alpha", "beta"]
    record = segment([
        message("system", "Sort toy tiles."),
        message("assistant", [dict(type="tool_use", id=k, name="sample", input={}) for k in ids]),
        message("user", [dict(type="tool_result", tool_use_id=k, content="blue") for k in returned]),
    ])
    with pytest.raises(ValueError, match="tool_order|unanswered_tool_call"):
        render(record, {})


@pytest.mark.parametrize("mode", ["drop", "summary"])
def test_codex_reasoning_policy_in_export(workspace, judge_stub, mode):
    path = next((workspace.parent / "run/calib-jobs/medium-0").rglob("rollout-*.jsonl"))
    events = codex_events()
    events.insert(-2, dict(type="response_item", payload=dict(
        type="reasoning", summary=[dict(type="summary_text", text="The tile colors agree.")],
        encrypted_content="synthetic-opaque-field")))
    write_lines(path, events)
    if mode == "summary":
        workspace.write_text(workspace.read_text().replace("[export]", '[export]\ncodex_reasoning="summary"'))
    assert main(["run", "--config", str(workspace)], transport=judge_stub) == 0
    row = next(r for r in lines(workspace.parent / "out/train.jsonl") if r["meta"]["harness"] == "codex")
    assert row["meta"]["codex_reasoning"] == mode
    assert row["meta"]["reasoning_source"] == ("summary" if mode == "summary" else "summary_dropped")
    assert row["messages"][-1]["reasoning_content"] == ("The tile colors agree." if mode == "summary" else "")


@pytest.mark.parametrize("tool,args", [
    ("Bash", {"command": "cat ../tests/palette.tsv"}),
    ("functions.exec_command", {"cmd": "find /opt/toy-palettes -iname '*truth*.tsv'"}),
    ("shell", {"command": ["sh", "-lc", "cat palette.tsv"], "workdir": "/tests"}),
    ("local_shell", {"action": {"command": ["cat", "palette.tsv"], "cwd": "/oracle"}}),
    ("pi.bash", {"script": "cat ../../tests/palette.tsv", "cwd": "/app/mosaic"}),
    ("mcp__pi__bash", {"snippet": "rg palette_truth .", "cwd": "/opt/toy-palettes"}),
])
def test_shell_tools_resolve_probe_paths(tool, args):
    src = _probe_segment("", "violet", name=tool)
    src["messages"][0]["content"][0]["input"] = args
    assert "answer_search" in inspect_segment(src, True)[1]


@pytest.mark.parametrize("tool,key", [("Bash", "command"), ("functions.exec_command", "cmd")])
@pytest.mark.parametrize("command,probe", [
    ("cd .. && cat tests/palette.tsv", True),
    ("pushd '../opt/toy tiles'; cd samples && rg palette_truth .", True),
    ("cd '../opt/toy tiles' && cd /app; rg palette_truth .", False),
    ("(cd ../opt); rg palette_truth .", False),
    ("cd ../opt | cat; rg palette_truth .", False),
])
def test_shell_sequential_cwd(tool, key, command, probe):
    src = _probe_segment("", "violet", name=tool)
    src["messages"][0]["content"][0]["input"] = {key: command}
    assert ("answer_search" in inspect_segment(src, True)[1]) is probe


@pytest.mark.parametrize("helper", ["_metrics", "budget"])
def test_task_judge_includes_timeout_helper(workspace, judge_stub, helper):
    graders = workspace.parent / "run/items/easy/pkg/tests"
    measure = f"from {helper} import run_solution\nrun_solution(1200)\n"
    (graders / "measure.py").write_text(measure)
    tail = "    return min(requested, remaining)\n"
    (graders / (helper + ".py")).write_text("# synthetic padding\n" * 150 + "def run_solution(requested):\n" + tail)
    workspace.write_text(workspace.read_text().replace("[judge]", "[judge]\nwindow_chars=700"))
    for stage in ("ingest", "difficulty", "filter-tasks"):
        assert main([stage, "--config", str(workspace)]) == 0
    assert main(["judge", "--kind", "task", "--config", str(workspace)], transport=judge_stub) == 0
    body, = [c for c in judge_stub.calls if "measure.py\n" in c["messages"][0]["content"]]
    prompt = body["messages"][0]["content"]
    assert len(prompt) <= 700 and measure in prompt and (graders / "test_outputs.py").read_text() in prompt
    assert helper + ".py\n" in prompt and "[omitted]" in prompt and tail in prompt
    assert "effective timeout" in body["system"]


def test_judge_windows_cover_long_blocks():
    from tbstrain.judge import windows

    text = "".join(f"{n}:violet " for n in range(2400))
    blocks = [dict(type="thinking", thinking=text), dict(type="tool_use", name="tile", input={"cmd": text}),
              dict(type="text", text=text)]
    parts = list(windows([segment([message("assistant", blocks)])], 713))
    assert max(map(len, parts)) <= 713
    assert "".join(parts).count(text) == 3


def test_minimum_window_quality_gates_easy(workspace, judge_stub):
    from tbstrain.core import dumps

    workspace.write_text(workspace.read_text().replace("[judge]", "[judge]\nwindow_chars=128"))
    async def rate(url, headers, body, timeout):
        response = await judge_stub(url, headers, body, timeout)
        if "Transcript window 1 of " in body["messages"][0]["content"]:
            response["content"][0]["text"] = dumps(dict(quality=3, findings=[], wasted_turn_ratio=0, rationale="Toy score"))
        return response

    for stage in ("ingest", "difficulty", "filter-tasks", "screen"):
        assert main([stage, "--config", str(workspace)]) == 0
    for kind in ("task", "trajectory"):
        assert main(["judge", "--kind", kind, "--config", str(workspace)], transport=rate) == 0
    easy = next(r for r in lines(workspace.parent / "out/judge-trajectory.jsonl") if r["task_id"] == "easy")
    assert {w["quality"] for w in easy["windows"]} == {3, 5} and easy["quality"] == 3
    assert main(["select", "--config", str(workspace)]) == 0
    assert any(r["reason"] == "easy_quality" for r in lines(workspace.parent / "out/select.ledger.jsonl"))


def test_unknown_blocking_repair_error_and_retry(workspace, judge_stub):
    from tbstrain.core import dumps

    workspace.write_text(workspace.read_text().replace("[judge]", "[judge]\nmax_error_ratio=0.5"))
    run_screen(workspace)
    bad_calls = []
    async def invalid(url, headers, body, timeout):
        response = await judge_stub(url, headers, body, timeout)
        if "Use the samples." in body["messages"][0]["content"]:
            bad_calls.append(len(body["messages"]))
            response["content"][0]["text"] = dumps(dict(
                quality=4, wasted_turn_ratio=0, rationale="Inspect the palette.",
                findings=[dict(type="unknown_kind", severity="blocking", quote="Use the samples.", why="Toy concern")]))
        return response

    args = ["judge", "--kind", "trajectory", "--config", str(workspace)]
    out = workspace.parent / "out"
    for attempt in range(2):
        assert main(args, transport=invalid) == 0
        errors = [r for r in lines(out / "judge-trajectory.jsonl") if r["verdict"] == "error"]
        assert len(errors) == 1 and "unknown blocking" in errors[0]["reasons"][0]
        assert bad_calls == [1, 3] * (attempt + 1)
    calls = len(judge_stub.calls)
    assert main(args, transport=judge_stub) == 0 and len(judge_stub.calls) == calls + 1
    assert all(r["verdict"] == "keep" for r in lines(out / "judge-trajectory.jsonl"))
    assert main(args, transport=judge_stub) == 0 and len(judge_stub.calls) == calls + 1


def test_dependency_rejects_old_code(completed):
    cfg, out = completed
    path = out / "reports/screen.json"
    atomic(path, {**read(path), "code": "previous-code"})
    assert main(["export", "--config", str(cfg)]) == 1
    assert "run screen again: code changed" in read(out / "reports/export.json")["error"]


def test_release_epoch_excludes_stale_trials(workspace, judge_stub):
    from shutil import copytree

    run = workspace.parent / "run"
    manifest = run / "release/manifest.jsonl"
    rows = list(lines(manifest))
    hard = next(r for r in rows if r["item_id"] == "hard")
    old = run / "calib-jobs/hard-0"
    current = old.with_name("hard-student-e8")
    old.rename(old.with_name("hard-student-e1"))
    old = old.with_name("hard-student-e1")
    copytree(old, current)
    hard["calib_runs"][0].update(job_name=current.name, reward=0, outcome="fail")
    write_lines(manifest, rows)
    result = read(current / "one/result.json")
    result.update(id="current-trial", verifier_result={"rewards": {"reward": 0}})
    atomic(current / "one/result.json", result)
    assert main(["run", "--config", str(workspace)], transport=judge_stub) == 0
    out = workspace.parent / "out"
    tier = next(r for r in lines(out / "difficulty.jsonl") if r["task_id"] == "hard")
    assert (tier["student_passes"], tier["student_attempts"], tier["teacher_passes"]) == (0, 3, 1)
    audit = {r.get("trial_id"): r["reason"] for r in lines(out / "ingest.ledger.jsonl")}
    assert audit["hard-trial"] == "stale_package" and audit["current-trial"] == "matched"
    stale = next(r for r in lines(out / "screen.jsonl") if r["trial_id"] == "hard-trial")
    assert stale["disposition"] == "drop" and "stale_package" in stale["reasons"]
    assert all(r["trial_id"] != "hard-trial" for r in lines(out / "judge-trajectory.jsonl"))


@pytest.mark.parametrize("binding,accepted", [
    ("path", True), ("package_hash", True), ("wrong_hash", False),
    ("wrong_path", False), ("harbor_hash", True), ("wrong_harbor_hash", False),
])
def test_rollout_release_binding(workspace, binding, accepted):
    from hashlib import sha256
    from shutil import copytree
    from tbstrain.core import tree_hash

    run = workspace.parent / "run"
    pkg = run / "items/easy/pkg"
    job = workspace.parent / "easy-0"  # external job collides with a calibration job
    trial = job / "two"
    copytree(run / "calib-jobs/easy-0/one", trial)
    result = read(trial / "result.json")
    result["id"] = "extra-trial"
    config = read(trial / "config.json")
    if binding == "wrong_path":
        config["task"]["path"] = str(copytree(pkg, workspace.parent / "old/easy/pkg"))
    if binding in {"package_hash", "wrong_hash"}:
        config["task"]["pkg_sha256"] = tree_hash(pkg) if accepted else "outdated"
    if binding in {"harbor_hash", "wrong_harbor_hash"}:
        def sha(text):
            return sha256(text.encode()).hexdigest()
        tests = sha("data:" + sha("assert measured_slope == 1") + "\0name:test_outputs.py")
        checksum = sha("data:" + sha("Estimate a synthetic slope.") + "\0name:instruction.md\0\0dirhash:"
                       + tests + "\0name:tests")
        result["task_checksum"] = checksum if accepted else "outdated"
    atomic(trial / "result.json", result)
    atomic(trial / "config.json", config)
    assert main(["ingest", "--config", str(workspace), "--job", str(job)]) == 0
    assert main(["difficulty", "--config", str(workspace)]) == 0
    out = workspace.parent / "out"
    row = next(r for r in lines(out / "trials.jsonl") if r["trial_id"] == "extra-trial")
    assert row["current_package"] is accepted
    easy = next(r for r in lines(out / "difficulty.jsonl") if r["task_id"] == "easy")
    assert easy["student_attempts"] == 3 + accepted


def test_binding_requires_calibration_directory_and_trial_name(workspace):
    from shutil import copytree

    job = workspace.parent / "run/calib-jobs/easy-0"
    extra = workspace.parent / "external"
    for path, tid in [(job / "different", "wrong-name"), (extra / "easy-0/one", "external-twin"),
                      (extra / "easy-1/one", "valid-twin")]:
        copytree(job / "one", path)
        atomic(path / "result.json", {**read(path / "result.json"), "id": tid})
        if tid == "external-twin":
            config = read(path / "config.json")
            config["task"]["pkg_sha256"] = "superseded"
            atomic(path / "config.json", config)
    assert main(["ingest", "--config", str(workspace), "--job", str(extra)]) == 0
    rows = {r["trial_id"]: r for r in lines(workspace.parent / "out/trials.jsonl")}
    assert rows["easy-trial"]["current_package"]
    assert rows["valid-twin"]["current_package"] and rows["valid-twin"]["who"] is None
    for tid in ("wrong-name", "external-twin"):
        assert not rows[tid]["current_package"] and rows[tid]["who"] is None
    assert main(["difficulty", "--config", str(workspace)]) == 0
    easy = next(r for r in lines(workspace.parent / "out/difficulty.jsonl") if r["task_id"] == "easy")
    assert easy["student_attempts"] == 4


@pytest.mark.parametrize("category", ["fake_result", "hardcode_output", "hidden_truth_access", "grader_tamper"])
def test_minor_integrity_finding_blocks(category):
    from tbstrain.judge import assess_trajectory

    evidence = "except ArithmeticError: save_result(initial_estimate)"
    finding = dict(type=category, severity="minor", quote=evidence, why="The transcript records the action.")
    answer = dict(quality=5, rationale="Synthetic finding.", wasted_turn_ratio=0, findings=[finding])
    assert assess_trajectory(answer, evidence)["findings"][0]["severity"] == "blocking"
    with pytest.raises(ValueError, match="quote"):
        assess_trajectory(answer, "Unrelated observation")


@pytest.mark.parametrize("harness", ["pi", "codex"])
def test_native_block_hint_drops_trial(workspace, harness):
    root = workspace.parent / "run/calib-jobs"
    if harness == "pi":
        path = next((root / "easy-0").rglob("tiny.jsonl"))
        events = list(lines(path))
        events[3]["event"]["message"]["content"][0]["hint"] = True
    else:
        path = next((root / "medium-0").rglob("rollout-*.jsonl"))
        events = list(lines(path))
        events[6]["payload"]["content"][0]["hint"] = True
    write_lines(path, events)
    row = run_screen(workspace)[harness]
    assert row["disposition"] == "drop" and "hint_run" in row["reasons"]


@pytest.mark.parametrize("proxy", ["request_id", "completion_time", "absent"])
def test_summary_model_and_exact_control_line(workspace, proxy):
    trial = workspace.parent / "run/calib-jobs/hard-0/one"
    path = next(trial.rglob("tiny.jsonl"))
    events = list(lines(path))
    summary = next(r for r in events if r.get("isCompactSummary"))
    summary.update(timestamp="2025-01-01T00:03:10.550Z", requestId="summary-request")
    instruction = "Return a short synthetic note."
    near = "The note says: " + instruction
    (workspace.parent / "compact.txt").write_text(instruction + "\n")
    summary["message"]["content"] = f"Summary:\n{instruction}\n{near}"
    write_lines(path, events)
    if proxy != "absent":
        entry = dict(model="summary-model", path="/v1/messages")
        if proxy == "request_id":
            entry["request_id"] = "summary-request"
        else:
            entry.update(ts="2025-01-01T00:03:00+0000", latency_s=10.0)
        log = trial / "agent/model-proxy/requests-test.jsonl"
        log.parent.mkdir()
        write_lines(log, [entry, dict(model="irrelevant-model", request_id="different")])
        row = run_screen(workspace)["claude-code"]
        assert "model_not_trainable" in row["reasons"]
        workspace.write_text(workspace.read_text().replace('["student-model"]', '["student-model", "summary-model"]'))
    row = run_screen(workspace)["claude-code"]
    segment = next(r for r in read(workspace.parent / "out/segments" / row["source"]) if r.get("compaction"))
    wire = render(segment, {})
    assert wire["meta"]["summary_model"] == ("student-model" if proxy == "absent" else "summary-model")
    assert wire["meta"]["summary_model_source"] == ("surrounding_assistant" if proxy == "absent" else "proxy")
    assert wire["messages"][-1]["content"] == "<summary>\n" + near + "\n</summary>"
    assert wire["messages"][-2]["content"] == instruction + "\n"



def test_summary_preserves_nonmatching_lines():
    from tbstrain.native import strip_instruction_lines

    body = "Keep the samples.\r\n  Keep the samples.\r\nFinal line.\n"
    assert strip_instruction_lines(body, "Keep the samples.") == "  Keep the samples.\r\nFinal line.\n"


def test_task_judge_evidence_ignores_markdown_emphasis():
    from tbstrain.judge import assess_task

    prompt = "instruction\nThe functional form of `N` is **not disclosed**. | `hidden_err` | max |"
    ok = {"verdict": "keep", "difficulty": "hard", "rationale": "r",
          "evidence": ["The functional form of N is not disclosed.", "hidden_err"]}
    assert assess_task(ok, prompt)["verdict"] == "keep"
    with pytest.raises(ValueError, match="evidence 0"):
        assess_task({**ok, "evidence": ["something else entirely"]}, prompt)


@pytest.mark.parametrize("source,altered", [
    ("score = err ** 2", "score = err * 2"),
    ("** 2", "* 2"),
    ("score = a*b", "score = a b"),
    ("score = a | b", "score = a b"),
    ("score = a > b", "score = a b"),
    ("score = a_b", "score = a b"),
])
def test_task_evidence_preserves_operators(source, altered):
    from tbstrain.judge import assess_task

    decision = dict(verdict="flag", difficulty="hard", rationale="Check the calculation.")
    with pytest.raises(ValueError, match="evidence 0"):
        assess_task({**decision, "evidence": [altered]}, source)
    assert assess_task({**decision, "evidence": [source]}, source)["evidence"] == [source]
    assert assess_task({**decision, "evidence": ["Use the blue tile."]},
                       "> | Use *the* _blue_ `tile` .")


@pytest.mark.parametrize("subdirs,expected", [([], 1), (["small", "large"], 2)])
def test_task_judge_package_layout(workspace, judge_stub, subdirs, expected):
    pkg = workspace.parent / "run/items/easy/pkg"
    hidden = pkg / "tests/hidden"
    hidden.mkdir()
    for name in subdirs:
        (hidden / name).mkdir()
    (hidden / "labels.py").write_text("secret = 718\n")
    (pkg / "tests/truth.py").write_text("secret = 824\n")
    (pkg / "tests/measure.py").write_text("from hidden import labels\nimport truth\n")
    for command in (["ingest"], ["difficulty"], ["filter-tasks"], ["judge", "--kind", "task"]):
        assert main([*command, "--config", str(workspace)], transport=judge_stub) == 0
    body, = [c for c in judge_stub.calls if "tests/hidden/labels.py" in c["messages"][0]["content"]]
    prompt = body["messages"][0]["content"]
    assert f"hidden_sets directory rule): {expected}" in prompt
    assert "tests/hidden/labels.py (13 bytes)" in prompt and "tests/truth.py (13 bytes)" in prompt
    assert all(f"tests/hidden/{name}/" in prompt for name in subdirs)
    assert "718" not in prompt and "824" not in prompt and str(pkg) not in prompt
    assert "released file tree" in body["system"]


@pytest.mark.parametrize("tool,key", [("Bash", "command"), ("functions.exec_command", "cmd")])
@pytest.mark.parametrize("command,note", [
    ("cd . || cd ..; cat tests/palette.tsv", True),
    ("cd swatches || cd samples; cat tests/palette.tsv", False),
    ("cd swatches || cd ..; cat tests/palette.tsv", True),
    ("cd . || cd /oracle; cat tiles", True),
    ("cd .. || cd ../oracle; cat tests/palette.tsv", True),
    ("cd . || cd ..; cd /app; cat tests/palette.tsv", False),
    ("(cd . || cd ..); cat tests/palette.tsv", False),
])
def test_conditional_cwd_candidates(tool, key, command, note):
    src = _probe_segment("", "violet", name=tool)
    src["messages"][0]["content"][0]["input"] = {key: command}
    flags, fatal, stripped = inspect_segment(src, True)
    assert not fatal and not stripped
    assert flags == ({"inconclusive"} if note else set())
