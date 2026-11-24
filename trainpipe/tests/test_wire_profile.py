"""Wire regressions use only hand-written prompts, schemas and native events."""

import json

import pytest

from conftest import block, cc_message, codex_events, link_events, pi_events, toy_profile, toy_tool
from tbstrain.claude_wire import Session, request
from tbstrain.cli import main
from tbstrain.native import claude, codex, pi, message, segment
from tbstrain.profile import ProfileError, extract, load_profile
from tbstrain.wire import check, render


def attachment(kind, *parts):
    return dict(type="attachment", attachment={"type": kind}, rendered=[{"content": t} for t in parts])


def wrapped(text):
    return "<system-reminder>\n" + text + "\n</system-reminder>"


def test_order_and_attachment_bursts():
    events = link_events([
        cc_message("user", "Question"),
        attachment("context", wrapped("one"), [block(wrapped("two"))]),
        attachment("date", wrapped("three")),
        attachment("prompt_snapshot", "private bookkeeping"),
        cc_message("assistant", [block("Answer")]),
    ])
    session = Session([events[i] for i in (1, 3, 2, 4, 0)])
    messages = session.state(session.leaf)
    assert [m["role"] for m in messages] == ["user", "system", "assistant"]
    assert messages[1]["content"] == "one\ntwo\n\nthree"


def test_attribution_first_user_merges_and_cache_breakpoint():
    events = link_events([
        cc_message("user", "Question"),
        attachment("remote_session_change", wrapped("Note")),
        cc_message("user", [block("Continue")]),
    ])
    session = Session(events)
    req = request(session, session.leaf, toy_profile(), {})
    assert req["messages"] == [dict(role="user", content=[
        block(wrapped("Note") + "\n"), block("Question\n"),
        {**block("Continue"), "cache_control": {"type": "ephemeral"}},
    ])]
    assert "cache_control" not in events[-1]["message"]["content"][0]


def call(name, tid):
    return dict(type="tool_use", name=name, id=tid, input={"n": 1})


def result(tid, content):
    return dict(type="tool_result", tool_use_id=tid, content=content)


def test_parallel_fragments_and_forked_results():
    events = link_events([
        cc_message("user", "Q"),
        cc_message("assistant", [call("sample", "a")]),
        cc_message("user", [result("a", "1")]),
        cc_message("assistant", [call("sample", "b")]),
        cc_message("user", [result("b", "2")]),
    ])
    events[3]["parentUuid"] = events[1]["uuid"]
    for i in (1, 3):
        events[i]["message"]["id"] = "shared"
    session = Session(events)
    ms = session.state(session.leaf)
    assert [m["role"] for m in ms] == ["user", "assistant", "user"]
    assert [b["id"] for b in ms[1]["content"]] == ["a", "b"]
    assert [b["tool_use_id"] for b in ms[2]["content"]] == ["a", "b"]


def test_compaction_active_branch_preserved_context_and_later_attribution():
    thought = dict(type="thinking", thinking="", signature="synthetic")
    events = link_events([
        cc_message("user", "Old task"),
        cc_message("assistant", [thought, call("sample", "a")]),
        cc_message("user", [result("a", "   data  \n")]),
        dict(type="system", subtype="compact_boundary"),
        attachment("session_context", wrapped("Context")),
        cc_message("user", "Summary:\nShort history.", isCompactSummary=True),
        attachment("remote_session_change", wrapped("Note")),
        cc_message("assistant", [block("Finished")]),
    ])
    boundary = events[3]
    boundary.update(parentUuid=None, logicalParentUuid="node-2", compactMetadata={
        "preservedMessages": {"anchorUuid": "node-5"},
        "preservedSegment": {"headUuid": "node-1", "tailUuid": "node-2"},
    })
    session = Session([*events[:7], dict(type="system", subtype="compact_boundary", uuid="other"), events[-1]])
    assert session.windows() == ["node-2", "node-7"]
    ms = session.state(session.leaf)
    assert [m["role"] for m in ms] == ["user", "system", "assistant", "user", "assistant"]
    assert ms[0]["content"] == "Summary:\nShort history."
    assert ms[1]["content"] == "Context"
    assert ms[2]["content"][0] == thought
    assert [m["loss"] for m in ms if m["role"] == "assistant"] == [0, 1]
    assert ms[3]["content"] == [result("a", "data\n\n" + wrapped("Note"))]
    assert events[2]["message"]["content"][0]["content"] == "   data  \n"


def test_row_mapping_and_trim():
    source = segment([
        message("system", "Prefix\n\nInstructions"),
        message("assistant", [dict(type="thinking", thinking=""), call("sample", "a")]),
        message("user", [block("a"), result("a", "observed"), block("b")]),
        message("assistant", [dict(type="thinking", thinking=x) for x in ("thought 1", "thought 2")]
                + [block("one"), block("two")]),
        message("user", "tail"),
    ], [toy_tool(), toy_tool("unused")])
    source["messages"][1]["loss"] = 0
    row = render(source, {})
    assert check(row, allow_masked=True) == []
    assert row["loss_mask"] == [0, 0, 0, 0, 1]
    assert row["messages"][1]["reasoning_content"] == ""
    assert row["messages"][2:] == [dict(role="tool", content="observed", tool_call_id="a"),
                                   dict(role="user", content="a\n\nb"),
                                   dict(role="assistant", content="one\ntwo", reasoning_content="thought 1\n\nthought 2")]
    assert row["meta"]["render_notes"]["dropped_tail"] == 1
    assert [t["function"]["name"] for t in row["tools"]] == ["sample", "unused"]


@pytest.mark.parametrize("field", ["system", "tools", "version", "schema_bytes"])
def test_extract_refuses_disagreement(tmp_path, field):
    profile = toy_profile()
    first, second = (tmp_path / name for name in ("first", "second"))
    first.write_text(json.dumps(profile))
    if field == "schema_bytes":
        second.write_text(first.read_text().replace('"properties":', '"properties" :'))
    else:
        profile[field] = {"system": [block("Different")], "tools": [], "version": "2"}[field]
        second.write_text(json.dumps(profile))
    out = tmp_path / "private"
    with pytest.raises(ProfileError, match="mismatch"):
        extract([first, second], out)
    assert not out.exists()


def test_profile_manifest_checksums_and_envelope_version(tmp_path):
    p = toy_profile()
    raw = tmp_path / "first"
    raw.write_text(json.dumps({"body": p, "headers": {"User-Agent": "claude-cli/1 (synthetic)"}}))
    out = tmp_path / "profile"
    assert main(["wire-profile", "extract", "--out", str(out), "--from", str(raw)]) == 0
    assert load_profile(out, "claude-code")["tools"] == p["tools"]
    (out / "tools.json").write_text("[]")
    with pytest.raises(ProfileError, match="checksum"):
        load_profile(out, "claude-code")


@pytest.mark.parametrize("version,profile", [("2", toy_profile()), ("1", None), (None, toy_profile())])
def test_missing_or_wrong_profile_version(version, profile, tmp_path):
    with pytest.raises(ProfileError):
        claude([], {"profile": profile}, {"agent_version": version, "trial": str(tmp_path)})


@pytest.mark.parametrize("location", ["session_meta", "turn_context", "additional_tools"])
def test_codex_recorded_instructions_and_full_menu(location):
    events = codex_events()
    del events[0]["payload"]["base_instructions"]
    events[1]["payload"]["instructions"] = "Use recorded instructions."
    del events[1]["payload"]["tools"]
    tools = [toy_tool(), toy_tool("unused")]
    if location == "additional_tools":
        events.insert(2, dict(type="response_item", payload=dict(type=location, tools=tools)))
    else:
        idx = int(location == "turn_context")
        events[idx]["payload"]["tools"] = tools
    record, = codex(events, {}, {})[0]
    row = render(record, {})
    assert row["messages"][0]["content"] == "Use recorded instructions."
    assert [t["function"]["name"] for t in row["tools"]] == ["sample", "unused"]
    assert check(row) == []


@pytest.mark.parametrize("harness", ["pi", "codex"])
def test_missing_menu_requires_profile_and_keeps_native_prompt(harness):
    if harness == "codex":
        events = codex_events()
        del events[1]["payload"]["tools"]
        convert = codex
        expected = "Use the synthetic samples."
    else:
        events = pi_events()
        del events[2]["body"]["tools"]
        convert = pi
        expected = "Use the samples."
    with pytest.raises(ProfileError):
        convert(events, {}, {})
    record, = convert(events, {"profile": toy_profile()}, {})[0]
    row = render(record, {})
    assert row["messages"][0]["content"] == expected
    assert len(row["tools"]) == 3
    assert check(row) == []


def test_pi_recorded_empty_menu_and_header():
    events = pi_events()
    del events[2]["body"]["messages"]
    events.insert(0, {"type": "session", "systemPrompt": "Native prompt"})
    row = render(pi(events, {}, {})[0][0], {})
    assert row["tools"] == []
    assert row["messages"][0]["content"] == "Native prompt"


@pytest.mark.parametrize("change,error", [("row", "row_keys"), ("system", "message_keys"),
                                         ("user", "message_keys"), ("schema", "tools_schema"),
                                         ("leading", "first_role"), ("offer", "missing_tool_schema")])
def test_strict_platform_shape(change, error):
    row = render(segment([message("system", "Prompt"), message("user", "Input"),
                          message("assistant", [call("sample", "a")]), message("user", [result("a", "1")]),
                          message("assistant", "Done")], [toy_tool()]), {})
    if change == "row":
        row["extra"] = True
    elif change in {"system", "user"}:
        row["messages"][int(change == "user")]["extra"] = True
    elif change == "schema":
        row["tools"][0]["function"]["parameters"] = {}
    elif change == "leading":
        row["messages"][0]["role"] = "user"
    else:
        row["tools"] = []
    assert error in check(row)


def test_validate_reference(workspace, completed):
    _, out = completed
    ref = workspace.parent / "accepted.json"
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    reference = next(r for r in rows if any(m.get("tool_calls") for m in r["messages"])
                     and any("reasoning_content" in m for m in r["messages"]))
    ref.write_text(json.dumps(reference))
    args = ["validate", str(out / "train.jsonl"), "--config", str(workspace), "--reference", str(ref)]
    assert main(args) == 0
    for m in reference["messages"]:
        m.pop("reasoning_content", None)
    ref.write_text(json.dumps(reference))
    assert main(args) == 1


def test_final_tool_turn_and_trimmed_result():
    src = segment([message("system", "Instructions"),
                   message("assistant", [call("sample", "final")]),
                   message("user", [result("final", "outside window")])], [toy_tool()])
    row = render(src, {})
    assert row["messages"][-1]["tool_calls"][0]["id"] == "final"
    assert row["meta"]["render_notes"]["dropped_tail"] == 1
    assert not check(row)


def test_export_refuses_changed_or_missing_profile(completed):
    cfg, out = completed
    directory = cfg.parent / "wire-profile"
    manifest = directory / "profile.json"
    data = json.loads(manifest.read_bytes())
    data["version"] = "newer"
    manifest.write_text(json.dumps(data))
    args = ["export", "--config", str(cfg)]
    assert main(args) == 1
    assert "version mismatch" in (out / "reports/export.json").read_text()
    cfg.write_text(cfg.read_text().replace('[export.wire.claude-code]\ndir = "wire-profile"', ''))
    assert main(args) == 1
    assert "wire profile required" in (out / "reports/export.json").read_text()


def test_codex_native_custom_tool_resolves_operator_schema():
    events = codex_events()
    events[1]["payload"]["tools"] = [{"type": "custom", "name": "sample"}]
    events[3]["payload"] = dict(type="custom_tool_call", name="sample", call_id="a", input="toy")
    events[5]["payload"]["type"] = "custom_tool_call_output"
    with pytest.raises(ProfileError):
        codex(events, {}, {})
    profile = toy_profile()
    profile["tools"][0]["input_schema"] = {"type": "object", "properties": {"input": {"type": "string"}}}
    row = render(codex(events, {"profile": profile}, {})[0][0], {})
    assert [t["function"]["parameters"] for t in row["tools"]] == [profile["tools"][0]["input_schema"]]
    assert check(row) == []


def test_wire_check_reports_body_mismatch_without_exposing_text(tmp_path, capsys):
    from tbstrain.core import write_lines

    projects = tmp_path / "agent/sessions/projects/toy"
    projects.mkdir(parents=True)
    records = link_events([
        cc_message("user", "ZZZ_RESUME_SENTINEL synthetic"),
        cc_message("assistant", [block("A synthetic answer.")]),
    ])
    write_lines(projects / "session.jsonl", records)
    proxy = tmp_path / "agent/resume-proxy/toy"
    proxy.mkdir(parents=True)
    gold = dict(model="student-model", stream=True, max_tokens=128000, thinking={"type": "adaptive"},
                context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
                output_config={"effort": "max"}, system=toy_profile()["system"], tools=toy_profile()["tools"],
                messages=[dict(role="user", content=[{**block(records[0]["message"]["content"]),
                                                     "cache_control": {"type": "ephemeral"}}])])
    path = proxy / "first_request_raw.json"
    raw = {"headers": {"User-Agent": "claude-cli/1"}, "body": gold}
    path.write_text(json.dumps(raw))
    args = ["wire-check", "--captures", str(tmp_path)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["first_request"]["exact"] == 1
    gold["max_tokens"] -= 1
    path.write_text(json.dumps(raw))
    assert main(args) == 1
    report = json.loads(capsys.readouterr().out)["first_request"]
    assert report["mismatch"] == 1 and report["first_difference"]["difference"] == "$.max_tokens value differs"
