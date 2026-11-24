"""Convert native journals into loss-bearing text, thinking, call and result blocks."""

from collections import deque
import json
from pathlib import Path
import re

from .claude_wire import Session, is_boundary
from .profile import ProfileError, required_profile, system_text
from .core import lines, loads, read
from .vendor.roller import compaction
from .vendor.roller.blobs import load_blobs
from .vendor.roller.codex_actf import codex_events_to_actf, session_tools
from .vendor.roller.sft import actf_to_sft, validate_sft_record


def string_content(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("unsupported_content")
    parts = []
    for b in value:
        if b["type"] == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
        elif b["type"] in {"image", "image_url"}:
            parts.append("[image omitted in export]")
        elif b["type"] == "tool_reference" and isinstance(b.get("tool_name"), str):
            parts.append(json.dumps(b, ensure_ascii=False))
        else:
            raise ValueError("unsupported_content_block")
    return "\n".join(parts)


def text_block(value):
    return {"type": "text", "text": value}


def message(role, content):
    return {
        "role": role,
        "content": content if isinstance(content, list) else [text_block(content)],
        "loss": int(role == "assistant"),
    }


def segment(messages, tools=()):
    return {"messages": messages, "tools": list(tools)}


def summary_target(text):
    match = re.search(r"(?:^|\n)Summary:\s*\n", text)
    if not match:
        raise ValueError("unrecognized_compaction_summary")
    body = re.split(
        r"\n(?:\s*\n)?(?:If you need specific details|Recent messages are preserved|Note: the earliest part|Continue the conversation from)",
        text[match.end() :],
    )[0].strip()
    if not body:
        raise ValueError("empty_compaction_summary")
    return "<summary>\n" + body + "\n</summary>"


def validate_pairs(messages):
    """Check observed results before replay removal, canonicalization or tail trimming."""
    pending = deque()
    for m in messages:
        if m["role"] == "assistant" and pending:
            raise ValueError("unanswered_tool_call")
        blocks = m["content"]
        for b in blocks if isinstance(blocks, list) else ():
            if b["type"] == "tool_use":
                pending.append(b["id"])
            elif b["type"] == "tool_result":
                if not pending or pending.popleft() != b["tool_use_id"]:
                    raise ValueError("tool_order")
    if pending:
        raise ValueError("unanswered_tool_call")


def claude(events, cfg, trial):
    rows = []
    profile = required_profile(cfg.get("profile"), "claude-code", trial.get("agent_version"))
    versions = {e["version"] for e in events if e.get("version") and not e.get("isSidechain")}
    if versions - {profile["version"]}:
        raise ProfileError("claude-code wire profile version mismatch in session")
    session = Session(events)
    marker = session.resume(cfg.get("resume_marker", "ZZZ_RESUME_SENTINEL"))
    if not marker and Path(trial["trial"], "agent/resume-proxy").exists():
        raise ValueError("unresolved_resume_prefix")
    historical = set()
    if marker:
        historical.update(r["uuid"] for r in session.chain(marker))
    ends = session.windows()
    if marker:
        start = next((i for i, end in enumerate(ends)
                      if any(r["uuid"] == marker for r in session.chain(end))), None)
        if start is None:
            raise ValueError("resume marker outside active branch")
        ends = ends[start:]
    models = []
    hinted = bool(trial.get("hint"))
    for i, end in enumerate(ends):
        chain = session.chain(end)
        for r in chain:
            content = r.get("message", {}).get("content")
            hinted |= bool(r.get("hint_run") or r.get("message", {}).get("hint_run")
                           or (isinstance(content, list) and any(b.get("hint") for b in content)))
            if r["type"] != "assistant":
                continue
            models.append(r["message"].get("model"))
            if r["uuid"] not in historical and r["message"].get("stop_reason") in {"max_tokens", "refusal", "error"}:
                raise ValueError("incomplete_assistant_response")
        messages = [message("system", system_text(profile["system"]))]
        skip_results = set()
        state = session.state(end, historical)
        validate_pairs(state)
        for m in state:
            content = m["content"]
            # The managed trainer supervises every assistant. Remove replay and its paired
            # results only for this target. Mask-aware targets keep the entire wire state.
            if cfg.get("train_on_what") == "all_assistant_messages":
                if m.get("loss") == 0 and m["role"] == "assistant":
                    if isinstance(content, list):
                        skip_results.update(b["id"] for b in content if b["type"] == "tool_use")
                    continue
                if isinstance(content, list) and m["role"] == "user":
                    content = [b for b in content if b.get("tool_use_id") not in skip_results]
                    if not content:
                        continue
            if m.get("prepended"):
                # The platform supplies the separator when joining text blocks.
                for block in content[:m["prepended"]]:
                    block["text"] = block["text"].rstrip("\n")
            entry = message(m["role"], content)
            entry["loss"] = m.get("loss", 0)
            messages.append(entry)
        row = segment(messages, profile["tools"])
        reasoning_sources = {
            ("native_thinking" if b["type"] == "thinking" else "redacted")
            for m in messages if m["role"] == "assistant" for b in m["content"]
            if b["type"] in {"thinking", "redacted_thinking"}
        }
        row["reasoning_source"] = "+".join(sorted(reasoning_sources)) or "none"
        if i + 1 < len(ends):
            boundary = next(r for r in reversed(session.chain(ends[i+1])) if is_boundary(r))
            summary = session.summary(boundary)["message"]["content"]
            instruction = profile.get("instruction") or cfg.get("instruction")
            if not instruction or not instruction.strip():
                raise ProfileError("compaction_instruction_required")
            messages.extend([message("user", instruction),
                             message("assistant", summary_target(string_content(summary)))])
            row["compaction"] = True
        rows.append(row)
    return rows, models, {"resume_prefix_events": len(historical), "hint_run": hinted}


def pi_segment(messages, system, tools):
    """Bind Pi's completed message events to Roller steps, then use its validated SFT path."""
    steps = []
    users = []
    errors = set()
    reasoning_sources = set()
    schemas = list(function_schemas(tools or []))
    for m in messages:
        role = m["role"]
        if role == "user":
            users.append(string_content(m["content"]))
        elif role == "toolResult":
            if not steps:
                raise ValueError("orphan_tool_result")
            steps[-1]["observation"].append(
                dict(
                    toolCallId=m["toolCallId"],
                    toolName=m.get("toolName"),
                    content=string_content(m["content"]),
                )
            )
            if m.get("isError"):
                errors.add(m["toolCallId"])
        else:
            text = []
            reasoning = []
            calls = []
            for b in m["content"]:
                if b["type"] == "text":
                    text.append(b["text"])
                elif b["type"] == "thinking":
                    if b.get("redacted"):
                        reasoning_sources.add("redacted")
                    else:
                        reasoning_sources.add("native_thinking")
                        reasoning.append(b["thinking"])
                elif b["type"] == "toolCall":
                    calls.append(dict(id=b["id"], name=b["name"], arguments=b["arguments"]))
                else:
                    raise ValueError("pi_unsupported_assistant_block")
            assistant = dict(
                content="\n".join(text), reasoning_content="\n".join(reasoning), tool_calls=calls
            )
            steps.append(dict(user_content="\n\n".join(users), assistant_content=assistant, observation=[]))
            users.clear()
    converted = from_sft(actf_to_sft(dict(system_prompt=system or "", tools=schemas, steps=steps)))
    for m in converted["messages"]:
        for b in m["content"]:
            if b.get("tool_use_id") in errors:
                b["is_error"] = True
    converted["reasoning_source"] = "+".join(sorted(reasoning_sources)) or "none"
    return converted


def pi(events, cfg, trial):
    current = []
    models = []
    tools = None
    end = None
    system = None
    request_model = None
    open_turn = None
    previous_id = None
    pending = deque()
    for index, e in enumerate(events):
        kind = e.get("type")
        if e.get("parentId") and previous_id is not None and e["parentId"] != previous_id:
            raise ValueError("pi_branched_history")
        previous_id = e.get("id", previous_id)
        if kind == "turn_start":
            if open_turn is not None:
                raise ValueError("pi_overlapping_turns")
            open_turn = e.get("turn")
        if kind == "turn_end":
            if open_turn != e.get("turn") or not end or end[1] != len(current):
                raise ValueError("pi_incomplete_interior_turn")
            open_turn = None
        if kind in {"request", "session", "context"}:
            body = e["body"] if kind == "request" else e
            if kind == "request":
                request_model = body.get("model")
            prompts = [string_content(m["content"]) for m in body.get("messages", [])
                       if m.get("role") in ("developer", "system")]
            prompt = body.get("systemPrompt", body.get("system")) or "\n\n".join(prompts)
            if prompt:
                prompt = string_content(prompt)
                if system is not None and system != prompt:
                    raise ValueError("pi_context_changed")
                system = prompt
            if "tools" in body:
                if tools is not None and tools != body["tools"]:
                    raise ValueError("pi_context_changed")
                tools = body["tools"]
        event = e.get("event", {}) if kind == "event" else e
        if event.get("type") in {"compaction", "compaction_start", "auto_retry_start", "branch_summary"}:
            raise ValueError("pi_nonlinear_context")
        if event.get("type") not in ("message", "message_end"):
            continue
        m = event["message"]
        role = m["role"]
        blocks = m["content"]
        if role == "toolResult":
            if not pending or pending.popleft() != m["toolCallId"]:
                raise ValueError("tool_order")
        elif pending:
            raise ValueError("unanswered_tool_call")
        if role == "assistant":
            pending.extend(b["id"] for b in blocks if b["type"] == "toolCall")
            models.append(m.get("model") or request_model)
            if m.get("stopReason") in {"aborted", "error", "length"}:
                if any(
                    (r.get("message") or r.get("event", {}).get("message") or {}).get("role") == "assistant"
                    for r in events[index + 1 :]
                ):
                    raise ValueError("pi_interior_incomplete_turn")
                break
        elif role not in {"toolResult", "user"}:
            raise ValueError("pi_unsupported_role")
        current.append(m)
        if (
            role == "assistant"
            and m.get("stopReason") in (None, "stop")
            and not any(b["type"] == "toolCall" for b in blocks)
        ):
            end = (index, len(current))
    if pending:
        raise ValueError("unanswered_tool_call")
    if not end:
        raise ValueError("pi_no_complete_turn")
    i, n = end
    tail = events[i + 1 :]
    # Journal bookkeeping may close the completed turn after its final response.
    while tail and tail[0].get("type") in {"turn_end", "context"}:
        tail = tail[1:]
    if system is None or tools is None:
        profile = required_profile(cfg.get("profile"), "pi", trial.get("agent_version"))
        system = system or system_text(profile["system"])
        if tools is None:
            tools = profile["tools"]
    tools = platform_schemas(tools, cfg.get("profile"), "pi", trial.get("agent_version"))
    return (
        [pi_segment(current[:n], system, tools)],
        models,
        {"cut_events": len(tail), "cut_messages": len(current) - n, "last_complete_event": i + 1},
    )


def function_schemas(tools, prefix=""):
    for tool in tools:
        if tool.get("type") == "namespace":
            yield from function_schemas(tool["tools"], "" if tool["name"] == "functions" else tool["name"] + ".")
            continue
        fn = tool.get("function", tool)
        if not fn.get("name"):
            raise ProfileError("tool has no captured function definition")
        parameters = fn.get("parameters", fn.get("input_schema", fn.get("inputSchema")))
        if not isinstance(parameters, dict) or not parameters:
            raise ProfileError("tool has no captured parameters: " + fn["name"])
        yield {
            "type": "function",
            "function": {
                "name": prefix + fn["name"],
                "description": fn.get("description", ""),
                "parameters": parameters,
            },
        }


def platform_schemas(tools, profile, harness, version):
    """Prefer recorded schemas. Resolve native custom/hosted tools by their offered names."""
    try:
        return list(function_schemas(tools))
    except ProfileError:
        supplied = required_profile(profile, harness, version)
        by_name = {t["function"]["name"]: t for t in function_schemas(supplied["tools"])}

    def resolve(items, prefix=""):
        for item in items:
            if item.get("type") == "namespace":
                group = "" if item["name"] == "functions" else item["name"] + "."
                yield from resolve(item["tools"], group)
                continue
            try:
                yield from function_schemas([item], prefix)
            except ProfileError:
                name = prefix + (item.get("function", item).get("name") or item.get("type", ""))
                if name not in by_name:
                    raise ProfileError(f"profile lacks parameters for offered {harness} tool {name}")
                yield by_name[name]

    return list(resolve(tools))


def from_sft(record, *, tools=None):
    validate_sft_record(record)
    messages = []
    start = record.get("target_start_index", 0)
    for i, m in enumerate(record["messages"]):
        role = m["role"]
        if i < start and role in ("assistant", "tool"):
            continue
        content = string_content(m["content"])
        if role == "tool":
            messages.append(
                message(
                    "user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content}]
                )
            )
        elif role == "assistant":
            blocks = []
            if m.get("reasoning_content"):
                blocks.append({"type": "thinking", "thinking": m["reasoning_content"]})
            if content:
                blocks.append(text_block(content))
            for call in m.get("tool_calls", []):
                f = call[call["type"]]
                name = (f["namespace"] + "." if f.get("namespace") else "") + f["name"]
                args = {"input": f["input"]} if call["type"] == "custom" else loads(f["arguments"])
                blocks.append({"type": "tool_use", "id": call["id"], "name": name, "input": args})
            messages.append(message(role, blocks))
        else:
            messages.append(message("system" if role == "developer" else role, content))
    return segment(messages, function_schemas(record["tools"]) if tools is None else tools)


def responses_tools(specs, offered=None):
    """Preserve Responses-native definitions and adapt other captured function shapes."""
    tools = []
    for spec in specs:
        kind = spec.get("type")
        if kind == "namespace" and spec.get("name") == "functions":
            tools.extend(responses_tools(spec["tools"], offered))
        elif kind and kind != "function":
            tools.append(spec)
        else:
            if offered is None:
                fn, = function_schemas([spec])
                function = fn["function"]
            else:
                function = offered[spec.get("function", spec)["name"]]
            tools.append(dict(type="function", **function))
    return tools


def codex(events, cfg, trial):
    current_model = None
    models = []
    pending = deque()
    reasoning_mode = cfg.get("codex_reasoning", "drop")
    received_result = False
    boundaries = []
    for e in events:
        p = e.get("payload") or {}
        if e["type"] == "event_msg" and p.get("type") in {"turn_aborted", "task_failed"}:
            raise ValueError("incomplete_codex_turn")
        kind = p.get("type")
        if e["type"] in ("session_meta", "turn_context") and p.get("model"):
            current_model = p["model"]
        if e["type"] == "compacted":
            boundaries.append(
                {"timestamp": e.get("timestamp"), "event_id": p.get("window_id"), "summary": p.get("message")}
            )
        if e["type"] != "response_item":
            continue
        if kind in {"function_call", "custom_tool_call", "reasoning"} or p.get("role") == "assistant":
            if received_result and pending:
                raise ValueError("unanswered_tool_call")
            models.append(p.get("model") or current_model)
        if kind in {"function_call", "custom_tool_call"}:
            received_result = False
            pending.append(p["call_id"])
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            if not pending or pending.popleft() != p["call_id"]:
                raise ValueError("tool_order")
            received_result = True
    if pending:
        raise ValueError("unanswered_tool_call")
    if boundaries and not trial.get("capture"):
        raise ValueError("codex_compaction_without_captured_calls")
    contexts = [e["payload"] for e in events if e["type"] in ("session_meta", "turn_context")]
    definitions = [p[key] for p in contexts for key in ("tools", "tool_specs", "dynamic_tools") if key in p]
    definitions.extend(e["payload"]["tools"] for e in events if e["type"] == "response_item"
                       and e["payload"].get("type") == "additional_tools")
    if not boundaries and definitions and any(t != definitions[0] for t in definitions):
        raise ValueError("codex_context_changed")
    profile = cfg.get("profile")
    if boundaries:
        # Only for aligning native steps. Output schemas come from the captured call.
        tools, status = session_tools(events)
    else:
        specs = definitions[0] if definitions else required_profile(profile, "codex", trial.get("agent_version"))["tools"]
        offered = platform_schemas(specs, profile, "codex", trial.get("agent_version"))
        tools = responses_tools(specs, {t["function"]["name"]: t["function"] for t in offered})
        status = "captured"
    actf = codex_events_to_actf(events, tools=tools, include_summary=reasoning_mode == "summary",
                                tools_capture={"source": "session", "status": status})
    provenance = {"codex_reasoning": reasoning_mode, "reasoning_source": actf["reasoning_source"]}
    if not boundaries:
        if not actf["system_prompt"]:
            actf["system_prompt"] = system_text(required_profile(profile, "codex", trial.get("agent_version"))["system"])
        record = actf_to_sft(actf)
        offered = platform_schemas(record["tools"], profile, "codex", trial.get("agent_version"))
        return [{**from_sft(record, tools=offered), **provenance}], models, {}
    path = Path(trial["capture"])
    calls = compaction._root_calls(list(lines(path)), read(path.parent / "manifest.json", {}))
    calls = load_blobs(calls, path.parent.parent / "blobs")
    for call in calls:
        context = call.get("context")
        if isinstance(context, dict) and "tools" not in context:
            specs = definitions[0] if definitions and all(t == definitions[0] for t in definitions) else None
            if specs is None:
                specs = required_profile(profile, "codex", trial.get("agent_version"))["tools"]
            if call.get("protocol") != "openai-responses":
                raise ValueError("missing tools in non-Responses capture")
            context["tools"] = responses_tools(specs)
    summaries = set()
    verified = compaction._calls_matching_main_steps(actf, calls, boundaries, summaries)
    records = compaction._segment_records(verified, boundaries, True, summaries)
    for call in calls:
        for field in ("context", "output"):
            if call.get(field, {}).get("model"):
                models.append(call[field]["model"])
    windows = []
    for metadata, record in records:
        if metadata["boundary_source"] == "summary_request" and windows:
            # A captured summary request can close a preceding tool-call window with
            # its actual tool results. Only join if the entire replayed prefix matches.
            prior = windows[-1]
            n = len(prior["messages"])
            start = record["target_start_index"]
            if (
                start >= n
                and record["tools"] == prior["tools"]
                and compaction._replayed_prefix_matches(record["messages"][:n], prior["messages"])
                and all(m["role"] != "assistant" for m in record["messages"][n:start])
            ):
                windows[-1] = {**record, "target_start_index": prior["target_start_index"]}
                continue
        windows.append(record)
    converted = [from_sft(r, tools=platform_schemas(r["tools"], profile, "codex", trial.get("agent_version")))
                 for r in windows]
    for row in converted:
        row.update(provenance)
        if row["messages"][0]["role"] != "system":
            prompt = actf["system_prompt"] or system_text(required_profile(
                profile, "codex", trial.get("agent_version"))["system"])
            row["messages"].insert(0, message("system", prompt))
    return converted, models, {"captured_windows": len(windows)}


CONVERTERS = {"claude-code": claude, "codex": codex, "pi": pi}
