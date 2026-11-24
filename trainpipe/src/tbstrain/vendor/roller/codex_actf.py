# SPDX-License-Identifier: Apache-2.0
# Adapted from Roller, revision cd4c7f2. See NOTICE and LICENSE.
"""Projection of Codex CLI native session JSONL into Roller ACTF."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actf import append_user_content, user_content


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return "" if value is None else str(value)
    parts: list[str] = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _instruction_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return text
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _instruction_text(item)))
    return ""


def _reasoning_text(payload: Mapping[str, Any], include_summary: bool) -> tuple[str, str]:
    visible = []
    summaries = []
    for block in payload.get("content") or []:
        if isinstance(block, str):
            visible.append(block)
        elif isinstance(block, Mapping) and block.get("text"):
            if block.get("type") == "summary_text":
                summaries.append(block["text"])
            elif block.get("type") in (None, "text", "reasoning_text"):
                visible.append(block["text"])
    if any(visible):
        return "\n".join(filter(None, visible)), "visible"
    for block in payload.get("summary") or []:
        if isinstance(block, str):
            summaries.append(block)
        elif isinstance(block, Mapping) and block.get("text"):
            summaries.append(block["text"])
    if any(summaries):
        return ("\n".join(filter(None, summaries)), "summary") if include_summary else ("", "summary_dropped")
    return "", "redacted" if payload.get("encrypted_content") else "none"


def _arguments(payload: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    payload_type = payload.get("type")
    raw = payload.get("arguments") if payload_type == "function_call" else payload.get("input")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}, raw
    else:
        parsed = raw
    if isinstance(parsed, dict):
        return deepcopy(parsed), raw
    if parsed is None:
        return {}, raw
    return {"value": deepcopy(parsed)}, raw


_OUTPUT_TEXT_BLOCK_TYPES = frozenset({"text", "input_text", "output_text"})
_OUTPUT_IMAGE_BLOCK_TYPES = frozenset({"image", "input_image", "image_url"})
_OUTPUT_UNSUPPORTED_BLOCK_TYPES = frozenset({"input_file", "input_audio"})


def _output_block_kind(value: Any) -> str | None:
    """Classify a declared output block without validating its fields.

    Field validation belongs to the ACTF-to-SFT converter, which can report
    malformed or unsupported media instead of silently turning a recognizable
    typed block into JSON text.
    """

    if not isinstance(value, Mapping):
        return None
    block_type = value.get("type")
    if block_type in _OUTPUT_TEXT_BLOCK_TYPES:
        return "text"
    if block_type in _OUTPUT_IMAGE_BLOCK_TYPES:
        return "image"
    if block_type in _OUTPUT_UNSUPPORTED_BLOCK_TYPES:
        return "unsupported"
    return None


def _is_output_content_block_list(value: Any) -> bool:
    # An empty JSON array has no structural evidence that it is a typed block
    # list. The caller handles a native list object separately so a genuine
    # empty content-block list can still remain empty.
    if not isinstance(value, list) or not value:
        return False
    # A fully discriminated list is a content-block list even when one block
    # has malformed fields. An explicit image discriminator is also decisive:
    # keep the media semantics while losslessly normalizing ordinary siblings,
    # and let malformed known blocks fail validation instead of downgrading the
    # whole result to JSON text.
    block_kinds = [_output_block_kind(item) for item in value]
    return all(kind is not None for kind in block_kinds) or "image" in block_kinds


def _json_text_block(value: Any) -> dict[str, Any]:
    return {"type": "text", "text": json.dumps(value, ensure_ascii=False)}


def _normalize_output_block(value: Any) -> dict[str, Any]:
    """Normalize one member of a recognized content-block list."""

    if isinstance(value, str):
        return {"type": "text", "text": value}

    block_kind = _output_block_kind(value)
    if block_kind == "text":
        return {"type": "text", "text": deepcopy(value.get("text"))}
    if block_kind == "image":
        image = deepcopy(dict(value))
        image["type"] = "image"
        image_url = image.pop("image_url", None)
        if "data" not in image and image_url is not None:
            image["data"] = image_url.get("url") if isinstance(image_url, Mapping) else image_url
        return image
    if block_kind == "unsupported":
        # Multimodal SFT cannot represent these typed Responses blocks. Keep
        # them in ACTF so conversion fails explicitly instead of flattening
        # them into text.
        return deepcopy(dict(value))
    return _json_text_block(value)


def _output_content_blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list):
        if not value:
            return []
        if not _is_output_content_block_list(value):
            return [_json_text_block(value)]
        # Classification recognizes block shapes; validation remains in the
        # ACTF-to-SFT converter so malformed known blocks fail explicitly.
        return [_normalize_output_block(item) for item in value]
    if value is None:
        return [{"type": "text", "text": ""}]
    if isinstance(value, Mapping):
        if _output_block_kind(value) is not None:
            return [_normalize_output_block(value)]
        return [_json_text_block(value)]
    return [_json_text_block(value)]


def _decode_output(value: Any) -> tuple[Any, bool]:
    """Decode JSON text while retaining whether the raw value was encoded."""

    if not isinstance(value, str):
        return value, False
    try:
        return json.loads(value), True
    except json.JSONDecodeError:
        return value, False


def _unwrap_output(value: Any) -> tuple[Any, dict[str, Any] | None]:
    """Extract the output/metadata envelope used by Codex tool results."""

    if not isinstance(value, Mapping):
        return value, None
    output = value.get("output") if "output" in value else value
    if output is None:
        output = json.dumps(value, ensure_ascii=False)
    metadata = value.get("metadata")
    return output, deepcopy(metadata) if isinstance(metadata, dict) else None


def _output_content(
    value: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    decoded, decoded_from_json_string = _decode_output(value)
    output, metadata = _unwrap_output(decoded)
    if decoded_from_json_string and isinstance(output, list) and not _is_output_content_block_list(output):
        # Preserve an encoded ordinary JSON array as one JSON value. In
        # particular, the string "[]" must not collapse into no output.
        output = json.dumps(output, ensure_ascii=False)
    return _output_content_blocks(output), metadata


def _user_image_blocks(value: Any) -> list[dict[str, Any]] | None:
    """Text and image blocks of a user message that carries an image, else None."""

    if not isinstance(value, list):
        return None
    kinds = [_output_block_kind(item) for item in value]
    if "image" not in kinds:
        return None
    return [_normalize_output_block(item) for item, kind in zip(value, kinds) if kind in {"text", "image"}]


def _usage_number(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def session_tools(events: Iterable[Mapping[str, Any]]) -> tuple[list[Any], str]:
    """The Responses tools of a session file and how they were found.

    Codex records its tools as ``additional_tools`` items. When they change in
    the session, the lists are merged: a tool joins in the order it first
    appears, with its last definition, and the status is ``captured_merged``.
    A session without these items has its tools inferred from the calls.
    """

    versions: set[str] = set()
    # Top-level tools and namespaces by name; the tools of a namespace by name.
    merged: dict[str, dict[str, Any]] = {}
    children: dict[str, dict[str, Any]] = {}
    for event in events:
        payload = event.get("payload")
        if event.get("type") != "response_item" or not isinstance(payload, Mapping):
            continue
        if payload.get("type") != "additional_tools":
            continue
        versions.add(json.dumps(payload.get("tools"), sort_keys=True))
        for tool in payload.get("tools") or []:
            name = tool.get("name") or tool.get("type")
            merged[name] = tool
            if tool.get("type") == "namespace":
                group = children.setdefault(name, {})
                for child in tool.get("tools") or []:
                    group[child.get("name") or child.get("type")] = child
    if versions:
        tools = []
        for name, tool in merged.items():
            if tool.get("type") != "namespace":
                tools.append(tool)
            elif name == "functions":
                # Codex calls the tools of this namespace without a namespace.
                tools += children[name].values()
            else:
                tools.append({**tool, "tools": list(children[name].values())})
        return deepcopy(tools), "captured_merged" if len(versions) > 1 else "captured"

    # Calls by namespace and name; a function lists the arguments it was given.
    inferred: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        payload = event.get("payload")
        if event.get("type") != "response_item" or not isinstance(payload, Mapping):
            continue
        key = (payload.get("namespace") or "", payload.get("name"))
        if payload.get("type") == "custom_tool_call":
            inferred.setdefault(key, {"type": "custom", "name": key[1]})
        elif payload.get("type") == "function_call":
            tool = inferred.setdefault(
                key,
                {
                    "type": "function",
                    "name": key[1],
                    "description": "Tool invoked by the Codex native harness.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            )
            for argument in _arguments(payload)[0]:
                tool["parameters"]["properties"].setdefault(argument, {})
    tools = []
    namespaces: dict[str, dict[str, Any]] = {}
    for (namespace, _), tool in inferred.items():
        if not namespace:
            tools.append(tool)
            continue
        if namespace not in namespaces:
            namespaces[namespace] = {
                "type": "namespace",
                "name": namespace,
                "tools": [],
            }
            tools.append(namespaces[namespace])
        namespaces[namespace]["tools"].append(tool)
    return tools, "inferred"


def native_response_id(event):
    payload = event.get("payload") or {}
    return (payload.get("response_id") or event.get("response_id")
            or payload.get("responseId") or event.get("responseId"))


def codex_events_to_actf(
    events: Iterable[dict[str, Any]],
    *,
    user_messages: Iterable[str] | None = None,
    keep_thinking: bool = True,
    include_summary: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tools_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one Codex native rollout stream to ``ACTF_v1.0``.

    Codex emits reasoning, messages and tool calls as separate response items.
    A ``token_count`` event closes the corresponding model request, so all
    response items before that boundary are grouped into one ACTF assistant
    step. An explicit response ID joins its items across intervening tool
    outputs. Without an ID, an output closes the preceding step. Tool outputs
    are paired by ``call_id``.
    """

    rows = list(events)
    expected_users = tuple(user_messages or ())
    user_index = 0
    pending_user: str | list[Any] = ""
    pending_user_time: str | None = None
    pending_reasoning: list[str] = []
    reasoning_sources = set()
    system_parts: list[str] = []
    system_seen: set[str] = set()
    groups: list[dict[str, Any]] = []
    identified_groups = {}
    current: dict[str, Any] | None = None
    calls_by_id: dict[str, dict[str, Any]] = {}
    first_time: str | None = None
    last_time: str | None = None

    def add_system(value: Any) -> None:
        text = _instruction_text(value)
        if text.strip() and text not in system_seen:
            system_seen.add(text)
            system_parts.append(text)

    def group(timestamp: str | None = None, response_id=None) -> dict[str, Any]:
        nonlocal current, pending_user, pending_user_time, pending_reasoning
        if response_id:
            if response_id in identified_groups:
                current = identified_groups[response_id]
            elif current and current.get("response_id") and current["response_id"] != response_id:
                close_group({}, timestamp)
        if current is None:
            current = {
                "user_content": pending_user,
                "user_time": pending_user_time,
                "started_at": pending_user_time or timestamp or first_time or "",
                "finished_at": timestamp or pending_user_time or first_time or "",
                "content": [],
                "reasoning": pending_reasoning,
                "tool_calls": [],
                "metric": {},
            }
            pending_user = ""
            pending_user_time = None
            pending_reasoning = []
            groups.append(current)
        elif pending_reasoning:
            current["reasoning"].extend(pending_reasoning)
            pending_reasoning = []
        if response_id:
            current["response_id"] = response_id
            identified_groups[response_id] = current
        if timestamp:
            current["finished_at"] = timestamp
        return current

    def close_group(payload: Mapping[str, Any], timestamp: str | None) -> None:
        nonlocal current
        if current is None:
            return
        if pending_reasoning:
            current["reasoning"].extend(pending_reasoning)
            pending_reasoning.clear()
        info = payload.get("info")
        last_usage = info.get("last_token_usage") if isinstance(info, Mapping) else None
        if isinstance(last_usage, Mapping):
            current["metric"] = {
                "prompt_tokens_len": _usage_number(last_usage.get("input_tokens")),
                "completion_tokens_len": _usage_number(last_usage.get("output_tokens")),
                "reasoning_tokens_len": _usage_number(last_usage.get("reasoning_output_tokens")),
                "cache_read_tokens_len": _usage_number(last_usage.get("cached_input_tokens")),
                "cache_write_tokens_len": _usage_number(last_usage.get("cache_write_input_tokens")),
            }
        if timestamp:
            current["finished_at"] = timestamp
        current = None

    for event in rows:
        timestamp = _timestamp(event.get("timestamp"))
        first_time = first_time or timestamp
        last_time = timestamp or last_time
        event_type = event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}

        if event_type in {"session_meta", "turn_context"}:
            add_system(payload.get("base_instructions"))
            add_system(payload.get("instructions"))
            add_system(payload.get("developer_instructions"))
            continue

        if event_type == "event_msg":
            payload_type = payload.get("type")
            if payload_type == "token_count":
                close_group(payload, timestamp)
            elif payload_type == "web_search_end":
                call_id = payload.get("call_id")
                if isinstance(call_id, str) and call_id:
                    action = payload.get("action")
                    arguments: dict[str, Any] = {}
                    if isinstance(payload.get("query"), str):
                        arguments["query"] = payload["query"]
                    if action is not None:
                        arguments["action"] = deepcopy(action)
                    call = {
                        "type": "toolCall",
                        "id": call_id,
                        "name": "web_search",
                        "arguments": arguments,
                        "output": json.dumps(payload.get("results", []), ensure_ascii=False),
                        "output_time": timestamp,
                        "is_error": False,
                    }
                    group(timestamp)["tool_calls"].append(call)
                    calls_by_id[call_id] = call
                    close_group({}, timestamp)
            continue

        if event_type == "compacted":
            # Local compaction records its summary response just before this
            # row and stores the last summary text as the suffix of ``message``.
            # That response answered the compaction prompt, not the task.
            summary = payload.get("message")
            last = groups[-1] if groups else None
            if (
                isinstance(summary, str)
                and last is not None
                and not last["tool_calls"]
                and last["content"]
                and summary.endswith(last["content"][-1])
            ):
                groups.pop()
                current = None
                if last["user_content"]:
                    pending_user, pending_user_time = (
                        last["user_content"],
                        last["user_time"],
                    )
            pending_reasoning = []
            continue

        if event_type != "response_item":
            continue

        payload_type = payload.get("type")
        response_id = native_response_id(event)
        if payload_type == "reasoning":
            reasoning, source = _reasoning_text(payload, include_summary)
            if source != "none":
                reasoning_sources.add(source)
            if reasoning:
                if response_id:
                    group(timestamp, response_id)["reasoning"].append(reasoning)
                else:
                    pending_reasoning.append(reasoning)
            continue

        if payload_type == "message":
            role = payload.get("role")
            content = _content_text(payload.get("content"))
            if role in {"developer", "system"}:
                add_system(content)
            elif role == "user":
                value: str | list[Any]
                expected = None
                if user_index < len(expected_users):
                    expected = expected_users[user_index]
                    normalized_content = content.strip()
                    normalized_expected = expected.strip()
                    if normalized_content == normalized_expected or (
                        normalized_expected and normalized_expected in normalized_content
                    ):
                        value = expected
                        user_index += 1
                    else:
                        if normalized_content.startswith("<environment_context>"):
                            add_system(content)
                        continue
                else:
                    value = content
                image_blocks = _user_image_blocks(payload.get("content"))
                if value or image_blocks:
                    if image_blocks is not None:
                        value = user_content(image_blocks, expected)
                    pending_user = append_user_content(pending_user, value)
                    pending_user_time = pending_user_time or timestamp
            elif role == "assistant":
                target = group(timestamp, response_id)
                if content:
                    target["content"].append(content)
            continue

        if payload_type in {"function_call", "custom_tool_call"}:
            call_id = payload.get("call_id")
            name = payload.get("name")
            if not isinstance(call_id, str) or not call_id:
                continue
            if not isinstance(name, str) or not name:
                name = "unknown_tool"
            arguments, raw_arguments = _arguments(payload)
            call = {
                "type": "toolCall",
                "id": call_id,
                "name": name,
                "arguments": arguments,
                "raw_arguments": raw_arguments,
                "native_type": payload_type,
                "namespace": payload.get("namespace"),
                "output": None,
                "output_time": None,
                "is_error": False,
            }
            group(timestamp, response_id)["tool_calls"].append(call)
            calls_by_id[call_id] = call
            continue

        if payload_type in {"function_call_output", "custom_tool_call_output"}:
            close_group({}, timestamp)
            call_id = payload.get("call_id")
            call = calls_by_id.get(call_id) if isinstance(call_id, str) else None
            if call is None:
                continue
            output, metadata = _output_content(payload.get("output"))
            call["output"] = output
            call["output_time"] = timestamp
            call["metadata"] = metadata
            exit_code = metadata.get("exit_code") if isinstance(metadata, Mapping) else None
            call["is_error"] = (
                isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0
            )

    if current is not None:
        current["finished_at"] = last_time or current["finished_at"]

    steps: list[dict[str, Any]] = []
    for raw_group in groups:
        calls: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        for raw_call in raw_group["tool_calls"]:
            call = {
                "type": "toolCall",
                "id": raw_call["id"],
                "name": raw_call["name"],
                "arguments": deepcopy(raw_call["arguments"]),
            }
            if raw_call.get("namespace"):
                call["namespace"] = raw_call["namespace"]
            if raw_call.get("native_type") == "custom_tool_call":
                call["native_type"] = "custom_tool_call"
                call["input"] = deepcopy(raw_call["raw_arguments"])
            calls.append(call)
            output = raw_call.get("output")
            missing = output is None
            if missing:
                content = [
                    {
                        "type": "text",
                        "text": "[Codex trace ended before this tool returned]",
                    }
                ]
            elif isinstance(output, list):
                content = deepcopy(output)
            else:
                content = _output_content_blocks(output)
            observations.append(
                {
                    "role": "toolResult",
                    "toolCallId": raw_call["id"],
                    "toolName": raw_call["name"],
                    "content": content,
                    "isError": bool(raw_call.get("is_error")) or missing,
                    **({"timestamp": raw_call["output_time"]} if raw_call.get("output_time") else {}),
                }
            )
        reasoning = "\n\n".join(item for item in raw_group["reasoning"] if isinstance(item, str) and item)
        metric = {
            "prompt_tokens_len": 0,
            "completion_tokens_len": 0,
            "llm_infer_ms": None,
            "env_action_ms": None,
            "stop_reason": "tool_use" if calls else "stop",
            **raw_group["metric"],
        }
        steps.append(
            {
                "step_id": len(steps) + 1,
                "response_id": raw_group.get("response_id"),
                "assistant_content": {
                    "content": "\n\n".join(raw_group["content"]),
                    "reasoning_content": reasoning if keep_thinking else "",
                    "tool_calls": calls,
                },
                "metric": metric,
                "user_content": raw_group["user_content"],
                "tools": deepcopy(calls),
                "observation": observations,
                "started_at": raw_group["started_at"],
                "finished_at": raw_group["finished_at"],
            }
        )

    if steps and not any(step["user_content"] for step in steps) and expected_users:
        steps[0]["user_content"] = expected_users[0]

    return {
        "schema_version": "ACTF_v1.0",
        "system_prompt": "\n\n".join(system_parts),
        "tools": deepcopy(tools),
        "tools_format": "responses",
        "tools_capture": deepcopy(tools_capture)
        if tools_capture is not None
        else {
            "source": "provided" if tools is not None else "none",
            "status": "captured" if tools is not None else "missing",
        },
        "steps": steps,
        "reasoning_source": "+".join(sorted(reasoning_sources)) or "none",
        "started_at": steps[0]["started_at"] if steps else first_time or "",
        "finished_at": steps[-1]["finished_at"] if steps else last_time or first_time or "",
    }
