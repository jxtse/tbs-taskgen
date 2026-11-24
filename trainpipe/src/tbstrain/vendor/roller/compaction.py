# SPDX-License-Identifier: Apache-2.0
# Adapted from Roller, revision cd4c7f2. See NOTICE and LICENSE.
from __future__ import annotations
import json
from datetime import datetime
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from .sft import (
    SFTConversionError,
    _arguments_json,
    _image_data_uri,
    _normalize_tool_call,
    validate_sft_record,
)

SFT_SEGMENTS_DIR = "sft_segments"


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SFTConversionError("compaction or call lacks a timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SFTConversionError(f"invalid compaction timestamp: {value}") from exc
    if timestamp.tzinfo is None:
        raise SFTConversionError(f"compaction timestamp lacks timezone: {value}")
    return timestamp


def _text_blocks(content: Any, *, kinds: set[str], separator: str = "") -> str:
    """The text blocks of one message, joined the way the endpoint joins them.

    A chat template emits the text parts of a message back to back, so a
    separator here would put a character in the window that the model was never
    served and the prompt would no longer tokenize to what it answered.
    """

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise SFTConversionError("model input contains unsupported content")
    texts = []
    for block in content:
        if (
            not isinstance(block, Mapping)
            or block.get("type") not in kinds
            or not isinstance(block.get("text"), str)
        ):
            raise SFTConversionError("model input contains non-text content")
        texts.append(block["text"])
    return separator.join(texts)


def _image_part(block: Mapping[str, Any], field: str) -> dict[str, Any]:
    """One Anthropic image block as the multimodal part an SFT record carries."""

    return {
        "type": "image_url",
        "image_url": {"url": _image_data_uri(dict(block), field)},
    }


def _joined(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Text-only parts as the plain string they were; anything else as parts.

    The text is concatenated, not separated: see :func:`_text_blocks`.
    """

    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    return list(parts)


def _anthropic_tool_result(identifier: str, content: Any, *, images_inline: bool) -> list[dict[str, Any]]:
    """One Anthropic tool result as the messages the served prompt held.

    An image a tool returned is a part of the tool message itself, which is
    where a chat template that renders tool content in place expects it.
    ``images_inline=False`` puts those images in a user message after it
    instead, as vLLM's own Messages endpoint arranges such a turn: a window
    rebuilt from captured Calls has to hold whichever arrangement the model was
    served, or a trainer tokenizes a prompt that never existed.
    """

    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        raise SFTConversionError("Anthropic tool result content is invalid")
    texts: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise SFTConversionError("Anthropic tool result block is invalid")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            images.append(_image_part(block, "Anthropic tool result image"))
        else:
            raise SFTConversionError(f"unsupported Anthropic tool result block: {kind}")
    if images_inline:
        return [
            {
                "role": "tool",
                "tool_call_id": identifier,
                "content": _joined(texts + images),
            }
        ]
    messages: list[dict[str, Any]] = [{"role": "tool", "tool_call_id": identifier, "content": _joined(texts)}]
    if images:
        messages.append({"role": "user", "content": images})
    return messages


def _tool_call(identifier: Any, name: Any, arguments: Any, kind: str = "function") -> dict[str, Any]:
    if not isinstance(identifier, str) or not identifier or not isinstance(name, str) or not name:
        raise SFTConversionError("captured tool call lacks an ID or name")
    if kind == "custom":
        if not isinstance(arguments, str):
            raise SFTConversionError("captured custom tool input is not text")
        return {
            "id": identifier,
            "type": "custom",
            "custom": {"name": name, "input": arguments},
        }
    return _normalize_tool_call(
        {"id": identifier, "name": name, "arguments": arguments}, "captured tool call"
    )


def _anthropic_message(message: Mapping[str, Any], *, images_inline: bool = False) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise SFTConversionError("Anthropic message content is invalid")
    if role == "system":
        return [{"role": "system", "content": _text_blocks(content, kinds={"text"})}]
    if role == "assistant":
        visible, reasoning, calls = [], [], []
        for block in content:
            if not isinstance(block, Mapping):
                raise SFTConversionError("Anthropic assistant block is invalid")
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                visible.append(block["text"])
            elif kind == "thinking" and isinstance(block.get("thinking"), str):
                reasoning.append(block["thinking"])
            elif kind == "tool_use":
                calls.append(_tool_call(block.get("id"), block.get("name"), block.get("input", {})))
            elif kind == "redacted_thinking":
                raise SFTConversionError("redacted Anthropic thinking cannot be represented in SFT")
            else:
                raise SFTConversionError(f"unsupported Anthropic assistant block: {kind}")
        return [
            {
                "role": "assistant",
                "reasoning_content": "\n\n".join(reasoning),
                "content": "\n\n".join(visible),
                "tool_calls": calls,
            }
        ]
    if role != "user":
        raise SFTConversionError(f"unsupported Anthropic role: {role}")
    result: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []

    def flush() -> None:
        if parts:
            result.append({"role": "user", "content": _joined(parts)})
            parts.clear()

    for block in content:
        if not isinstance(block, Mapping):
            raise SFTConversionError("Anthropic user block is invalid")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            parts.append(_image_part(block, "Anthropic user image"))
        elif kind == "tool_result":
            flush()
            identifier = block.get("tool_use_id")
            if not isinstance(identifier, str) or not identifier:
                raise SFTConversionError("Anthropic tool result lacks an ID")
            result.extend(
                _anthropic_tool_result(
                    identifier,
                    block.get("content", ""),
                    images_inline=images_inline,
                )
            )
        else:
            raise SFTConversionError(f"unsupported Anthropic user block: {kind}")
    flush()
    return result


def _responses_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    kind = item.get("type")
    if kind == "message":
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise SFTConversionError(f"unsupported Responses role: {role}")
        value = _text_blocks(item.get("content", []), kinds={"input_text", "output_text"}, separator="")
        if role == "assistant":
            return [
                {
                    "role": role,
                    "reasoning_content": "",
                    "content": value,
                    "tool_calls": [],
                }
            ]
        return [{"role": role, "content": value}]
    if kind in {"function_call", "custom_tool_call"}:
        tool_kind = "custom" if kind == "custom_tool_call" else "function"
        call = _tool_call(
            item.get("call_id"),
            item.get("name"),
            item.get("input") if tool_kind == "custom" else item.get("arguments", {}),
            tool_kind,
        )
        return [
            {
                "role": "assistant",
                "reasoning_content": "",
                "content": "",
                "tool_calls": [call],
            }
        ]
    if kind in {"function_call_output", "custom_tool_call_output"}:
        identifier = item.get("call_id")
        if not isinstance(identifier, str) or not identifier:
            raise SFTConversionError("Responses tool output lacks a call ID")
        output = item.get("output")
        if not isinstance(output, str):
            raise SFTConversionError("Responses tool output is not text")
        return [{"role": "tool", "tool_call_id": identifier, "content": output}]
    if kind == "reasoning":
        # An empty placeholder contributes no prompt content. Any populated
        # reasoning item still needs a representation we do not have.
        if (
            item.get("summary") in (None, [])
            and item.get("content") in (None, [])
            and item.get("encrypted_content") in (None, "")
        ):
            return []
        raise SFTConversionError("Responses reasoning item cannot be represented in SFT")
    raise SFTConversionError(f"unsupported Responses input item: {kind}")


def _chat_content(content: Any) -> str | list[dict[str, Any]]:
    """OpenAI multimodal content list when images are present, plain text otherwise."""
    if not isinstance(content, list):
        return _text_blocks(content, kinds={"text"})
    parts: list[dict[str, Any]] = []
    has_image = False
    for block in content:
        image = block.get("image_url") if isinstance(block, Mapping) else None
        if isinstance(block, Mapping) and block.get("type") == "image_url" and isinstance(image, Mapping):
            uri = _image_data_uri({"data": image.get("url")}, "image_url")
            parts.append({"type": "image_url", "image_url": {"url": uri}})
            has_image = True
        else:
            parts.append({"type": "text", "text": _text_blocks([block], kinds={"text"})})
    if not has_image:
        return "\n".join(part["text"] for part in parts)
    return parts


def _chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise SFTConversionError(f"unsupported Chat Completions role: {role}")
    content = message.get("content")
    if content is None and role == "assistant":
        content = ""
    value = (
        _text_blocks(content, kinds={"text"})
        if role in {"system", "developer", "assistant"}
        else _chat_content(content)
    )
    if role == "tool":
        identifier = message.get("tool_call_id")
        if not isinstance(identifier, str) or not identifier:
            raise SFTConversionError("Chat Completions tool result lacks a call ID")
        return {"role": role, "tool_call_id": identifier, "content": value}
    if role == "assistant":
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise SFTConversionError("Chat Completions reasoning is not text")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise SFTConversionError("Chat Completions tool calls are invalid")
        calls = []
        for call in raw_calls:
            if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                raise SFTConversionError("Chat Completions tool call is invalid")
            function = call["function"]
            calls.append(_tool_call(call.get("id"), function.get("name"), function.get("arguments", {})))
        return {
            "role": role,
            "reasoning_content": reasoning,
            "content": value,
            "tool_calls": calls,
        }
    return {"role": role, "content": value}


def _merge_response_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Codex stores one response as separate text and tool items; ACTF keeps one step."""
    merged: list[dict[str, Any]] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if previous is not None and previous["role"] == message["role"] == "assistant":
            previous["content"] = "\n\n".join(
                text for text in (previous["content"], message["content"]) if text
            )
            previous["tool_calls"].extend(message["tool_calls"])
        else:
            merged.append(message)
    return merged


def _prompt_from_call(
    call: Mapping[str, Any], *, images_inline: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = call.get("context")
    if not isinstance(context, Mapping):
        raise SFTConversionError("captured Call lacks context")
    protocol = call.get("protocol")
    messages: list[dict[str, Any]] = []
    tools_value = context.get("tools", [])
    if not isinstance(tools_value, list):
        raise SFTConversionError("captured tools are invalid")
    if protocol == "anthropic-messages":
        system = context.get("system", "")
        if system:
            messages.append({"role": "system", "content": _text_blocks(system, kinds={"text"})})
        items = context.get("messages", [])
        if not isinstance(items, list):
            raise SFTConversionError("Anthropic request messages are invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Anthropic request has an invalid message")
            messages.extend(_anthropic_message(item, images_inline=images_inline))
        tools = []
        for tool in tools_value:
            if not isinstance(tool, Mapping) or not isinstance(tool.get("name"), str):
                raise SFTConversionError("Anthropic tool schema is invalid")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": deepcopy(tool.get("input_schema", {})),
                    },
                }
            )
    elif protocol == "openai-responses":
        instructions = context.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append({"role": "system", "content": instructions})
        items = context.get("input", [])
        if not isinstance(items, list):
            raise SFTConversionError("Responses request input is invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Responses request has an invalid item")
            messages.extend(_responses_item(item))
        messages = _merge_response_items(messages)
        tools = deepcopy(tools_value)
    elif protocol == "openai-chat-completions":
        items = context.get("messages", [])
        if not isinstance(items, list):
            raise SFTConversionError("Chat Completions request messages are invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Chat Completions request has an invalid message")
            messages.append(_chat_message(item))
        tools = deepcopy(tools_value)
    else:
        raise SFTConversionError(f"unsupported compaction Call protocol: {protocol}")
    if not isinstance(tools, list):
        raise SFTConversionError("captured tools are invalid")
    return messages, tools


def _target_messages_from_call(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = call.get("output")
    if not isinstance(output, Mapping):
        raise SFTConversionError("captured Call lacks output")
    if call.get("protocol") == "anthropic-messages":
        return _anthropic_message({"role": "assistant", "content": output.get("content", [])})
    if call.get("protocol") == "openai-responses":
        result: list[dict[str, Any]] = []
        items = output.get("output", [])
        if not isinstance(items, list):
            raise SFTConversionError("Responses output is invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Responses output contains an invalid item")
            if item.get("type") == "reasoning":
                # The Responses API may return encrypted reasoning. It is not
                # a supervised text target. A later request containing it is
                # rejected because the actual prompt cannot be represented.
                continue
            result.extend(_responses_item(item))
        result = _merge_response_items(result)
        if len(result) != 1 or result[0]["role"] != "assistant":
            raise SFTConversionError("Responses output has no representable assistant target")
        return result
    if call.get("protocol") == "openai-chat-completions":
        choices = output.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise SFTConversionError("Chat Completions output lacks one choice")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise SFTConversionError("Chat Completions output lacks an assistant target")
        target = _chat_message(message)
        # vLLM returns the thinking as ``reasoning``; a main step takes it
        # from ACTF instead, a kept summary from here.
        if not target["reasoning_content"] and isinstance(message.get("reasoning"), str):
            target["reasoning_content"] = message["reasoning"]
        return [target]
    raise SFTConversionError("unsupported compaction response protocol")


def _assistant_signature(
    message: Mapping[str, Any],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Compare model outputs with ACTF main steps without harness-specific tool IDs."""
    content = message.get("content")
    calls = message.get("tool_calls") or []
    if not isinstance(content, str) or not isinstance(calls, list):
        raise SFTConversionError("ACTF assistant output cannot be matched to a Call")
    tools = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise SFTConversionError("ACTF tool call cannot be matched to a Call")
        kind = call.get("type")
        if kind == "function":
            function = call.get("function")
        elif kind == "custom":
            function = call.get("custom")
        else:
            function = call
        if not isinstance(function, Mapping):
            raise SFTConversionError("ACTF tool call cannot be matched to a Call")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise SFTConversionError("ACTF tool call lacks a name")
        if kind == "custom" or call.get("native_type") == "custom_tool_call":
            arguments = function.get("input")
            if not isinstance(arguments, str):
                raise SFTConversionError("ACTF custom tool input is not text")
            tools.append((name, arguments))
        else:
            encoded = _arguments_json(
                function.get("arguments", function.get("input", {})),
                "ACTF tool arguments",
            )
            tools.append(
                (
                    name,
                    json.dumps(json.loads(encoded), ensure_ascii=False, sort_keys=True),
                )
            )
    return content, tuple(tools)


def _tool_arguments_match(observed: str, native: str) -> bool:
    """A Harness may record a tool input after applying its schema defaults.

    Claude Code stores ``Edit`` with ``replace_all: false`` in its native
    session even when the model never emitted the key, so the native step
    may carry keys the wire response does not. Every key the model did send
    must still be present and equal; keys only the Harness added are ignored.
    Non-JSON inputs (Codex custom tools) keep comparing exactly.
    """

    if observed == native:
        return True
    try:
        sent, recorded = json.loads(observed), json.loads(native)
    except (TypeError, ValueError):
        return False
    if not isinstance(sent, dict) or not isinstance(recorded, dict):
        return False
    return not set(sent) - set(recorded) and all(recorded[key] == value for key, value in sent.items())


def _signature_matches(
    observed: tuple[str, tuple[tuple[str, str], ...]],
    native: tuple[str, tuple[tuple[str, str], ...]],
) -> bool:
    """Bind a captured Call to a native step; text and tool order stay exact."""

    if observed == native:
        return True
    observed_content, observed_tools = observed
    native_content, native_tools = native
    if observed_content != native_content or len(observed_tools) != len(native_tools):
        return False
    return all(
        observed_name == native_name and _tool_arguments_match(observed_args, native_args)
        for (observed_name, observed_args), (native_name, native_args) in zip(observed_tools, native_tools)
    )


def _is_summary_call(
    call: Mapping[str, Any],
    messages: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
) -> bool:
    """The next native boundary recorded this Call's text as its summary."""
    started = _time(call.get("started_at"))
    boundary = next((item for item in boundaries if _time(item["timestamp"]) > started), None)
    summary = boundary["summary"] if boundary else None
    return (
        isinstance(summary, str)
        and len(messages) == 1
        and not messages[0]["tool_calls"]
        and bool(messages[0]["content"])
        and messages[0]["content"] in summary
    )


def _calls_matching_main_steps(
    trajectory: Mapping[str, Any],
    calls: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    summaries: set[str] | None = None,
    native_summaries: frozenset[str] = frozenset(),
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Bind root Calls to ACTF main steps in order; only native summaries are skipped.

    Codex and OpenClaw send compaction requests under the main identity. A
    Call is skipped only when it is named in ``native_summaries`` or it does
    not match the next main step and its text is recorded verbatim in the
    following native compaction row. With ``summaries``, such a Call is kept
    as a target and its ID added there.
    """
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    if not isinstance(steps, list) or not steps or any(not isinstance(step, Mapping) for step in steps):
        raise SFTConversionError("main ACTF steps are required to bind captured Calls")
    targets = [_assistant_signature(step.get("assistant_content", {})) for step in steps]
    verified = []
    for call in calls:
        if call.get("export_status") != "available":
            raise SFTConversionError(f"main Call {call.get('call_id')} is incomplete")
        messages = _target_messages_from_call(call)
        index = len(verified) - len(summaries or ())
        if (
            index < len(steps)
            and len(messages) == 1
            and _signature_matches(_assistant_signature(messages[0]), targets[index])
        ):
            # ACTF applies keep_thinking and retains visible Codex reasoning. Use
            # that canonical value while the Call verifies the output and prompt.
            reasoning = steps[index]["assistant_content"].get("reasoning_content")
            if isinstance(reasoning, str):
                messages[0]["reasoning_content"] = reasoning
            verified.append((call, messages))
        elif not (call["call_id"] in native_summaries or _is_summary_call(call, messages, boundaries)):
            raise SFTConversionError(
                f"root Calls do not correspond one-to-one with ACTF main steps; Call "
                f"{call.get('call_id')} matches neither the next step nor a native summary"
            )
        elif summaries is not None:
            summaries.add(call["call_id"])
            verified.append((call, messages))
    if len(verified) - len(summaries or ()) != len(steps):
        raise SFTConversionError(
            "root Calls do not correspond one-to-one with ACTF main steps; "
            f"{len(steps) - len(verified)} steps have no Call"
        )
    return verified


def _root_calls(calls: list[dict[str, Any]], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    participants = manifest.get("participants")
    if not isinstance(participants, list):
        raise SFTConversionError("capture manifest lacks participants")
    roots = {
        p.get("participant_id") for p in participants if isinstance(p, Mapping) and p.get("role") == "root"
    }
    if len(roots) != 1 or not all(isinstance(root, str) and root for root in roots):
        raise SFTConversionError("compaction requires one observed root participant")
    return [
        call
        for call in sorted(calls, key=lambda c: c.get("request_seq", 0))
        if call.get("participant_id") in roots
    ]


def _without_tool_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenClaw rewrites tool-call IDs when replaying history to some providers."""
    return [
        {
            **message,
            "tool_calls": [{**call, "id": ""} for call in message["tool_calls"]],
        }
        if message["role"] == "assistant"
        else {**message, "tool_call_id": ""}
        if message["role"] == "tool"
        else message
        for message in messages
    ]


def _replayed_prefix_matches(replayed: list[dict[str, Any]], stored: list[dict[str, Any]]) -> bool:
    """A window's own messages against the same messages in a later prompt.

    A stored window ends with the target as the model sent it, while the next
    prompt replays that turn as the Harness recorded it: the same arguments
    can come back with the schema defaults applied and the keys in another
    order. Compare tool arguments by value, so that only a real context
    rewrite ends the window.
    """

    normalized_replayed = _without_tool_ids(replayed)
    normalized_stored = _without_tool_ids(stored)
    if len(normalized_replayed) != len(normalized_stored):
        return False
    for replayed_message, stored_message in zip(normalized_replayed, normalized_stored):
        if replayed_message == stored_message:
            continue
        replayed_calls = replayed_message.get("tool_calls") or []
        stored_calls = stored_message.get("tool_calls") or []
        if len(replayed_calls) != len(stored_calls) or _without_tool_calls(
            replayed_message
        ) != _without_tool_calls(stored_message):
            return False
        for replayed_call, stored_call in zip(replayed_calls, stored_calls):
            if _without_arguments(replayed_call) != _without_arguments(
                stored_call
            ) or not _tool_arguments_match(_call_arguments(stored_call), _call_arguments(replayed_call)):
                return False
    return True


def _without_tool_calls(message: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if key != "tool_calls"}


def _without_arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return dict(call)
    return {
        **call,
        "function": {key: value for key, value in function.items() if key != "arguments"},
    }


def _call_arguments(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    return arguments if isinstance(arguments, str) else ""


def _segment_records(
    calls: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    boundaries: list[dict[str, Any]],
    keep_thinking: bool,
    summaries: set[str] = frozenset(),
    *,
    images_inline: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Use one continuation rule; window_index counts emitted SFT records.

    A summary Call in ``summaries`` is a record of its own, so the summary
    never extends the window before it or the one after it.
    """
    times = [_time(boundary["timestamp"]) for boundary in boundaries]
    if any(not isinstance(boundary["event_id"], str) or not boundary["event_id"] for boundary in boundaries):
        raise SFTConversionError("compaction boundary lacks a native event ID")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise SFTConversionError("compaction boundaries are not in chronological order")
    exported: list[tuple[dict[str, Any], dict[str, Any]]] = []
    native_index = -1
    for call, target in calls:
        when = _time(call.get("started_at")) if boundaries else None
        if when in times:
            raise SFTConversionError("main Call starts at an ambiguous compaction boundary")
        current_native = sum(when > boundary for boundary in times) if when else 0
        prompt, tools = _prompt_from_call(call, images_inline=images_inline)
        if not prompt:
            raise SFTConversionError("compaction segment lacks its actual input")
        if not keep_thinking:
            for message in [*prompt, *target]:
                if message["role"] == "assistant":
                    message["reasoning_content"] = ""
        previous = exported[-1][1] if exported else None
        prefix = previous["messages"] if previous else []
        summary = call["call_id"] in summaries
        continuation = (
            previous is not None
            and not summary
            and exported[-1][0]["first_call_id"] not in summaries
            and current_native == native_index
            and tools == previous["tools"]
            and _replayed_prefix_matches(prompt[: len(prefix)], prefix)
            and not any(item["role"] == "assistant" for item in prompt[len(prefix) :])
        )
        if continuation:
            # The newest prompt carries the IDs the harness actually replayed.
            previous["messages"] = prompt + target
            exported[-1][0]["last_call_id"] = call["call_id"]
        else:
            boundary = (
                boundaries[current_native - 1] if current_native > native_index and current_native else None
            )
            metadata = {
                "path": f"{SFT_SEGMENTS_DIR}/{len(exported) + 1:04d}.json",
                "window_index": len(exported),
                "boundary_source": (
                    "summary_request"
                    if summary
                    else "native_boundary"
                    if boundary
                    else "observed_context_change"
                    if exported
                    else "initial"
                ),
                "boundary_event_id": boundary["event_id"] if boundary else None,
                "first_call_id": call["call_id"],
                "last_call_id": call["call_id"],
            }
            record = {
                "messages": prompt + target,
                "tools": tools,
                "target_start_index": len(prompt),
            }
            # Only assistant messages at/after this index enter the loss;
            # intervening user/tool messages are context for later targets.
            exported.append((metadata, record))
        native_index = current_native
    if not exported:
        raise SFTConversionError("compaction has no complete main-agent Calls")
    for _, record in exported:
        validate_sft_record(record)
    return exported
