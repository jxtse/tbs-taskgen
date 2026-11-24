# SPDX-License-Identifier: Apache-2.0
# Adapted from Roller, revision cd4c7f2. See NOTICE and LICENSE.
from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .codex_tools import tool_index


class SFTConversionError(ValueError):
    """Raised when an ACTF trajectory cannot form a valid SFT conversation."""


class SFTUnsupportedContentError(SFTConversionError):
    """Raised when valid ACTF content cannot be represented without data loss."""


# Multimodal SFT contract: messages with images use standard OpenAI multimodal
# content lists (`{"type": "text", "text": ...}` and
# `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}`).
# Text-only messages keep a plain string ``content``.
_LEGACY_IMAGE_PLACEHOLDER = "<image>"
_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise SFTConversionError(f"{field} must be a string")
    return value


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    # ``None`` means the wire trace did not expose a tools schema. It is still
    # safe to export trajectories that never called a tool; calls are checked
    # against this empty list below.
    if value is None:
        return []
    if not isinstance(value, list):
        raise SFTConversionError("tools must be a list or null")

    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise SFTConversionError(f"tools[{index}] must be a function tool")
        function = tool.get("function")
        if function is None:
            # The Responses API flattens function fields onto each tool item;
            # Chat Completions nests them below ``function``. This legacy path
            # uses the nested representation; captured Codex tools bypass it.
            function = {key: deepcopy(item) for key, item in tool.items() if key != "type"}
        elif not isinstance(function, dict):
            raise SFTConversionError(f"tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise SFTConversionError(f"tools[{index}].function.name must be non-empty")
        if name in names:
            raise SFTConversionError(f"duplicate tool name: {name}")
        names.add(name)
        tools.append({"type": "function", "function": deepcopy(function)})
    return tools


def _arguments_json(value: Any, field: str) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SFTConversionError(f"{field} must contain valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise SFTConversionError(f"{field} must encode a JSON object")
    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SFTConversionError(f"{field} is not JSON-serializable") from exc


def _normalize_tool_call(call: Any, field: str) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise SFTConversionError(f"{field} must be an object")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise SFTConversionError(f"{field}.id must be non-empty")

    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise SFTConversionError(f"{field} tool name must be non-empty")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _arguments_json(arguments, f"{field}.arguments"),
        },
    }


def _responses_tool_call(call: Any, field: str) -> dict[str, Any]:
    if isinstance(call, dict) and call.get("native_type") == "custom_tool_call":
        call_id = call.get("id")
        name = call.get("name")
        value = call.get("input")
        if not all(isinstance(x, str) and x for x in (call_id, name)) or not isinstance(value, str):
            raise SFTConversionError(f"{field} has invalid custom tool input")
        result = {
            "id": call_id,
            "type": "custom",
            "custom": {"name": name, "input": value},
        }
    else:
        result = _normalize_tool_call(call, field)
    if isinstance(call, dict) and "namespace" in call:
        namespace = call["namespace"]
        if not isinstance(namespace, str) or not namespace:
            raise SFTConversionError(f"{field}.namespace must be a non-empty string")
        result[result["type"]]["namespace"] = namespace
    return result


def _image_media_type(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise SFTConversionError(f"{field} must be a string")
    media_type = value.strip().lower()
    if media_type not in _IMAGE_MEDIA_TYPES:
        supported = ", ".join(sorted(_IMAGE_MEDIA_TYPES))
        raise SFTConversionError(f"{field} must be one of {supported}; got {value!r}")
    return media_type


def _image_data_uri(block: Any, field: str) -> str:
    """Normalize one ACTF image block to a canonical ``data:`` URI.

    Accepts the base64 ``source`` form (Claude Code), the flat
    ``data`` + ``mimeType`` form (OpenClaw/Pi native), and an inline
    ``data:...;base64,...`` URI. The converter never reads files or networks;
    blocks that only reference a path keep failing conversion.
    """
    if not isinstance(block, dict):
        raise SFTConversionError(f"{field} must be an object")
    source = block.get("source")
    if isinstance(source, dict):
        if source.get("type") != "base64":
            raise SFTConversionError(f"{field}.source.type must be 'base64'")
        media_type = _image_media_type(source.get("media_type"), f"{field}.source.media_type")
        payload = source.get("data")
    elif isinstance(block.get("data"), str):
        payload = block["data"]
        if payload.startswith("data:"):
            header, separator, encoded = payload.partition(",")
            if not separator or not header.endswith(";base64"):
                raise SFTConversionError(f"{field}.data must be a base64 data URI")
            media_type = _image_media_type(header[5:-7], f"{field}.data media type")
            payload = encoded
        else:
            media_type = _image_media_type(block.get("mimeType", block.get("mime_type")), f"{field}.mimeType")
    else:
        raise SFTConversionError(f"{field} must carry base64 image bytes in source.data or data")
    if not isinstance(payload, str) or not payload:
        raise SFTConversionError(f"{field} image data must be non-empty base64")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SFTConversionError(f"{field} image data is not valid base64") from exc
    # Decoding accepts some redundant padding; never persist it verbatim.
    canonical = base64.b64encode(decoded).decode("ascii")
    return f"data:{media_type};base64,{canonical}"


def parse_image_data_uri(uri: str, field: str = "image_url.url") -> tuple[str, bytes]:
    """Validate a canonical ``data:<mime>;base64,<payload>`` URI and decode its bytes."""
    if not isinstance(uri, str) or not uri.startswith("data:"):
        raise SFTConversionError(f"{field} must be a base64 image data URI")
    header, separator, payload = uri.partition(",")
    if not separator or not header.endswith(";base64"):
        raise SFTConversionError(f"{field} must be a base64 image data URI")
    media_type = _image_media_type(header[5:-7], f"{field} media type")
    if not payload:
        raise SFTConversionError(f"{field} image data must be non-empty base64")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SFTConversionError(f"{field} image data is not valid base64") from exc
    return media_type, decoded


def _message_content(value: Any, field: str) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise SFTConversionError(f"{field} must be a string or content-block list")

    parts: list[dict[str, Any]] = []
    has_image = False
    for index, block in enumerate(value):
        block_field = f"{field}[{index}]"
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            raise SFTUnsupportedContentError(f"{block_field} is non-text; SFT export would lose data")
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise SFTConversionError(f"{block_field}.text must be a string")
            parts.append({"type": "text", "text": text})
        elif block.get("type") == "image":
            uri = _image_data_uri(block, block_field)
            parts.append({"type": "image_url", "image_url": {"url": uri}})
            has_image = True
        elif block.get("type") == "image_url":
            image = block.get("image_url")
            if not isinstance(image, dict):
                raise SFTConversionError(f"{block_field}.image_url must be an object")
            uri = _image_data_uri({"data": image.get("url")}, block_field)
            parts.append({"type": "image_url", "image_url": {"url": uri}})
            has_image = True
        elif block.get("type") == "tool_reference":
            # A tool search result names the tool it loaded; keep it as text.
            parts.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
        else:
            raise SFTUnsupportedContentError(f"{block_field} is non-text; SFT export would lose data")
    if not has_image:
        return "\n".join(part["text"] for part in parts)
    return parts


def _validate_content_part(part: Any, field: str) -> None:
    if not isinstance(part, Mapping):
        raise ValueError(f"{field} must be an object")
    part_type = part.get("type")
    if part_type == "text":
        if not isinstance(part.get("text"), str):
            raise ValueError(f"{field}.text must be a string")
        return
    if part_type == "image_url":
        image = part.get("image_url")
        if not isinstance(image, Mapping):
            raise ValueError(f"{field}.image_url must be an object")
        uri = image.get("url")
        if not isinstance(uri, str) or not uri.startswith("data:"):
            raise ValueError(f"{field}.image_url.url must be a base64 image data URI")
        try:
            canonical = _image_data_uri({"data": uri}, f"{field}.image_url.url")
        except SFTConversionError as exc:
            raise ValueError(f"{field}.image_url.url must be a base64 image data URI: {exc}") from exc
        if canonical != uri:
            raise ValueError(f"{field}.image_url.url must be a canonical base64 image data URI")
        return
    raise ValueError(f"{field}.type must be 'text' or 'image_url'; got {part_type!r}")


def _legacy_placeholder_count(message: Mapping[str, Any]) -> int:
    return json.dumps(message, ensure_ascii=False).count(_LEGACY_IMAGE_PLACEHOLDER)


def validate_sft_record(record: Any) -> None:
    """Check the top-level shape and OpenAI multimodal content contract.

    Records carry ``messages`` + ``tools`` (and ``target_start_index`` on
    compaction segments). Text-only messages store a string ``content``;
    multimodal messages store an OpenAI content-part list of ``text`` and
    ``image_url`` blocks with canonical base64 data URIs.
    """
    if not isinstance(record, Mapping):
        raise ValueError("SFT record must be a JSON object")
    keys = set(record)
    target_start = record.get("target_start_index")
    if "target_start_index" in keys:
        if (
            not isinstance(target_start, int)
            or isinstance(target_start, bool)
            or target_start < 1
            or not isinstance(record.get("messages"), list)
            or target_start >= len(record["messages"])
        ):
            raise ValueError("target_start_index must identify a target after the prompt")
        target = record["messages"][target_start]
        if not isinstance(target, Mapping) or target.get("role") != "assistant":
            raise ValueError("target_start_index must point to an assistant target")
        keys.remove("target_start_index")
    if keys != {"messages", "tools"}:
        raise ValueError("SFT record may only contain messages and tools")
    messages = record["messages"]
    if not isinstance(messages, list):
        raise ValueError("SFT messages must be a list")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if isinstance(content, list):
            if not content:
                raise ValueError(f"messages[{index}].content list must not be empty")
            for part_index, part in enumerate(content):
                _validate_content_part(part, f"messages[{index}].content[{part_index}]")
            continue
        raise ValueError(f"messages[{index}].content must be a string or content-part list")


def normalize_sft_record(record: Any) -> dict[str, Any]:
    """Validate an SFT record, upgrading legacy ``<image>`` + ``images`` records."""
    if not isinstance(record, Mapping):
        raise ValueError("SFT record must be a JSON object")
    if "images" not in record:
        validate_sft_record(record)
        return deepcopy(dict(record))

    keys = set(record) - {"target_start_index"}
    if keys != {"messages", "tools", "images"}:
        raise ValueError("SFT record may only contain messages, tools, and images")
    images = record["images"]
    if not isinstance(images, list) or not images:
        raise ValueError("SFT images must be a non-empty list")
    for index, uri in enumerate(images):
        field = f"images[{index}]"
        if not isinstance(uri, str) or not uri.startswith("data:"):
            raise ValueError(f"{field} must be a base64 image data URI")
        try:
            canonical = _image_data_uri({"data": uri}, field)
        except SFTConversionError as exc:
            raise ValueError(f"{field} must be a base64 image data URI: {exc}") from exc
        if canonical != uri:
            raise ValueError(f"{field} must be a canonical base64 image data URI")

    raw_messages = record["messages"]
    if not isinstance(raw_messages, list):
        raise ValueError("SFT messages must be a list")
    placeholders = sum(
        _legacy_placeholder_count(message) for message in raw_messages if isinstance(message, Mapping)
    )
    if placeholders != len(images):
        raise ValueError("SFT <image> placeholder count does not match the images length")

    upgraded_messages: list[dict[str, Any]] = []
    image_index = 0
    for message in raw_messages:
        if not isinstance(message, Mapping):
            raise ValueError("SFT message must be an object")
        copy_msg = deepcopy(dict(message))
        content = copy_msg.get("content")
        if isinstance(content, str) and _LEGACY_IMAGE_PLACEHOLDER in content:
            pieces = content.split(_LEGACY_IMAGE_PLACEHOLDER)
            blocks: list[dict[str, Any]] = []
            for piece_index, piece in enumerate(pieces):
                # Strip the single newline legacy export inserted around <image>.
                if piece_index < len(pieces) - 1 and piece.endswith("\n"):
                    piece = piece[:-1]
                if piece_index > 0 and piece.startswith("\n"):
                    piece = piece[1:]
                if piece:
                    blocks.append({"type": "text", "text": piece})
                if piece_index < len(pieces) - 1:
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": images[image_index]},
                        }
                    )
                    image_index += 1
            copy_msg["content"] = blocks
        upgraded_messages.append(copy_msg)

    if image_index != len(images):
        raise ValueError("SFT <image> placeholder count in message content does not match the images length")

    upgraded: dict[str, Any] = {
        "messages": upgraded_messages,
        "tools": deepcopy(record["tools"]),
    }
    if "target_start_index" in record:
        upgraded["target_start_index"] = record["target_start_index"]
    validate_sft_record(upgraded)
    return upgraded


def actf_to_sft(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one ACTF trajectory to a minimal messages/tools SFT record."""

    if trajectory.get("sft_unavailable_reason"):
        raise SFTConversionError(str(trajectory["sft_unavailable_reason"]))
    raw_steps = trajectory.get("steps")
    for step in raw_steps if isinstance(raw_steps, list) else []:
        extra = step.get("extra") if isinstance(step, Mapping) else None
        if isinstance(extra, Mapping) and extra.get("is_sidechain"):
            raise SFTConversionError("mixed subagent steps cannot be exported as one assistant")
        if isinstance(extra, Mapping) and extra.get("model") == "<synthetic>":
            raise SFTConversionError("synthetic assistant is not a model output")

    system_prompt = trajectory.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise SFTConversionError("system_prompt must be a string")
    responses_tools = trajectory.get("tools_format") == "responses"
    native_names: dict[tuple[str, str], str] = {}
    if responses_tools:
        capture = trajectory.get("tools_capture", {})
        if not isinstance(capture, Mapping):
            raise SFTConversionError("Codex tools_capture must be an object")
        # Collection captures the tools of each request; an imported session
        # has them merged from its own records or inferred from its calls.
        if capture.get("status") != "captured" and not (
            capture.get("source") == "session" and capture.get("status") in {"captured_merged", "inferred"}
        ):
            raise SFTConversionError(
                capture.get("error") or "Codex tools were not captured; schema inference is disabled"
            )
        tools = deepcopy(trajectory.get("tools"))
        try:
            native_names = tool_index(tools)
        except ValueError as exc:
            raise SFTConversionError(str(exc)) from exc
        tool_names = set()
    else:
        tools = _normalize_tools(trajectory.get("tools"))
        tool_names = {tool["function"]["name"] for tool in tools}

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise SFTConversionError("steps must be a list")

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    seen_call_ids: set[str] = set()
    user_count = 0
    assistant_count = 0

    for step_index, step in enumerate(steps):
        prefix = f"steps[{step_index}]"
        if not isinstance(step, dict):
            raise SFTConversionError(f"{prefix} must be an object")

        user_content = _message_content(step.get("user_content", ""), f"{prefix}.user_content")
        if user_content:
            messages.append({"role": "user", "content": user_content})
            user_count += 1

        assistant = step.get("assistant_content")
        if not isinstance(assistant, dict):
            raise SFTConversionError(f"{prefix}.assistant_content must be an object")
        reasoning = _string(
            assistant.get("reasoning_content", ""),
            f"{prefix}.assistant_content.reasoning_content",
        )
        content = _string(assistant.get("content", ""), f"{prefix}.assistant_content.content")
        raw_calls = assistant.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise SFTConversionError(f"{prefix}.assistant_content.tool_calls must be a list")
        normalize_call = _responses_tool_call if responses_tools else _normalize_tool_call
        calls = [
            normalize_call(call, f"{prefix}.assistant_content.tool_calls[{index}]")
            for index, call in enumerate(raw_calls)
        ]
        if not reasoning and not content and not calls:
            raise SFTConversionError(f"{prefix} contains an empty assistant message")

        expected_ids: set[str] = set()
        call_names: dict[str, str] = {}
        for call in calls:
            call_id = call["id"]
            details = call[call["type"]]
            name = details["name"]
            if call_id in seen_call_ids:
                raise SFTConversionError(f"duplicate tool call id: {call_id}")
            if responses_tools:
                key = (details.get("namespace", ""), name)
                if key not in native_names and not key[0] and "." in name:
                    namespace, leaf = name.rsplit(".", 1)
                    key = (namespace, leaf)
                if key not in native_names:
                    raise SFTConversionError(f"tool call references missing captured schema: {name}")
                kind = native_names[key]
                if (kind == "custom") != (call["type"] == "custom"):
                    raise SFTConversionError(f"tool call type differs from captured schema: {name}")
            elif name not in tool_names:
                raise SFTConversionError(f"tool call references missing schema: {name}")
            seen_call_ids.add(call_id)
            expected_ids.add(call_id)
            call_names[call_id] = name

        observations = step.get("observation", [])
        if not isinstance(observations, list):
            raise SFTConversionError(f"{prefix}.observation must be a list")
        observed_ids: set[str] = set()
        results_by_id: dict[str, tuple[dict[str, Any], str]] = {}
        for observation_index, observation in enumerate(observations):
            field = f"{prefix}.observation[{observation_index}]"
            if not isinstance(observation, dict):
                raise SFTConversionError(f"{field} must be an object")
            call_id = observation.get("toolCallId", observation.get("tool_call_id"))
            if not isinstance(call_id, str) or not call_id:
                raise SFTConversionError(f"{field}.toolCallId must be non-empty")
            if call_id in observed_ids:
                raise SFTConversionError(f"duplicate tool result for call id: {call_id}")
            if call_id not in expected_ids:
                raise SFTConversionError(f"tool result has no matching call: {call_id}")
            observed_name = observation.get("toolName")
            if isinstance(observed_name, str) and observed_name and observed_name != call_names[call_id]:
                raise SFTConversionError(
                    f"tool result name {observed_name} does not match call {call_names[call_id]}"
                )
            observed_ids.add(call_id)
            results_by_id[call_id] = (observation, field)
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            raise SFTConversionError(f"tool calls missing results: {', '.join(missing)}")

        # Parallel tools can finish in any order. Canonicalize only the SFT
        # output, matching by ID rather than completion order or tool name.
        tool_messages: list[dict[str, Any]] = []
        for call in calls:
            observation, field = results_by_id[call["id"]]
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _message_content(observation.get("content", []), f"{field}.content"),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": content,
                "tool_calls": calls,
            }
        )
        messages.extend(tool_messages)
        assistant_count += 1

    if user_count == 0:
        raise SFTConversionError("trajectory contains no user message")
    if assistant_count == 0:
        raise SFTConversionError("trajectory contains no assistant message")
    return {"messages": messages, "tools": tools}
