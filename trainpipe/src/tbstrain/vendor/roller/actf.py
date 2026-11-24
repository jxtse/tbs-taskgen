# SPDX-License-Identifier: Apache-2.0
# Adapted from Roller, revision cd4c7f2. See NOTICE and LICENSE.
from copy import deepcopy
from typing import Any


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    return "" if value is None else str(value)


def user_content(value: Any, expected: str | None = None) -> str | list[Any]:
    """Keep native user media while replacing transport-decorated prompt text."""

    if not isinstance(value, list):
        return expected if expected is not None else _text(value)
    blocks: list[Any] = []
    has_non_text = False
    inserted_expected = False
    for item in value:
        if isinstance(item, str):
            if expected is not None:
                if not inserted_expected:
                    blocks.append({"type": "text", "text": expected})
                    inserted_expected = True
            else:
                blocks.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            blocks.append(deepcopy(item))
            has_non_text = True
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            if expected is not None:
                if not inserted_expected:
                    blocks.append({"type": "text", "text": expected})
                    inserted_expected = True
            else:
                blocks.append(deepcopy(item))
            continue
        blocks.append(deepcopy(item))
        has_non_text = True
    if expected is not None and not inserted_expected:
        blocks.insert(0, {"type": "text", "text": expected})
    if not has_non_text:
        return expected if expected is not None else _text(value)
    return blocks


def append_user_content(current: str | list[Any], value: str | list[Any]) -> str | list[Any]:
    if not current:
        return deepcopy(value)
    if not value:
        return current
    if isinstance(current, str) and isinstance(value, str):
        return f"{current}\n\n{value}"

    def blocks(item: str | list[Any]) -> list[Any]:
        if isinstance(item, list):
            return deepcopy(item)
        return [{"type": "text", "text": item}]

    head, tail = blocks(current), blocks(value)
    last, first = head[-1], tail[0]
    if isinstance(last, dict) and isinstance(first, dict) and last.get("type") == first.get("type") == "text":
        # Adjacent prompts keep the paragraph break that plain strings get.
        last["text"] = f"{last['text']}\n\n{first['text']}"
        tail = tail[1:]
    return [*head, *tail]
