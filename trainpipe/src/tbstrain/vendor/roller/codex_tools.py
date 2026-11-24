# SPDX-License-Identifier: Apache-2.0
# Adapted from Roller, revision cd4c7f2. See NOTICE and LICENSE.
"""Capture Codex Responses tool definitions without inferring or merging schemas."""

from __future__ import annotations

from typing import Any


def tool_index(tools: Any) -> dict[tuple[str, str], str]:
    """Validate native definitions and index callable names; do not rewrite them."""
    if not isinstance(tools, list):
        raise ValueError("Responses tools must be a list")
    names: dict[tuple[str, str], str] = {}

    def visit(items: list[Any], namespace: str = "") -> None:
        for tool in items:
            if not isinstance(tool, dict) or not isinstance(tool.get("type"), str):
                raise ValueError("Responses tool definition must have a string type")
            kind = tool["type"]
            if kind == "namespace":
                name = tool.get("name")
                children = tool.get("tools")
                if namespace or not isinstance(name, str) or not name or not isinstance(children, list):
                    raise ValueError("Invalid Responses tool namespace")
                visit(children, name)
                continue
            name = tool.get("name") if kind in {"function", "custom"} else kind
            if not isinstance(name, str) or not name:
                raise ValueError("Responses tool definition has no callable name")
            key = (namespace, name)
            if key in names:
                raise ValueError(f"Duplicate Responses tool: {key}")
            names[key] = kind

    visit(tools)
    return names
