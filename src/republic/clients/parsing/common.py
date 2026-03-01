"""Common parsing utilities shared by completion and responses adapters."""

from __future__ import annotations

import json
from typing import Any


def field(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def expand_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for call in calls:
        expanded.extend(_expand_tool_call(call))
    return expanded


def _expand_tool_call(call: dict[str, Any]) -> list[dict[str, Any]]:
    function = field(call, "function")
    if not isinstance(function, dict):
        return [dict(call)]

    arguments = field(function, "arguments")
    if not isinstance(arguments, str):
        return [dict(call)]

    chunks = _split_concatenated_json_objects(arguments)
    if not chunks:
        return [dict(call)]

    call_id = field(call, "id")
    expanded: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        cloned = dict(call)
        cloned_function = dict(function)
        cloned_function["arguments"] = chunk
        cloned["function"] = cloned_function
        if isinstance(call_id, str) and call_id and index > 0:
            cloned["id"] = f"{call_id}__{index + 1}"
        expanded.append(cloned)
    return expanded


def _split_concatenated_json_objects(raw: str) -> list[str]:
    decoder = json.JSONDecoder()
    chunks: list[str] = []
    position = 0
    total = len(raw)
    while position < total:
        while position < total and raw[position].isspace():
            position += 1
        if position >= total:
            break
        try:
            parsed, end = decoder.raw_decode(raw, position)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, dict):
            return []
        chunks.append(raw[position:end])
        position = end

    if len(chunks) <= 1:
        return []
    return chunks
