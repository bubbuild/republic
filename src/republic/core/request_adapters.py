"""Request-shape adapters for different upstream APIs."""

from __future__ import annotations

from typing import Any


def normalize_responses_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize completion-style kwargs into responses-compatible shapes."""
    tool_choice = kwargs.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return kwargs

    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return kwargs

    function_name = function.get("name")
    if not isinstance(function_name, str) or not function_name:
        return kwargs

    normalized_tool_choice = dict(tool_choice)
    normalized_tool_choice.pop("function", None)
    normalized_tool_choice["type"] = normalized_tool_choice.get("type", "function")
    normalized_tool_choice["name"] = function_name
    return {**kwargs, "tool_choice": normalized_tool_choice}
