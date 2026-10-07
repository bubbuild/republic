from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from republic._content import Tool

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""How much the model reasons before answering. Providers accept different subsets."""

ToolChoice = Literal["auto", "none", "required"] | Tool
"""Let the model decide, forbid tools, require some tool, or require the given tool."""


class ChatOptions(TypedDict, total=False):
    """Generation options shared by chat requests.

    Each API format maps them to its own fields and raises
    :class:`~republic.UnsupportedFeatureError` for options it cannot express.
    Use ``extra_body`` for anything provider-specific; it is merged last.
    """

    tools: Sequence[Tool]
    tool_choice: ToolChoice
    parallel_tool_calls: bool
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    stop: Sequence[str]
    seed: int
    reasoning_effort: ReasoningEffort
    include_reasoning: bool
    """Ask for readable reasoning (or its summary) where the provider hides it by default."""
    extra_body: Mapping[str, Any]
