from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from republic._content import Tool
from republic.tools import BuiltinTool

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""How much the model reasons before answering. Providers accept different subsets."""

ToolChoice = Literal["auto", "none", "required"] | Tool
"""Let the model decide, forbid tools, require some tool, or require the given tool."""


class ChatOptions(TypedDict, total=False):
    """Generation options shared by chat requests.

    Each API format maps them to its own fields and raises
    :class:`~republic.errors.UnsupportedFeatureError` for options it cannot express.
    Use ``extra_body`` for anything provider-specific; it is merged last.
    """

    tools: Sequence[Tool | BuiltinTool]
    """Function tool schemas and built-in tools from :mod:`republic.tools`."""
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


class ImageOptions(TypedDict, total=False):
    """Options for image generation and editing.

    Values follow each service's vocabulary and are validated by the service.
    API formats raise :class:`~republic.errors.UnsupportedFeatureError` for
    options they cannot express. Use ``extra_body`` for anything else, such as
    ``input_fidelity``; it is merged last.
    """

    n: int
    """How many images to return."""
    size: str
    """Such as ``"1024x1024"`` or ``"auto"`` for OpenAI, or ``"1K"``, ``"2K"``, or ``"4K"`` for Gemini."""
    aspect_ratio: str
    """Such as ``"1:1"`` or ``"16:9"``; Gemini only."""
    quality: str
    """Such as ``"low"``, ``"medium"``, ``"high"``, or ``"auto"``."""
    background: Literal["transparent", "opaque", "auto"]
    output_format: Literal["png", "jpeg", "webp"]
    output_compression: int
    """Compression level from 0 to 100 for ``jpeg`` and ``webp`` output."""
    moderation: Literal["low", "auto"]
    extra_body: Mapping[str, Any]
