"""Events yielded while streaming a chat response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from republic._content import Image, ToolCall
from republic._response import Response, TokenUsage

__all__ = [
    "Completed",
    "Event",
    "ImageReady",
    "ReasoningDelta",
    "RefusalDelta",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallReady",
    "UsageDelta",
]


@dataclass(frozen=True)
class TextDelta:
    chunk: str


@dataclass(frozen=True)
class ReasoningDelta:
    """Reasoning text, or a summary of it, as the provider exposes it."""

    chunk: str


@dataclass(frozen=True)
class RefusalDelta:
    """Text explaining why the model declines to answer."""

    chunk: str


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of tool call arguments. ``ToolCallReady`` follows once the call is complete."""

    call_id: str
    name: str
    chunk: str


@dataclass(frozen=True)
class UsageDelta:
    """Usage added since the previous usage event. The deltas sum to the final usage."""

    usage: TokenUsage


@dataclass(frozen=True)
class ImageReady:
    image: Image


@dataclass(frozen=True)
class ToolCallReady:
    """A tool call whose arguments are complete."""

    call: ToolCall


@dataclass(frozen=True)
class Completed:
    """The last event of a stream, carrying the full response."""

    response: Response[Any]


Event = TextDelta | ReasoningDelta | RefusalDelta | UsageDelta | ImageReady | ToolCallDelta | ToolCallReady | Completed
