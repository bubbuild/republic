"""Events yielded while streaming a chat response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from republic._content import Image, ToolCall
from republic._response import BuiltinToolCall, Citation, Response, TokenUsage

__all__ = [
    "BuiltinToolCallReady",
    "CitationAdded",
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
class BuiltinToolCallReady:
    """A built-in tool finished running on the provider's side."""

    call: BuiltinToolCall


@dataclass(frozen=True)
class CitationAdded:
    citation: Citation


@dataclass(frozen=True)
class Completed:
    """The last event of a stream, carrying the full response."""

    response: Response[Any]


Event = (
    TextDelta
    | ReasoningDelta
    | RefusalDelta
    | UsageDelta
    | ImageReady
    | CitationAdded
    | ToolCallDelta
    | ToolCallReady
    | BuiltinToolCallReady
    | Completed
)
