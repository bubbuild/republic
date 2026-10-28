"""Events yielded while streaming a chat response, and the events middleware sees for every call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, final

from republic._content import Image, ToolCall
from republic._response import BuiltinToolCall, Citation, Response, TokenUsage

__all__ = [
    "BuiltinToolCallReady",
    "CallEvent",
    "CitationAdded",
    "Completed",
    "Event",
    "ImageReady",
    "ReasoningDelta",
    "RefusalDelta",
    "Retry",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallReady",
    "UsageDelta",
]


@final
@dataclass(frozen=True)
class TextDelta:
    chunk: str


@final
@dataclass(frozen=True)
class ReasoningDelta:
    """Reasoning text, or a summary of it, as the provider exposes it."""

    chunk: str


@final
@dataclass(frozen=True)
class RefusalDelta:
    """Text explaining why the model declines to answer."""

    chunk: str


@final
@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of tool call arguments. ``ToolCallReady`` follows once the call is complete."""

    call_id: str
    name: str
    chunk: str


@final
@dataclass(frozen=True)
class UsageDelta:
    """Usage added since the previous usage event. The deltas sum to the final usage."""

    usage: TokenUsage


@final
@dataclass(frozen=True)
class ImageReady:
    image: Image


@final
@dataclass(frozen=True)
class ToolCallReady:
    """A tool call whose arguments are complete."""

    call: ToolCall


@final
@dataclass(frozen=True)
class BuiltinToolCallReady:
    """A built-in tool finished running on the provider's side."""

    call: BuiltinToolCall


@final
@dataclass(frozen=True)
class CitationAdded:
    citation: Citation


ResponseT = TypeVar("ResponseT")


@final
@dataclass(frozen=True)
class Completed(Generic[ResponseT]):
    """The last event of a call, carrying the full response.

    Streams carry the chat ``Response``. Middleware also sees it for other calls,
    carrying their result, such as an ``EmbeddingResponse`` or a list of models.
    """

    response: ResponseT


@final
@dataclass(frozen=True)
class Retry:
    """A failed attempt that will be retried after ``delay`` seconds. Only middleware sees it."""

    error: Exception
    attempt: int
    """Starts at one for each HTTP retry loop, including each page of model listing."""
    delay: float


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
    | Completed[Response[Any]]
)

CallEvent = Event | Retry
"""What middleware receives and yields: ``Retry`` events, stream events, and the final ``Completed``."""
