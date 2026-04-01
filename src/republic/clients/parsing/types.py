"""Shared parser typing and validation primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

TransportKind = Literal["completion", "responses", "messages"]


@dataclass(frozen=True)
class ParsedResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    metadata_only: bool = False


@dataclass(frozen=True)
class ParsedChunk:
    text_delta: str = ""
    tool_call_deltas: list[Any] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    response_completed: bool = False
    output_item_type: str | None = None


class BaseTransportParser(ABC):
    @abstractmethod
    def is_non_stream_response(self, response: Any) -> bool:
        """Return True if the response is a non-streaming response, False otherwise."""

    @abstractmethod
    def parse_chunk(self, chunk: Any) -> ParsedChunk:
        """Parse one streaming chunk into typed data."""

    @abstractmethod
    def parse_response(self, response: Any) -> ParsedResponse:
        """Parse one full response into typed data."""
