"""Shared parser typing and validation primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

TransportKind = Literal["completion", "responses", "messages"]


class BaseTransportParser(ABC):
    @abstractmethod
    def is_non_stream_response(self, response: Any) -> bool:
        """Return True if the response is a non-streaming response, False otherwise."""

    @abstractmethod
    def extract_chunk_tool_call_deltas(self, chunk: Any) -> list[Any]:
        """Extract tool call deltas from a response chunk, if present."""

    @abstractmethod
    def extract_chunk_text(self, chunk: Any) -> str:
        """Extract text from a response chunk."""

    @abstractmethod
    def extract_text(self, response: Any) -> str:
        """Extract text from a response."""

    @abstractmethod
    def extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from a response, if present."""

    @abstractmethod
    def extract_usage(self, response: Any) -> dict[str, Any] | None:
        """Extract usage information from a response, if present."""
