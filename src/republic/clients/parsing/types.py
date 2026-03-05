"""Shared parser typing and validation primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

TransportKind = Literal["completion", "responses", "messages"]


@runtime_checkable
class BaseTransportParser(Protocol):
    def is_non_stream_response(self, response: Any) -> bool: ...

    def extract_chunk_tool_call_deltas(self, chunk: Any) -> list[Any]: ...

    def extract_chunk_text(self, chunk: Any) -> str: ...

    def extract_text(self, response: Any) -> str: ...

    def extract_tool_calls(self, response: Any) -> list[dict[str, Any]]: ...

    def extract_usage(self, response: Any) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class FunctionTransportParser:
    is_non_stream_response_fn: Callable[[Any], bool]
    extract_chunk_tool_call_deltas_fn: Callable[[Any], list[Any]]
    extract_chunk_text_fn: Callable[[Any], str]
    extract_text_fn: Callable[[Any], str]
    extract_tool_calls_fn: Callable[[Any], list[dict[str, Any]]]
    extract_usage_fn: Callable[[Any], dict[str, Any] | None]

    def is_non_stream_response(self, response: Any) -> bool:
        return self.is_non_stream_response_fn(response)

    def extract_chunk_tool_call_deltas(self, chunk: Any) -> list[Any]:
        return self.extract_chunk_tool_call_deltas_fn(chunk)

    def extract_chunk_text(self, chunk: Any) -> str:
        return self.extract_chunk_text_fn(chunk)

    def extract_text(self, response: Any) -> str:
        return self.extract_text_fn(response)

    def extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        return self.extract_tool_calls_fn(response)

    def extract_usage(self, response: Any) -> dict[str, Any] | None:
        return self.extract_usage_fn(response)
