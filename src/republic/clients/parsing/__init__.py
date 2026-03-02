"""Parsing helpers for provider response payloads."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from republic.clients.parsing import completion, messages, responses

TransportKind = Literal["completion", "responses", "messages"]


class TransportParser(Protocol):
    @staticmethod
    def is_non_stream_response(response: Any) -> bool: ...

    @staticmethod
    def extract_chunk_tool_call_deltas(chunk: Any) -> list[Any]: ...

    @staticmethod
    def extract_chunk_text(chunk: Any) -> str: ...

    @staticmethod
    def extract_text(response: Any) -> str: ...

    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]: ...

    @staticmethod
    def extract_usage(response: Any) -> dict[str, Any] | None: ...


def parser_for_transport(transport: TransportKind) -> TransportParser:
    if transport == "responses":
        return responses
    if transport == "messages":
        return messages
    return completion


__all__ = ["TransportKind", "TransportParser", "parser_for_transport"]
