"""OpenAI chat-completions shape parsing."""

from __future__ import annotations

from typing import Any

from republic.clients.parsing.common import expand_tool_calls, field
from republic.clients.parsing.types import BaseTransportParser, ParsedChunk, ParsedResponse


class CompletionTransportParser(BaseTransportParser):
    def is_non_stream_response(self, response: Any) -> bool:
        return isinstance(response, str) or field(response, "choices") is not None

    def parse_chunk(self, chunk: Any) -> ParsedChunk:
        choices = field(chunk, "choices")
        if not choices:
            return ParsedChunk()
        delta = field(choices[0], "delta")
        if delta is None:
            return ParsedChunk()
        return ParsedChunk(
            text_delta=field(delta, "content", "") or "",
            tool_call_deltas=field(delta, "tool_calls") or [],
            usage=self._extract_usage(response=chunk),
        )

    def parse_response(self, response: Any) -> ParsedResponse:
        return ParsedResponse(
            text=self._extract_text(response),
            tool_calls=self._extract_tool_calls(response),
            usage=self._extract_usage(response),
            metadata_only=bool(field(response, "republic_metadata_only", False)),
        )

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response

        choices = field(response, "choices")
        if not choices:
            return ""
        message = field(choices[0], "message")
        if message is None:
            return ""
        return field(message, "content", "") or ""

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        choices = field(response, "choices")
        if not choices:
            return []
        message = field(choices[0], "message")
        if message is None:
            return []
        tool_calls = field(message, "tool_calls") or []
        calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = field(tool_call, "function")
            if function is None:
                continue
            entry: dict[str, Any] = {
                "function": {
                    "name": field(function, "name"),
                    "arguments": field(function, "arguments"),
                }
            }
            call_id = field(tool_call, "id")
            if call_id:
                entry["id"] = call_id
            call_type = field(tool_call, "type")
            if call_type:
                entry["type"] = call_type
            if isinstance(tool_call, dict):
                entry.update({key: value for key, value in tool_call.items() if key not in {"id", "type", "function"}})
            calls.append(entry)
        return expand_tool_calls(calls)

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        usage = field(response, "usage")
        if usage is None:
            return None
        if isinstance(usage, dict):
            payload = dict(usage)
        elif hasattr(usage, "model_dump"):
            payload = usage.model_dump()
        else:
            return None
        normalized: dict[str, Any] = {}
        if "input_tokens" in payload:
            normalized["input_tokens"] = payload["input_tokens"]
        elif "prompt_tokens" in payload:
            normalized["input_tokens"] = payload["prompt_tokens"]
        if "output_tokens" in payload:
            normalized["output_tokens"] = payload["output_tokens"]
        elif "completion_tokens" in payload:
            normalized["output_tokens"] = payload["completion_tokens"]
        if "total_tokens" in payload:
            normalized["total_tokens"] = payload["total_tokens"]
        if "requests" in payload:
            normalized["requests"] = payload["requests"]
        return normalized or None
