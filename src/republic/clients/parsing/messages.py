"""Anthropic-style messages shape parsing."""

from __future__ import annotations

import json
from typing import Any

from republic.clients.parsing.common import expand_tool_calls, field
from republic.clients.parsing.types import BaseTransportParser


class MessageTransportParser(BaseTransportParser):
    def is_non_stream_response(self, response: Any) -> bool:
        return field(response, "type") == "message"

    def extract_chunk_tool_call_deltas(self, chunk: Any) -> list[Any]:
        event_type = field(chunk, "type")
        if event_type == "content_block_start":
            content_block = field(chunk, "content_block")
            if field(content_block, "type") != "tool_use":
                return []
            return [
                {
                    "id": field(content_block, "id"),
                    "index": field(chunk, "index"),
                    "type": "function",
                    "function": {
                        "name": field(content_block, "name") or "",
                        "arguments": "",
                    },
                }
            ]

        if event_type == "content_block_delta" and field(field(chunk, "delta"), "type") == "input_json_delta":
            return [
                {
                    "index": field(chunk, "index"),
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": field(field(chunk, "delta"), "partial_json") or "",
                    },
                }
            ]
        return []

    def extract_chunk_text(self, chunk: Any) -> str:
        if field(chunk, "type") != "content_block_delta":
            return ""
        delta = field(chunk, "delta")
        if field(delta, "type") != "text_delta":
            return ""
        text = field(delta, "text")
        return text if isinstance(text, str) else ""

    def extract_text(self, response: Any) -> str:
        content = field(response, "content")
        if not isinstance(content, list):
            return ""
        return "".join(
            text
            for block in content
            if field(block, "type") == "text" and isinstance(text := field(block, "text"), str)
        )

    def extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        content = field(response, "content")
        if not isinstance(content, list):
            return []

        calls: list[dict[str, Any]] = []
        for block in content:
            if field(block, "type") != "tool_use":
                continue
            name = field(block, "name")
            if not name:
                continue
            arguments = field(block, "input")
            arguments_payload = json.dumps(arguments) if isinstance(arguments, dict) else "{}"
            entry: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_payload,
                },
            }
            call_id = field(block, "id")
            if call_id:
                entry["id"] = call_id
            calls.append(entry)
        return expand_tool_calls(calls)

    def extract_usage(self, response: Any) -> dict[str, Any] | None:
        usage = field(response, "usage")
        if usage is None and field(response, "type") == "message_start":
            usage = field(field(response, "message"), "usage")
        if usage is None:
            return None

        payload: dict[str, Any] = {}
        for usage_field in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "total_tokens",
        ):
            value = field(usage, usage_field)
            if value not in (None, 0):
                payload[usage_field] = value

        if "total_tokens" not in payload and ("input_tokens" in payload or "output_tokens" in payload):
            payload["total_tokens"] = payload.get("input_tokens", 0) + payload.get("output_tokens", 0)
        return payload or None
