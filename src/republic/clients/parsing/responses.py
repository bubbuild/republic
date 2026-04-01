"""OpenAI responses shape parsing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from republic.clients.parsing.common import expand_tool_calls, field
from republic.clients.parsing.types import BaseTransportParser, ParsedChunk, ParsedResponse

RESPONSES_METADATA_ONLY_ITEM_TYPES = frozenset({"reasoning", "compaction"})


class ResponseTransportParser(BaseTransportParser):
    def is_non_stream_response(self, response: Any) -> bool:
        return (
            isinstance(response, str)
            or field(response, "choices") is not None
            or field(response, "output") is not None
            or field(response, "output_text") is not None
        )

    def _tool_delta_from_args_event(self, chunk: Any, event_type: str) -> list[Any]:
        item_id = field(chunk, "item_id")
        if not item_id:
            return []
        arguments = field(chunk, "delta")
        if event_type == "response.function_call_arguments.done":
            arguments = field(chunk, "arguments")
        if not isinstance(arguments, str):
            return []

        call_id = field(chunk, "call_id")
        payload: dict[str, Any] = {
            "index": item_id,
            "type": "function",
            "function": SimpleNamespace(name=field(chunk, "name") or "", arguments=arguments),
            "arguments_complete": event_type == "response.function_call_arguments.done",
        }
        if call_id:
            payload["id"] = call_id
        return [SimpleNamespace(**payload)]

    def _tool_delta_from_output_item_event(self, chunk: Any, event_type: str) -> list[Any]:
        item = field(chunk, "item")
        if field(item, "type") != "function_call":
            return []

        item_id = field(item, "id")
        call_id = field(item, "call_id") or item_id
        if not call_id:
            return []
        arguments = field(item, "arguments")
        if not isinstance(arguments, str):
            arguments = ""
        return [
            SimpleNamespace(
                id=call_id,
                index=item_id or call_id,
                type="function",
                function=SimpleNamespace(name=field(item, "name") or "", arguments=arguments),
                arguments_complete=event_type == "response.output_item.done",
            )
        ]

    def parse_chunk(self, chunk: Any) -> ParsedChunk:
        event_type = field(chunk, "type")
        output_item_type: str | None = None
        if event_type in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
            tool_call_deltas = self._tool_delta_from_args_event(chunk, event_type)
        elif event_type in {"response.output_item.added", "response.output_item.done"}:
            tool_call_deltas = self._tool_delta_from_output_item_event(chunk, event_type)
            output_item_type = self._output_item_type(chunk)
        else:
            tool_call_deltas = []
            output_item_type = self._output_item_type(chunk)

        text_delta = ""
        if field(chunk, "type") == "response.output_text.delta":
            delta = field(chunk, "delta")
            if isinstance(delta, str):
                text_delta = delta

        return ParsedChunk(
            text_delta=text_delta,
            tool_call_deltas=tool_call_deltas,
            usage=self._extract_usage(chunk),
            response_completed=field(chunk, "type") == "response.completed",
            output_item_type=output_item_type,
        )

    def parse_response(self, response: Any) -> ParsedResponse:
        usage = self._extract_usage(response)
        text = self._extract_text(response)
        tool_calls = self._extract_tool_calls(response)
        metadata_only = self._metadata_only_response(response)
        return ParsedResponse(text=text, tool_calls=tool_calls, usage=usage, metadata_only=metadata_only)

    def _extract_text_from_output(self, output: Any) -> str:
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            if field(item, "type") != "message":
                continue
            content = field(item, "content") or []
            for entry in content:
                if field(entry, "type") == "output_text":
                    text = field(entry, "text")
                    if text:
                        parts.append(text)
        return "".join(parts)

    def _extract_text(self, response: Any) -> str:
        output_text = field(response, "output_text")
        if isinstance(output_text, str):
            return output_text
        return self._extract_text_from_output(field(response, "output"))

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        output = response if isinstance(response, list) else field(response, "output")
        if not isinstance(output, list):
            return []
        calls: list[dict[str, Any]] = []
        for item in output:
            if field(item, "type") != "function_call":
                continue
            name = field(item, "name")
            arguments = field(item, "arguments")
            if not name:
                continue
            entry: dict[str, Any] = {"function": {"name": name, "arguments": arguments or ""}}
            call_id = field(item, "call_id") or field(item, "id")
            if call_id:
                entry["id"] = call_id
            entry["type"] = "function"
            calls.append(entry)
        return expand_tool_calls(calls)

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        event_type = field(response, "type")
        if event_type in {"response.completed", "response.in_progress", "response.failed", "response.incomplete"}:
            usage = field(field(response, "response"), "usage")
        else:
            usage = field(response, "usage")

        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return dict(usage)

        data: dict[str, Any] = {}
        for usage_field in ("input_tokens", "output_tokens", "total_tokens", "requests"):
            value = field(usage, usage_field)
            if value is not None:
                data[usage_field] = value
        return data or None

    def _metadata_only_response(self, response: Any) -> bool:
        if field(response, "status") != "completed":
            return False
        if field(response, "incomplete_details") is not None:
            return False
        output = field(response, "output")
        if not isinstance(output, list) or not output:
            return False
        return all(
            isinstance(item_type := field(item, "type"), str) and item_type in RESPONSES_METADATA_ONLY_ITEM_TYPES
            for item in output
        )

    @staticmethod
    def _output_item_type(chunk: Any) -> str | None:
        if field(chunk, "type") not in {"response.output_item.added", "response.output_item.done"}:
            return None
        item_type = field(field(chunk, "item"), "type")
        return item_type if isinstance(item_type, str) else None
