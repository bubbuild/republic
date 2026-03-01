"""OpenAI responses shape parsing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from republic.clients.parsing.common import field


def is_non_stream_response(response: Any) -> bool:
    return (
        isinstance(response, str)
        or field(response, "choices") is not None
        or field(response, "output") is not None
        or field(response, "output_text") is not None
    )


def _tool_delta_from_args_event(chunk: Any, event_type: str) -> list[Any]:
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


def _tool_delta_from_output_item_event(chunk: Any, event_type: str) -> list[Any]:
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


def extract_chunk_tool_call_deltas(chunk: Any) -> list[Any]:
    event_type = field(chunk, "type")
    if event_type in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
        return _tool_delta_from_args_event(chunk, event_type)
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        return _tool_delta_from_output_item_event(chunk, event_type)
    return []


def extract_chunk_text(chunk: Any) -> str:
    if field(chunk, "type") != "response.output_text.delta":
        return ""
    delta = field(chunk, "delta")
    if isinstance(delta, str):
        return delta
    return ""


def extract_text_from_output(output: Any) -> str:
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


def extract_text(response: Any) -> str:
    output_text = field(response, "output_text")
    if isinstance(output_text, str):
        return output_text
    return extract_text_from_output(field(response, "output"))


def extract_tool_calls(output: Any) -> list[dict[str, Any]]:
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
    return calls


def extract_usage(response: Any) -> dict[str, Any] | None:
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
