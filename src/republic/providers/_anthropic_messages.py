# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: explicit input validation, no repair, inclusive usage.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""Anthropic Messages conversion for text, thinking and caller-owned tools."""

import json
from typing import Any

from republic.errors import ProviderError, UnsupportedRequestError
from republic.providers._files import base64_data, file_id, file_url, text_data
from republic.types import (
    FilePart,
    FinishReason,
    Message,
    Part,
    ProviderMetadata,
    ReasoningPart,
    Request,
    Response,
    TextPart,
    Tool,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
)

_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_call",
    "refusal": "content_filter",
}
_RECORD = {"stop_reason", "stop_sequence", "stop_details"}
_OPTIONS = {
    "thinking",
    "top_k",
    "metadata",
    "service_tier",
    "output_config",
    "cache_control",
    "extra_headers",
    "timeout",
}
_BLOCK_FIELDS = {
    "text": {"type", "text", "citations"},
    "thinking": {"type", "thinking", "signature"},
    "redacted_thinking": {"type", "data"},
    "tool_use": {"type", "id", "name", "input", "caller"},
}


def invalid_response(*, detail: str) -> ProviderError:
    return ProviderError(
        f"Invalid Anthropic Messages response: {detail}", provider="anthropic", code="invalid_response"
    )


def metadata(value: ProviderMetadata | None, allowed: set[str], field: str) -> dict[str, Any]:
    if not value:
        return {}
    if set(value) != {"anthropic"} or not isinstance(value["anthropic"], dict):
        raise UnsupportedRequestError(field, "expected an anthropic metadata object")
    data = value["anthropic"]
    if unknown := data.keys() - allowed:
        raise UnsupportedRequestError(field, f"unrecognized metadata keys: {sorted(unknown)}")
    return dict(data)


def _cache(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "ephemeral" or value.keys() - {"type", "ttl"}:
        raise UnsupportedRequestError("cache_control", "expected ephemeral cache control with optional ttl")
    if "ttl" in value and value["ttl"] not in ("5m", "1h"):
        raise UnsupportedRequestError("cache_control.ttl", "expected 5m or 1h")
    return value


def _part_options(part: Part) -> dict[str, Any]:
    allowed = {"cache_control", "caller"} if isinstance(part, ToolCallPart) else {"cache_control"}
    info = metadata(part.provider_metadata, allowed, "part.metadata")
    if "cache_control" in info:
        _cache(info["cache_control"])
    if "caller" in info and info["caller"] != {"type": "direct"}:
        raise UnsupportedRequestError("tool.caller", "only direct function calls are supported")
    return info


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(key)
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(value)


def _tool_input(arguments: str) -> dict[str, Any]:
    try:
        value = json.loads(arguments, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except ValueError as exc:
        raise UnsupportedRequestError("tool_args", "expected JSON with unique keys and finite values") from exc
    if not isinstance(value, dict):
        raise UnsupportedRequestError("tool_args", "Anthropic tool_use input must be a JSON object")
    # Also rejects overflowed JSON numbers such as 1e999, without changing them.
    try:
        json.dumps(value, allow_nan=False)
    except ValueError as exc:
        raise UnsupportedRequestError("tool_args", "expected finite JSON numbers") from exc
    return value


def _reasoning(part: ReasoningPart) -> dict[str, Any]:
    info = metadata(part.provider_metadata, {"signature", "redacted_data"}, "reasoning.metadata")
    if set(info) == {"signature"} and isinstance(info["signature"], str) and info["signature"]:
        return {"type": "thinking", "thinking": part.text, "signature": info["signature"]}
    if set(info) == {"redacted_data"} and isinstance(info["redacted_data"], str) and not part.text:
        return {"type": "redacted_thinking", "data": info["redacted_data"]}
    raise UnsupportedRequestError("reasoning", "expected signed thinking or empty text with opaque redacted_data")


def _file(part: FilePart) -> dict[str, Any]:
    image = part.media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if not image and part.media_type not in {"application/pdf", "text/plain"}:
        raise UnsupportedRequestError(
            "file.media_type", "Messages supports JPEG/PNG/GIF/WebP images, PDF and plain-text documents"
        )
    allowed = {"cache_control"} if image else {"cache_control", "title", "context", "citations"}
    info = metadata(part.provider_metadata, allowed, "file.metadata")
    if "cache_control" in info:
        _cache(info["cache_control"])
    for field in ("title", "context"):
        if field in info and not isinstance(info[field], str):
            raise UnsupportedRequestError("file.metadata", "title/context must be strings")
    if "citations" in info and (
        not isinstance(info["citations"], dict)
        or set(info["citations"]) != {"enabled"}
        or not isinstance(info["citations"]["enabled"], bool)
    ):
        raise UnsupportedRequestError("file.citations", "expected an enabled boolean")
    if part.filename is not None:
        raise UnsupportedRequestError("file.filename", "Messages uses document title metadata, not filename")
    if part.encoding == "file_id":
        source = {"type": "file", "file_id": file_id(part)}
    elif part.encoding == "base64" or part.data.startswith("data:"):
        source = (
            {"type": "text", "media_type": "text/plain", "data": text_data(part)}
            if part.media_type == "text/plain"
            else {"type": "base64", "media_type": part.media_type, "data": base64_data(part)}
        )
    else:
        source = {"type": "url", "url": file_url(part)}
    return {"type": "image" if image else "document", "source": source, **info}


def _part(part: Part, role: str) -> dict[str, Any]:
    if isinstance(part, ReasoningPart) and role == "assistant":
        return _reasoning(part)
    if isinstance(part, FilePart) and role == "user":
        return _file(part)
    info = _part_options(part)
    if isinstance(part, TextPart) and role != "tool":
        return {"type": "text", "text": part.text, **info}
    if isinstance(part, ToolCallPart) and role == "assistant":
        if not part.tool_call_id or not part.tool_name:
            raise UnsupportedRequestError("tool.call", "ID and name must be nonempty")
        return {
            "type": "tool_use",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "input": _tool_input(part.tool_args),
            **info,
        }
    if isinstance(part, ToolResultPart) and role in ("user", "tool"):
        if not part.tool_call_id:
            raise UnsupportedRequestError("tool.result", "call ID must be nonempty")
        return {
            "type": "tool_result",
            "tool_use_id": part.tool_call_id,
            "is_error": part.is_error,
            "content": part.result if isinstance(part.result, str) else json.dumps(part.result, ensure_ascii=False),
            **info,
        }
    raise UnsupportedRequestError("message.part", f"{part.kind} cannot be sent as {role}")


def _messages(messages: list[Message]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system = []
    result = []
    for message in messages:
        metadata(message.provider_metadata, _RECORD if message.role == "assistant" else set(), "message.metadata")
        if message.role == "system":
            if result:
                raise UnsupportedRequestError(
                    "system", "only leading system messages are supported; history is not reordered"
                )
            system.extend(_part(part, "system") for part in message.parts)
        else:
            result.append({
                "role": "user" if message.role == "tool" else message.role,
                "content": [_part(part, message.role) for part in message.parts],
            })
    return system, result


def _tool(tool: Tool) -> dict[str, Any]:
    info = metadata(tool.provider_metadata, {"cache_control", "strict"}, "tool.metadata")
    if "cache_control" in info:
        _cache(info["cache_control"])
    if not tool.name or ("strict" in info and not isinstance(info["strict"], bool)):
        raise UnsupportedRequestError("tool", "expected a name and boolean strict option")
    result = {"name": tool.name, "input_schema": tool.parameters, **info}
    if tool.description is not None:
        result["description"] = tool.description
    return result


def _tool_choice(request: Request) -> dict[str, Any] | None:
    choice = request.options.tool_choice
    parallel = request.options.parallel_tool_calls
    if isinstance(choice, ToolChoice) and not choice.name:
        raise UnsupportedRequestError("tool_choice.name", "expected a nonempty tool name")
    if choice is None and parallel is None:
        return None
    result = (
        {"type": "tool", "name": choice.name}
        if isinstance(choice, ToolChoice)
        else {"type": "any" if choice == "required" else choice or "auto"}
    )
    if parallel is not None:
        if choice == "none":
            raise UnsupportedRequestError("parallel_tool_calls", "cannot combine with tool_choice=none")
        result["disable_parallel_tool_use"] = not parallel
    return result


def _options(request: Request) -> dict[str, Any]:
    options = dict(request.options.provider_options)
    if unknown := options.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", f"unsupported or managed keys: {sorted(unknown)}")
    if "cache_control" in options:
        _cache(options["cache_control"])
    headers = options.get("extra_headers", {})
    if not isinstance(headers, dict) or any(not isinstance(v, str) for v in headers.values()):
        raise UnsupportedRequestError("extra_headers", "expected string header values")
    timeout = options.get("timeout", 1)
    if not isinstance(timeout, (float, int)) or isinstance(timeout, bool) or timeout <= 0:
        raise UnsupportedRequestError("timeout", "expected positive seconds")
    return options


def request_payload(request: Request) -> dict[str, Any]:
    if request.options.max_output_tokens is None:
        raise UnsupportedRequestError("max_output_tokens", "Anthropic requires an explicit output limit")
    system, messages = _messages(request.messages)
    payload = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.options.max_output_tokens,
        **_options(request),
    }
    if system:
        payload["system"] = system
    if request.tools:
        payload["tools"] = [_tool(tool) for tool in request.tools]
    if (choice := _tool_choice(request)) is not None:
        payload["tool_choice"] = choice
    for source, dest in (("temperature", "temperature"), ("top_p", "top_p"), ("stop", "stop_sequences")):
        if (value := getattr(request.options, source)) is not None:
            payload[dest] = value
    return payload


def string(raw: dict[str, Any], key: str, *, nonempty: bool = False) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (nonempty and not value):
        raise invalid_response(detail=f"expected {'nonempty ' if nonempty else ''}string {key}")
    return value


def block_part(block: dict[str, Any]) -> TextPart | ReasoningPart | ToolCallPart:
    kind = block.get("type")
    if kind not in _BLOCK_FIELDS or block.keys() - _BLOCK_FIELDS[kind]:
        raise invalid_response(detail=f"unsupported content block or fields: {kind!r}")
    match kind:
        case "text":
            if block.get("citations"):
                raise invalid_response(detail="citations are not supported")
            return TextPart(text=string(block, "text"))
        case "thinking":
            return ReasoningPart(
                text=string(block, "thinking"),
                provider_metadata={"anthropic": {"signature": string(block, "signature")}},
            )
        case "redacted_thinking":
            return ReasoningPart(text="", provider_metadata={"anthropic": {"redacted_data": string(block, "data")}})
        case "tool_use":
            if not isinstance(block.get("input"), dict):
                raise invalid_response(detail="tool_use input must be an object")
            caller = block.get("caller")
            if caller is not None and caller != {"type": "direct"}:
                raise invalid_response(detail="only direct function calls are supported")
            return ToolCallPart(
                tool_call_id=string(block, "id", nonempty=True),
                tool_name=string(block, "name", nonempty=True),
                tool_args=json.dumps(block["input"], ensure_ascii=False, allow_nan=False),
                provider_metadata={"anthropic": {"caller": caller}} if caller is not None else None,
            )
    raise invalid_response(detail="unsupported content block")


def _count(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise invalid_response(detail=f"invalid usage count {key}")
    return value


def usage(raw: dict[str, Any] | None) -> Usage | None:
    if raw is None:
        return None
    input_count = _count(raw, "input_tokens")
    read = _count(raw, "cache_read_input_tokens")
    write = _count(raw, "cache_creation_input_tokens")
    total_input = None if input_count is None or read is None or write is None else input_count + read + write
    return Usage(
        input_tokens=total_input,
        output_tokens=_count(raw, "output_tokens"),
        cache_read_tokens=read,
        cache_write_tokens=write,
        reasoning_tokens=_count(raw.get("output_tokens_details") or {}, "thinking_tokens"),
        raw=raw,
    )


def message_metadata(raw: dict[str, Any]) -> ProviderMetadata:
    return {"anthropic": {key: raw[key] for key in _RECORD if key in raw}}


def finish_reason(raw: dict[str, Any]) -> FinishReason:
    return _FINISH.get(string(raw, "stop_reason", nonempty=True), "other")


def check_message(raw: dict[str, Any]) -> None:
    if raw.get("role") != "assistant" or raw.get("type") != "message":
        raise invalid_response(detail="expected an assistant Message")
    allowed = _RECORD | {"id", "model", "role", "type", "content", "usage"}
    if unknown := raw.keys() - allowed:
        raise invalid_response(detail=f"unsupported message fields: {sorted(unknown)}")
    string(raw, "id", nonempty=True)
    string(raw, "model", nonempty=True)


def response(raw: dict[str, Any]) -> Response:
    check_message(raw)
    parts = [block_part(block) for block in raw["content"]]
    calls = [part.tool_call_id for part in parts if isinstance(part, ToolCallPart)]
    if len(set(calls)) != len(calls):
        raise invalid_response(detail="duplicate tool call ID")
    return Response(
        message=Message(role="assistant", parts=parts, provider_metadata=message_metadata(raw)),
        usage=usage(raw.get("usage")),
        finish_reason=finish_reason(raw),
        response_id=raw["id"],
        response_model=raw["model"],
    )
