# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: Chat Completions only, explicit validation, no repair.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""Conversion between Republic data and Chat Completions wire data."""

import base64
import binascii
import json
from typing import Any, cast

from republic.errors import ProviderError, UnsupportedRequestError
from republic.types import (
    FilePart,
    FinishReason,
    Message,
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

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
}
_OPTIONS = {
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "reasoning_effort",
    "user",
    "store",
    "service_tier",
    "metadata",
    "extra_headers",
    "extra_body",
    "timeout",
}
_MANAGED = {
    "model",
    "messages",
    "tools",
    "stream",
    "stream_options",
    "n",
    "temperature",
    "top_p",
    "stop",
    "tool_choice",
    "parallel_tool_calls",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "functions",
    "function_call",
    "max_retries",
}
# These are response records, not history wire fields. All remain persisted.
_RECORD_FIELDS = {"finish_reason", "system_fingerprint", "service_tier", "logprobs"}


def _metadata(value: ProviderMetadata | None, allowed: set[str], field: str) -> dict[str, Any]:
    if not value:
        return {}
    if set(value) != {"openai"} or not isinstance(value["openai"], dict):
        raise UnsupportedRequestError(field, "expected an openai metadata object")
    data = value["openai"]
    if unknown := data.keys() - allowed:
        raise UnsupportedRequestError(field, f"unrecognized metadata keys: {sorted(unknown)}")
    return dict(data)


def _image(part: FilePart) -> dict[str, Any]:
    options = _metadata(part.provider_metadata, {"detail"}, "file metadata")
    if not part.media_type.startswith("image/") or part.filename is not None:
        raise UnsupportedRequestError("file", "only user images without a filename are supported")
    detail = options.get("detail", "auto")
    if not isinstance(detail, str) or detail not in {"auto", "low", "high"}:
        raise UnsupportedRequestError("image.detail", "expected auto, low or high")
    data = part.data
    if part.encoding == "base64":
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise UnsupportedRequestError("file.data", "invalid base64") from exc
        data = f"data:{part.media_type};base64,{data}"
    elif not data.startswith(("https://", "http://", "data:image/")):
        raise UnsupportedRequestError("file.URL", "expected an HTTP(S) or image data URL")
    return {"type": "image_url", "image_url": {"url": data, **options}}


def _text(part: TextPart) -> str:
    _metadata(part.provider_metadata, set(), "text metadata")
    return part.text


def _assistant(message: Message, metadata: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": ""}
    calls = []
    for part in message.parts:
        if isinstance(part, TextPart):
            entry["content"] += _text(part)
        elif isinstance(part, ReasoningPart):
            info = _metadata(part.provider_metadata, {"field"}, "reasoning metadata")
            field = info.get("field", "reasoning")
            if not isinstance(field, str) or field not in {"reasoning", "reasoning_content"}:
                raise UnsupportedRequestError("reasoning.field", "expected reasoning or reasoning_content")
            entry[field] = entry.get(field, "") + part.text
        elif isinstance(part, ToolCallPart):
            calls.append(_history_tool_call(part))
        else:
            raise UnsupportedRequestError("assistant.part", part.kind)
    if calls:
        entry["tool_calls"] = calls
        if not entry["content"]:
            entry["content"] = None
    if "refusal" in metadata:
        if not isinstance(metadata["refusal"], str):
            raise UnsupportedRequestError("refusal", "expected a string")
        entry["refusal"] = metadata["refusal"]
    return entry


def _history_tool_call(part: ToolCallPart) -> dict[str, Any]:
    _metadata(part.provider_metadata, set(), "tool call metadata")
    if not part.tool_call_id or not part.tool_name:
        raise UnsupportedRequestError("tool.call", "ID and name must be nonempty")
    return {
        "id": part.tool_call_id,
        "type": "function",
        "function": {"name": part.tool_name, "arguments": part.tool_args},
    }


def _messages(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        metadata = _metadata(message.provider_metadata, _RECORD_FIELDS | {"refusal"}, "message metadata")
        if metadata and message.role != "assistant":
            raise UnsupportedRequestError("message.metadata", "only assistant response records are supported")
        if message.role == "assistant":
            result.append(_assistant(message, metadata))
        elif message.role == "tool":
            result.extend(_tool_results(message))
        else:
            parts: list[dict[str, Any]] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": _text(part)})
                elif isinstance(part, FilePart) and message.role == "user":
                    parts.append(_image(part))
                else:
                    raise UnsupportedRequestError(f"{message.role}.parts", part.kind)
            content = parts if any(p["type"] != "text" for p in parts) else "".join(p["text"] for p in parts)
            result.append({"role": message.role, "content": content})
    return result


def _tool_results(message: Message) -> list[dict[str, Any]]:
    if not message.parts:
        raise UnsupportedRequestError("tool.message", "requires at least one tool result")
    result = []
    for part in message.parts:
        if not isinstance(part, ToolResultPart):
            raise UnsupportedRequestError("tool.message.part", part.kind)
        _metadata(part.provider_metadata, set(), "tool result metadata")
        if part.is_error:
            raise UnsupportedRequestError(
                "tool.result.is_error", "Chat Completions has no error flag; encode it in result"
            )
        if not part.tool_call_id or not part.tool_name:
            raise UnsupportedRequestError("tool.result", "ID and name must be nonempty")
        content = part.result if isinstance(part.result, str) else json.dumps(part.result, ensure_ascii=False)
        result.append({"role": "tool", "tool_call_id": part.tool_call_id, "content": content})
    return result


def _tool(tool: Tool) -> dict[str, Any]:
    metadata = _metadata(tool.provider_metadata, {"strict"}, "tool metadata")
    if "strict" in metadata and not isinstance(metadata["strict"], bool):
        raise UnsupportedRequestError("tool.strict", "expected a boolean")
    function = {"name": tool.name, "parameters": tool.parameters, **metadata}
    if tool.description is not None:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _provider_options(request: Request) -> dict[str, Any]:
    extra = dict(request.options.provider_options)
    if unknown := extra.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", f"unsupported or managed keys: {sorted(unknown)}")
    body = extra.get("extra_body", {})
    if not isinstance(body, dict):
        raise UnsupportedRequestError("extra_body", "expected an object")
    if conflict := body.keys() & (_MANAGED | _OPTIONS):
        raise UnsupportedRequestError("extra_body", f"cannot override managed or standard fields: {sorted(conflict)}")
    headers = extra.get("extra_headers", {})
    if not isinstance(headers, dict) or any(not isinstance(v, str) for v in headers.values()):
        raise UnsupportedRequestError("extra_headers", "expected string header values")
    if "timeout" in extra and (
        not isinstance(extra["timeout"], (float, int)) or isinstance(extra["timeout"], bool) or extra["timeout"] <= 0
    ):
        raise UnsupportedRequestError("timeout", "expected positive seconds")
    return extra


def request_payload(request: Request) -> dict[str, Any]:
    """Build one request without changing or repairing caller history."""
    options = request.options
    extra = _provider_options(request)
    payload = {"model": request.model, "messages": _messages(request.messages), **extra}
    for name in ("temperature", "top_p", "stop", "parallel_tool_calls"):
        value = getattr(options, name)
        if value is not None:
            payload[name] = value
    if options.max_output_tokens is not None:
        payload["max_completion_tokens"] = options.max_output_tokens
    if options.tool_choice is not None:
        choice = options.tool_choice
        payload["tool_choice"] = (
            {"type": "function", "function": {"name": choice.name}} if isinstance(choice, ToolChoice) else choice
        )
    if request.tools:
        payload["tools"] = [_tool(tool) for tool in request.tools]
    return payload


def invalid_response(*, detail: str) -> ProviderError:
    return ProviderError(f"Invalid Chat Completions response: {detail}", provider="openai", code="invalid_response")


def finish_reason(reason: str) -> FinishReason:
    return _FINISH_REASONS.get(reason, "other")


def response_metadata(raw: dict[str, Any], reason: str) -> ProviderMetadata | None:
    data = {k: raw[k] for k in ("system_fingerprint", "service_tier") if raw.get(k) is not None}
    if reason not in _FINISH_REASONS:
        data["finish_reason"] = reason
    return {"openai": data} if data else None


def usage(raw: dict[str, Any] | None) -> Usage | None:
    if raw is None:
        return None
    return Usage(
        input_tokens=raw.get("prompt_tokens"),
        output_tokens=raw.get("completion_tokens"),
        reasoning_tokens=(raw.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        cache_read_tokens=(raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
        raw=raw,
    )


def check_message_fields(data: dict[str, Any]) -> None:
    """Fail visibly for output kinds that this increment cannot preserve/reuse."""
    for name in ("function_call", "audio", "images", "annotations", "reasoning_details"):
        if data.get(name):
            raise invalid_response(detail=f"unsupported output field {name}")
    if data.get("role", "assistant") != "assistant":
        raise invalid_response(detail="expected assistant role")


def response(raw: dict[str, Any]) -> Response:
    choices = raw.get("choices", [])
    if len(choices) != 1 or choices[0].get("index") != 0:
        raise invalid_response(detail="expected exactly one choice at index 0")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if not isinstance(reason, str) or not reason:
        raise invalid_response(detail="missing finish reason")
    data = choice.get("message")
    if not isinstance(data, dict) or data.get("role") != "assistant":
        raise invalid_response(detail="missing assistant message")
    message = _response_message(data)
    metadata = response_metadata(raw, reason) or {"openai": {}}
    details = cast(dict[str, Any], metadata["openai"])
    if data.get("refusal") is not None:
        if not isinstance(data["refusal"], str):
            raise invalid_response(detail="refusal must be a string")
        details["refusal"] = data["refusal"]
    if choice.get("logprobs") is not None:
        details["logprobs"] = choice["logprobs"]
    message.provider_metadata = metadata if details else None
    return Response(
        message=message,
        usage=usage(raw.get("usage")),
        finish_reason=finish_reason(reason),
        response_id=raw.get("id"),
        response_model=raw.get("model"),
    )


def _response_message(data: dict[str, Any]) -> Message:
    check_message_fields(data)
    message = Message(role="assistant", parts=[])
    for field in ("reasoning", "reasoning_content"):
        if data.get(field) is not None:
            message.parts.append(ReasoningPart(text=data[field], provider_metadata={"openai": {"field": field}}))
    if data.get("content") is not None:
        message.parts.append(TextPart(text=data["content"]))
    message.parts.extend(_response_tools(data.get("tool_calls") or []))
    return message


def _response_tools(calls: list[dict[str, Any]]) -> list[ToolCallPart]:
    result = []
    ids: set[str] = set()
    for call in calls:
        function = call.get("function") or {}
        if call.get("type") != "function" or not call.get("id") or not function.get("name"):
            raise invalid_response(detail="tool call needs function type, ID and name")
        if call["id"] in ids or not isinstance(function.get("arguments"), str):
            raise invalid_response(detail="duplicate tool ID or non-string arguments")
        ids.add(call["id"])
        result.append(
            ToolCallPart(tool_call_id=call["id"], tool_name=function["name"], tool_args=function["arguments"])
        )
    return result
