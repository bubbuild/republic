# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: self-contained history, native items, no history repair.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""Conversion for the supported Responses text, reasoning and function items."""

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from republic.errors import ProviderError, UnsupportedRequestError
from republic.providers._files import detail_option, file_id, file_url, text_data
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

_RECORD = {"response_id", "model", "status", "incomplete_details", "error", "service_tier"}
_OPTIONS = {
    "text",
    "reasoning",
    "store",
    "include",
    "truncation",
    "metadata",
    "user",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "extra_headers",
    "timeout",
    "extra_body",
    "extra_query",
    "previous_response_id",
    "instructions",
}


def invalid_response(*, detail: str) -> ProviderError:
    return ProviderError(f"Invalid Responses response: {detail}", provider="openai", code="invalid_response")


def metadata(value: ProviderMetadata | None, allowed: set[str], field: str) -> dict[str, Any]:
    if not value:
        return {}
    if set(value) != {"openai"} or not isinstance(value["openai"], dict):
        raise UnsupportedRequestError(field, "expected an openai metadata object")
    data = value["openai"]
    if unknown := data.keys() - allowed:
        raise UnsupportedRequestError(field, f"unrecognized metadata keys: {sorted(unknown)}")
    return dict(data)


def required_string(raw: dict[str, Any], key: str, *, nonempty: bool = False) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (nonempty and not value):
        raise invalid_response(detail=f"expected {'nonempty ' if nonempty else ''}string {key}")
    return value


def item_metadata(item: dict[str, Any]) -> ProviderMetadata:
    return {"openai": {"item_id": item["id"], "raw_item": deepcopy(item)}}


def _texts(parts: list[dict[str, Any]], kind: str) -> str:
    return "".join(required_string(part, "text") for part in parts if part.get("type") == kind)


def item_part(item: dict[str, Any]) -> TextPart | ReasoningPart | ToolCallPart:
    """One Republic part per output item; raw metadata retains content boundaries."""
    required_string(item, "id", nonempty=True)
    info = item_metadata(item)
    match item.get("type"):
        case "message":
            if item.get("role") != "assistant":
                raise invalid_response(detail="expected assistant output role")
            for content in item["content"]:
                if content.get("type") not in {"output_text", "refusal"}:
                    raise invalid_response(detail="unsupported message content")
                required_string(content, "refusal" if content["type"] == "refusal" else "text")
            return TextPart(text=_texts(item["content"], "output_text"), provider_metadata=info)
        case "reasoning":
            for field, kind in (("summary", "summary_text"), ("content", "reasoning_text")):
                if any(part.get("type") != kind for part in item.get(field, [])):
                    raise invalid_response(detail=f"unsupported reasoning {field}")
                _texts(item.get(field, []), kind)
            return ReasoningPart(text=_texts(item["summary"], "summary_text"), provider_metadata=info)
        case "function_call":
            return ToolCallPart(
                tool_call_id=required_string(item, "call_id", nonempty=True),
                tool_name=required_string(item, "name", nonempty=True),
                tool_args=required_string(item, "arguments"),
                provider_metadata=info,
            )
        case _:
            raise invalid_response(detail=f"unsupported output item {item.get('type')!r}")


def _assistant_part(part: Part) -> dict[str, Any]:
    info = metadata(part.provider_metadata, {"item_id", "raw_item"}, "assistant.metadata")
    if info:
        raw = info.get("raw_item")
        if not isinstance(raw, dict) or info.get("item_id") != raw.get("id"):
            raise UnsupportedRequestError("assistant.metadata", "expected matching item_id and raw_item")
        try:
            restored = item_part(raw)
        except (ProviderError, KeyError, TypeError, AttributeError) as exc:
            raise UnsupportedRequestError("assistant.raw_item", str(exc)) from exc
        if restored.model_dump(exclude={"provider_metadata"}) != part.model_dump(exclude={"provider_metadata"}):
            raise UnsupportedRequestError("assistant.raw_item", "visible part differs from its retained native item")
        return deepcopy(raw)
    if isinstance(part, TextPart):
        return {"role": "assistant", "content": part.text}
    if isinstance(part, ToolCallPart) and part.tool_call_id and part.tool_name:
        return {
            "type": "function_call",
            "call_id": part.tool_call_id,
            "name": part.tool_name,
            "arguments": part.tool_args,
        }
    raise UnsupportedRequestError("assistant.part", "expected text, a named function call, or a retained native item")


def _tool_result(part: Part) -> dict[str, Any]:
    metadata(part.provider_metadata, set(), "tool_result.metadata")
    if not isinstance(part, ToolResultPart) or not part.tool_call_id:
        raise UnsupportedRequestError("tool.part", "expected tool result with call ID")
    if part.is_error:
        raise UnsupportedRequestError("tool_result.is_error", "encode the error in result explicitly")
    return {
        "type": "function_call_output",
        "call_id": part.tool_call_id,
        "output": part.result if isinstance(part.result, str) else json.dumps(part.result, ensure_ascii=False),
    }


def _file(part: FilePart) -> dict[str, Any]:
    options = metadata(part.provider_metadata, {"detail"}, "file.metadata")
    detail_option(options)
    if part.media_type.startswith("text/") and part.encoding != "file_id":
        if options or part.filename is not None:
            raise UnsupportedRequestError("file", "inline text conversion cannot retain detail/filename")
        return {"type": "input_text", "text": text_data(part)}
    if not part.media_type.startswith(("image/", "text/")) and part.media_type != "application/pdf":
        raise UnsupportedRequestError("file", "implemented Responses inputs are images, PDF and text")
    image = part.media_type.startswith("image/")
    result = {"type": "input_image" if image else "input_file", **options}
    if part.encoding == "file_id":
        result["file_id"] = file_id(part)
    elif image:
        result["image_url"] = file_url(part)
    elif part.encoding == "base64" or part.data.startswith("data:"):
        result["file_data"] = file_url(part)
        if not part.filename and part.media_type != "application/pdf":
            raise UnsupportedRequestError("file.filename", "inline documents need a filename")
        result["filename"] = part.filename or "document.pdf"
    else:
        result["file_url"] = file_url(part)
    if part.filename is not None and "filename" not in result:
        raise UnsupportedRequestError("file.filename", "only inline document data has a filename field")
    return result


def _messages(messages: list[Message], file_part: Callable[[FilePart], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        metadata(message.provider_metadata, _RECORD if message.role == "assistant" else set(), "message.metadata")
        if message.role == "assistant":
            result.extend(_assistant_part(part) for part in message.parts)
        elif message.role == "tool":
            result.extend(_tool_result(part) for part in message.parts)
        else:
            content = []
            for part in message.parts:
                if isinstance(part, FilePart) and message.role == "user":
                    content.append(file_part(part))
                elif isinstance(part, TextPart):
                    metadata(part.provider_metadata, set(), "text.metadata")
                    content.append({"type": "input_text", "text": part.text})
                else:
                    raise UnsupportedRequestError("message.part", "expected text or user media")
            result.append({"role": message.role, "content": content})
    return result


def _tool(tool: Tool) -> dict[str, Any]:
    info = metadata(tool.provider_metadata, {"strict"}, "tool.metadata")
    if not tool.name or ("strict" in info and not isinstance(info["strict"], bool)):
        raise UnsupportedRequestError("tool", "expected a name and boolean strict option")
    result = {"type": "function", "name": tool.name, "parameters": tool.parameters, **info}
    if tool.description is not None:
        result["description"] = tool.description
    return result


def _options(request: Request) -> dict[str, Any]:
    options = dict(request.options.provider_options)
    if unknown := options.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", f"unsupported or managed keys: {sorted(unknown)}")
    body = options.get("extra_body", {})
    managed = {
        "model",
        "input",
        "tools",
        "stream",
        "temperature",
        "top_p",
        "max_output_tokens",
        "tool_choice",
        "parallel_tool_calls",
    }
    if not isinstance(body, dict) or body.keys() & (managed | (_OPTIONS - {"extra_body"})):
        raise UnsupportedRequestError("extra_body", "expected native fields without managed or duplicate keys")
    includes = options.get("include", ["reasoning.encrypted_content"])
    if not isinstance(includes, list) or any(not isinstance(value, str) for value in includes):
        raise UnsupportedRequestError("include", "expected a list of native include names")
    headers = options.get("extra_headers", {})
    if not isinstance(headers, dict) or any(not isinstance(v, str) for v in headers.values()):
        raise UnsupportedRequestError("extra_headers", "expected string header values")
    timeout = options.get("timeout", 1)
    if not isinstance(timeout, (float, int)) or isinstance(timeout, bool) or timeout <= 0:
        raise UnsupportedRequestError("timeout", "expected positive seconds")
    return {"store": False, "truncation": "disabled", "include": includes, **options}


def request_payload(request: Request, *, file_part: Callable[[FilePart], dict[str, Any]] = _file) -> dict[str, Any]:
    if request.options.stop is not None:
        raise UnsupportedRequestError("stop", "Responses has no stop-sequence option")
    payload = {"model": request.model, "input": _messages(request.messages, file_part), **_options(request)}
    for field in ("temperature", "top_p", "max_output_tokens", "parallel_tool_calls"):
        if (value := getattr(request.options, field)) is not None:
            payload[field] = value
    if (choice := request.options.tool_choice) is not None:
        payload["tool_choice"] = {"type": "function", "name": choice.name} if isinstance(choice, ToolChoice) else choice
    if request.tools:
        payload["tools"] = [_tool(tool) for tool in request.tools]
    return payload


def finish_reason(raw: dict[str, Any]) -> FinishReason:
    match raw.get("status"):
        case "completed":
            return "tool_call" if any(item["type"] == "function_call" for item in raw["output"]) else "stop"
        case "incomplete":
            reason = (raw.get("incomplete_details") or {}).get("reason")
            reasons: dict[str, FinishReason] = {"max_output_tokens": "length", "content_filter": "content_filter"}
            return reasons.get(reason, "other") if isinstance(reason, str) else "other"
        case "failed":
            return "error"
        case _:
            raise invalid_response(detail=f"nonterminal response status {raw.get('status')!r}")


def response(raw: dict[str, Any]) -> Response:
    reason = finish_reason(raw)
    info = {key: raw[key] for key in _RECORD - {"response_id"} if key in raw}
    info["response_id"] = required_string(raw, "id", nonempty=True)
    raw_usage = raw.get("usage")
    usage = (
        None
        if raw_usage is None
        else Usage(
            input_tokens=raw_usage.get("input_tokens"),
            output_tokens=raw_usage.get("output_tokens"),
            reasoning_tokens=(raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            cache_read_tokens=(raw_usage.get("input_tokens_details") or {}).get("cached_tokens"),
            raw=raw_usage,
        )
    )
    parts = [item_part(item) for item in raw["output"]]
    ids = [item["id"] for item in raw["output"]]
    calls = [part.tool_call_id for part in parts if isinstance(part, ToolCallPart)]
    if len(set(ids)) != len(ids) or len(set(calls)) != len(calls):
        raise invalid_response(detail="duplicate item or call ID")
    return Response(
        message=Message(role="assistant", parts=parts, provider_metadata={"openai": info}),
        usage=usage,
        finish_reason=reason,
        response_id=info["response_id"],
        response_model=required_string(raw, "model", nonempty=True),
    )
