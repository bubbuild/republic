"""Builtin Gemini provider hook."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from republic.conversation import ContentPart, Conversation, ToolCallPart, ToolResultPart
from republic.core.execution import TransportResponse
from republic.providers.sse import aiter_sse_json, iter_sse_json
from republic.providers.types import (
    COMPLETION_TRANSPORTS,
    ChatRequest,
    EmbedRequest,
    ProviderBackend,
    ProviderCapabilities,
    ProviderContext,
    ProviderHook,
)

DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def resolve_gemini_api_base(api_base: str | None) -> str:
    return (api_base or DEFAULT_GEMINI_API_BASE).rstrip("/")


class GeminiBackend(ProviderBackend):
    capabilities = ProviderCapabilities(
        transports=COMPLETION_TRANSPORTS,
        completion_max_tokens_arg="max_output_tokens",
    )

    def __init__(self, context: ProviderContext) -> None:
        self._context = context
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        return {key: value for key, value in self._context.client_args.items() if key != "default_headers"}

    def _sync(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(**self._client_kwargs())
        return self._sync_client

    def _async(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._client_kwargs())
        return self._async_client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "x-goog-api-key": str(self._context.api_key or "")}
        default_headers = self._context.client_args.get("default_headers")
        if isinstance(default_headers, dict):
            headers.update({str(key): str(value) for key, value in default_headers.items()})
        return headers

    def _request_payload(self, request: ChatRequest) -> dict[str, Any]:
        kwargs = dict(request.kwargs)
        kwargs.pop("extra_headers", None)
        system_instruction, contents = _conversation_to_gemini_contents(request.conversation)
        generation_config = _gemini_generation_config(kwargs)
        payload = {
            "contents": contents,
            "tools": _to_gemini_tools(request.tools),
            "toolConfig": _to_gemini_tool_config(kwargs.pop("tool_choice", None)),
            "systemInstruction": system_instruction,
            "generationConfig": generation_config,
        }
        payload.update(kwargs)
        return {key: value for key, value in payload.items() if value is not None}

    def chat(self, request: ChatRequest) -> TransportResponse:
        model = request.model.removeprefix("models/")
        if request.stream:
            return TransportResponse(
                transport="completion",
                payload=self._stream_chat(model, request),
            )
        response = self._sync().post(
            f"{resolve_gemini_api_base(self._context.api_base)}/models/{model}:generateContent",
            json=self._request_payload(request),
            headers=self._headers(),
        )
        response.raise_for_status()
        return TransportResponse(transport="completion", payload=_normalize_gemini_response(response.json()))

    async def achat(self, request: ChatRequest) -> TransportResponse:
        model = request.model.removeprefix("models/")
        if request.stream:
            return TransportResponse(
                transport="completion",
                payload=await self._astream_chat(model, request),
            )
        response = await self._async().post(
            f"{resolve_gemini_api_base(self._context.api_base)}/models/{model}:generateContent",
            json=self._request_payload(request),
            headers=self._headers(),
        )
        response.raise_for_status()
        return TransportResponse(transport="completion", payload=_normalize_gemini_response(response.json()))

    def _stream_chat(self, model: str, request: ChatRequest) -> Iterator[dict[str, Any]]:
        def _generator() -> Iterator[dict[str, Any]]:
            with self._sync().stream(
                "POST",
                f"{resolve_gemini_api_base(self._context.api_base)}/models/{model}:streamGenerateContent",
                params={"alt": "sse"},
                json=self._request_payload(request),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                for item in iter_sse_json(response.iter_lines()):
                    yield _normalize_gemini_chunk(item)

        return _generator()

    async def _astream_chat(self, model: str, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        async def _generator() -> AsyncIterator[dict[str, Any]]:
            async with self._async().stream(
                "POST",
                f"{resolve_gemini_api_base(self._context.api_base)}/models/{model}:streamGenerateContent",
                params={"alt": "sse"},
                json=self._request_payload(request),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for item in aiter_sse_json(response.aiter_lines()):
                    yield _normalize_gemini_chunk(item)

        return _generator()

    def embed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("gemini embeddings are not implemented")

    async def aembed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("gemini embeddings are not implemented")

    def list_models(self, **kwargs: Any) -> Any:
        response = self._sync().get(
            f"{resolve_gemini_api_base(self._context.api_base)}/models",
            params=kwargs or None,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def alist_models(self, **kwargs: Any) -> Any:
        response = await self._async().get(
            f"{resolve_gemini_api_base(self._context.api_base)}/models",
            params=kwargs or None,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()


def _conversation_to_gemini_contents(
    conversation: Conversation,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}

    for message in conversation.messages:
        if message.role in {"system", "developer"}:
            system_parts.extend(_gemini_text_parts_from_message(message))
            continue

        if message.role == "user":
            if parts := _gemini_text_parts_from_message(message):
                contents.append({"role": "user", "parts": parts})
            continue

        if message.role == "assistant":
            if parts := _gemini_assistant_parts(message, tool_names):
                contents.append({"role": "model", "parts": parts})
            continue

        if message.role == "tool" and (parts := _gemini_tool_result_parts(message, tool_names)):
            contents.append({"role": "user", "parts": parts})

    system_instruction = {"parts": system_parts} if system_parts else None
    return system_instruction, contents


def _gemini_assistant_parts(
    message: Any,
    tool_names: dict[str, str],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, ContentPart):
            parts.extend(_gemini_text_parts(part.content))
            continue
        if isinstance(part, ToolCallPart):
            parts.append(_gemini_function_call_part(part, tool_names))
    return parts


def _gemini_function_call_part(
    part: ToolCallPart,
    tool_names: dict[str, str],
) -> dict[str, Any]:
    function_call = {
        "name": part.name,
        "args": _json_value(part.arguments),
    }
    call_id = part.call_id
    if call_id:
        function_call["id"] = call_id
        tool_names[call_id] = part.name
    thought_signature = part.metadata.get("thought_signature")
    if not thought_signature:
        return {"functionCall": function_call}
    function_call["id"] = call_id or part.name
    return {"functionCall": function_call, "thoughtSignature": thought_signature}


def _gemini_tool_result_parts(
    message: Any,
    tool_names: dict[str, str],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if not isinstance(part, ToolResultPart):
            continue
        parts.append({
            "functionResponse": {
                "name": tool_names.get(part.call_id, part.call_id),
                "response": _gemini_function_response(part.output),
            }
        })
    return parts


def _gemini_text_parts_from_message(message: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, ContentPart):
            parts.extend(_gemini_text_parts(part.content))
    return parts


def _gemini_text_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            parts.extend(_gemini_text_parts(item))
        return parts
    if isinstance(content, str):
        return [{"text": content}]
    return [{"text": json.dumps(content, ensure_ascii=True, separators=(",", ":"))}]


def _to_gemini_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        declarations.append({
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": _gemini_schema(function.get("parameters", {})),
        })
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _gemini_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_gemini_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    converted = {key: _gemini_schema(item) for key, item in value.items()}
    schema_type = converted.get("type")
    if isinstance(schema_type, str):
        converted["type"] = schema_type.upper()
    return converted


def _to_gemini_tool_config(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name")
        function = tool_choice.get("function")
        if name is None and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str) and name:
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [name],
                }
            }
    return None


def _gemini_generation_config(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    if (max_tokens := kwargs.pop("max_output_tokens", None)) is not None:
        config["maxOutputTokens"] = max_tokens
    if (temperature := kwargs.pop("temperature", None)) is not None:
        config["temperature"] = temperature
    if (top_p := kwargs.pop("top_p", None)) is not None:
        config["topP"] = top_p
    if (top_k := kwargs.pop("top_k", None)) is not None:
        config["topK"] = top_k
    if (stop_sequences := kwargs.pop("stop", None)) is not None:
        config["stopSequences"] = stop_sequences if isinstance(stop_sequences, list) else [stop_sequences]
    return config or None


def _gemini_function_response(output: Any) -> dict[str, Any]:
    if output in (None, ""):
        return {}
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return {"result": output}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    return {"result": output}


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return value


def _normalize_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    message = _gemini_message(payload)
    usage = _gemini_usage(payload.get("usageMetadata"))
    normalized = {"choices": [{"message": message}], "usage": usage}
    if not message["content"] and not message["tool_calls"] and usage is not None:
        normalized["republic_metadata_only"] = True
    return normalized


def _normalize_gemini_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    message = _gemini_message(payload)
    delta: dict[str, Any] = {}
    content = message.get("content")
    if content:
        delta["content"] = content
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        delta["tool_calls"] = tool_calls
    chunk = {"choices": [{"delta": delta}]}
    usage = _gemini_usage(payload.get("usageMetadata"))
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _gemini_message(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return {"role": "assistant", "content": "", "tool_calls": []}
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if text := part.get("text"):
            text_parts.append(text)
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        tool_call: dict[str, Any] = {
            "id": function_call.get("id") or function_call.get("name"),
            "type": "function",
            "function": {
                "name": function_call.get("name"),
                "arguments": json.dumps(function_call.get("args", {}), ensure_ascii=True, separators=(",", ":")),
            },
        }
        if thought_signature := part.get("thoughtSignature"):
            tool_call["thought_signature"] = thought_signature
        tool_calls.append(tool_call)
    return {
        "role": "assistant",
        "content": "".join(text_parts),
        "tool_calls": tool_calls,
    }


def _gemini_usage(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "input_tokens": payload.get("promptTokenCount", 0),
        "output_tokens": payload.get("candidatesTokenCount", 0),
        "total_tokens": payload.get("totalTokenCount", 0),
    }


class GeminiProviderHook(ProviderHook):
    name = "gemini"

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        return GeminiBackend(context)
