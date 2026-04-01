"""Builtin Anthropic provider hook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from republic.clients.parsing.common import field
from republic.core.execution import TransportResponse
from republic.providers.codecs import conversation_to_anthropic_messages
from republic.providers.types import (
    MESSAGES_TRANSPORTS,
    ChatRequest,
    EmbedRequest,
    ProviderBackend,
    ProviderCapabilities,
    ProviderContext,
    ProviderHook,
)


@dataclass
class _ToolDeltaState:
    tool_id: str
    name: str
    arguments: str = ""


class AnthropicBackend(ProviderBackend):
    capabilities = ProviderCapabilities(transports=MESSAGES_TRANSPORTS)

    def __init__(self, context: ProviderContext) -> None:
        self._context = context
        self._sync_client: Any | None = None
        self._async_client: Any | None = None

    def _sync(self) -> Any:
        if self._sync_client is None:
            from anthropic import Anthropic

            self._sync_client = Anthropic(
                api_key=self._context.api_key,
                base_url=self._context.api_base,
                **self._context.client_args,
            )
        return self._sync_client

    def _async(self) -> Any:
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            self._async_client = AsyncAnthropic(
                api_key=self._context.api_key,
                base_url=self._context.api_base,
                **self._context.client_args,
            )
        return self._async_client

    def chat(self, request: ChatRequest) -> TransportResponse:
        payload = self._build_payload(request)
        response = self._sync().messages.create(**payload)
        if request.stream:
            return TransportResponse(transport="messages", payload=self._normalize_stream(response))
        return TransportResponse(transport="messages", payload=self._normalize_message(response))

    async def achat(self, request: ChatRequest) -> TransportResponse:
        payload = self._build_payload(request)
        response = await self._async().messages.create(**payload)
        if request.stream:
            return TransportResponse(transport="messages", payload=self._normalize_async_stream(response))
        return TransportResponse(transport="messages", payload=self._normalize_message(response))

    def embed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("anthropic does not support embeddings")

    async def aembed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("anthropic does not support embeddings")

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        kwargs = dict(request.kwargs)
        system, messages = conversation_to_anthropic_messages(request.conversation)
        payload = {
            "model": request.model,
            "messages": messages,
            "system": system,
            "tools": _to_anthropic_tools(request.tools),
            "stream": request.stream,
            **kwargs,
        }
        payload["tool_choice"] = _to_anthropic_tool_choice(payload.get("tool_choice"))
        return {key: value for key, value in payload.items() if value is not None}

    def _normalize_message(self, message: Any) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in field(message, "content") or []:
            block_type = field(block, "type")
            if block_type == "text":
                text = field(block, "text")
                if text:
                    text_parts.append(text)
                continue
            if block_type != "tool_use":
                continue
            arguments = field(block, "input")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, separators=(",", ":"))
            tool_calls.append({
                "id": field(block, "id"),
                "type": "function",
                "function": {
                    "name": field(block, "name"),
                    "arguments": arguments,
                },
            })

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "".join(text_parts),
                        "tool_calls": tool_calls,
                    }
                }
            ],
            "usage": _anthropic_usage(field(message, "usage")),
        }

    def _normalize_stream(self, response: Any):
        states: dict[int, _ToolDeltaState] = {}
        usage: dict[str, Any] | None = None
        for event in response:
            chunk, usage = _normalize_stream_event(event, states, usage)
            if chunk is not None:
                yield chunk
        if usage:
            yield {"choices": [{"delta": {}}], "usage": usage}

    async def _normalize_async_stream(self, response: Any):
        states: dict[int, _ToolDeltaState] = {}
        usage: dict[str, Any] | None = None
        async for event in response:
            chunk, usage = _normalize_stream_event(event, states, usage)
            if chunk is not None:
                yield chunk
        if usage:
            yield {"choices": [{"delta": {}}], "usage": usage}


def _normalize_stream_event(
    event: Any,
    states: dict[int, _ToolDeltaState],
    usage: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    event_type = field(event, "type")
    if event_type == "message_start":
        return None, _anthropic_usage(field(field(event, "message"), "usage")) or usage
    if event_type == "content_block_start":
        return _normalize_content_block_start(event, states, usage)
    if event_type == "content_block_delta":
        return _normalize_content_block_delta(event, states, usage)
    if event_type == "message_delta" and (delta_usage := _anthropic_usage(field(field(event, "delta"), "usage"))):
        return None, delta_usage
    return None, usage


def _normalize_content_block_start(
    event: Any,
    states: dict[int, _ToolDeltaState],
    usage: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    block = field(event, "content_block")
    if field(block, "type") != "tool_use":
        return None, usage
    index = field(event, "index", len(states))
    state = _ToolDeltaState(
        tool_id=field(block, "id") or f"tool_{index}",
        name=field(block, "name") or "",
    )
    initial_input = field(block, "input")
    if initial_input:
        state.arguments = json.dumps(initial_input, separators=(",", ":"))
    states[index] = state
    if not state.name:
        return None, usage
    return _tool_chunk(index=index, tool_id=state.tool_id, name=state.name, arguments=""), usage


def _normalize_content_block_delta(
    event: Any,
    states: dict[int, _ToolDeltaState],
    usage: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    delta = field(event, "delta")
    delta_type = field(delta, "type")
    if delta_type == "text_delta":
        text = field(delta, "text")
        if text:
            return _text_chunk(text), usage
        return None, usage
    if delta_type != "input_json_delta":
        return None, usage
    index = field(event, "index", 0)
    partial = field(delta, "partial_json") or ""
    state = states.get(index)
    if state is None:
        return None, usage
    state.arguments += partial
    return _tool_chunk(index=index, tool_id=state.tool_id, name=state.name, arguments=partial), usage


def _text_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def _tool_chunk(*, index: int, tool_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ]
                }
            }
        ]
    }


def _anthropic_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    input_tokens = field(usage, "input_tokens")
    output_tokens = field(usage, "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    input_value = int(input_tokens or 0)
    output_value = int(output_tokens or 0)
    return {
        "input_tokens": input_value,
        "output_tokens": output_value,
        "total_tokens": input_value + output_value,
    }


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        converted.append({
            "name": function.get("name"),
            "description": function.get("description", ""),
            "input_schema": function.get("parameters", {}),
        })
    return converted or None


def _to_anthropic_tool_choice(tool_choice: Any) -> Any:
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or field(tool_choice.get("function"), "name")
        if name:
            return {"type": "tool", "name": name}
    return tool_choice


class AnthropicProviderHook(ProviderHook):
    name = "anthropic"

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        return AnthropicBackend(context)
