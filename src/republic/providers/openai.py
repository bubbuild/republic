"""Builtin OpenAI provider hooks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from republic.auth.openai_codex import (
    extract_openai_codex_account_id,
    login_openai_codex_oauth,
    openai_codex_oauth_resolver,
)
from republic.core.errors import ErrorKind, RepublicError
from republic.core.execution import TransportResponse
from republic.providers.codecs import conversation_to_completion_messages, conversation_to_openai_responses_input
from republic.providers.types import (
    OPENAI_TRANSPORTS,
    OPENROUTER_TRANSPORTS,
    RESPONSES_TRANSPORTS,
    ChatRequest,
    EmbedRequest,
    ProviderBackend,
    ProviderCapabilities,
    ProviderContext,
    ProviderHook,
    ResponsesRequest,
)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CODEX_ORIGINATOR = "republic"
DEFAULT_CODEX_INCLUDE = ("reasoning.encrypted_content",)
DEFAULT_CODEX_INSTRUCTIONS = "You are Codex."
DEFAULT_CODEX_TEXT_CONFIG = {"verbosity": "medium"}
_OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "openrouter"})


class OpenAICodexTransportError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OpenAIBackend(ProviderBackend):
    capabilities = ProviderCapabilities(
        transports=OPENAI_TRANSPORTS,
        supports_embeddings=True,
        preserves_responses_extra_headers=True,
        default_completion_stream_usage=True,
        completion_max_tokens_arg="max_completion_tokens",
    )

    def __init__(self, context: ProviderContext) -> None:
        self._context = context
        self._sync_client: Any | None = None
        self._async_client: Any | None = None
        if context.provider == "openrouter":
            self.capabilities = ProviderCapabilities(
                transports=OPENROUTER_TRANSPORTS,
                supports_embeddings=True,
                default_completion_stream_usage=True,
                completion_max_tokens_arg="max_tokens",
            )

    def validate_chat_request(self, request: ChatRequest) -> None:
        if self._context.provider != "openrouter":
            return
        if request.transport == "messages" and not request.model.startswith("anthropic/"):
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                f"{self._context.provider}:{request.model}: messages format is only supported for Anthropic models",
            )
        if request.transport == "responses" and request.model.startswith("anthropic/") and request.tools:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                (
                    f"{self._context.provider}:{request.model}: responses format is not supported "
                    "for this model when tools are enabled"
                ),
            )

    def _sync(self) -> Any:
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(
                api_key=self._context.api_key,
                base_url=self._resolved_api_base(),
                **self._context.client_args,
            )
        return self._sync_client

    def _async(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self._context.api_key,
                base_url=self._resolved_api_base(),
                **self._context.client_args,
            )
        return self._async_client

    def _resolved_api_base(self) -> str | None:
        if self._context.api_base is not None:
            return self._context.api_base
        if self._context.provider == "openrouter":
            return DEFAULT_OPENROUTER_BASE_URL
        return None

    @staticmethod
    def _clean_payload(**payload: Any) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

    def chat(self, request: ChatRequest) -> TransportResponse:
        client = self._sync()
        if request.transport == "responses":
            instructions, input_items = conversation_to_openai_responses_input(request.conversation)
            payload = client.responses.create(
                **self._clean_payload(
                    model=request.model,
                    input=input_items,
                    tools=request.tools,
                    stream=request.stream,
                    instructions=instructions,
                    **request.kwargs,
                )
            )
            return TransportResponse(transport="responses", payload=payload)

        payload = client.chat.completions.create(
            **self._clean_payload(
                model=request.model,
                messages=conversation_to_completion_messages(request.conversation),
                tools=request.tools,
                stream=request.stream,
                reasoning_effort=request.reasoning_effort,
                **request.kwargs,
            )
        )
        return TransportResponse(transport=request.transport, payload=payload)

    async def achat(self, request: ChatRequest) -> TransportResponse:
        client = self._async()
        if request.transport == "responses":
            instructions, input_items = conversation_to_openai_responses_input(request.conversation)
            payload = await client.responses.create(
                **self._clean_payload(
                    model=request.model,
                    input=input_items,
                    tools=request.tools,
                    stream=request.stream,
                    instructions=instructions,
                    **request.kwargs,
                )
            )
            return TransportResponse(transport="responses", payload=payload)

        payload = await client.chat.completions.create(
            **self._clean_payload(
                model=request.model,
                messages=conversation_to_completion_messages(request.conversation),
                tools=request.tools,
                stream=request.stream,
                reasoning_effort=request.reasoning_effort,
                **request.kwargs,
            )
        )
        return TransportResponse(transport=request.transport, payload=payload)

    def embed(self, request: EmbedRequest) -> Any:
        return self._sync().embeddings.create(model=request.model, input=request.inputs, **request.kwargs)

    async def aembed(self, request: EmbedRequest) -> Any:
        return await self._async().embeddings.create(model=request.model, input=request.inputs, **request.kwargs)

    def responses(self, request: ResponsesRequest) -> Any:
        return self._sync().responses.create(model=request.model, input=request.input_data, **request.kwargs)

    async def aresponses(self, request: ResponsesRequest) -> Any:
        return await self._async().responses.create(model=request.model, input=request.input_data, **request.kwargs)

    def list_models(self, **kwargs: Any) -> Any:
        return self._sync().models.list(**kwargs)

    async def alist_models(self, **kwargs: Any) -> Any:
        return await self._async().models.list(**kwargs)

    def create_batch(self, **kwargs: Any) -> Any:
        return self._sync().batches.create(**kwargs)

    async def acreate_batch(self, **kwargs: Any) -> Any:
        return await self._async().batches.create(**kwargs)

    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self._sync().batches.retrieve(batch_id, **kwargs)

    async def aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self._async().batches.retrieve(batch_id, **kwargs)

    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self._sync().batches.cancel(batch_id, **kwargs)

    async def acancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self._async().batches.cancel(batch_id, **kwargs)

    def list_batches(self, **kwargs: Any) -> Any:
        return self._sync().batches.list(**kwargs)

    async def alist_batches(self, **kwargs: Any) -> Any:
        return await self._async().batches.list(**kwargs)


class OpenAICodexBackend(ProviderBackend):
    capabilities = ProviderCapabilities(
        transports=RESPONSES_TRANSPORTS,
        preferred_transport="responses",
        preserves_responses_extra_headers=True,
    )

    def __init__(self, context: ProviderContext) -> None:
        self._context = context
        self._sync_client: Any | None = None
        self._async_client: Any | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        return {key: value for key, value in self._context.client_args.items() if key != "default_headers"}

    def _default_headers(self) -> dict[str, str]:
        default_headers = dict(self._context.client_args.get("default_headers", {}))
        default_headers.update(build_openai_codex_default_headers(self._context.api_key or ""))
        return default_headers

    def _sync(self) -> Any:
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(
                api_key=self._context.api_key,
                base_url=resolve_openai_codex_api_base(self._context.api_base),
                default_headers=self._default_headers(),
                **self._client_kwargs(),
            )
        return self._sync_client

    def _async(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self._context.api_key,
                base_url=resolve_openai_codex_api_base(self._context.api_base),
                default_headers=self._default_headers(),
                **self._client_kwargs(),
            )
        return self._async_client

    def chat(self, request: ChatRequest) -> TransportResponse:
        response = self._sync().responses.create(**self._build_payload(request))
        if request.stream:
            return TransportResponse(transport="responses", payload=response)
        return TransportResponse(transport="responses", payload=self._collect_responses_events(response))

    async def achat(self, request: ChatRequest) -> TransportResponse:
        response = await self._async().responses.create(**self._build_payload(request))
        if request.stream:
            return TransportResponse(transport="responses", payload=response)
        return TransportResponse(
            transport="responses",
            payload=self._collect_responses_events(await self._collect_async_events(response)),
        )

    def responses(self, request: ResponsesRequest) -> Any:
        return self._sync().responses.create(model=request.model, input=request.input_data, **request.kwargs)

    async def aresponses(self, request: ResponsesRequest) -> Any:
        return await self._async().responses.create(model=request.model, input=request.input_data, **request.kwargs)

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        instructions, input_items = conversation_to_openai_responses_input(request.conversation)
        payload = {
            "model": request.model,
            "input": input_items,
            "tools": request.tools,
            "stream": True,
            "instructions": instructions or DEFAULT_CODEX_INSTRUCTIONS,
            "store": False,
            "include": list(DEFAULT_CODEX_INCLUDE),
            "text": dict(DEFAULT_CODEX_TEXT_CONFIG),
            **request.kwargs,
        }
        payload.pop("max_output_tokens", None)
        payload.pop("max_tokens", None)
        text = payload.get("text")
        if isinstance(text, dict):
            payload["text"] = {**DEFAULT_CODEX_TEXT_CONFIG, **text}
        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    async def _collect_async_events(response: AsyncIterator[Any]) -> list[Any]:
        events: list[Any] = []
        async for event in response:
            events.append(event)
        return events

    @staticmethod
    def _collect_responses_events(response: Iterator[Any] | list[Any]) -> dict[str, Any]:
        events = list(response) if not isinstance(response, list) else response
        collected = _ResponsesEventCollector()
        for event in events:
            collected.consume(event)
        return collected.build_response()


class _ResponsesEventCollector:
    def __init__(self) -> None:
        text_parts: list[str] = []
        tool_calls: dict[str, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None
        completed_response: Any | None = None

        self.text_parts = text_parts
        self.tool_calls = tool_calls
        self.usage = usage
        self.completed_response = completed_response

    def consume(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            self._consume_text_delta(event)
            return
        if event_type == "response.output_item.done":
            self._consume_output_item(event)
            return
        if event_type == "response.function_call_arguments.done":
            self._consume_arguments_done(event)
            return
        if event_type == "response.completed":
            self._consume_completed(event)
            return
        self._consume_event_usage(event)

    def _consume_text_delta(self, event: Any) -> None:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            self.text_parts.append(delta)

    def _consume_output_item(self, event: Any) -> None:
        if call := _function_call_from_output_item(event):
            self.tool_calls[call["call_id"]] = call

    def _consume_arguments_done(self, event: Any) -> None:
        if call := _function_call_from_arguments_done(event):
            self.tool_calls[call["call_id"]] = call

    def _consume_completed(self, event: Any) -> None:
        self.completed_response = getattr(event, "response", None)
        completed_usage = getattr(self.completed_response, "usage", None)
        if completed_usage is not None:
            self.usage = _normalize_response_usage(completed_usage)

    def _consume_event_usage(self, event: Any) -> None:
        event_usage = getattr(event, "usage", None)
        if event_usage is not None:
            self.usage = _normalize_response_usage(event_usage)

    def build_response(self) -> dict[str, Any]:
        completed_response = self.completed_response
        text_parts = self.text_parts
        tool_calls = self.tool_calls
        usage = self.usage
        existing_output = getattr(completed_response, "output", None)
        if isinstance(existing_output, list) and existing_output:
            output = existing_output
        else:
            output = []
            if text_parts:
                output.append({
                    "id": "msg_codex",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "".join(text_parts), "annotations": []}],
                })
            output.extend(
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments", ""),
                    "status": "completed",
                }
                for call in tool_calls.values()
            )

        return {
            "id": getattr(completed_response, "id", None) or "resp_codex",
            "created_at": getattr(completed_response, "created_at", None) or 0,
            "model": getattr(completed_response, "model", None) or "gpt-5-codex",
            "object": getattr(completed_response, "object", None) or "response",
            "output": output,
            "parallel_tool_calls": getattr(completed_response, "parallel_tool_calls", None) or False,
            "tool_choice": getattr(completed_response, "tool_choice", None) or "auto",
            "tools": getattr(completed_response, "tools", None) or [],
            "usage": usage or _normalize_response_usage(getattr(completed_response, "usage", None)),
        }


def _normalize_response_usage(usage: Any) -> dict[str, Any]:
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        usage = model_dump()
    payload = dict(usage) if isinstance(usage, dict) else {}
    input_tokens = payload.get("input_tokens", 0)
    output_tokens = payload.get("output_tokens", 0)
    total_tokens = payload.get("total_tokens")
    if not isinstance(total_tokens, int):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": payload.get("input_tokens_details") or {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": payload.get("output_tokens_details") or {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }


def _function_call_from_output_item(event: Any) -> dict[str, Any] | None:
    item = getattr(event, "item", None)
    if getattr(item, "type", None) != "function_call":
        return None
    call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
    if not isinstance(call_id, str) or not call_id:
        return None
    return {
        "type": "function_call",
        "call_id": call_id,
        "id": getattr(item, "id", None),
        "name": getattr(item, "name", None),
        "arguments": getattr(item, "arguments", "") or "",
    }


def _function_call_from_arguments_done(event: Any) -> dict[str, Any] | None:
    call_id = getattr(event, "call_id", None) or getattr(event, "item_id", None)
    if not isinstance(call_id, str) or not call_id:
        return None
    return {
        "type": "function_call",
        "call_id": call_id,
        "id": getattr(event, "item_id", None),
        "name": getattr(event, "name", None),
        "arguments": getattr(event, "arguments", "") or "",
    }


def resolve_openai_codex_api_base(api_base: str | None) -> str:
    raw = (api_base or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if raw.endswith("/responses"):
        raw = raw[: -len("/responses")]
    if raw.endswith("/codex"):
        return raw
    return f"{raw}/codex"


def build_openai_codex_default_headers(
    api_key: str,
    *,
    originator: str = DEFAULT_CODEX_ORIGINATOR,
) -> dict[str, str]:
    account_id = extract_openai_codex_account_id(api_key)
    if account_id is None:
        raise OpenAICodexTransportError(None, "OpenAI Codex OAuth token is missing chatgpt_account_id")
    return {
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
    }


class OpenAIProviderHook(ProviderHook):
    name = "openai"

    def __init__(self) -> None:
        self._codex_resolver = openai_codex_oauth_resolver()

    def matches(self, provider: str) -> bool:
        return provider in _OPENAI_COMPATIBLE_PROVIDERS

    def resolve_api_key(self, provider: str) -> str | None:
        return self._codex_resolver(provider)

    def login(self, **kwargs: Any) -> Any:
        return login_openai_codex_oauth(**kwargs)

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        if (
            context.provider == "openai"
            and context.api_key
            and extract_openai_codex_account_id(context.api_key) is not None
        ):
            return OpenAICodexBackend(context)
        return OpenAIBackend(context)
