"""Builtin Mistral provider hook."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from republic.core.execution import TransportResponse
from republic.providers.codecs import conversation_to_completion_messages
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

DEFAULT_MISTRAL_API_BASE = "https://api.mistral.ai/v1"


def resolve_mistral_api_base(api_base: str | None) -> str:
    return (api_base or DEFAULT_MISTRAL_API_BASE).rstrip("/")


class MistralBackend(ProviderBackend):
    capabilities = ProviderCapabilities(
        transports=COMPLETION_TRANSPORTS,
        completion_max_tokens_arg="max_tokens",
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

    def _headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._context.api_key}",
            "Content-Type": "application/json",
        }
        default_headers = self._context.client_args.get("default_headers")
        if isinstance(default_headers, dict):
            headers.update({str(key): str(value) for key, value in default_headers.items()})
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        kwargs = dict(request.kwargs)
        kwargs.pop("extra_headers", None)
        payload = {
            "model": request.model,
            "messages": conversation_to_completion_messages(request.conversation),
            "tools": request.tools,
            "stream": request.stream,
            **kwargs,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def chat(self, request: ChatRequest) -> TransportResponse:
        if request.stream:
            return TransportResponse(transport="completion", payload=self._stream_chat(request))
        response = self._sync().post(
            f"{resolve_mistral_api_base(self._context.api_base)}/chat/completions",
            json=self._payload(request),
            headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
        )
        response.raise_for_status()
        return TransportResponse(transport="completion", payload=response.json())

    async def achat(self, request: ChatRequest) -> TransportResponse:
        if request.stream:
            return TransportResponse(transport="completion", payload=await self._astream_chat(request))
        response = await self._async().post(
            f"{resolve_mistral_api_base(self._context.api_base)}/chat/completions",
            json=self._payload(request),
            headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
        )
        response.raise_for_status()
        return TransportResponse(transport="completion", payload=response.json())

    def _stream_chat(self, request: ChatRequest) -> Iterator[dict[str, Any]]:
        def _generator() -> Iterator[dict[str, Any]]:
            with self._sync().stream(
                "POST",
                f"{resolve_mistral_api_base(self._context.api_base)}/chat/completions",
                json=self._payload(request),
                headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
            ) as response:
                response.raise_for_status()
                yield from iter_sse_json(response.iter_lines())

        return _generator()

    async def _astream_chat(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        async def _generator() -> AsyncIterator[dict[str, Any]]:
            async with self._async().stream(
                "POST",
                f"{resolve_mistral_api_base(self._context.api_base)}/chat/completions",
                json=self._payload(request),
                headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
            ) as response:
                response.raise_for_status()
                async for item in aiter_sse_json(response.aiter_lines()):
                    yield item

        return _generator()

    def embed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("mistral embeddings are not implemented")

    async def aembed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("mistral embeddings are not implemented")

    def list_models(self, **kwargs: Any) -> Any:
        response = self._sync().get(
            f"{resolve_mistral_api_base(self._context.api_base)}/models",
            params=kwargs or None,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def alist_models(self, **kwargs: Any) -> Any:
        response = await self._async().get(
            f"{resolve_mistral_api_base(self._context.api_base)}/models",
            params=kwargs or None,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()


class MistralProviderHook(ProviderHook):
    name = "mistral"

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        return MistralBackend(context)
