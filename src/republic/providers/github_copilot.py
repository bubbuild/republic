"""Builtin GitHub Copilot provider hook."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from republic.auth.github_copilot import github_copilot_oauth_resolver, login_github_copilot_oauth
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

DEFAULT_GITHUB_COPILOT_API_BASE = "https://models.github.ai/inference"
DEFAULT_GITHUB_API_VERSION = "2022-11-28"


def resolve_github_copilot_api_base(api_base: str | None) -> str:
    raw = (api_base or DEFAULT_GITHUB_COPILOT_API_BASE).rstrip("/")
    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")]
    if raw.endswith("/inference"):
        return raw
    if raw.endswith("models.github.ai"):
        return f"{raw}/inference"
    return raw


def build_github_copilot_default_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": DEFAULT_GITHUB_API_VERSION,
    }


class GitHubCopilotBackend(ProviderBackend):
    capabilities = ProviderCapabilities(
        transports=COMPLETION_TRANSPORTS,
        default_completion_stream_usage=True,
    )

    def __init__(self, context: ProviderContext) -> None:
        self._context = context
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def chat(self, request: ChatRequest) -> TransportResponse:
        if request.stream:
            return TransportResponse(
                transport="completion",
                payload=self._stream_chat(request),
            )
        return TransportResponse(transport="completion", payload=self._post_chat(request))

    async def achat(self, request: ChatRequest) -> TransportResponse:
        if request.stream:
            return TransportResponse(
                transport="completion",
                payload=await self._astream_chat(request),
            )
        return TransportResponse(transport="completion", payload=await self._apost_chat(request))

    def embed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("github-copilot does not support embeddings")

    async def aembed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError("github-copilot does not support embeddings")

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

    def _base_url(self) -> str:
        return resolve_github_copilot_api_base(self._context.api_base)

    def _headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            **build_github_copilot_default_headers(),
            "Authorization": f"Bearer {self._context.api_key}",
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

    def _post_chat(self, request: ChatRequest) -> Any:
        response = self._sync().post(
            f"{self._base_url()}/chat/completions",
            json=self._payload(request),
            headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
        )
        response.raise_for_status()
        return response.json()

    async def _apost_chat(self, request: ChatRequest) -> Any:
        response = await self._async().post(
            f"{self._base_url()}/chat/completions",
            json=self._payload(request),
            headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
        )
        response.raise_for_status()
        return response.json()

    def _stream_chat(self, request: ChatRequest) -> Iterator[dict[str, Any]]:
        def _generator() -> Iterator[dict[str, Any]]:
            with self._sync().stream(
                "POST",
                f"{self._base_url()}/chat/completions",
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
                f"{self._base_url()}/chat/completions",
                json=self._payload(request),
                headers=self._headers(extra_headers=request.kwargs.get("extra_headers")),
            ) as response:
                response.raise_for_status()
                async for item in aiter_sse_json(response.aiter_lines()):
                    yield item

        return _generator()


class GitHubCopilotProviderHook(ProviderHook):
    name = "github-copilot"

    def __init__(self) -> None:
        self._resolver = github_copilot_oauth_resolver()

    def resolve_api_key(self, provider: str) -> str | None:
        return self._resolver(provider)

    def login(self, **kwargs: Any) -> Any:
        return login_github_copilot_oauth(**kwargs)

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        return GitHubCopilotBackend(context)


def should_use_github_copilot_backend(provider: str) -> bool:
    return provider == "github-copilot"
