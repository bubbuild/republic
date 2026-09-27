# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: native generate, explicit ownership and no SDK retries.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""OpenAI-compatible Chat Completions via the official asynchronous client."""

from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Self

import httpx
import openai

from republic import events
from republic.errors import ProviderError, UnsupportedRequestError
from republic.providers import _openai_chat as chat
from republic.providers._openai_stream import ChatStream
from republic.types import Request, Response


def _provider_error(exc: Exception) -> ProviderError:
    code = getattr(exc, "code", None)
    return ProviderError(
        str(exc),
        provider="openai",
        status_code=getattr(exc, "status_code", None),
        code=code if isinstance(code, str) else None,
        request_id=getattr(exc, "request_id", None),
    )


class OpenAIChatCompletions:
    """One request per operation, with owned or borrowed AsyncOpenAI clients.

    Without client, Republic creates and owns an AsyncOpenAI(max_retries=0).
    With client, it borrows the transport through with_options(max_retries=0),
    leaving the caller's settings and lifetime unchanged. Custom transports and
    endpoints must not independently retry if one HTTP attempt is required.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        if client is not None and (api_key is not None or base_url is not None):
            raise UnsupportedRequestError("client.configuration", "use client or api_key/base_url, not both")
        self._owns_client = client is None
        self._client = (
            openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
            if client is None
            else client.with_options(max_retries=0)
        )
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderError("closed", provider="openai", code="closed")

    async def generate(self, request: Request) -> Response:
        """Perform one non-streaming POST /chat/completions."""
        self._ensure_open()
        payload = chat.request_payload(request)
        try:
            result = await self._client.chat.completions.create(**payload, stream=False)
            return chat.response(result.model_dump(mode="json", exclude_none=True))
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise _provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise chat.invalid_response(detail=str(exc)) from exc

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """Stream one request; read trailing usage before emitting StreamEnd."""
        self._ensure_open()
        payload = chat.request_payload(request)
        state = ChatStream()
        try:
            source = await self._client.chat.completions.create(
                **payload,
                stream=True,
                stream_options={"include_usage": True},
            )
            async with source:
                async for chunk in source:
                    for event in state.feed(chunk.model_dump(mode="json", exclude_none=True)):
                        yield event
                terminal = state.end()
            # Close the wire response before handing completion to the caller.
            yield terminal
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise _provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise chat.invalid_response(detail=str(exc)) from exc

    async def aclose(self) -> None:
        """Close this adapter and its owned client; borrowed clients stay open."""
        if not self._closed:
            if self._owns_client:
                await self._client.close()
            self._closed = True

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
