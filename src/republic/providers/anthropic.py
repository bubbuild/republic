# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: native generate, owned/borrowed clients, caller-configurable retries.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""Explicit Anthropic Messages adapter using the official asynchronous client."""

from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Any, Self

import anthropic
import httpx

from republic import events
from republic.errors import ProviderError, UnsupportedRequestError
from republic.providers import _anthropic_messages as wire
from republic.providers._anthropic_stream import MessagesStream
from republic.types import Request, Response


def _provider_error(exc: Exception) -> ProviderError:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("type") if isinstance(error, dict) else None
    return ProviderError(
        str(exc),
        provider="anthropic",
        code=code if isinstance(code, str) else None,
        status_code=getattr(exc, "status_code", None),
        request_id=getattr(exc, "request_id", None),
    )


class AnthropicMessages:
    """One Messages request with explicit credentials and client ownership.

    Pass api_key (and optionally base_url) to create an owned AsyncAnthropic, or
    client to borrow its transport without changing its settings/lifetime. Owned clients default to zero retries; borrowed settings are retained. Custom transports/services must not independently
    retry if one HTTP attempt is required. No credential discovery is performed
    by Republic; a borrowed client's authentication remains caller-controlled.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        if client is None and not api_key:
            raise UnsupportedRequestError("api_key", "pass an API key explicitly or supply a client")
        options: dict[str, Any] = {}
        if api_key is not None:
            options["api_key"] = api_key
        if base_url is not None:
            options["base_url"] = base_url
        if headers is not None:
            options["default_headers"] = headers
        if timeout is not None:
            options["timeout"] = timeout
        if max_retries is not None or client is None:
            options["max_retries"] = 0 if max_retries is None else max_retries
        self._owns_client = client is None
        self._client = anthropic.AsyncAnthropic(**options) if client is None else client.with_options(**options)
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderError("closed", provider="anthropic", code="closed")

    async def generate(self, request: Request) -> Response:
        """Perform one non-streaming POST /v1/messages."""
        self._ensure_open()
        payload = wire.request_payload(request)
        try:
            result = await self._client.messages.create(**payload, stream=False)
            return wire.response(result.model_dump(mode="json", exclude_none=True))
        except (anthropic.AnthropicError, httpx.HTTPError) as exc:
            raise _provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise wire.invalid_response(detail=str(exc)) from exc

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """Stream raw Messages events, retaining argument fragments verbatim."""
        self._ensure_open()
        payload = wire.request_payload(request)
        state = MessagesStream()
        try:
            source = await self._client.messages.create(**payload, stream=True)
            async with source:
                async for chunk in source:
                    for event in state.feed(chunk.model_dump(mode="json", exclude_none=True)):
                        yield event
                    if state.terminal is not None:
                        break
                terminal = state.end()
            yield terminal
        except (anthropic.AnthropicError, httpx.HTTPError) as exc:
            raise _provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise wire.invalid_response(detail=str(exc)) from exc

    async def aclose(self) -> None:
        """Close owned clients only; active streams should be closed first."""
        if not self._closed:
            if self._owns_client:
                await self._client.close()
            self._closed = True

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.aclose()
