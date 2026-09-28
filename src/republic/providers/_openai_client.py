"""Shared OpenAI client ownership and transport error mapping."""

from types import TracebackType
from typing import Any, Self

import httpx
import openai

from republic.errors import ProviderError


def oauth_client(
    access_token: str,
    base_url: str,
    headers: dict[str, str],
    client: openai.AsyncOpenAI | None,
    *,
    endpoint: str | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> tuple[openai.AsyncOpenAI, bool]:
    """Service defaults < borrowed settings < explicit overrides, on a private copy."""
    merged = httpx.Headers(headers)
    if client is not None:
        merged.update(client._custom_headers)
    merged.update(extra_headers or {})
    canonical = {
        "user-agent": "User-Agent",
        "authorization": "Authorization",
        "accept": "Accept",
        "content-type": "Content-Type",
    }
    configured = {canonical.get(key, key): value for key, value in merged.items()}
    options: dict[str, Any] = {"api_key": access_token}
    options["default_headers" if client is None else "set_default_headers"] = configured
    if endpoint is not None or client is None:
        options["base_url"] = endpoint or base_url
    if max_retries is not None or client is None:
        options["max_retries"] = 0 if max_retries is None else max_retries
    if timeout is not None:
        options["timeout"] = timeout
    if client is None:
        return openai.AsyncOpenAI(**options, http_client=openai.DefaultAsyncHttpxClient(follow_redirects=False)), True
    return client.with_options(**options), False


def oauth_error(exc: Exception, *, provider: str) -> ProviderError:
    """Fixed diagnostics; never retain an OAuth-bearing SDK error as a cause."""
    status = getattr(exc, "status_code", None)
    code = {401: "unauthorized", 403: "forbidden", 429: "rate_limit"}.get(status, "request_failed")
    if isinstance(exc, ProviderError):
        code = "invalid_response" if exc.code == "invalid_response" else "stream_error"
    return ProviderError(f"{provider} request: {code}", provider=provider, status_code=status, code=code)


def provider_error(exc: Exception) -> ProviderError:
    code = getattr(exc, "code", None)
    return ProviderError(
        str(exc),
        provider="openai",
        status_code=getattr(exc, "status_code", None),
        code=code if isinstance(code, str) else None,
        request_id=getattr(exc, "request_id", None),
    )


class OpenAIClient:
    """Internal lifetime helper shared by the two explicit protocols."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
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
        self._client = openai.AsyncOpenAI(**options) if client is None else client.with_options(**options)
        self._closed = False

    def _request_headers(self, payload: dict[str, Any]) -> None:
        # The SDK merges dictionaries before creating case-insensitive HTTP headers.
        # Match existing key casing so a lower-case override does not get appended.
        if "extra_headers" in payload:
            names = {key.lower(): key for key in self._client.default_headers}
            payload["extra_headers"] = {
                names.get(key.lower(), key): value for key, value in payload["extra_headers"].items()
            }

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderError("closed", provider="openai", code="closed")

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
