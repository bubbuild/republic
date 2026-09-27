"""Shared OpenAI client ownership and transport error mapping."""

from types import TracebackType
from typing import Self

import openai

from republic.errors import ProviderError, UnsupportedRequestError


def oauth_client(
    access_token: str,
    base_url: str,
    headers: dict[str, str],
    client: openai.AsyncOpenAI | None,
) -> tuple[openai.AsyncOpenAI, bool]:
    """Private SDK copy with explicit OAuth routing, no retries or redirects."""
    if client is not None and client._client.follow_redirects:
        raise UnsupportedRequestError("client.follow_redirects", "disable redirects for a single HTTP attempt")
    if client is None:
        result = openai.AsyncOpenAI(
            api_key=access_token,
            base_url=base_url,
            max_retries=0,
            default_headers=headers,
            http_client=openai.DefaultAsyncHttpxClient(follow_redirects=False),
        )
    else:
        result = client.with_options(
            api_key=access_token, base_url=base_url, max_retries=0, set_default_headers=headers, set_default_query={}
        )
    # SDK copy(None) inherits these; change only the new, private object.
    result.organization = None
    result.project = None
    return result, client is None


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
