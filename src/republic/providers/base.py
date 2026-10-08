"""The base class shared by built-in and custom providers."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import httpx2

from republic._errors import APIStatusError, UnsupportedApiFormatError
from republic._formats import API_FORMATS, ApiFormatName
from republic._formats.base import ApiFormat, ChatApiFormat, DecisionApiFormat, EmbeddingApiFormat, HttpRequest
from republic._formats.sse import ServerSentEvent, iter_events

if TYPE_CHECKING:
    from republic._models import ChatModel, DecisionModel, EmbeddingModel
    from republic.history import HistoryProtocol

DEFAULT_TIMEOUT = httpx2.Timeout(600, connect=10)

_FormatT = TypeVar("_FormatT", bound=ApiFormat)


class HeaderAuth(httpx2.Auth):
    """Send a fixed header, such as an API key, with every request."""

    def __init__(self, name: str, value: str) -> None:
        self._name = name
        self._value = value

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        request.headers[self._name] = self._value
        yield request


class Provider:
    """An AI service reachable through one or more API formats.

    Credentials fall back to the ``{env_prefix}_API_KEY`` and
    ``{env_prefix}_API_BASE`` environment variables. The prefix defaults to
    ``REPUBLIC_{NAME}``, for example ``REPUBLIC_OPENAI``.
    """

    name: ClassVar[str]
    DEFAULT_API_BASE: ClassVar[str]
    SUPPORTED_API_FORMATS: ClassVar[Sequence[ApiFormatName]]

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        auth: httpx2.Auth | None = None,
        api_format: ApiFormatName | None = None,
        headers: Mapping[str, str] | None = None,
        env_prefix: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
        timeout: httpx2.Timeout | float = DEFAULT_TIMEOUT,
    ) -> None:
        try:
            from republic.__version__ import __version__
        except ImportError:
            __version__ = "0.0.0"

        env_prefix = env_prefix or f"REPUBLIC_{self.name.upper()}"
        api_key = api_key or os.getenv(f"{env_prefix}_API_KEY")
        self.api_base = (api_base or os.getenv(f"{env_prefix}_API_BASE") or self.DEFAULT_API_BASE).rstrip("/")
        self.auth = auth or (self._api_key_auth(api_key) if api_key else None)
        if api_format is not None and api_format not in self.SUPPORTED_API_FORMATS:
            raise UnsupportedApiFormatError(
                f"Provider {self.name!r} supports {list(self.SUPPORTED_API_FORMATS)}, not {api_format!r}"
            )
        self.api_format = api_format
        """The preferred format for models of its kind; other kinds use their default."""
        self.headers = dict(headers or {})
        """Sent with every request, such as beta flags or gateway attribution headers."""
        self.headers.setdefault("User-Agent", f"python-republic/{__version__}")
        self.timeout = timeout
        self._http_client = http_client

    def get_model(self, name: str, *, history: HistoryProtocol | None = None) -> ChatModel:
        from republic._models import ChatModel

        return ChatModel(self, name, self._select_api_format(ChatApiFormat), history=history)

    def get_embedding_model(self, name: str) -> EmbeddingModel:
        from republic._models import EmbeddingModel

        return EmbeddingModel(self, name, self._select_api_format(EmbeddingApiFormat))

    def get_decision_model(self, name: str) -> DecisionModel:
        from republic._models import DecisionModel

        return DecisionModel(self, name, self._select_api_format(DecisionApiFormat))

    def _api_key_auth(self, api_key: str) -> httpx2.Auth:
        """Authenticate requests with the API key. Override for other header schemes."""
        return HeaderAuth("Authorization", f"Bearer {api_key}")

    def _select_api_format(self, format_kind: type[_FormatT]) -> _FormatT:
        candidates = [self.api_format] if self.api_format is not None else []
        candidates.extend(name for name in API_FORMATS if name in self.SUPPORTED_API_FORMATS)
        for name in candidates:
            if isinstance(api_format := API_FORMATS[name], format_kind):
                return api_format
        raise UnsupportedApiFormatError(f"Provider {self.name!r} supports no {format_kind.kind} API format")

    async def _post(self, api_format: ApiFormat, request: HttpRequest) -> Any:
        async with self._client() as client:
            response = await self._send(client, api_format, request, stream=False)
            if response.is_error:
                raise APIStatusError(response.status_code, response.text)
            return response.json()

    @asynccontextmanager
    async def _stream(
        self, api_format: ApiFormat, request: HttpRequest
    ) -> AsyncGenerator[AsyncIterator[ServerSentEvent]]:
        async with self._client() as client:
            response = await self._send(client, api_format, request, stream=True)
            try:
                if response.is_error:
                    await response.aread()
                    raise APIStatusError(response.status_code, response.text)
                end_marker = "[DONE]" if api_format.name in {"chat", "responses"} else None
                yield iter_events(response.aiter_lines(), end_marker=end_marker)
            finally:
                await response.aclose()

    @asynccontextmanager
    async def _client(self) -> AsyncGenerator[httpx2.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        # A client per call keeps providers usable across event loops.
        async with httpx2.AsyncClient(timeout=self.timeout) as client:
            yield client

    async def _send(
        self, client: httpx2.AsyncClient, api_format: ApiFormat, request: HttpRequest, *, stream: bool
    ) -> httpx2.Response:
        built = client.build_request(
            "POST",
            f"{self.api_base}{request.path}",
            json=request.body,
            params=request.params,
            headers={**api_format.headers, **self.headers},
        )
        auth = httpx2.USE_CLIENT_DEFAULT if self.auth is None else self.auth
        return await client.send(built, auth=auth, stream=stream)
