"""The base class shared by built-in and custom providers."""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass, field, replace
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeAlias, TypeVar

import httpx2

from republic._response import ModelInfo
from republic.auth import Auth, HeaderAuth
from republic.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    UnsupportedApiFormatError,
    UnsupportedFeatureError,
)
from republic.events import CallEvent, Completed, Retry
from republic.formats import _API_FORMATS, ApiFormatName
from republic.formats._base import (
    ApiFormat,
    ChatApiFormat,
    DecisionApiFormat,
    EmbeddingApiFormat,
    HttpRequest,
    ImageApiFormat,
)
from republic.formats._sse import ServerSentEvent, iter_events

if TYPE_CHECKING:
    from republic._models import ChatModel, DecisionModel, EmbeddingModel, ImageModel
    from republic.history import HistoryProtocol

DEFAULT_TIMEOUT = httpx2.Timeout(600, connect=10)

_FormatT = TypeVar("_FormatT", bound=ApiFormat)
_T = TypeVar("_T")


@dataclass(frozen=True)
class Call:
    """The public call a middleware wraps."""

    provider: str
    operation: str
    """``chat``, ``stream``, ``embed``, ``generate``, ``edit``, ``decide`` or ``list_models``."""
    model: str | None
    """``None`` for ``list_models``."""
    events: AsyncGenerator[CallEvent, None] = field(compare=False, repr=False)
    """The events of the next layer in: ``Retry`` events, stream events, and a final ``Completed``.

    Errors are raised from the iteration.
    """


Middleware: TypeAlias = Callable[[Call], AsyncGenerator[CallEvent, None]]
"""An async generator function that iterates ``call.events`` and yields the events passed on."""


@dataclass(frozen=True)
class HttpStream:
    events: AsyncIterator[ServerSentEvent]
    headers: Mapping[str, str]


class MiddlewareStack:
    """A provider's middlewares and retry policy, shared by all of its calls."""

    def __init__(self, provider: Provider, middlewares: Sequence[Middleware]) -> None:
        self.provider = provider
        self.middlewares = tuple(middlewares)

    async def wrap(self, call: Call) -> AsyncGenerator[CallEvent, None]:
        """Pass the call's own events through every middleware; the first is outermost.

        Each middleware gets a copy of ``call`` whose ``events`` are those of the next
        layer in. Closing the returned generator closes every layer, outermost first,
        so each sees an early exit at its ``yield``.
        """
        async with AsyncExitStack() as layers:
            events = call.events
            layers.push_async_callback(events.aclose)
            for middleware in reversed(self.middlewares):
                events = middleware(replace(call, events=events))
                layers.push_async_callback(events.aclose)
            async for event in events:
                yield event

    async def run(self, call: Call) -> Any:
        """Run a call to the end and return the response of the last ``Completed`` event."""
        completed: Completed[Any] | None = None
        async with aclosing(self.wrap(call)) as events:
            async for event in events:
                if isinstance(event, Completed):
                    completed = event
        if completed is None:
            raise RuntimeError("Middleware finished the call without a Completed event")
        return completed.response

    async def retry(self, send: Callable[[], Awaitable[_T]]) -> AsyncGenerator[Retry | _T, None]:
        """Yield a ``Retry`` before each wait, then the result of the first successful attempt.

        ``send`` makes one attempt and raises ``APIStatusError`` for an error status.
        Retry only the request, never replay a successful streaming response.
        """
        provider = self.provider
        for attempt in range(provider.max_retries + 1):
            try:
                result = await send()
            except httpx2.RequestError as exc:
                retryable = isinstance(exc, (httpx2.TimeoutException, httpx2.NetworkError, httpx2.RemoteProtocolError))
                if not retryable or attempt == provider.max_retries:
                    raise _connection_error(exc) from exc
                error: Exception = _connection_error(exc)
                error.__cause__ = exc
                headers: Mapping[str, str] = {}
            except APIStatusError as exc:
                if attempt == provider.max_retries or not (
                    exc.status_code in {408, 409, 429} or exc.status_code >= 500
                ):
                    raise
                error, headers = exc, exc.headers
            else:
                yield result
                return
            delay = _retry_after(headers)
            if delay is None:
                delay = min(provider.max_retry_delay, provider.retry_delay * 2 ** min(attempt, 30))
                delay *= random.uniform(0.75, 1)  # noqa: S311 - retry jitter, not cryptography
            delay = min(delay, provider.max_retry_delay)
            yield Retry(error, attempt + 1, delay)
            await asyncio.sleep(delay)


class Provider:
    """An AI service reachable through one or more API formats.

    Credentials fall back to the ``{env_prefix}_API_KEY`` and
    ``{env_prefix}_API_BASE`` environment variables. The prefix defaults to
    ``REPUBLIC_{NAME}`` with hyphens as underscores, for example
    ``REPUBLIC_OPENAI`` or ``REPUBLIC_AZURE_OPENAI``.
    """

    name: ClassVar[str]
    DEFAULT_API_BASE: ClassVar[str]
    SUPPORTED_API_FORMATS: ClassVar[Sequence[ApiFormatName]]
    """The API formats this provider speaks, in order of preference within each kind of model."""
    MODELS_PATH: ClassVar[str | None] = "/models"
    """Where the model list is served, relative to ``api_base``; ``None`` when the service has none."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        auth: Auth | None = None,
        api_format: ApiFormatName | None = None,
        headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        env_prefix: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
        timeout: httpx2.Timeout | float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        max_retry_delay: float = 60,
        middlewares: Sequence[Middleware] = (),
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if any(not math.isfinite(value) or value < 0 for value in (retry_delay, max_retry_delay)):
            raise ValueError("Retry delays must be finite and nonnegative")
        try:
            from republic.__version__ import __version__
        except ImportError:
            __version__ = "0.0.0"

        env_prefix = env_prefix or f"REPUBLIC_{self.name.upper().replace('-', '_')}"
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
        self.extra_body = dict(extra_body or {})
        """Merged into every chat request body, under the ``extra_body`` of each call."""
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self._middleware_stack = MiddlewareStack(self, middlewares)
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._open_contexts: list[Provider | ChatModel | EmbeddingModel | ImageModel | DecisionModel] = []

    @property
    def is_closed(self) -> bool:
        """Whether this provider has been closed, independently of an external client."""
        return self._closed

    async def __aenter__(self) -> Self:
        self._open_context(self)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._close_context(self)

    async def close(self) -> None:
        """Close the provider and its internally created client; leave supplied clients open.

        Finish requests and exit stream contexts before closing. Close an owned
        client on the event loop that first used it. Repeated closes are harmless.
        """
        if self._closed:
            return
        self._check_client_loop()
        self._closed = True
        self._open_contexts.clear()
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()

    def _open_context(self, context: Provider | ChatModel | EmbeddingModel | ImageModel | DecisionModel) -> None:
        self._ensure_open()
        self._check_client_loop()
        # Record each entry so nested contexts for the same object stay open.
        self._open_contexts.append(context)

    async def _close_context(self, context: Provider | ChatModel | EmbeddingModel | ImageModel | DecisionModel) -> None:
        self._check_client_loop()
        if context not in self._open_contexts:
            return
        self._open_contexts.remove(context)
        if not self._open_contexts:
            await self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Provider is closed")

    def _check_client_loop(self) -> None:
        if self._client_loop is not None and self._client_loop is not asyncio.get_running_loop():
            raise RuntimeError("Use and close the provider on the event loop that created its HTTP client")

    def get_model(self, name: str, *, history: HistoryProtocol | None = None) -> ChatModel:
        from republic._models import ChatModel

        return ChatModel(self, name, self.select_api_format(ChatApiFormat, name), history=history)

    def get_embedding_model(self, name: str) -> EmbeddingModel:
        from republic._models import EmbeddingModel

        return EmbeddingModel(self, name, self.select_api_format(EmbeddingApiFormat, name))

    def get_image_model(self, name: str) -> ImageModel:
        from republic._models import ImageModel

        return ImageModel(self, name, self.select_api_format(ImageApiFormat, name))

    def get_decision_model(self, name: str) -> DecisionModel:
        from republic._models import DecisionModel

        return DecisionModel(self, name, self.select_api_format(DecisionApiFormat, name))

    async def list_models(self) -> list[ModelInfo]:
        """List the models the service offers to these credentials, following every page."""
        return await self._middleware_stack.run(Call(self.name, "list_models", None, self._list_models()))

    async def _list_models(self) -> AsyncGenerator[CallEvent, None]:
        if self.MODELS_PATH is None:
            raise UnsupportedFeatureError(f"Provider {self.name!r} cannot list models")
        api_format = _API_FORMATS[self.api_format or self.SUPPORTED_API_FORMATS[0]]
        headers = {**api_format.headers, **self.headers}
        auth = httpx2.USE_CLIENT_DEFAULT if self.auth is None else self.auth
        models: list[ModelInfo] = []
        params = self._models_params(None)
        client = self._client()

        async def get(params: dict[str, str]) -> httpx2.Response:
            response = await client.get(f"{self.api_base}{self.MODELS_PATH}", params=params, headers=headers, auth=auth)
            if response.is_error:
                raise APIStatusError(response.status_code, response.text, headers=response.headers)
            return response

        while params is not None:
            async for response in self._middleware_stack.retry(lambda params=params: get(params)):
                if isinstance(response, Retry):
                    yield response
                    continue
                data = response.json()
                models.extend(self._parse_models(data))
                params = self._models_params(data)
        yield Completed(models)

    def _models_params(self, previous: Any) -> dict[str, str] | None:
        """Query parameters for the next page of models, or ``None`` when ``previous`` was the last.

        ``previous`` is ``None`` before the first page. Override for paginated lists.
        """
        return {} if previous is None else None

    def _parse_models(self, data: Any) -> list[ModelInfo]:
        """Read one page of models. Accepts the common ``data`` and ``models`` list shapes."""
        items = data if isinstance(data, list) else data.get("data") or data.get("models") or []
        models = []
        for item in items:
            model_id = item.get("id") or item.get("name")
            display_name = item.get("display_name") or item.get("displayName") or item.get("name")
            models.append(ModelInfo(model_id, None if display_name == model_id else display_name, item))
        return models

    def _api_key_auth(self, api_key: str) -> Auth:
        """Authenticate requests with the API key. Override for other header schemes."""
        return HeaderAuth("Authorization", f"Bearer {api_key}")

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        """Pick the API format used by ``model`` for one kind of model.

        The requested ``api_format`` wins, then the first format of the kind in
        ``SUPPORTED_API_FORMATS``. Override to return a format subclass whose
        hooks match this service, or to vary the format by model.
        """
        candidates = [self.api_format] if self.api_format is not None else []
        candidates.extend(self.SUPPORTED_API_FORMATS)
        for name in candidates:
            if isinstance(api_format := _API_FORMATS[name], format_kind):
                return api_format
        raise UnsupportedApiFormatError(f"Provider {self.name!r} supports no {format_kind.kind} API format")

    async def _post(self, api_format: ApiFormat, request: HttpRequest) -> httpx2.Response:
        """Send one request; raise ``APIStatusError`` for an error status."""
        response = await self._send(self._client(), api_format, request, stream=False)
        if response.is_error:
            raise APIStatusError(response.status_code, response.text, headers=response.headers)
        return response

    @asynccontextmanager
    async def _stream(self, api_format: ApiFormat, request: HttpRequest) -> AsyncGenerator[HttpStream]:
        """Open one streaming request; raise ``APIStatusError`` for an error status."""
        response = await self._send(self._client(), api_format, request, stream=True)
        try:
            if response.is_error:
                await response.aread()
                raise APIStatusError(response.status_code, response.text, headers=response.headers)
            end_marker = "[DONE]" if api_format.name in {"chat", "responses"} else None
            yield HttpStream(_response_events(response, end_marker), dict(response.headers))
        except httpx2.RequestError as exc:
            raise _connection_error(exc, headers=response.headers) from exc
        finally:
            await response.aclose()

    def _client(self) -> httpx2.AsyncClient:
        self._ensure_open()
        self._check_client_loop()
        if self._http_client is None:
            # No await between checking and creating: concurrent tasks on this
            # event loop share the same client, including their first request.
            self._http_client = httpx2.AsyncClient(timeout=self.timeout)
            self._client_loop = asyncio.get_running_loop()
        return self._http_client

    async def _send(
        self, client: httpx2.AsyncClient, api_format: ApiFormat, request: HttpRequest, *, stream: bool
    ) -> httpx2.Response:
        content: dict[str, Any] = (
            {"data": request.body, "files": list(request.files)} if request.files else {"json": request.body}
        )
        built = client.build_request(
            "POST",
            f"{self.api_base}{request.path}",
            params=request.params,
            headers={**api_format.headers, **self.headers},
            **content,
        )
        auth = httpx2.USE_CLIENT_DEFAULT if self.auth is None else self.auth
        return await client.send(built, auth=auth, stream=stream)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Accept Retry-After seconds or an HTTP date, and the millisecond variant."""
    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        if (value := headers.get(name)) is None:
            continue
        try:
            delay = float(value) / divisor
        except ValueError:
            if name != "retry-after":
                continue
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(delay):
            return max(0, delay)
    return None


def _connection_error(exc: httpx2.RequestError, *, headers: Mapping[str, str] | None = None) -> APIConnectionError:
    error_type = APITimeoutError if isinstance(exc, httpx2.TimeoutException) else APIConnectionError
    return error_type(
        "Provider request timed out" if error_type is APITimeoutError else "Provider connection failed", headers=headers
    )


async def _response_events(response: httpx2.Response, end_marker: str | None) -> AsyncIterator[ServerSentEvent]:
    try:
        async for event in iter_events(response.aiter_lines(), end_marker=end_marker, yield_end_marker=True):
            yield event
    except httpx2.RequestError as exc:
        raise _connection_error(exc, headers=response.headers) from exc
