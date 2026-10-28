from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Mapping, Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import replace
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, Unpack, overload

import pydantic

from republic._content import Image, Input, Message, ToolCall, to_messages
from republic._options import ChatOptions, ImageOptions
from republic._response import EmbeddingResponse, ImageResponse, OutputT, Response, TokenUsage
from republic.decisions import DecisionResponse, JSONValue, Question
from republic.errors import APIResponseError, StreamNotFinishedError, request_id
from republic.events import CallEvent, Completed, Event, Retry
from republic.formats._base import (
    ChatApiFormat,
    ChatRequest,
    DecisionApiFormat,
    EmbeddingApiFormat,
    HttpRequest,
    ImageApiFormat,
    OutputSchema,
    ResponseBuilder,
    deep_merge,
    normalize,
)
from republic.providers.base import Call

if TYPE_CHECKING:
    from republic.formats._base import ApiFormat
    from republic.history import HistoryProtocol
    from republic.providers import Provider

_ResultT = TypeVar("_ResultT", bound=EmbeddingResponse | ImageResponse | DecisionResponse)


class ChatModel:
    def __init__(
        self,
        provider: Provider,
        name: str,
        api_format: ChatApiFormat,
        *,
        history: HistoryProtocol | None = None,
    ) -> None:
        self.provider = provider
        self.name = name
        self.api_format = api_format
        self.history = history

    async def __aenter__(self) -> Self:
        self.provider._open_context(self)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.provider._close_context(self)

    @overload
    async def chat(
        self, prompt: Input, *, output_schema: None = None, **options: Unpack[ChatOptions]
    ) -> Response[None]: ...

    @overload
    async def chat(
        self, prompt: Input, *, output_schema: type[OutputT], **options: Unpack[ChatOptions]
    ) -> Response[OutputT]: ...

    async def chat(
        self, prompt: Input, *, output_schema: type[Any] | None = None, **options: Unpack[ChatOptions]
    ) -> Response[Any]:
        """Send the conversation and wait for the complete response.

        ``output_schema`` is a Pydantic model or any type Pydantic can validate;
        the parsed result is ``response.output``. See :class:`ChatOptions` for
        the other options.
        """
        events = self._chat(prompt, output_schema, ChatOptions(**options))
        return await self.provider._middleware_stack.run(Call(self.provider.name, "chat", self.name, events))

    async def _chat(
        self, prompt: Input, output_schema: type[Any] | None, options: ChatOptions
    ) -> AsyncGenerator[CallEvent, None]:
        new_messages = to_messages(prompt)
        adapter = _adapter(output_schema)
        request = self.api_format.chat_request(await self._request(new_messages, options, adapter), stream=False)
        async for response in self.provider._middleware_stack.retry(
            lambda: self.provider._post(self.api_format, request)
        ):
            if isinstance(response, Retry):
                yield response
                continue
            with _response_errors(response.headers):
                builder = ResponseBuilder()
                for delta in self.api_format.parse_chat(response.json()):
                    builder.add(delta)
                result = await self._finish(new_messages, builder, adapter, headers=response.headers)
            yield Completed(result)

    @overload
    def stream(self, prompt: Input, *, output_schema: None = None, **options: Unpack[ChatOptions]) -> Stream[None]: ...

    @overload
    def stream(
        self, prompt: Input, *, output_schema: type[OutputT], **options: Unpack[ChatOptions]
    ) -> Stream[OutputT]: ...

    def stream(
        self, prompt: Input, *, output_schema: type[Any] | None = None, **options: Unpack[ChatOptions]
    ) -> Stream[Any]:
        """Stream the response. Use as ``async with model.stream(...) as stream``."""
        return Stream(self, to_messages(prompt), ChatOptions(**options), _adapter(output_schema))

    async def _request(
        self, new_messages: list[Message], options: ChatOptions, adapter: pydantic.TypeAdapter[Any] | None
    ) -> ChatRequest:
        if self.provider.extra_body:
            options = ChatOptions(**options)
            options["extra_body"] = deep_merge(self.provider.extra_body, options.get("extra_body", {}))
        past_messages = await self.history.read() if self.history is not None else []
        return ChatRequest(
            model=self.name,
            messages=normalize([*past_messages, *new_messages]),
            options=options,
            output_schema=None if adapter is None else _output_schema(adapter),
        )

    async def _finish(
        self,
        new_messages: list[Message],
        builder: ResponseBuilder,
        adapter: pydantic.TypeAdapter[Any] | None,
        *,
        headers: Mapping[str, str],
    ) -> Response[Any]:
        response = replace(builder.response(), headers=dict(headers), request_id=request_id(headers))
        if adapter is not None and response.refusal is None:
            response = replace(response, output=adapter.validate_json(response.text))
        if self.history is not None:
            await self.history.write([*new_messages, response.message])
        return response


class Stream(Generic[OutputT]):
    """An in-progress response, iterated for events.

    After the iteration ends, the full response is available as
    :attr:`response`, with shortcuts such as :attr:`text` and :attr:`output`.
    """

    def __init__(
        self,
        model: ChatModel,
        new_messages: list[Message],
        options: ChatOptions,
        adapter: pydantic.TypeAdapter[Any] | None,
    ) -> None:
        self._model = model
        self._new_messages = new_messages
        self._options = options
        self._adapter = adapter
        self._operation = "stream"
        self._events: AsyncGenerator[CallEvent, None] | None = None
        self._received: list[CallEvent] = []
        self._opened = False
        self._iterated = False
        self._response: Response[OutputT] | None = None
        self.headers: Mapping[str, str] = {}
        self.request_id: str | None = None

    async def __aenter__(self) -> Self:
        provider = self._model.provider
        self._events = provider._middleware_stack.wrap(
            Call(provider.name, self._operation, self._model.name, self._source())
        )
        try:
            # Open the HTTP stream now, so its headers are available once entered.
            while not self._opened:
                self._received.append(await anext(self._events))
        except StopAsyncIteration:
            pass
        except BaseException:
            await self._events.aclose()
            raise
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if self._events is not None:
            await self._events.aclose()

    async def __aiter__(self) -> AsyncIterator[Event]:
        if self._events is None:
            raise RuntimeError("Enter the stream with 'async with' before iterating it")
        if self._iterated:
            raise RuntimeError("A stream can only be iterated once; read stream.response instead")
        self._iterated = True
        received, self._received = self._received, []
        for event in received:
            if not isinstance(event, Retry):
                yield self._record(event)
        async for event in self._events:
            if not isinstance(event, Retry):
                yield self._record(event)

    def _record(self, event: Event) -> Event:
        if isinstance(event, Completed):
            self._response = event.response
        return event

    async def _source(self) -> AsyncGenerator[CallEvent, None]:
        model = self._model
        api_format = model.api_format
        request = api_format.chat_request(
            await model._request(self._new_messages, self._options, self._adapter), stream=True
        )
        async with AsyncExitStack() as http:
            async for item in model.provider._middleware_stack.retry(
                lambda: http.enter_async_context(model.provider._stream(api_format, request))
            ):
                if isinstance(item, Retry):
                    yield item
                    continue
                self.headers = item.headers
                self.request_id = request_id(self.headers)
                self._opened = True
                parser = api_format.stream_parser()
                builder = ResponseBuilder()
                with _response_errors(self.headers):
                    async for event in item.events:
                        for delta in parser.feed(event.event, event.data):
                            for public_event in builder.add(delta):
                                yield public_event
                    parser.finish()
                    for public_event in builder.release_tool_calls():
                        yield public_event
                    response = await model._finish(self._new_messages, builder, self._adapter, headers=self.headers)
                yield Completed(response)

    @property
    def response(self) -> Response[OutputT]:
        if self._response is None:
            raise StreamNotFinishedError("The stream has not been fully consumed")
        return self._response

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def reasoning(self) -> str:
        return self.response.reasoning

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.response.tool_calls

    @property
    def output(self) -> OutputT | None:
        return self.response.output

    @property
    def token_usage(self) -> TokenUsage:
        return self.response.token_usage


class EmbeddingModel:
    def __init__(self, provider: Provider, name: str, api_format: EmbeddingApiFormat) -> None:
        self.provider = provider
        self.name = name
        self.api_format = api_format

    async def __aenter__(self) -> Self:
        self.provider._open_context(self)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.provider._close_context(self)

    async def embed(self, text: str, *, dimensions: int | None = None) -> EmbeddingResponse:
        """Embed one text; the vector is ``response.vector``."""
        return await self.embed_many([text], dimensions=dimensions)

    async def embed_many(self, texts: Sequence[str], *, dimensions: int | None = None) -> EmbeddingResponse:
        """Embed several texts in one request; ``response.vectors`` follows input order.

        ``dimensions`` shortens vectors on models that support it.
        """
        return await _post(
            self.provider,
            "embed",
            self.name,
            self.api_format,
            lambda: self.api_format.embedding_request(self.name, texts, dimensions=dimensions),
            self.api_format.parse_embedding,
        )


class ImageModel:
    """Generate and edit images."""

    def __init__(self, provider: Provider, name: str, api_format: ImageApiFormat) -> None:
        self.provider = provider
        self.name = name
        self.api_format = api_format

    async def __aenter__(self) -> Self:
        self.provider._open_context(self)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.provider._close_context(self)

    async def generate(self, prompt: str, **options: Unpack[ImageOptions]) -> ImageResponse:
        """Generate images from ``prompt``. See :class:`ImageOptions` for the options."""
        return await _post(
            self.provider,
            "generate",
            self.name,
            self.api_format,
            lambda: self.api_format.generation_request(self.name, prompt, ImageOptions(**options)),
            self.api_format.parse_images,
        )

    async def edit(
        self,
        prompt: str,
        images: Image | Sequence[Image],
        *,
        mask: Image | None = None,
        **options: Unpack[ImageOptions],
    ) -> ImageResponse:
        """Edit or combine ``images`` as ``prompt`` describes.

        Build images with :func:`republic.image`. Transparent areas of ``mask``
        mark where the first image is edited, on formats that support masks.
        """
        images = [images] if isinstance(images, Image) else list(images)

        def request() -> HttpRequest:
            if not images:
                raise ValueError("Pass at least one image to edit")
            return self.api_format.edit_request(self.name, prompt, images, mask=mask, options=ImageOptions(**options))

        return await _post(self.provider, "edit", self.name, self.api_format, request, self.api_format.parse_images)


class DecisionModel:
    """Answer typed questions about a piece of state, without generating text."""

    def __init__(self, provider: Provider, name: str, api_format: DecisionApiFormat) -> None:
        self.provider = provider
        self.name = name
        self.api_format = api_format

    async def __aenter__(self) -> Self:
        self.provider._open_context(self)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.provider._close_context(self)

    async def decide(self, state: JSONValue, *, questions: Mapping[str, Question]) -> DecisionResponse:
        """Answer every question about ``state``. Answers come back under the same ids."""
        return await _post(
            self.provider,
            "decide",
            self.name,
            self.api_format,
            lambda: self.api_format.decision_request(self.name, state, questions),
            self.api_format.parse_decision,
        )


async def _post(
    provider: Provider,
    operation: str,
    model: str,
    api_format: ApiFormat,
    request: Callable[[], HttpRequest],
    parse: Callable[[Any], _ResultT],
) -> _ResultT:
    """Run a one-response call: retry the request, then parse the result with its diagnostics."""

    async def events() -> AsyncGenerator[CallEvent, None]:
        http_request = request()
        async for response in provider._middleware_stack.retry(lambda: provider._post(api_format, http_request)):
            if isinstance(response, Retry):
                yield response
                continue
            with _response_errors(response.headers):
                result = replace(
                    parse(response.json()), headers=dict(response.headers), request_id=request_id(response.headers)
                )
            yield Completed(result)

    return await provider._middleware_stack.run(Call(provider.name, operation, model, events()))


@contextmanager
def _response_errors(headers: Mapping[str, str]) -> Generator[None]:
    try:
        yield
    except APIResponseError as exc:
        exc._set_headers(headers)
        raise


def _adapter(output_schema: type[Any] | None) -> pydantic.TypeAdapter[Any] | None:
    return None if output_schema is None else pydantic.TypeAdapter(output_schema)


def _output_schema(adapter: pydantic.TypeAdapter[Any]) -> OutputSchema:
    schema = adapter.json_schema()
    return OutputSchema(name=schema.get("title", "output"), schema=schema)
