from __future__ import annotations

from collections.abc import AsyncIterator, Generator, Mapping, Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import replace
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Self, Unpack, overload

import pydantic

from republic._content import Input, Message, ToolCall, to_messages
from republic._options import ChatOptions
from republic._response import EmbeddingResponse, OutputT, Response, TokenUsage
from republic.decisions import DecisionResponse, JSONValue, Question
from republic.errors import APIResponseError, StreamNotFinishedError, request_id
from republic.events import Completed, Event
from republic.formats._base import (
    ChatApiFormat,
    ChatRequest,
    DecisionApiFormat,
    EmbeddingApiFormat,
    OutputSchema,
    ResponseBuilder,
    deep_merge,
    normalize,
)
from republic.formats._sse import ServerSentEvent

if TYPE_CHECKING:
    from republic.history import HistoryProtocol
    from republic.providers import Provider


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
        new_messages = to_messages(prompt)
        adapter = _adapter(output_schema)
        request = await self._request(new_messages, ChatOptions(**options), adapter)
        response = await self.provider._post(self.api_format, self.api_format.chat_request(request, stream=False))
        with _response_errors(response.headers):
            builder = ResponseBuilder()
            for delta in self.api_format.parse_chat(response.json()):
                builder.add(delta)
            return await self._finish(new_messages, builder, adapter, headers=response.headers)

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
        self._exit_stack = AsyncExitStack()
        self._events: AsyncIterator[ServerSentEvent] | None = None
        self._iterated = False
        self._response: Response[OutputT] | None = None
        self.headers: Mapping[str, str] = {}
        self.request_id: str | None = None

    async def __aenter__(self) -> Self:
        request = await self._model._request(self._new_messages, self._options, self._adapter)
        api_format = self._model.api_format
        response = await self._exit_stack.enter_async_context(
            self._model.provider._stream(api_format, api_format.chat_request(request, stream=True))
        )
        self._events = response.events
        self.headers = response.headers
        self.request_id = request_id(self.headers)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._exit_stack.aclose()

    async def __aiter__(self) -> AsyncIterator[Event]:
        if self._events is None:
            raise RuntimeError("Enter the stream with 'async with' before iterating it")
        if self._iterated:
            raise RuntimeError("A stream can only be iterated once; read stream.response instead")
        self._iterated = True
        parser = self._model.api_format.stream_parser()
        builder = ResponseBuilder()
        with _response_errors(self.headers):
            async for event in self._events:
                for delta in parser.feed(event.event, event.data):
                    for public_event in builder.add(delta):
                        yield public_event
            parser.finish()
            for public_event in builder.release_tool_calls():
                yield public_event
            self._response = await self._model._finish(self._new_messages, builder, self._adapter, headers=self.headers)
        yield Completed(self._response)

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

    async def embed(self, text: str, *, dimensions: int | None = None) -> EmbeddingResponse:
        """Embed one text; the vector is ``response.vector``."""
        return await self.embed_many([text], dimensions=dimensions)

    async def embed_many(self, texts: Sequence[str], *, dimensions: int | None = None) -> EmbeddingResponse:
        """Embed several texts in one request; ``response.vectors`` follows input order.

        ``dimensions`` shortens vectors on models that support it.
        """
        request = self.api_format.embedding_request(self.name, texts, dimensions=dimensions)
        response = await self.provider._post(self.api_format, request)
        with _response_errors(response.headers):
            return replace(
                self.api_format.parse_embedding(response.json()),
                headers=dict(response.headers),
                request_id=request_id(response.headers),
            )


class DecisionModel:
    """Answer typed questions about a piece of state, without generating text."""

    def __init__(self, provider: Provider, name: str, api_format: DecisionApiFormat) -> None:
        self.provider = provider
        self.name = name
        self.api_format = api_format

    async def decide(self, state: JSONValue, *, questions: Mapping[str, Question]) -> DecisionResponse:
        """Answer every question about ``state``. Answers come back under the same ids."""
        request = self.api_format.decision_request(self.name, state, questions)
        response = await self.provider._post(self.api_format, request)
        with _response_errors(response.headers):
            return replace(
                self.api_format.parse_decision(response.json()),
                headers=dict(response.headers),
                request_id=request_id(response.headers),
            )


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
