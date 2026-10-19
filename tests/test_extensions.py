from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from typing import Any, cast

import httpx2
import pytest

import republic
from republic.extensions import Extension, Observer, Request, ResponseValue
from tests.conftest import FakeService
from tests.test_http_reliability import sse


class RecordingCall(Observer):
    def __init__(self, request: Request) -> None:
        self.request = request
        self.retries: list[tuple[Exception, int, float]] = []
        self.finishes: list[tuple[ResponseValue | None, BaseException | None]] = []
        self.retry_started = asyncio.Event()

    def on_retry(self, error: Exception, attempt: int, delay: float) -> None:
        self.retries.append((error, attempt, delay))
        self.retry_started.set()

    def on_finish(self, response: ResponseValue | None, error: BaseException | None) -> None:
        self.finishes.append((response, error))


class Recording(Extension):
    def __init__(self, operation: str | None = None) -> None:
        self.operation = operation
        self.calls: list[RecordingCall] = []
        self.started = asyncio.Event()

    def on_start(self, request: Request) -> Observer | None:
        if self.operation is not None and request.operation != self.operation:
            return None
        call = RecordingCall(request)
        self.calls.append(call)
        self.started.set()
        return call


def reply(text: str = "ok") -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def stream_reply(text: str = "ok", *, complete: bool = True) -> httpx2.Response:
    events: list[Any] = [
        {"choices": [{"delta": {"content": text}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    ]
    return sse([*events, "[DONE]"] if complete else events)


@pytest.mark.parametrize("operation", ["chat", "embed", "embed_many", "generate", "edit", "decide", "list_models"])
async def test_observes_public_results(operation: str, service: FakeService) -> None:
    recording = Recording()
    async with service.client() as client:
        options: dict[str, Any] = {"http_client": client, "extensions": [recording]}
        if operation == "chat":
            service.reply_json(reply())
            result = await republic.get_model("openai:test", api_format="chat", **options).chat("Hi")
        elif operation in {"embed", "embed_many"}:
            service.reply_json({"data": [{"index": 0, "embedding": [0.1]}]})
            model = republic.get_embedding_model("openai:test", **options)
            result = await model.embed("Hi") if operation == "embed" else await model.embed_many(["Hi"])
        elif operation in {"generate", "edit"}:
            service.reply_json({"data": [{"b64_json": "cG5n"}]})
            model = republic.get_image_model("openai:test", **options)
            result = (
                await model.generate("Hi")
                if operation == "generate"
                else await model.edit("Hi", republic.image(b"png", media_type="image/png"))
            )
        elif operation == "decide":
            service.reply_json({"answers": {}})
            result = await republic.get_decision_model("typesafe:test", **options).decide("Hi", questions={})
        else:
            service.reply_json({"data": []})
            result = await republic.get_provider("openai", **options).list_models()
    assert len(recording.calls) == 1
    call = recording.calls[0]
    assert call.request == Request(
        "typesafe" if operation == "decide" else "openai",
        "embed" if operation == "embed_many" else operation,
        None if operation == "list_models" else "test",
    )
    assert len(call.finishes) == 1
    assert call.finishes[0][0] is result
    assert call.finishes[0][1] is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["validation", "history"])
async def test_postprocessing_error_keeps_response_and_usage(streaming: bool, failure: str) -> None:
    recording = Recording()
    history_error = RuntimeError("history unavailable")

    class History(republic.history.InMemoryHistory):
        async def write(self, messages: Sequence[republic.Message]) -> None:
            raise history_error

    history = History() if failure == "history" else republic.history.InMemoryHistory()
    text = '"ok"' if failure == "history" else "ok"
    response = stream_reply(text) if streaming else httpx2.Response(200, json=reply(text))
    schema = str if failure == "history" else int
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, history=history, extensions=[recording]
        )
        with pytest.raises((ValueError, RuntimeError)) as caught:
            if streaming:
                async with model.stream("Hi", output_schema=schema) as stream:
                    [event async for event in stream]
            else:
                await model.chat("Hi", output_schema=schema)
    assert len(recording.calls[0].finishes) == 1
    prepared, error = recording.calls[0].finishes[0]
    assert isinstance(prepared, republic.Response)
    assert prepared.text == text
    assert prepared.token_usage.total_tokens == 5
    assert prepared.output == ("ok" if failure == "history" else None)
    assert error is caught.value
    if failure == "history":
        assert error is history_error
    assert await history.read() == []


async def test_input_error_is_observed_without_a_response(service: FakeService) -> None:
    recording = Recording()
    async with service.client() as client:
        model = republic.get_image_model("openai:test", http_client=client, extensions=[recording])
        with pytest.raises(ValueError, match="at least one image") as caught:
            await model.edit("Hi", [])
    assert recording.calls[0].finishes == [(None, caught.value)]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_retry_diagnostics_and_final_outcome(recover: bool, streaming: bool, service: FakeService) -> None:
    recording = Recording()
    service.reply_json({}, status_code=503)
    if recover and streaming:
        service.reply_events(["[DONE]"])
    else:
        service.reply_json(reply(), status_code=200 if recover else 503)
    async with service.client() as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            max_retries=1,
            max_retry_delay=0,
            extensions=[recording],
        )

        async def run():
            if streaming:
                async with model.stream("Hi") as stream:
                    [event async for event in stream]
                return stream.response
            return await model.chat("Hi")

        if recover:
            result = await run()
            assert recording.calls[0].finishes == [(result, None)]
        else:
            with pytest.raises(republic.errors.APIStatusError) as caught:
                await run()
            assert recording.calls[0].finishes == [(None, caught.value)]
    [(error, attempt, delay)] = recording.calls[0].retries
    assert isinstance(error, republic.errors.APIStatusError)
    assert (error.status_code, attempt, delay) == (503, 1, 0)


async def test_model_listing_reports_page_retries_in_one_call(service: FakeService) -> None:
    recording = Recording()
    service.reply_json({}, status_code=503)
    service.reply_json({"data": [{"id": "a"}], "has_more": True, "last_id": "a"})
    service.reply_json({}, status_code=503)
    service.reply_json({"data": [{"id": "b"}], "has_more": False})
    async with service.client() as client:
        provider = republic.get_provider("anthropic", http_client=client, max_retry_delay=0, extensions=[recording])
        result = await provider.list_models()
    assert [model.id for model in result] == ["a", "b"]
    assert len(recording.calls) == 1
    assert [attempt for _, attempt, _ in recording.calls[0].retries] == [1, 1]
    assert recording.calls[0].finishes == [(result, None)]


async def test_stream_reports_completion_before_context_exit() -> None:
    recording = Recording()
    history = republic.history.InMemoryHistory()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: stream_reply())) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, history=history, extensions=[recording]
        )
        with pytest.raises(RuntimeError, match="consumer failed"):
            async with model.stream("Hi") as stream:
                async for event in stream:
                    if isinstance(event, republic.events.Completed):
                        assert recording.calls[0].finishes == [(event.response, None)]
                        assert event.response is stream.response
                        assert len(await history.read()) == 2
                raise RuntimeError("consumer failed")
    assert recording.calls[0].finishes == [(stream.response, None)]


@pytest.mark.parametrize("mode", ["unread", "early", "consumer_error"])
async def test_unfinished_stream_has_no_response_or_provider_error(mode: str) -> None:
    recording = Recording()
    history = republic.history.InMemoryHistory()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: stream_reply())) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, history=history, extensions=[recording]
        )
        stream = model.stream("Hi")
        assert recording.calls == []
        expected = pytest.raises(RuntimeError, match="consumer failed") if mode == "consumer_error" else nullcontext()
        with expected:
            async with stream:
                if mode == "consumer_error":
                    raise RuntimeError("consumer failed")
                if mode == "early":
                    async for _ in stream:
                        break
    assert recording.calls[0].finishes == [(None, None)]
    assert await history.read() == []
    with pytest.raises(republic.errors.StreamNotFinishedError):
        _ = stream.response


async def test_stream_failure_is_visible_when_caught_inside_context() -> None:
    recording = Recording()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: stream_reply(complete=False))
    ) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, extensions=[recording])
        async with model.stream("Hi") as stream:
            with pytest.raises(republic.errors.StreamIncompleteError) as caught:
                [event async for event in stream]
            assert recording.calls[0].finishes == [(None, caught.value)]
    assert recording.calls[0].finishes == [(None, caught.value)]


@pytest.mark.parametrize("phase", ["send", "stream", "backoff"])
async def test_cancellation_is_observed_and_propagates(phase: str) -> None:
    recording = Recording()
    entered = asyncio.Event()

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b""

    async def send(request: httpx2.Request) -> httpx2.Response:
        if phase == "send":
            entered.set()
            await asyncio.Event().wait()
        if phase == "backoff":
            return httpx2.Response(503, headers={"Retry-After": "60"})
        return httpx2.Response(200, stream=Body(), headers={"content-type": "text/event-stream"})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, extensions=[recording])

        async def run():
            if phase == "stream":
                async with model.stream("Hi") as stream:
                    [event async for event in stream]
            else:
                await model.chat("Hi")

        task = asyncio.create_task(run())
        await recording.started.wait()
        await (recording.calls[0].retry_started.wait() if phase == "backoff" else entered.wait())
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    assert recording.calls[0].finishes == [(None, caught.value)]


async def test_extension_is_reusable_across_overlapping_calls() -> None:
    recording = Recording()
    entered, release = asyncio.Event(), asyncio.Event()
    first_attempt = True

    async def send(request: httpx2.Request) -> httpx2.Response:
        nonlocal first_attempt
        name = json.loads(request.content)["model"]
        if name == "first" and first_attempt:
            first_attempt = False
            entered.set()
            await release.wait()
            return httpx2.Response(503)
        return httpx2.Response(200, json=reply(name))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        provider = republic.get_provider(
            "openai", api_format="chat", http_client=client, max_retry_delay=0, extensions=[recording]
        )
        task = asyncio.create_task(provider.get_model("first").chat("Hi"))
        await entered.wait()
        try:
            second = await provider.get_model("second").chat("Hi")
        finally:
            release.set()
        first = await task
    calls = {call.request.model: call for call in recording.calls}
    assert calls["first"].finishes == [(first, None)]
    assert calls["second"].finishes == [(second, None)]
    assert len(calls["first"].retries) == 1
    assert calls["second"].retries == []


async def test_nested_calls_keep_their_own_results(service: FakeService) -> None:
    recording = Recording()
    service.reply_json(reply("inner"))
    service.reply_json(reply("outer"))
    async with service.client() as client:
        provider = republic.get_provider("openai", api_format="chat", http_client=client, extensions=[recording])
        inner = provider.get_model("inner")
        inner_results = []

        class History(republic.history.InMemoryHistory):
            async def read(self) -> list[republic.Message]:
                inner_results.append(await inner.chat("Hi"))
                return []

        result = await provider.get_model("outer", history=History()).chat("Hi")
    calls = {call.request.model: call for call in recording.calls}
    assert calls["outer"].finishes == [(result, None)]
    assert calls["inner"].finishes == [(inner_results[0], None)]


async def test_stream_observation_survives_task_changes(service: FakeService) -> None:
    recording = Recording()
    service.reply_events(["[DONE]"])
    async with service.client() as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, extensions=[recording])
        stack = AsyncExitStack()
        stream = await asyncio.create_task(stack.enter_async_context(model.stream("Hi")))

        async def consume():
            return [event async for event in stream]

        await asyncio.create_task(consume())
        await asyncio.create_task(stack.aclose())
    assert recording.calls[0].finishes == [(stream.response, None)]


class Broken(Extension, Observer):
    def __init__(self, phase: str, error: BaseException) -> None:
        self.phase, self.error = phase, error

    def on_start(self, request: Request) -> Observer:
        if self.phase == "start":
            raise self.error
        return self

    def on_retry(self, error: Exception, attempt: int, delay: float) -> None:
        if self.phase == "retry":
            raise self.error

    def on_finish(self, response: ResponseValue | None, error: BaseException | None) -> None:
        if self.phase == "finish":
            raise self.error


@pytest.mark.parametrize("phase", ["start", "retry", "finish"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_observer_failure_does_not_change_result(
    phase: str, error_type: type[BaseException], service: FakeService
) -> None:
    first, last = Recording(), Recording()
    service.reply_json({}, status_code=503)
    service.reply_json(reply())
    history = republic.history.InMemoryHistory()
    async with service.client() as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            max_retry_delay=0,
            history=history,
            extensions=[first, Broken(phase, error_type("observer failed")), last],
        )
        result = await model.chat("Hi")
    assert first.calls[0].finishes == last.calls[0].finishes == [(result, None)]
    assert len(await history.read()) == 2


async def test_observer_failure_does_not_replace_public_error(service: FakeService) -> None:
    recording = Recording()
    service.reply_json({}, status_code=401)
    async with service.client() as client:
        model = republic.get_model(
            "openai:test", http_client=client, extensions=[Broken("finish", RuntimeError("observer failed")), recording]
        )
        with pytest.raises(republic.errors.APIStatusError) as caught:
            await model.chat("Hi")
    assert recording.calls[0].finishes == [(None, caught.value)]


async def test_async_observer_does_not_run_or_change_result(
    service: FakeService, recwarn: pytest.WarningsRecorder
) -> None:
    broken, recording = Extension(), Recording()

    async def asynchronous(request: Request) -> Observer:
        pytest.fail("async observers are unsupported")

    broken.on_start = cast(Any, asynchronous)
    service.reply_json(reply())
    async with service.client() as client:
        result = await republic.get_model(
            "openai:test", api_format="chat", http_client=client, extensions=[broken, recording]
        ).chat("Hi")
    assert recording.calls[0].finishes == [(result, None)]
    assert not recwarn


async def test_extensions_can_select_calls_and_use_default_methods(service: FakeService) -> None:
    recording = Recording("embed")
    service.reply_json(reply())
    service.reply_json({"data": [{"index": 0, "embedding": [0.1]}]})
    async with service.client() as client:
        provider = republic.get_provider(
            "openai", api_format="chat", http_client=client, extensions=[Extension(), recording]
        )
        assert (await provider.get_model("test").chat("Hi")).text == "ok"
        result = await provider.get_embedding_model("test").embed("Hi")
    assert len(recording.calls) == 1
    assert recording.calls[0].finishes == [(result, None)]


async def test_existing_provider_subclass_can_observe_calls(service: FakeService) -> None:
    class CustomProvider(republic.providers.OpenAI):
        async def _post(self, api_format, request):
            return await super()._post(api_format, request)

        @asynccontextmanager
        async def _stream(self, api_format, request):
            async with super()._stream(api_format, request) as stream:
                yield stream

    recording = Recording()
    service.reply_json(reply())
    service.reply_events(["[DONE]"])
    async with service.client() as client:
        model = CustomProvider(api_format="chat", http_client=client, extensions=[recording]).get_model("test")
        result = await model.chat("Hi")
        async with model.stream("Hi") as stream:
            [event async for event in stream]
    assert [call.finishes for call in recording.calls] == [[(result, None)], [(stream.response, None)]]
