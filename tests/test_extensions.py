from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import AsyncExitStack, nullcontext
from typing import Any

import httpx2
import pytest

import republic
from republic.extensions import Extension, Request, ResponseValue
from tests.conftest import FakeService
from tests.test_http_reliability import sse


class Recording(Extension):
    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.retries: list[tuple[Request, Exception, int, float]] = []
        self.finishes: list[tuple[Request, ResponseValue | None, BaseException | None]] = []
        self.retry_started = asyncio.Event()

    def on_start(self, request: Request) -> None:
        self.requests.append(request)

    def on_retry(self, request: Request, error: Exception, attempt: int, delay: float) -> None:
        self.retries.append((request, error, attempt, delay))
        self.retry_started.set()

    def on_finish(self, request: Request, response: ResponseValue | None, error: BaseException | None) -> None:
        self.finishes.append((request, response, error))


def reply(text: str = "ok", *, streaming: bool = False, complete: bool = True) -> httpx2.Response:
    usage = {"prompt_tokens": 3, "completion_tokens": 2}
    if streaming:
        events: list[Any] = [{"choices": [{"delta": {"content": text}}], "usage": usage}]
        return sse([*events, "[DONE]"] if complete else events)
    return httpx2.Response(200, json={"choices": [{"message": {"content": text}}], "usage": usage})


@pytest.mark.parametrize("operation", ["chat", "embed", "embed_many", "generate", "edit", "decide", "list_models"])
async def test_one_extension_observes_public_operations(operation: str, service: FakeService) -> None:
    recording = Recording()
    async with service.client() as client:
        options: dict[str, Any] = {"http_client": client, "extensions": [recording]}
        if operation == "chat":
            service.reply_json({"choices": [{"message": {"content": "ok"}}]})
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
    [(request, observed, error)] = recording.finishes
    assert recording.requests == [request]
    assert (request.provider, request.operation, request.model) == (
        "typesafe" if operation == "decide" else "openai",
        "embed" if operation == "embed_many" else operation,
        None if operation == "list_models" else "test",
    )
    assert observed is result and error is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["validation", "history"])
async def test_postprocessing_failure_keeps_available_usage(streaming: bool, failure: str) -> None:
    recording = Recording()
    original = RuntimeError("history unavailable")

    class History(republic.history.InMemoryHistory):
        async def write(self, messages: Sequence[republic.Message]) -> None:
            raise original

    history = History() if failure == "history" else republic.history.InMemoryHistory()
    response = reply('"ok"' if failure == "history" else "ok", streaming=streaming)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, history=history, extensions=[recording]
        )
        with pytest.raises((ValueError, RuntimeError)) as caught:
            if streaming:
                async with model.stream("Hi", output_schema=str if failure == "history" else int) as stream:
                    [event async for event in stream]
            else:
                await model.chat("Hi", output_schema=str if failure == "history" else int)
    [(request, prepared, error)] = recording.finishes
    assert recording.requests == [request] and error is caught.value
    assert isinstance(prepared, republic.Response)
    assert prepared.token_usage.total_tokens == 5
    assert prepared.output == ("ok" if failure == "history" else None)
    assert await history.read() == []
    if failure == "history":
        assert error is original


@pytest.mark.parametrize("recover", [False, True])
async def test_retries_and_outcomes_share_the_same_request(recover: bool) -> None:
    recording = Recording()
    responses = [httpx2.Response(503), reply() if recover else httpx2.Response(401)]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, max_retry_delay=0, extensions=[recording]
        )
        if recover:
            result = await model.chat("Hi")
            assert recording.finishes == [(recording.requests[0], result, None)]
        else:
            with pytest.raises(republic.errors.APIStatusError) as caught:
                await model.chat("Hi")
            assert recording.finishes == [(recording.requests[0], None, caught.value)]
    [(request, error, attempt, delay)] = recording.retries
    assert request is recording.requests[0]
    assert isinstance(error, republic.errors.APIStatusError)
    assert (error.status_code, attempt, delay) == (503, 1, 0)


@pytest.mark.parametrize("mode", ["complete", "unread", "early", "consumer_error", "truncated"])
async def test_stream_outcomes_are_reported_once(mode: str) -> None:
    recording = Recording()
    response = reply(streaming=True, complete=mode != "truncated")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, extensions=[recording])
        stream = model.stream("Hi")
        assert recording.requests == []
        expected = (
            pytest.raises(RuntimeError, match="consumer failed")
            if mode in {"complete", "consumer_error"}
            else nullcontext()
        )
        with expected:
            async with stream:
                if mode == "consumer_error":
                    raise RuntimeError("consumer failed")
                if mode == "truncated":
                    with pytest.raises(republic.errors.StreamIncompleteError) as caught:
                        [event async for event in stream]
                    assert recording.finishes == [(recording.requests[0], None, caught.value)]
                elif mode != "unread":
                    async for event in stream:
                        if mode == "early":
                            break
                        if isinstance(event, republic.events.Completed):
                            assert recording.finishes == [(recording.requests[0], event.response, None)]
                    if mode == "complete":
                        raise RuntimeError("consumer failed")
    [(request, observed, error)] = recording.finishes
    assert recording.requests == [request]
    if mode == "complete":
        assert observed is stream.response and error is None
    elif mode != "truncated":
        assert observed is None and error is None


@pytest.mark.parametrize("phase", ["send", "stream", "backoff"])
async def test_cancellation_reaches_the_extension_and_caller(phase: str) -> None:
    recording, entered = Recording(), asyncio.Event()

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b""

    async def send(request: httpx2.Request) -> httpx2.Response:
        if phase == "send":
            entered.set()
            await asyncio.Event().wait()
        return (
            httpx2.Response(503, headers={"Retry-After": "60"})
            if phase == "backoff"
            else httpx2.Response(200, stream=Body())
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, extensions=[recording])

        async def run():
            if phase == "stream":
                async with model.stream("Hi") as stream:
                    [event async for event in stream]
            else:
                await model.chat("Hi")

        task = asyncio.create_task(run())
        await (recording.retry_started.wait() if phase == "backoff" else entered.wait())
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    assert recording.finishes == [(recording.requests[0], None, caught.value)]


@pytest.mark.parametrize("streaming", [False, True])
async def test_identical_overlapping_calls_have_distinct_request_keys(streaming: bool) -> None:
    recording = Recording()
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(request: httpx2.Request) -> httpx2.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if prompt == "first":
            entered.set()
            await release.wait()
        return reply(prompt, streaming=streaming)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, extensions=[recording])

        async def run(prompt):
            if streaming:
                async with model.stream(prompt) as stream:
                    [event async for event in stream]
                return stream.response
            return await model.chat(prompt)

        task = asyncio.create_task(run("first"))
        await entered.wait()
        try:
            second = await run("second")
        finally:
            release.set()
        first = await task
    results = {request: response for request, response, _ in recording.finishes}
    assert len(results) == 2
    assert results[recording.requests[0]] is first
    assert results[recording.requests[1]] is second


@pytest.mark.parametrize("method", ["on_start", "on_retry", "on_finish"])
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_extension_failure_does_not_change_public_outcome(method: str, failure: type[BaseException]) -> None:
    class Broken(Extension):
        def on_start(self, request):
            if method == "on_start":
                raise failure("extension failed")

        def on_retry(self, request, error, attempt, delay):
            if method == "on_retry":
                raise failure("extension failed")

        def on_finish(self, request, response, error):
            if method == "on_finish":
                raise failure("extension failed")

    recording = Recording()
    responses = [httpx2.Response(503), httpx2.Response(401)]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        model = republic.get_model(
            "openai:test", http_client=client, max_retry_delay=0, extensions=[Broken(), Extension(), recording]
        )
        with pytest.raises(republic.errors.APIStatusError) as caught:
            await model.chat("Hi")
    assert recording.finishes == [(recording.requests[0], None, caught.value)]


async def test_stream_can_be_used_across_tasks(service: FakeService) -> None:
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
    assert recording.finishes == [(recording.requests[0], stream.response, None)]
