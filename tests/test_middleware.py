from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, nullcontext
from dataclasses import replace
from typing import Any

import httpx2
import pytest

import republic
from republic.events import CallEvent, Completed, Retry, TextDelta
from republic.providers import Call
from tests.conftest import FakeService
from tests.test_codex import reply as codex_reply
from tests.test_http_reliability import sse


class Recording:
    """Pass every event on and record the call, its events and how it ended."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.events: list[CallEvent] = []
        self.outcomes: list[tuple[Call, Any, BaseException | str | None]] = []

    async def __call__(self, call: Call) -> AsyncGenerator[CallEvent, None]:
        await asyncio.sleep(0)
        self.calls.append(call)
        response = None
        try:
            async for event in call.events:
                await asyncio.sleep(0)
                self.events.append(event)
                if isinstance(event, Completed):
                    response = event.response
                yield event
        except GeneratorExit:
            self.outcomes.append((call, response, "closed"))
            raise
        except BaseException as error:
            self.outcomes.append((call, response, error))
            raise
        self.outcomes.append((call, response, None))

    @property
    def retries(self) -> list[Retry]:
        return [event for event in self.events if isinstance(event, Retry)]


def reply(text: str = "ok", *, streaming: bool = False, complete: bool = True) -> httpx2.Response:
    usage = {"prompt_tokens": 3, "completion_tokens": 2}
    if streaming:
        events: list[Any] = [{"choices": [{"delta": {"content": text}}], "usage": usage}]
        return sse([*events, "[DONE]"] if complete else events)
    return httpx2.Response(200, json={"choices": [{"message": {"content": text}}], "usage": usage})


async def run(model: republic.ChatModel, prompt: str = "Hi", *, streaming: bool, **options: Any) -> Any:
    if streaming:
        async with model.stream(prompt, **options) as stream:
            [event async for event in stream]
        return stream.response
    return await model.chat(prompt, **options)


@pytest.mark.parametrize("operation", ["chat", "embed", "embed_many", "generate", "edit", "decide", "list_models"])
async def test_one_middleware_wraps_public_operations(operation: str, service: FakeService) -> None:
    recording = Recording()
    async with service.client() as client:
        options: dict[str, Any] = {"http_client": client, "middlewares": [recording]}
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
    assert [(call.provider, call.operation, call.model) for call in recording.calls] == [
        (
            "typesafe" if operation == "decide" else "openai",
            "embed" if operation == "embed_many" else operation,
            None if operation == "list_models" else "test",
        )
    ]
    assert recording.events == [Completed(result)]
    [(_, observed, error)] = recording.outcomes
    assert observed is result and error is None


@pytest.mark.parametrize("operation", ["chat", "stream"])
@pytest.mark.parametrize("outcome", ["success", "retry", "http_error", "validation_error"])
async def test_codex_wraps_the_public_operation(operation: str, outcome: str, service: FakeService) -> None:
    recording = Recording()
    if outcome == "retry":
        service.reply_json({}, status_code=503)
    if outcome == "http_error":
        service.reply_json({}, status_code=401)
    else:
        codex_reply(service, "ok")
    expected = (
        pytest.raises(republic.errors.APIStatusError)
        if outcome == "http_error"
        else pytest.raises(ValueError)
        if outcome == "validation_error"
        else nullcontext()
    )
    async with service.client() as client:
        model = republic.get_model(
            "codex:test", api_key="test-key", http_client=client, max_retry_delay=0, middlewares=[recording]
        )
        with expected as caught:
            schema = int if outcome == "validation_error" else None
            result = await run(model, streaming=operation == "stream", output_schema=schema)
    assert [(call.provider, call.operation, call.model) for call in recording.calls] == [("codex", operation, "test")]
    assert [retry.attempt for retry in recording.retries] == ([1] if outcome == "retry" else [])
    [(_, observed, error)] = recording.outcomes
    if caught is not None:
        assert error is caught.value and observed is None
    else:
        assert observed is result and error is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_retries_reach_middleware_as_events(streaming: bool, recover: bool) -> None:
    recording = Recording()
    final = reply(streaming=streaming) if recover else httpx2.Response(401)
    responses = [httpx2.Response(503), final]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, max_retry_delay=0, middlewares=[recording]
        )
        if recover:
            consumed: list[Any] = []
            if streaming:
                async with model.stream("Hi") as stream:
                    consumed = [event async for event in stream]
                result = stream.response
            else:
                result = await model.chat("Hi")
            assert recording.outcomes == [(recording.calls[0], result, None)]
            assert not any(isinstance(event, Retry) for event in consumed)
        else:
            with pytest.raises(republic.errors.APIStatusError) as caught:
                await run(model, streaming=streaming)
            assert recording.outcomes == [(recording.calls[0], None, caught.value)]
    [retry] = recording.retries
    assert recording.events[0] is retry
    assert isinstance(retry.error, republic.errors.APIStatusError)
    assert (retry.error.status_code, retry.attempt, retry.delay) == (503, 1, 0)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_middlewares_wrap_in_order(streaming: bool, fail: bool) -> None:
    trace: list[str] = []

    def named(name: str):
        async def middleware(call: Call) -> AsyncGenerator[CallEvent, None]:
            trace.append(f"{name} start")
            try:
                async for event in call.events:
                    trace.append(f"{name} {type(event).__name__}")
                    yield event
            except republic.errors.APIStatusError:
                trace.append(f"{name} error")
                raise
            trace.append(f"{name} end")

        return middleware

    response = httpx2.Response(401) if fail else reply(streaming=streaming)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, middlewares=[named("outer"), named("inner")]
        )
        with pytest.raises(republic.errors.APIStatusError) if fail else nullcontext():
            await run(model, streaming=streaming)
    if fail:
        middle = ["inner error", "outer error"]
    else:
        names = ["TextDelta", "UsageDelta", "Completed"] if streaming else ["Completed"]
        middle = [line for name in names for line in (f"inner {name}", f"outer {name}")]
        middle += ["inner end", "outer end"]
    assert trace == ["outer start", "inner start", *middle]


async def test_middleware_rewrites_and_drops_stream_events(service: FakeService) -> None:
    async def shout(call: Call) -> AsyncGenerator[CallEvent, None]:
        async for event in call.events:
            if isinstance(event, TextDelta):
                yield TextDelta(event.chunk.upper())
            elif not isinstance(event, republic.events.UsageDelta):
                yield event

    service.reply_events([{"choices": [{"delta": {"content": "hi"}}], "usage": {"prompt_tokens": 1}}, "[DONE]"])
    async with service.client() as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[shout])
        async with model.stream("Hi") as stream:
            events = [event async for event in stream]
    assert events == [TextDelta("HI"), Completed(stream.response)]
    assert stream.text == "hi"


@pytest.mark.parametrize("streaming", [False, True])
async def test_middleware_replaces_the_response(streaming: bool) -> None:
    async def rewrite(call: Call) -> AsyncGenerator[CallEvent, None]:
        async for event in call.events:
            if isinstance(event, Completed):
                yield Completed(replace(event.response, message=republic.assistant("rewritten")))
            else:
                yield event

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply(streaming=streaming))) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[rewrite])
        result = await run(model, streaming=streaming)
    assert result.text == "rewritten"


@pytest.mark.parametrize("streaming", [False, True])
async def test_middleware_answers_without_sending(streaming: bool) -> None:
    cached = republic.Response(republic.assistant("cached"))

    async def cache(call: Call) -> AsyncGenerator[CallEvent, None]:
        yield TextDelta("cached")
        yield Completed(cached)

    def send(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("A cached call sends no request")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[cache])
        assert await run(model, streaming=streaming) is cached


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("phase", ["start", "end"])
async def test_middleware_failure_fails_the_call(streaming: bool, phase: str) -> None:
    failure = RuntimeError("middleware failed")
    requests: list[httpx2.Request] = []

    async def broken(call: Call) -> AsyncGenerator[CallEvent, None]:
        if phase == "start":
            raise failure
        async for event in call.events:
            yield event
        raise failure

    def send(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return reply(streaming=streaming)

    recording = Recording()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, middlewares=[recording, broken]
        )
        with pytest.raises(RuntimeError) as caught:
            await run(model, streaming=streaming)
    assert caught.value is failure
    assert len(requests) == (0 if phase == "start" else 1)
    [(_, _, error)] = recording.outcomes
    assert error is failure


async def test_call_needs_a_completed_event() -> None:
    async def swallow(call: Call) -> AsyncGenerator[CallEvent, None]:
        try:
            async for event in call.events:
                yield event
        except republic.errors.APIStatusError:
            return

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(401))) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[swallow])
        with pytest.raises(RuntimeError, match="without a Completed event"):
            await model.chat("Hi")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["validation", "history"])
async def test_postprocessing_failure_reaches_middleware(streaming: bool, failure: str) -> None:
    recording = Recording()
    original = RuntimeError("history unavailable")

    class History(republic.history.InMemoryHistory):
        async def write(self, messages: Sequence[republic.Message]) -> None:
            raise original

    history = History() if failure == "history" else republic.history.InMemoryHistory()
    response = reply('"ok"' if failure == "history" else "ok", streaming=streaming)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, history=history, middlewares=[recording]
        )
        with pytest.raises((ValueError, RuntimeError)) as caught:
            await run(model, streaming=streaming, output_schema=str if failure == "history" else int)
    [(_, observed, error)] = recording.outcomes
    assert error is caught.value and observed is None
    assert await history.read() == []
    if failure == "history":
        assert error is original


@pytest.mark.parametrize("mode", ["complete", "unread", "early", "consumer_error", "truncated"])
async def test_stream_outcomes_are_reported_once(mode: str) -> None:
    recording = Recording()
    response = reply(streaming=True, complete=mode != "truncated")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[recording])
        stream = model.stream("Hi")
        assert recording.calls == []
        expected = (
            pytest.raises(RuntimeError, match="consumer failed")
            if mode in {"complete", "consumer_error"}
            else nullcontext()
        )
        with expected:
            async with stream:
                assert stream.request_id == "req_1"
                if mode == "consumer_error":
                    raise RuntimeError("consumer failed")
                if mode == "truncated":
                    with pytest.raises(republic.errors.StreamIncompleteError) as caught:
                        [event async for event in stream]
                    assert recording.outcomes == [(recording.calls[0], None, caught.value)]
                elif mode != "unread":
                    async for event in stream:
                        if mode == "early":
                            break
                        if isinstance(event, Completed):
                            assert recording.outcomes == []
                    if mode == "complete":
                        assert recording.outcomes == [(recording.calls[0], stream.response, None)]
                        raise RuntimeError("consumer failed")
    [(call, observed, outcome)] = recording.outcomes
    assert call is recording.calls[0]
    if mode == "complete":
        assert observed is stream.response and outcome is None
    elif mode != "truncated":
        assert observed is None and outcome == "closed"


@pytest.mark.parametrize("phase", ["send", "stream", "backoff"])
async def test_cancellation_reaches_the_middleware_and_caller(phase: str) -> None:
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
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[recording])
        task = asyncio.create_task(run(model, streaming=phase == "stream"))
        if phase == "backoff":
            while not recording.retries:
                await asyncio.sleep(0)
        else:
            await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    assert recording.outcomes == [(recording.calls[0], None, caught.value)]


@pytest.mark.parametrize("streaming", [False, True])
async def test_overlapping_calls_keep_their_own_events(streaming: bool) -> None:
    recording = Recording()
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(request: httpx2.Request) -> httpx2.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if prompt == "first":
            entered.set()
            await release.wait()
        return reply(prompt, streaming=streaming)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(send)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[recording])
        task = asyncio.create_task(run(model, "first", streaming=streaming))
        await entered.wait()
        try:
            second = await run(model, "second", streaming=streaming)
        finally:
            release.set()
        first = await task
    assert sorted((response.text, error) for _, response, error in recording.outcomes) == [
        ("first", None),
        ("second", None),
    ]
    assert {id(response) for _, response, _ in recording.outcomes} == {id(first), id(second)}


async def test_stream_can_be_used_across_tasks(service: FakeService) -> None:
    recording = Recording()
    service.reply_events(["[DONE]"])
    async with service.client() as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, middlewares=[recording])
        stack = AsyncExitStack()
        stream = await asyncio.create_task(stack.enter_async_context(model.stream("Hi")))

        async def consume():
            return [event async for event in stream]

        await asyncio.create_task(consume())
        await asyncio.create_task(stack.aclose())
    assert recording.outcomes == [(recording.calls[0], stream.response, None)]
