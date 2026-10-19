from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest

import republic
from republic.extensions import Call, Extension
from republic.providers import base
from tests.conftest import FakeService
from tests.test_http_reliability import CHAT_REPLY, HEADERS, sse


async def test_observes_public_result_with_per_call_state(service: FakeService) -> None:
    calls: list[Call] = []
    order: list[str] = []

    def start(call: Call) -> None:
        assert call.status == "pending"
        assert call.response is None
        call.state["span"] = object()
        calls.append(call)
        order.append("start")

    async def finish(call: Call) -> None:
        assert call.state["span"] is calls[0].state["span"]
        order.append("finish")

    service.reply_json({**CHAT_REPLY, "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    extension = Extension(on_start=start, on_finish=finish)
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client(), extensions=[extension])
    result = await model.chat("Hi")

    assert order == ["start", "finish"]
    (call,) = calls
    assert (call.provider, call.model, call.operation, call.api_format, call.streaming) == (
        "openai",
        "test",
        "chat",
        "chat",
        False,
    )
    assert (call.status, call.attempts, call.error) == ("success", 1, None)
    assert call.response is result
    assert result.token_usage.total_tokens == 5
    assert call.elapsed >= 0


@pytest.mark.parametrize("operation", ["embed", "generate", "edit", "decide", "list_models"])
async def test_other_operations_use_same_extensions(operation: str, service: FakeService) -> None:
    calls: list[Call] = []
    options = {"http_client": service.client(), "extensions": [Extension(on_finish=calls.append)]}
    if operation == "embed":
        service.reply_json({"data": [{"index": 0, "embedding": [0.1]}]})
        result = await republic.get_embedding_model("openai:test", **options).embed("Hi")
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
    (call,) = calls
    assert call.operation == operation
    assert call.response is result
    assert call.status == "success"


@pytest.mark.parametrize("failure", ["http", "json", "shape", "schema", "auth", "options"])
async def test_observes_original_error_and_available_metadata(failure: str, service: FakeService) -> None:
    calls: list[Call] = []
    options: dict[str, Any] = {"http_client": service.client(), "extensions": [Extension(on_finish=calls.append)]}
    arguments: dict[str, Any] = {}
    if failure == "http":
        service.reply_json({"error": "no"}, status_code=400)
    elif failure == "json":
        service.reply_bytes(b"not json", content_type="text/plain")
    elif failure == "shape":
        service.reply_json({})
    elif failure == "schema":
        service.reply_json(CHAT_REPLY)
        arguments["output_schema"] = int
    elif failure == "auth":

        class BrokenAuth(httpx2.Auth):
            def auth_flow(self, request: httpx2.Request):
                raise republic.errors.AuthenticationError("login unavailable")
                yield request

        options["auth"] = BrokenAuth()
    else:
        arguments["parallel_tool_calls"] = True
    model = republic.get_model(
        "google:test" if failure == "options" else "openai:test",
        api_format=None if failure == "options" else "chat",
        **options,
    )
    with pytest.raises(Exception) as caught:
        await model.chat("Hi", **arguments)
    (call,) = calls
    assert call.status == "error"
    assert call.error is caught.value
    assert (call.response is not None) == (failure == "schema")
    assert call.attempts == (0 if failure == "options" else 1)


async def test_stream_finishes_after_cleanup_with_full_response() -> None:
    calls: list[Call] = []
    reply = sse([
        {"choices": [{"delta": {"content": "ok"}}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        "[DONE]",
    ])

    def finish(call: Call) -> None:
        assert reply.is_closed
        calls.append(call)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, extensions=[Extension(on_finish=finish)]
        )
        async with model.stream("Hi") as stream:
            assert calls == []
            events = [event async for event in stream]
            assert calls == []
            assert isinstance(events[-1], republic.events.Completed)
        await stream.__aexit__(None, None, None)
    (call,) = calls
    assert call.response is stream.response
    assert call.streaming
    assert (call.status, call.request_id) == ("success", "req_1")
    assert stream.token_usage.total_tokens == 5


@pytest.mark.parametrize(
    "mode", ["unread", "early", "truncated", "caught", "cancelled", "cancelled_body", "consumer_error"]
)
async def test_stream_terminal_outcomes(mode: str) -> None:
    calls: list[Call] = []
    reply = sse(
        [{"choices": [{"delta": {"content": "partial"}}]}] + ([] if mode in {"truncated", "caught"} else ["[DONE]"])
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, extensions=[Extension(on_finish=calls.append)]
        )
        error = None
        try:
            async with model.stream("Hi") as stream:
                await consume_stream(stream, mode)
        except (Exception, asyncio.CancelledError) as exc:
            error = exc
    (call,) = calls
    expected = {"truncated": "error", "caught": "error", "cancelled": "cancelled", "cancelled_body": "cancelled"}.get(
        mode, "closed"
    )
    assert call.status == expected
    if mode == "caught":
        assert isinstance(call.error, republic.errors.StreamIncompleteError)
    elif expected in {"error", "cancelled"}:
        assert call.error is error
    else:
        assert call.error is None
    assert call.response is None
    assert reply.is_closed


async def consume_stream(stream: republic.Stream[Any], mode: str) -> None:
    if mode == "unread":
        return
    if mode == "consumer_error":
        raise ValueError("consumer failed")
    if mode == "cancelled_body":
        raise asyncio.CancelledError
    if mode == "caught":
        with pytest.raises(republic.errors.StreamIncompleteError):
            [event async for event in stream]
        return
    async for _event in stream:
        if mode == "early":
            break
        if mode == "cancelled":
            raise asyncio.CancelledError


async def test_failed_stream_entry_finishes_once(service: FakeService) -> None:
    calls: list[Call] = []
    service.reply_json({"error": "no"}, status_code=401)
    model = republic.get_model(
        "openai:test", http_client=service.client(), extensions=[Extension(on_finish=calls.append)]
    )
    stream = model.stream("Hi")
    with pytest.raises(republic.errors.APIStatusError) as caught:
        async with stream:
            pytest.fail("stream must fail to open")
    await stream.__aexit__(None, None, None)
    (call,) = calls
    assert call.error is caught.value
    assert call.status == "error"


async def test_concurrent_and_nested_calls_keep_state_and_response_local() -> None:
    started: list[Call] = []
    finished: list[Call] = []

    async def start(call: Call) -> None:
        call.state["marker"] = object()
        started.append(call)
        await asyncio.sleep(0)

    def handle(request: httpx2.Request) -> httpx2.Response:
        name = json.loads(request.content)["messages"][0]["content"]
        return (
            sse(["[DONE]"], headers={"x-request-id": name})
            if json.loads(request.content).get("stream")
            else httpx2.Response(200, json=CHAT_REPLY, headers={"x-request-id": name})
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            extensions=[Extension(on_start=start, on_finish=finished.append)],
        )
        await asyncio.gather(model.chat("a"), model.chat("b"))
        async with model.stream("c") as first, model.stream("d") as second:
            [event async for event in first]
            [event async for event in second]
    assert [call.request_id for call in finished] == ["a", "b", "d", "c"]
    assert len({id(call.state["marker"]) for call in started}) == 4
    assert set(started) == set(finished)
    assert all(call.status == "success" for call in finished)


async def test_retry_policy_and_observers_compose_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    responses = [httpx2.Response(429, headers=HEADERS), httpx2.Response(200, json=CHAT_REPLY)]
    failed = responses[0]
    seen: list[tuple[str, float | None]] = []
    calls: list[Call] = []

    def adjust(call: Call, error: Exception, delay: float | None) -> float:
        assert isinstance(error, republic.errors.APIStatusError)
        assert (call.attempts, error.status_code, delay) == (1, 429, 3)
        seen.append(("policy1", delay))
        return 2

    async def adjust_again(call: Call, error: Exception, delay: float | None) -> float:
        seen.append(("policy2", delay))
        return 0.25

    def retry(call: Call, error: Exception, delay: float) -> None:
        assert failed.is_closed
        seen.append(("retry", delay))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            extensions=[
                Extension(retry_delay=adjust),
                Extension(retry_delay=adjust_again, on_retry=retry, on_finish=calls.append),
            ],
        )
        await model.chat("Hi")
    assert seen == [("policy1", 3), ("policy2", 2), ("retry", 0.25)]
    sleep.assert_awaited_once_with(0.25)
    assert calls[0].attempts == 2
    assert calls[0].request_id is None


@pytest.mark.parametrize("mode", ["stop", "custom_status", "disabled", "cap"])
async def test_retry_customization_respects_limits(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(418 if mode == "custom_status" else 503)

    def policy(call: Call, error: Exception, delay: float | None) -> float | None:
        return None if mode == "stop" else 100

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model(
            "openai:test",
            http_client=client,
            extensions=[Extension(retry_delay=policy)],
            max_retries=0 if mode == "disabled" else 1,
            max_retry_delay=0.1,
        )
        with pytest.raises(republic.errors.APIStatusError):
            await model.chat("Hi")
    assert len(requests) == (1 if mode in {"stop", "disabled"} else 2)
    if len(requests) == 2:
        sleep.assert_awaited_once_with(0.1)
    else:
        sleep.assert_not_awaited()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "wrong"])
async def test_invalid_retry_delay_closes_response_and_stops(value: Any) -> None:
    reply = httpx2.Response(503)
    calls: list[Call] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test",
            http_client=client,
            extensions=[Extension(retry_delay=lambda *args: value, on_finish=calls.append)],
        )
        with pytest.raises((ValueError, TypeError)) as caught:
            await model.chat("Hi")
    assert reply.is_closed
    (call,) = calls
    assert call.error is caught.value
    assert call.attempts == 1


async def test_observer_failures_do_not_change_result_or_retry(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    calls: list[Call] = []

    def broken(*args: Any) -> None:
        raise RuntimeError("telemetry unavailable")

    async def broken_async(*args: Any) -> None:
        raise RuntimeError("async telemetry unavailable")

    responses = [httpx2.Response(503), httpx2.Response(200, json=CHAT_REPLY)]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            extensions=[
                Extension(on_start=broken, on_retry=broken_async, on_finish=broken),
                Extension(on_finish=calls.append),
            ],
        )
        assert (await model.chat("Hi")).text == "ok"
    assert calls[0].status == "success"
    assert caplog.text.count("Provider extension") == 3
    sleep.assert_awaited_once()


@pytest.mark.parametrize("stage", ["start", "policy", "retry", "send", "backoff"])
async def test_cancellation_propagates_and_finishes_once(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Call] = []
    reply = httpx2.Response(503)

    def cancel(*args: Any) -> Any:
        raise asyncio.CancelledError

    def handle(request: httpx2.Request) -> httpx2.Response:
        if stage == "send":
            cancel()
        return reply

    monkeypatch.setattr(base.asyncio, "sleep", cancel if stage == "backoff" else AsyncMock())
    extension = Extension(
        on_start=cancel if stage == "start" else None,
        retry_delay=cancel if stage == "policy" else None,
        on_retry=cancel if stage == "retry" else None,
        on_finish=calls.append,
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model("openai:test", http_client=client, extensions=[extension])
        with pytest.raises(asyncio.CancelledError):
            await model.chat("Hi")
    (call,) = calls
    assert call.status == "cancelled"
    assert isinstance(call.error, asyncio.CancelledError)
    assert call.attempts == (0 if stage == "start" else 1)
    if stage in {"policy", "retry", "backoff"}:
        assert reply.is_closed


async def test_open_stream_failure_cannot_be_retried_by_extension() -> None:
    calls: list[Call] = []
    policy = AsyncMock(return_value=0)
    reply = sse([{"choices": [{"delta": {"content": "partial"}}]}])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            extensions=[Extension(retry_delay=policy, on_finish=calls.append)],
        )
        with pytest.raises(republic.errors.StreamIncompleteError):
            async with model.stream("Hi") as stream:
                [event async for event in stream]
    policy.assert_not_awaited()
    assert calls[0].attempts == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_validation_failure_retains_usage_without_writing_history(streaming: bool) -> None:
    calls: list[Call] = []
    history = republic.history.InMemoryHistory()
    usage = {"prompt_tokens": 3, "completion_tokens": 2}
    reply = (
        sse([{"choices": [{"delta": {"content": "invalid"}}]}, {"choices": [], "usage": usage}, "[DONE]"])
        if streaming
        else httpx2.Response(200, json={**CHAT_REPLY, "usage": usage}, headers=HEADERS)
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            history=history,
            extensions=[Extension(on_finish=calls.append)],
        )
        with pytest.raises(ValueError) as caught:
            if streaming:
                async with model.stream("Hi", output_schema=int) as stream:
                    [event async for event in stream]
            else:
                await model.chat("Hi", output_schema=int)
    (call,) = calls
    assert call.error is caught.value
    assert call.status == "error"
    assert isinstance(call.response, republic.Response)
    assert call.response.output is None
    assert call.response.token_usage.total_tokens == 5
    assert call.request_id == "req_1"
    assert await history.read() == []


async def test_model_listing_observes_all_pages_as_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.asyncio, "sleep", AsyncMock())
    calls: list[Call] = []
    retries: list[int] = []
    responses = [
        httpx2.Response(503),
        httpx2.Response(200, json={"data": [{"id": "one"}], "has_more": True, "last_id": "one"}),
        httpx2.Response(429),
        httpx2.Response(200, json={"data": [{"id": "two"}]}),
    ]

    def retry(call: Call, error: Exception, delay: float) -> None:
        retries.append(call.attempts)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        provider = republic.get_provider(
            "anthropic", http_client=client, extensions=[Extension(on_retry=retry, on_finish=calls.append)]
        )
        models = await provider.list_models()
    (call,) = calls
    assert [model.id for model in models] == ["one", "two"]
    assert call.response is models
    assert call.attempts == 4
    assert retries == [1, 3]


async def test_task_cancellation_is_observed_without_retrying() -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()
    calls: list[Call] = []
    retries = AsyncMock()

    async def handle(request: httpx2.Request) -> httpx2.Response:
        started.set()
        await blocked.wait()
        return httpx2.Response(200, json=CHAT_REPLY)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model(
            "openai:test", http_client=client, extensions=[Extension(on_retry=retries, on_finish=calls.append)]
        )
        task = asyncio.create_task(model.chat("Hi"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    (call,) = calls
    assert call.status == "cancelled"
    assert call.error is caught.value
    assert call.attempts == 1
    retries.assert_not_awaited()
