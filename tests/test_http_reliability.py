from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from email.utils import formatdate
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest

import republic
from republic.history import InMemoryHistory
from republic.providers import base

CHAT_REPLY = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
HEADERS = {"X-Request-ID": "req_1", "Retry-After": "3"}


def sse(events: list[Any], *, headers: dict[str, str] | None = None) -> httpx2.Response:
    body = "".join(f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n" for event in events)
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream", **(headers or HEADERS)})


@pytest.fixture
def sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    sleep = AsyncMock()
    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    monkeypatch.setattr(base.random, "uniform", lambda low, high: 1)
    return sleep


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_retries_status_and_closes_response_before_waiting(status: int, sleep: AsyncMock) -> None:
    responses = [httpx2.Response(status, text="busy", headers=HEADERS), httpx2.Response(200, json=CHAT_REPLY)]
    failed = responses[0]
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return responses.pop(0)

    async def waited(delay: float) -> None:
        assert failed.is_closed
        assert delay == 3

    sleep.side_effect = waited
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client)
        assert (await model.chat("Hi")).text == "ok"
        assert not client.is_closed
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    sleep.assert_awaited_once_with(3)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_does_not_retry_permanent_errors(status: int, sleep: AsyncMock) -> None:
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status, text="bad request", headers=HEADERS)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model("openai:test", http_client=client)
        with pytest.raises(republic.APIStatusError) as caught:
            await model.chat("Hi")
    assert len(requests) == 1
    assert caught.value.status_code == status
    assert caught.value.body == "bad request"
    assert caught.value.request_id == "req_1"
    assert caught.value.headers["retry-after"] == "3"
    sleep.assert_not_awaited()


@pytest.mark.parametrize("max_retries", [0, 2])
async def test_exhausted_retries_keep_last_response(max_retries: int, sleep: AsyncMock) -> None:
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(503, text=f"failure {len(requests)}", headers={"request-id": str(len(requests))})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model("openai:test", http_client=client, max_retries=max_retries)
        with pytest.raises(republic.APIStatusError) as caught:
            await model.chat("Hi")
    assert len(requests) == max_retries + 1
    assert caught.value.body == f"failure {max_retries + 1}"
    assert caught.value.request_id == str(max_retries + 1)
    assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1][:max_retries]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "2.5"}, 2.5),
        ({"retry-after-ms": "1250"}, 1.25),
        ({"retry-after": formatdate(1003, usegmt=True)}, 3),
        ({"retry-after": formatdate(999, usegmt=True)}, 0),
        ({"retry-after": "1000"}, 60),
        ({"retry-after": "invalid"}, 0.5),
        ({"retry-after": "NaN"}, 0.5),
        ({"retry-after-ms": "invalid", "retry-after": "4"}, 4),
    ],
)
async def test_retry_after_formats(
    headers: dict[str, str], expected: float, sleep: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base.time, "time", lambda: 1000)
    responses = [httpx2.Response(429, headers=headers), httpx2.Response(200, json=CHAT_REPLY)]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        await republic.get_model("openai:test", api_format="chat", http_client=client).chat("Hi")
    sleep.assert_awaited_once_with(expected)


@pytest.mark.parametrize("failure", [httpx2.ConnectError, httpx2.ReadTimeout, httpx2.RemoteProtocolError])
async def test_transient_network_errors_can_recover(failure: type[httpx2.RequestError], sleep: AsyncMock) -> None:
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if len(requests) == 1:
            raise failure("temporary", request=request)
        return httpx2.Response(200, json=CHAT_REPLY)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        assert (await republic.get_model("openai:test", api_format="chat", http_client=client).chat("Hi")).text == "ok"
    assert len(requests) == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [(httpx2.ConnectError, republic.APIConnectionError), (httpx2.ReadTimeout, republic.APITimeoutError)],
)
async def test_network_error_normalization_preserves_cause(
    failure: type[httpx2.RequestError], expected: type[republic.APIConnectionError], sleep: AsyncMock
) -> None:
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        raise failure("original", request=request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        with pytest.raises(expected) as caught:
            await republic.get_model("openai:test", http_client=client).chat("Hi")
    assert len(requests) == 3
    assert isinstance(caught.value.__cause__, failure)
    assert caught.value.request_id is None
    assert sleep.await_count == 2


async def test_model_listing_retries_each_page(sleep: AsyncMock) -> None:
    responses = [
        httpx2.Response(503),
        httpx2.Response(200, json={"data": [{"id": "one"}], "has_more": True, "last_id": "one"}),
        httpx2.Response(429),
        httpx2.Response(200, json={"data": [{"id": "two"}]}),
    ]
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return responses.pop(0)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        models = await republic.get_provider("anthropic", http_client=client).list_models()
    assert [model.id for model in models] == ["one", "two"]
    assert requests[2].url.params["after_id"] == requests[3].url.params["after_id"] == "one"
    assert sleep.await_count == 2


@pytest.mark.parametrize("kind", ["chat", "embedding", "decision"])
async def test_successful_responses_preserve_headers(kind: str) -> None:
    payloads = {
        "chat": CHAT_REPLY,
        "embedding": {"data": [{"index": 0, "embedding": [0.1]}]},
        "decision": {"answers": {"ok": {"type": "noul", "noul": 0.9}}},
    }
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=payloads[kind], headers=HEADERS))
    ) as client:
        if kind == "chat":
            response = await republic.get_model("openai:test", api_format="chat", http_client=client).chat("Hi")
        elif kind == "embedding":
            response = await republic.get_embedding_model("openai:test", http_client=client).embed("Hi")
        else:
            response = await republic.get_decision_model("typesafe:test", http_client=client).decide(
                "Hi", questions={"ok": republic.decisions.Noul("OK?")}
            )
    assert response.request_id == "req_1"
    assert response.headers["retry-after"] == "3"


@pytest.mark.parametrize("streaming", [False, True])
async def test_api_response_errors_preserve_headers(streaming: bool) -> None:
    error = {"error": {"message": "failed"}}
    response = sse([error]) if streaming else httpx2.Response(200, json=error, headers=HEADERS)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client)
        with pytest.raises(republic.APIResponseError) as caught:
            if streaming:
                async with model.stream("Hi") as stream:
                    [event async for event in stream]
            else:
                await model.chat("Hi")
    assert caught.value.request_id == "req_1"
    assert caught.value.headers["retry-after"] == "3"


# Each case has content followed by the format's terminal event. A finish reason
# in a Chat/ Messages delta alone is insufficient to establish clean completion.
STREAM_CASES = [
    ("openai", "chat", {"choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}]}, "[DONE]"),
    (
        "openai",
        "responses",
        {"type": "response.output_text.delta", "delta": "partial"},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    ),
    (
        "anthropic",
        "messages",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
        {"type": "message_stop"},
    ),
    (
        "google",
        "gemini",
        {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]},
        {"candidates": [{"finishReason": "STOP"}]},
    ),
]


@pytest.mark.parametrize(("provider", "api_format", "chunk", "terminal"), STREAM_CASES)
@pytest.mark.parametrize("complete", [False, True])
async def test_stream_completion_and_history(
    provider: str, api_format: str, chunk: Any, terminal: Any, complete: bool
) -> None:
    response = sse([chunk, terminal] if complete else [chunk])
    history = InMemoryHistory()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        model = republic.get_model(f"{provider}:test", api_format=api_format, http_client=client, history=history)
        events = []
        async with model.stream("Hi") as stream:
            assert stream.request_id == "req_1"
            if complete:
                events = [event async for event in stream]
                assert isinstance(events[-1], republic.events.Completed)
                assert stream.response.request_id == "req_1"
                assert stream.response.headers == stream.headers
            else:
                with pytest.raises(republic.StreamIncompleteError) as caught:
                    async for event in stream:
                        events.append(event)
                assert caught.value.request_id == "req_1"
                assert all(not isinstance(event, republic.events.Completed) for event in events)
                with pytest.raises(republic.StreamNotFinishedError):
                    _ = stream.response
    assert await history.read() == ([republic.user("Hi"), stream.response.message] if complete else [])
    assert response.is_closed


@pytest.mark.parametrize("failure", [None, httpx2.ReadError, httpx2.ReadTimeout])
async def test_does_not_replay_open_stream_even_before_first_chunk(
    failure: type[httpx2.RequestError] | None, sleep: AsyncMock
) -> None:
    class BrokenStream(httpx2.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            if failure is None:
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise httpx2.ReadError("disconnected")
            raise failure("failed before any output")
            yield b""  # pragma: no cover

        async def aclose(self) -> None:
            self.closed = True

    body = BrokenStream()
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, stream=body, headers=HEADERS)

    history = InMemoryHistory()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client, history=history)
        expected = republic.APITimeoutError if failure is httpx2.ReadTimeout else republic.APIConnectionError
        with pytest.raises(expected) as caught:
            async with model.stream("Hi") as stream:
                [event async for event in stream]
        assert not client.is_closed
    assert len(requests) == 1
    assert caught.value.request_id == "req_1"
    assert isinstance(caught.value.__cause__, httpx2.RequestError)
    assert await history.read() == []
    assert body.closed
    sleep.assert_not_awaited()


async def test_stream_retries_initial_http_error(sleep: AsyncMock) -> None:
    responses = [httpx2.Response(503), sse(["[DONE]"])]
    failed = responses[0]
    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client,
        republic.get_model("openai:test", api_format="chat", http_client=client).stream("Hi") as stream,
    ):
        events = [event async for event in stream]
    assert isinstance(events[-1], republic.events.Completed)
    assert failed.is_closed
    sleep.assert_awaited_once_with(0.5)


async def test_cancelled_request_is_not_retried(sleep: AsyncMock) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise asyncio.CancelledError

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        with pytest.raises(asyncio.CancelledError):
            await republic.get_model("openai:test", http_client=client).chat("Hi")
    sleep.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [
        {"max_retries": -1},
        {"max_retries": True},
        {"retry_delay": float("nan")},
        {"max_retry_delay": float("inf")},
        {"retry_delay": -1},
    ],
)
def test_rejects_invalid_retry_settings(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        republic.get_provider("openai", **options)


async def test_concurrent_stream_metadata_belongs_to_each_call() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        return sse(["[DONE]"], headers={"request-id": prompt})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model("openai:test", api_format="chat", http_client=client)
        async with model.stream("first") as first, model.stream("second") as second:
            [event async for event in second]
            [event async for event in first]
            assert first.request_id == first.response.request_id == "first"
            assert second.request_id == second.response.request_id == "second"
            assert first.headers["request-id"] == "first"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_empty_stream_is_incomplete(provider: str) -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: sse([]))) as client:
        with pytest.raises(republic.StreamIncompleteError):
            async with republic.get_model(f"{provider}:test", http_client=client).stream("Hi") as stream:
                [event async for event in stream]


async def test_messages_stop_reason_without_message_stop_is_incomplete() -> None:
    reply = sse([{"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        with pytest.raises(republic.StreamIncompleteError):
            async with republic.get_model("anthropic:test", http_client=client).stream("Hi") as stream:
                [event async for event in stream]


async def test_response_output_limit_is_terminal_not_truncated_transport() -> None:
    reply = sse([
        {"type": "response.output_text.delta", "delta": "partial"},
        {
            "type": "response.incomplete",
            "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
        },
    ])
    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client,
        republic.get_model("openai:test", http_client=client).stream("Hi") as stream,
    ):
        events = [event async for event in stream]
    assert stream.response.finish_reason == "length"
    assert isinstance(events[-1], republic.events.Completed)


async def test_truncated_tool_arguments_are_not_released_as_ready() -> None:
    reply = sse([
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": '{"city":'}}
                        ]
                    }
                }
            ]
        }
    ])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        events = []
        with pytest.raises(republic.StreamIncompleteError):
            async with republic.get_model("openai:test", api_format="chat", http_client=client).stream("Hi") as stream:
                async for event in stream:
                    events.append(event)
    assert any(isinstance(event, republic.events.ToolCallDelta) for event in events)
    assert all(not isinstance(event, (republic.events.ToolCallReady, republic.events.Completed)) for event in events)


async def test_cancellation_during_backoff_closes_failed_response(sleep: AsyncMock) -> None:
    response = httpx2.Response(503)
    sleep.side_effect = asyncio.CancelledError
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response)) as client:
        with pytest.raises(asyncio.CancelledError):
            await republic.get_model("openai:test", http_client=client).chat("Hi")
    assert response.is_closed
    sleep.assert_awaited_once()
