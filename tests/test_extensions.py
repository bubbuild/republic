from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx2
import pytest

import republic
from republic.extensions import Extension
from republic.providers import base
from tests.conftest import FakeService
from tests.test_http_reliability import CHAT_REPLY, HEADERS, sse


@pytest.mark.parametrize("operation", ["chat", "embed", "generate", "edit", "decide", "list_models"])
async def test_observes_public_result(operation: str, service: FakeService) -> None:
    seen: list[Any] = []
    options: dict[str, Any] = {"http_client": service.client(), "extensions": [Extension(on_response=seen.append)]}
    if operation == "chat":
        service.reply_json(CHAT_REPLY)
        result = await republic.get_model("openai:test", api_format="chat", **options).chat("Hi")
    elif operation == "embed":
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
    assert len(seen) == 1
    assert seen[0] is result


@pytest.mark.parametrize("enabled", [False, True])
async def test_existing_provider_override_signatures_remain_usable(enabled: bool, service: FakeService) -> None:
    class ExistingProvider(republic.providers.OpenAI):
        async def _post(self, api_format, request):
            return await super()._post(api_format, request)

        @asynccontextmanager
        async def _stream(self, api_format, request):
            async with super()._stream(api_format, request) as stream:
                yield stream

    seen: list[Any] = []
    service.reply_json(CHAT_REPLY)
    service.reply_events(["[DONE]"])
    model = ExistingProvider(
        api_format="chat",
        http_client=service.client(),
        extensions=[Extension(on_response=seen.append)] if enabled else [],
    ).get_model("test")
    await model.chat("Hi")
    async with model.stream("Hi") as stream:
        [event async for event in stream]
    assert len(seen) == (2 if enabled else 0)


async def test_stream_notifies_on_completion_before_context_exit() -> None:
    seen: list[Any] = []
    reply = sse([{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
    history = republic.history.InMemoryHistory()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            history=history,
            extensions=[Extension(on_response=seen.append)],
        )
        async with model.stream("Hi") as stream:
            assert seen == []
            async for event in stream:
                if isinstance(event, republic.events.Completed):
                    assert seen == [event.response]
                    assert seen[0] is stream.response
                    assert len(await history.read()) == 2
            assert len(seen) == 1
        assert len(seen) == 1


@pytest.mark.parametrize("mode", ["http", "invalid", "truncated", "early", "unread"])
async def test_no_response_notification_without_completed_result(mode: str) -> None:
    seen: list[Any] = []
    reply = (
        httpx2.Response(400)
        if mode == "http"
        else sse([{"choices": [{"delta": {"content": "partial"}}]}] + ([] if mode == "truncated" else ["[DONE]"]))
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: reply)) as client:
        model = republic.get_model(
            "openai:test", api_format="chat", http_client=client, extensions=[Extension(on_response=seen.append)]
        )
        if mode == "http":
            with pytest.raises(republic.errors.APIStatusError):
                await model.chat("Hi")
        elif mode == "invalid":
            with pytest.raises(ValueError):
                async with model.stream("Hi", output_schema=int) as stream:
                    [event async for event in stream]
        elif mode == "truncated":
            async with model.stream("Hi") as stream:
                with pytest.raises(republic.errors.StreamIncompleteError):
                    [event async for event in stream]
        else:
            async with model.stream("Hi") as stream:
                if mode == "early":
                    async for _event in stream:
                        break
    assert seen == []
    assert reply.is_closed


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_observer_failure_does_not_change_success_or_history(
    failure: type[BaseException], service: FakeService, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[Any] = []

    def broken(response: Any) -> None:
        raise failure("observer failed")

    service.reply_json(CHAT_REPLY)
    history = republic.history.InMemoryHistory()
    model = republic.get_model(
        "openai:test",
        api_format="chat",
        http_client=service.client(),
        history=history,
        extensions=[Extension(on_response=broken), Extension(on_response=seen.append)],
    )
    result = await model.chat("Hi")
    assert seen == [result]
    assert len(await history.read()) == 2
    assert "on_response failed" in caplog.text


async def test_retry_callbacks_run_after_failed_response_is_released(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    failed = httpx2.Response(503, headers=HEADERS)
    responses = [failed, sse(["[DONE]"])]
    seen = []

    def policy(error: Exception, attempt: int, delay: float) -> float:
        assert failed.is_closed
        assert isinstance(error, republic.errors.APIStatusError)
        assert (error.status_code, error.request_id, attempt, delay) == (503, "req_1", 1, 3)
        return 0.1

    def retry(error: Exception, attempt: int, delay: float) -> None:
        assert failed.is_closed
        seen.append((attempt, delay))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: responses.pop(0))) as client:
        model = republic.get_model(
            "openai:test",
            api_format="chat",
            http_client=client,
            extensions=[
                Extension(retry_delay=policy),
                Extension(retry_delay=lambda error, attempt, delay: delay * 2, on_retry=retry),
            ],
        )
        async with model.stream("Hi") as stream:
            [event async for event in stream]
    sleep.assert_awaited_once_with(0.2)
    assert seen == [(1, 0.2)]


@pytest.mark.parametrize("status", [400, 503])
async def test_extensions_keep_existing_retry_eligibility_and_limits(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base.asyncio, "sleep", AsyncMock())
    seen = []
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status)

    def retry(error: Exception, attempt: int, delay: float) -> None:
        seen.append((attempt, delay))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = republic.get_model(
            "openai:test",
            http_client=client,
            max_retries=1,
            max_retry_delay=0.1,
            extensions=[Extension(retry_delay=lambda *args: 100, on_retry=retry)],
        )
        with pytest.raises(republic.errors.APIStatusError):
            await model.chat("Hi")
    assert len(requests) == (1 if status == 400 else 2)
    assert seen == ([] if status == 400 else [(1, 0.1)])


@pytest.mark.parametrize("value", [None, True, -1, float("inf"), float("nan")])
async def test_invalid_policy_stops_after_response_cleanup(value: Any) -> None:
    failed = httpx2.Response(503)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: failed)) as client:
        model = republic.get_model(
            "openai:test", http_client=client, extensions=[Extension(retry_delay=lambda *args: value)]
        )
        with pytest.raises(ValueError):
            await model.chat("Hi")
    assert failed.is_closed


async def test_async_observer_is_rejected_without_affecting_response(
    service: FakeService, caplog: pytest.LogCaptureFixture, recwarn: pytest.WarningsRecorder
) -> None:
    async def observer(response: Any) -> None:
        pytest.fail("async observer must not run")

    options: dict[str, Any] = {"extensions": [Extension(on_response=cast(Any, observer))]}
    service.reply_json(CHAT_REPLY)
    result = await republic.get_model("openai:test", api_format="chat", http_client=service.client(), **options).chat(
        "Hi"
    )
    assert result.text == "ok"
    assert "must be synchronous" in caplog.text
    assert not recwarn
