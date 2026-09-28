# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: real SDK + HTTP transport fixtures; see NOTICE.
import asyncio
from typing import Any

import httpx
import openai
import pytest

from republic import (
    FilePart,
    IncompleteStreamError,
    Message,
    ProviderError,
    ReasoningPart,
    RequestOptions,
    Response,
    TextPart,
    Tool,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    UnsupportedRequestError,
    events,
    generate,
    stream,
)
from republic.providers.openai import OpenAIChatCompletions
from tests.http_fixtures import Bytes, streaming
from tests.openai_fixtures import Wire, chunk, completion, request, sse

pytestmark = pytest.mark.asyncio


async def test_generate_request_options_payload_and_response() -> None:
    req = request()
    req.messages.insert(0, Message(role="system", parts=[TextPart(text="Be brief")]))
    req.tools = [
        Tool(
            name="weather",
            description="Current weather",
            parameters={"type": "object"},
            provider_metadata={"openai": {"strict": True}},
        )
    ]
    req.options = RequestOptions(
        temperature=0,
        top_p=0.9,
        max_output_tokens=50,
        stop=["END"],
        tool_choice=ToolChoice(name="weather"),
        parallel_tool_calls=False,
        provider_options={
            "seed": 42,
            "response_format": {"type": "json_object"},
            "extra_headers": {"X-Title": "Fixture"},
            "extra_body": {"provider": {"sort": "price"}},
        },
    )
    raw = completion(
        usage={
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
        system_fingerprint="fp",
    )
    async with Wire([httpx.Response(200, json=raw)]) as wire:
        result = await generate(wire.provider, req)
        assert str(wire.requests[0].url) == "https://unit.test/api/v1/chat/completions"
        assert wire.requests[0].headers["authorization"] == "Bearer fixture-key"
        assert wire.requests[0].headers["x-title"] == "Fixture"
        assert wire.payload() == {
            "model": "vendor/model",
            "messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Current weather",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
            "temperature": 0,
            "top_p": 0.9,
            "max_completion_tokens": 50,
            "stop": ["END"],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "parallel_tool_calls": False,
            "seed": 42,
            "response_format": {"type": "json_object"},
            "provider": {"sort": "price"},
            "stream": False,
        }
        assert len(wire.requests) == 1
        assert wire.client.max_retries == 3
    assert result.message.text == "Hello"
    assert result.response_id == "chat-1"
    assert result.response_model == "resolved-model"
    assert result.usage is not None
    assert result.usage.total_tokens == 12
    assert result.usage.cache_read_tokens == 0
    assert result.usage.reasoning_tokens == 2
    assert result.usage.raw == raw["usage"]
    assert Response.model_validate_json(result.model_dump_json()) == result


async def test_tool_result_second_request_is_explicit_and_history_not_repaired() -> None:
    call = {"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"上海"}'}}
    req = request()
    original = req.model_dump_json()
    async with Wire([
        httpx.Response(200, json=completion({"tool_calls": [call]}, finish="tool_calls")),
        httpx.Response(200, json=completion()),
    ]) as wire:
        first = await generate(wire.provider, req)
        assert len(wire.requests) == 1
        assert req.model_dump_json() == original
        assert first.message.tool_calls[0].tool_args == '{"city":"上海"}'
        # Only the caller adds a result and chooses to make a second request.
        req.messages.extend([
            Message.model_validate_json(first.message.model_dump_json()),
            Message(
                role="tool",
                parts=[
                    ToolResultPart(tool_call_id="call-1", tool_name="weather", result={"temperature": 20}),
                    ToolResultPart(tool_call_id="unmatched", tool_name="unknown", result=None),
                ],
            ),
        ])
        before = req.model_dump_json()
        await generate(wire.provider, req)
        assert len(wire.requests) == 2
        assert wire.payload(1)["messages"][1:] == [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature": 20}'},
            {"role": "tool", "tool_call_id": "unmatched", "content": "null"},
        ]
        assert req.model_dump_json() == before


async def test_images_and_reasoning_history_are_converted_without_fetching() -> None:
    req = request()
    req.messages[0].parts.extend([
        FilePart(
            data="https://example.test/image.png",
            media_type="image/png",
            provider_metadata={"openai": {"detail": "low"}},
        ),
        FilePart.from_bytes(b"\xfb\xff\x00", media_type="image/png"),
    ])
    raw = completion({"content": "answer", "reasoning_content": "consider", "refusal": "refused"})
    async with Wire([httpx.Response(200, json=raw), httpx.Response(200, json=completion())]) as wire:
        result = await generate(wire.provider, req)
        assert wire.payload()["messages"][0]["content"] == [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "https://example.test/image.png", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,+/8A"}},
        ]
        req.messages.append(Message.model_validate_json(result.message.model_dump_json()))
        await generate(wire.provider, req)
        assert wire.payload(1)["messages"][-1] == {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "consider",
            "refusal": "refused",
        }


async def test_interleaved_tools_delayed_headers_empty_arguments_and_tail_usage() -> None:
    pieces = [
        chunk({"role": "assistant"}),
        chunk({"tool_calls": [{"index": 0}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "time", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "a"}]}),
        chunk({"reasoning": "think"}),
        chunk({"tool_calls": [{"index": 0, "function": {"name": "weather"}}]}),
        chunk({"content": "checking"}),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"zone":"'}}]}),
        chunk({
            "tool_calls": [
                {"index": 0, "function": {"arguments": '"上海"}'}},
                {"index": 1, "function": {"arguments": 'UTC"}'}},
            ]
        }),
        chunk(finish="tool_calls"),
        chunk(choices=[], usage={"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}),
    ]
    # Split inside UTF-8/SSE frames too, so the actual SDK decoder is exercised.
    encoded = b"".join(sse(p) for p in pieces) + sse("[DONE]")
    body = Bytes([encoded[i : i + 7] for i in range(0, len(encoded), 7)])
    async with Wire([streaming(body)]) as wire:
        async with stream(wire.provider, request()) as output:
            received = [event async for event in output]
            assert body.closed == 1
        assert wire.payload()["stream_options"] == {"include_usage": True}
        assert wire.payload()["stream"] is True
        assert len(wire.requests) == 1
        assert not wire.client.is_closed()
    assert [e.tool_name for e in received if isinstance(e, events.ToolStart)] == ["weather", "time"]
    assert [e.chunk for e in received if isinstance(e, events.ToolDelta)] == [
        '{"city":',
        "",
        '{"zone":"',
        '"上海"}',
        'UTC"}',
    ]
    assert output.response is not None
    assert output.response.usage is not None
    assert output.response.usage.total_tokens == 19
    assert output.response.finish_reason == "tool_call"
    assert output.message.text == "checking"
    assert [c.tool_args for c in output.message.tool_calls] == ['{"city":"上海"}', '{"zone":"UTC"}']
    assert sum(isinstance(event, events.StreamEnd) for event in received) == 1


@pytest.mark.parametrize(
    "finish, expected",
    [
        ("stop", "stop"),
        ("tool_calls", "tool_call"),
        ("length", "length"),
        ("content_filter", "content_filter"),
        ("new_reason", "other"),
    ],
)
async def test_finish_reasons_and_partial_arguments_are_data(finish: str, expected: str) -> None:
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"partial":'}}
    body = Bytes([sse(chunk({"tool_calls": [{"index": 0, **call}]}, finish=finish)), sse("[DONE]")])
    async with Wire([
        httpx.Response(200, json=completion({"tool_calls": [call]}, finish=finish)),
        streaming(body),
    ]) as wire:
        response = await generate(wire.provider, request())
        async with stream(wire.provider, request()) as output:
            async for _ in output:
                pass
        assert output.response == response
    assert response.finish_reason == expected
    assert response.message.tool_calls[0].tool_args == '{"partial":'
    if expected == "other":
        assert response.message.provider_metadata == {"openai": {"finish_reason": "new_reason"}}


@pytest.mark.parametrize("with_done", [False, True])
async def test_missing_finish_reason_is_incomplete_even_with_done(with_done: bool) -> None:
    body = Bytes([sse(chunk({"content": "partial"}))] + ([sse("[DONE]")] if with_done else []))
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(IncompleteStreamError):
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert output.status == "incomplete"
        assert output.message.text == "partial"
        assert output.response is None
        assert body.closed == 1
        assert len(wire.requests) == 1


async def test_early_exit_closes_response_but_borrowed_client_can_be_reused() -> None:
    body = Bytes([sse(chunk({"content": "partial"})), sse(chunk(finish="stop")), sse("[DONE]")])
    async with Wire([streaming(body), httpx.Response(200, json=completion())]) as wire:
        async with stream(wire.provider, request()) as output:
            async for event in output:
                if isinstance(event, events.TextDelta):
                    break
        assert body.closed == 1
        assert body.consumed == 1
        assert output.status == "closed"
        assert output.response is None
        await wire.provider.aclose()
        await wire.provider.aclose()
        assert not wire.client.is_closed()
        assert wire.client.max_retries == 3
        second = OpenAIChatCompletions(client=wire.client)
        await generate(second, request())
        await second.aclose()


@pytest.mark.parametrize("after_finish", [False, True])
async def test_cancellation_while_reading_or_waiting_for_tail_releases_response(after_finish: bool) -> None:
    body = Bytes([sse(chunk({"content": "partial"}, finish="stop" if after_finish else None))], wait=True)
    async with Wire([streaming(body)]) as wire:
        async with stream(wire.provider, request()) as output:

            async def consume() -> None:
                async for _ in output:
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(body.waiting.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert body.closed == 1
        assert output.status == "cancelled"
        assert output.response is None
        assert not wire.client.is_closed()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
@pytest.mark.parametrize("streaming_call", [False, True])
async def test_http_errors_preserve_cause_status_request_id_and_never_retry(status: int, streaming_call: bool) -> None:
    reply = httpx.Response(
        status,
        json={"error": {"message": "denied", "type": "test_error", "code": "test_code"}},
        headers={"x-request-id": "req-1", "retry-after": "0"},
    )
    async with Wire([reply]) as wire:
        with pytest.raises(ProviderError) as raised:
            if streaming_call:
                async with stream(wire.provider, request()) as output:
                    async for _ in output:
                        pass
            else:
                await generate(wire.provider, request())
        assert len(wire.requests) == 1
        assert raised.value.status_code == status
        assert raised.value.code == "test_code"
        assert raised.value.request_id == "req-1"
        assert isinstance(raised.value.__cause__, openai.APIStatusError)
        assert reply.is_closed


@pytest.mark.parametrize("error", [httpx.ConnectError("offline"), httpx.ReadTimeout("timeout")])
async def test_connection_errors_are_not_retried(error: httpx.HTTPError) -> None:
    async with Wire([error]) as wire:
        with pytest.raises(ProviderError) as raised:
            await generate(wire.provider, request())
        assert len(wire.requests) == 1
        assert isinstance(raised.value.__cause__, openai.APIConnectionError)
        assert raised.value.__cause__.__cause__ is error


@pytest.mark.parametrize("finished", [False, True])
async def test_sse_error_after_partial_output_closes_without_retry(finished: bool) -> None:
    body = Bytes([
        sse(chunk({"content": "partial"}, finish="stop" if finished else None)),
        sse({"error": {"message": "remote error", "code": "remote"}}),
    ])
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(ProviderError) as raised:
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert isinstance(raised.value.__cause__, openai.APIError)
        assert output.message.text == "partial"
        assert output.status == "failed"
        assert body.closed == 1
        assert len(wire.requests) == 1


async def test_transport_failure_mid_stream_keeps_original_cause() -> None:
    error = httpx.ReadError("connection lost")
    body = Bytes([sse(chunk({"content": "partial"}))], error=error)
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(ProviderError) as raised:
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert raised.value.__cause__ is error
        assert body.closed == 1
        assert len(wire.requests) == 1


@pytest.mark.parametrize("status", [200, 500])
async def test_owned_client_constructor_context_and_no_retry(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    body = completion() if status == 200 else {"error": {"message": "down"}}
    async with Wire([httpx.Response(status, json=body)]) as wire:
        real_constructor = openai.AsyncOpenAI
        created = []

        def constructor(**kwargs: Any) -> openai.AsyncOpenAI:
            client = real_constructor(**kwargs, http_client=wire.http)
            created.append(client)
            return client

        monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
        async with OpenAIChatCompletions(api_key="owned-key", base_url="https://owned.test/v1") as provider:
            if status == 500:
                with pytest.raises(ProviderError):
                    await generate(provider, request())
            else:
                await generate(provider, request())
            assert created[0].max_retries == 0
            assert not created[0].is_closed()
            assert str(wire.requests[0].url) == "https://owned.test/v1/chat/completions"
            assert wire.requests[0].headers["authorization"] == "Bearer owned-key"
        assert created[0].is_closed()
        assert len(wire.requests) == 1
        await provider.aclose()
        with pytest.raises(ProviderError, match="closed"):
            await generate(provider, request())


@pytest.mark.parametrize(
    "options",
    [
        {"model": "other"},
        {"messages": []},
        {"tools": []},
        {"stream": False},
        {"stream_options": {}},
        {"n": 2},
        {"temperature": 1},
        {"max_tokens": 10},
        {"max_retries": 4},
        {"not_an_option": True},
        {"extra_body": {"messages": []}},
        {"extra_body": {"stream": False}},
        {"extra_body": {"seed": 5}},
        {"extra_body": {"max_completion_tokens": 1}},
        {"extra_body": []},
        {"extra_headers": {"X": 1}},
        {"timeout": 0},
        {"timeout": True},
    ],
)
async def test_conflicting_or_unsupported_options_fail_before_http(options: dict[str, Any]) -> None:
    req = request()
    req.options.provider_options = options
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        async with stream(wire.provider, req) as output:
            with pytest.raises(UnsupportedRequestError):
                await anext(output)
        assert wire.requests == []


@pytest.mark.parametrize(
    "message",
    [
        Message(role="system", parts=[ReasoningPart(text="hidden")]),
        Message(role="user", parts=[ToolCallPart(tool_call_id="c", tool_name="f", tool_args="{}")]),
        Message(role="assistant", parts=[FilePart(data="https://example.test/a.png", media_type="image/png")]),
        Message(role="tool", parts=[TextPart(text="unlinked")]),
        Message(role="tool", parts=[]),
        Message(role="tool", parts=[ToolResultPart(tool_call_id="c", tool_name="f", result="failed", is_error=True)]),
        Message(role="user", parts=[FilePart(data="https://example.test/a.pdf", media_type="application/pdf")]),
        Message(role="user", parts=[TextPart(text="x", provider_metadata={"anthropic": {"signature": "x"}})]),
        Message(role="assistant", parts=[ReasoningPart(text="x", provider_metadata={"openai": {"encrypted": "x"}})]),
    ],
)
async def test_unrepresentable_history_is_rejected_not_dropped(message: Message) -> None:
    req = request()
    req.messages.append(message)
    before = req.model_dump_json()
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert wire.requests == []
        assert req.model_dump_json() == before


@pytest.mark.parametrize(
    "pieces",
    [
        [chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, finish="tool_calls")],
        [
            chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]}),
            chunk({"tool_calls": [{"index": 0, "id": "other"}]}),
        ],
        [chunk({"tool_calls": [{"index": -1}]})],
        [
            chunk({
                "tool_calls": [
                    {"index": 0, "id": "c", "function": {"name": "f"}},
                    {"index": 1, "id": "c", "function": {"name": "g"}},
                ]
            })
        ],
        [chunk(finish="stop"), chunk({"content": "late"})],
        [chunk({"content": "x"}), chunk(id="different")],
        [chunk({"audio": {"id": "a"}})],
        [chunk({"reasoning_details": [{"type": "reasoning.unknown", "data": "x"}]})],
        [chunk(choices=[{"index": 1, "delta": {"content": "wrong choice"}}])],
    ],
)
async def test_invalid_stream_data_is_visible_and_response_is_closed(pieces: list[dict[str, Any]]) -> None:
    body = Bytes([sse(piece) for piece in pieces])
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(ProviderError):
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert output.response is None
        assert body.closed == 1
        assert len(wire.requests) == 1


async def test_malformed_logprobs_warning_and_error_are_visible() -> None:
    body = Bytes([sse(chunk(logprobs={"content": "invalid array"}))])
    async with Wire([streaming(body)]) as wire:
        with (
            pytest.warns(UserWarning, match="Pydantic serializer warnings"),
            pytest.raises(ProviderError, match="arrays"),
        ):
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert output.response is None
        assert body.closed == 1
        assert len(wire.requests) == 1


@pytest.mark.parametrize(
    "raw",
    [
        completion(finish=None),
        completion(choices=[]),
        completion(choices=[{"index": 0, "finish_reason": "stop"}]),
        completion({"role": "user", "content": "bad role"}),
        completion({"audio": {"id": "a"}}),
        completion({"function_call": {"name": "legacy", "arguments": "{}"}}),
    ],
)
async def test_invalid_generate_response_is_visible(raw: dict[str, Any]) -> None:
    async with Wire([httpx.Response(200, json=raw)]) as wire:
        with pytest.raises(ProviderError):
            await generate(wire.provider, request())
        assert len(wire.requests) == 1


async def test_stream_preserves_reasoning_refusal_logprobs_and_empty_call() -> None:
    body = Bytes([
        sse(
            chunk(
                {"reasoning_content": "think", "refusal": "can"},
                logprobs={"content": [{"token": "a", "logprob": -1.0}]},
            )
        ),
        sse(
            chunk(
                {"reasoning_content": " more", "refusal": "not"},
                logprobs={"content": [{"token": "b", "logprob": -2.0}]},
            )
        ),
        sse(chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]}, finish="tool_calls")),
        sse(chunk(choices=[], usage={"completion_tokens": 0})),
        sse("[DONE]"),
    ])
    async with Wire([streaming(body), httpx.Response(200, json=completion())]) as wire:
        async with stream(wire.provider, request()) as output:
            async for _ in output:
                pass
        result = output.response
        assert result is not None
        assert result.message.tool_calls[0].tool_args == ""
        assert result.usage is not None
        assert result.usage.input_tokens is None
        assert result.usage.output_tokens == 0
        assert result.message.provider_metadata == {
            "openai": {
                "refusal": "cannot",
                "logprobs": {"content": [{"token": "a", "logprob": -1.0}, {"token": "b", "logprob": -2.0}]},
            }
        }
        req = request()
        req.messages.append(Message.model_validate_json(result.message.model_dump_json()))
        await generate(wire.provider, req)
        assert wire.payload(1)["messages"][-1]["reasoning_content"] == "think more"
        assert wire.payload(1)["messages"][-1]["refusal"] == "cannot"


@pytest.mark.parametrize("streaming_call", [False, True])
async def test_cancellation_during_request_creation_propagates(streaming_call: bool) -> None:
    entered = asyncio.Event()
    released = asyncio.Event()

    async def handle(req: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()
        return httpx.Response(200, json=completion())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http,
        openai.AsyncOpenAI(api_key="fixture", http_client=http) as client,
        OpenAIChatCompletions(client=client) as provider,
    ):

        async def run() -> None:
            if streaming_call:
                async with stream(provider, request()) as output:
                    await anext(output)
            else:
                await generate(provider, request())

        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert released.is_set()
        assert not client.is_closed()


@pytest.mark.parametrize("streaming_call", [False, True])
async def test_malformed_wire_json_preserves_parser_error(streaming_call: bool) -> None:
    body = Bytes([sse("{broken-json")])
    reply = (
        streaming(body)
        if streaming_call
        else httpx.Response(200, content=b"{broken-json", headers={"content-type": "application/json"})
    )
    async with Wire([reply]) as wire:
        with pytest.raises(ProviderError) as raised:
            if streaming_call:
                async with stream(wire.provider, request()) as output:
                    async for _ in output:
                        pass
            else:
                await generate(wire.provider, request())
        assert isinstance(raised.value.__cause__, ValueError)
        assert reply.is_closed
        assert len(wire.requests) == 1


async def test_injected_client_configuration_cannot_be_overridden() -> None:
    async with Wire([]) as wire:
        override = OpenAIChatCompletions(client=wire.client, api_key="conflict")
        assert override._client.api_key == "conflict"
        assert wire.client.api_key == "fixture-key"
        await override.aclose()
        override = OpenAIChatCompletions(client=wire.client, base_url="https://other.test")
        assert str(override._client.base_url) == "https://other.test"
        await override.aclose()
        assert not wire.client.is_closed()
