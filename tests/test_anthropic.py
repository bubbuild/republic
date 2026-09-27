# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Selected Messages scenarios adapted from ai-python c788059d; see NOTICE.
"""Anthropic payloads, reasoning, usage and lifetime through the real SDK."""

import asyncio
import json
from typing import Any

import anthropic
import httpx
import pytest

from republic import (
    FilePart,
    IncompleteStreamError,
    Message,
    Provider,
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
    generate,
    stream,
)
from republic.providers.anthropic import AnthropicMessages
from tests.anthropic_fixtures import Wire, block, delta, finish, message, request, sse, start, stop
from tests.http_fixtures import Bytes, streaming

pytestmark = pytest.mark.asyncio


async def collect(provider: Provider) -> Response:
    async with stream(provider, request()) as output:
        async for _ in output:
            pass
    assert output.response is not None
    return output.response


async def test_generate_system_tools_results_cache_and_options() -> None:
    cache = {"type": "ephemeral", "ttl": "1h"}
    req = request()
    req.messages = [
        Message(
            role="system", parts=[TextPart(text="Rules", provider_metadata={"anthropic": {"cache_control": cache}})]
        ),
        Message(role="system", parts=[TextPart(text="More rules")]),
        *req.messages,
        Message(
            role="assistant",
            parts=[
                ReasoningPart(text="Think", provider_metadata={"anthropic": {"signature": "sig"}}),
                ToolCallPart(tool_call_id="call_1", tool_name="lookup", tool_args='{"n":1}'),
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(
                    tool_call_id="call_1",
                    tool_name="lookup",
                    result={"error": "missing"},
                    is_error=True,
                    provider_metadata={"anthropic": {"cache_control": cache}},
                )
            ],
        ),
        Message(role="user", parts=[TextPart(text="Continue")]),
    ]
    req.tools = [
        Tool(
            name="lookup",
            description="Lookup",
            parameters={"type": "object"},
            provider_metadata={"anthropic": {"cache_control": cache, "strict": True}},
        )
    ]
    req.options = RequestOptions(
        max_output_tokens=4096,
        temperature=0.2,
        top_p=0.7,
        stop=["END"],
        tool_choice=ToolChoice(name="lookup"),
        parallel_tool_calls=False,
        provider_options={
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "top_k": 10,
            "output_config": {"effort": "low"},
            "service_tier": "standard_only",
            "metadata": {"user_id": "fixture-user"},
            "cache_control": {"type": "ephemeral"},
            "extra_headers": {"x-fixture": "yes"},
            "timeout": 2,
        },
    )
    original = req.model_dump_json()
    async with Wire([httpx.Response(200, json=message())]) as w:
        result = await generate(w.provider, req)
        payload = w.payload()
        assert str(w.requests[0].url) == "https://unit.test/api/v1/messages"
        assert w.requests[0].headers["x-api-key"] == "fixture-key"
        assert w.requests[0].headers["x-fixture"] == "yes"
        assert payload == {
            "model": "fixture-model",
            "stream": False,
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.7,
            "stop_sequences": ["END"],
            "top_k": 10,
            "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "low"},
            "service_tier": "standard_only",
            "metadata": {"user_id": "fixture-user"},
            "cache_control": {"type": "ephemeral"},
            "system": [
                {"type": "text", "text": "Rules", "cache_control": cache},
                {"type": "text", "text": "More rules"},
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup",
                    "input_schema": {"type": "object"},
                    "strict": True,
                    "cache_control": cache,
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Think", "signature": "sig"},
                        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"n": 1}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "is_error": True,
                            "content": '{"error": "missing"}',
                            "cache_control": cache,
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Continue"}]},
            ],
        }
        assert result.message.text == "Hello" and result.finish_reason == "stop"
        assert result.response_id == "msg_fixture" and result.response_model == "resolved-model"
        assert req.model_dump_json() == original and len(w.requests) == 1
        assert result.usage is not None and result.usage.total_tokens == 13


@pytest.mark.parametrize("streamed", [False, True])
async def test_thinking_signature_redaction_json_round_trip_and_tool_result(streamed: bool) -> None:
    content = [
        {"type": "thinking", "thinking": "Think carefully", "signature": "initial-A-B"},
        {"type": "redacted_thinking", "data": "opaque-encrypted"},
        {"type": "thinking", "thinking": "", "signature": "hidden-signature"},
        {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"n": 1}, "caller": {"type": "direct"}},
    ]
    body = Bytes([
        start(),
        block({"type": "thinking", "thinking": "Think", "signature": "initial-"}),
        delta("thinking_delta", thinking=" carefully"),
        delta("signature_delta", signature="A-"),
        delta("signature_delta", signature="B"),
        stop(),
        sse("ping"),
        block(content[1], 1),
        stop(1),
        block({"type": "thinking", "thinking": "", "signature": ""}, 2),
        delta("thinking_delta", 2, thinking=""),
        delta("signature_delta", 2, signature="hidden-"),
        delta("signature_delta", 2, signature="signature"),
        stop(2),
        block({**content[3], "input": {}}, 3),
        delta("input_json_delta", 3, partial_json='{"n":'),
        delta("input_json_delta", 3, partial_json="1}"),
        stop(3),
        finish("tool_use"),
        sse("message_stop"),
    ])
    reply = streaming(body) if streamed else httpx.Response(200, json=message(content, stop_reason="tool_use"))
    async with Wire([reply, httpx.Response(200, json=message())]) as w:
        result = await collect(w.provider) if streamed else await generate(w.provider, request())
        restored = Response.model_validate_json(result.model_dump_json())
        assert restored == result and result.finish_reason == "tool_call"
        assert restored.message.parts[:3] == [
            ReasoningPart(text="Think carefully", provider_metadata={"anthropic": {"signature": "initial-A-B"}}),
            ReasoningPart(text="", provider_metadata={"anthropic": {"redacted_data": "opaque-encrypted"}}),
            ReasoningPart(text="", provider_metadata={"anthropic": {"signature": "hidden-signature"}}),
        ]
        req = request()
        req.messages += [
            restored.message,
            Message(role="tool", parts=[ToolResultPart(tool_call_id="tool_1", tool_name="lookup", result="found")]),
        ]
        await generate(w.provider, req)
        assert w.payload(1)["messages"][1]["content"] == content
        assert w.payload(1)["messages"][2]["content"] == [
            {"type": "tool_result", "tool_use_id": "tool_1", "content": "found", "is_error": False}
        ]
        assert len(w.requests) == 2
        if streamed:
            assert body.closed == 1


async def test_interleaved_fragments_do_not_include_initial_placeholder() -> None:
    body = Bytes([
        start(),
        block({"type": "tool_use", "id": "a", "name": "first", "input": {}}),
        block({"type": "tool_use", "id": "b", "name": "second", "input": {}}, 1),
        delta("input_json_delta", partial_json='{"text":'),
        delta("input_json_delta", 1, partial_json="{bad"),
        delta("input_json_delta", partial_json='"中"}'),
        stop(1),
        stop(),
        finish("max_tokens"),
        sse("message_stop"),
    ])
    # Break UTF-8 and SSE framing across actual HTTP byte chunks.
    body.pieces = [bytes([byte]) for byte in b"".join(body.pieces)]
    async with Wire([streaming(body)]) as w:
        result = await collect(w.provider)
        assert [c.tool_args for c in result.message.tool_calls] == ['{"text":"中"}', "{bad"]
        assert [c.tool_call_id for c in result.message.tool_calls] == ["a", "b"]
        assert result.finish_reason == "length" and body.closed == 1
        assert len(w.requests) == 1 and w.payload()["stream"] is True


@pytest.mark.parametrize("initial", [{}, {"value": 2}])
async def test_tool_input_without_deltas_uses_initial_object(initial: dict[str, Any]) -> None:
    body = Bytes([
        start(),
        block({"type": "tool_use", "id": "a", "name": "f", "input": initial}),
        stop(),
        finish("tool_use"),
        sse("message_stop"),
    ])
    async with Wire([streaming(body)]) as w:
        result = await collect(w.provider)
        assert json.loads(result.message.tool_calls[0].tool_args) == initial


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            {
                "input_tokens": 10,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 30,
                "output_tokens": 4,
                "cache_creation": {"ephemeral_5m_input_tokens": 25, "ephemeral_1h_input_tokens": 5},
                "output_tokens_details": {"thinking_tokens": 2},
            },
            (60, 4, 20, 30, 2, 64),
        ),
        (
            {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 0},
            (0, 0, 0, 0, None, 0),
        ),
        ({"input_tokens": 10, "output_tokens": 4}, (None, 4, None, None, None, None)),
        (
            {"cache_read_input_tokens": 20, "cache_creation_input_tokens": 0, "output_tokens": 4},
            (None, 4, 20, 0, None, None),
        ),
        (
            {"input_tokens": 10, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 0},
            (30, None, 20, 0, None, None),
        ),
    ],
)
async def test_usage_is_inclusive_with_unknowns(raw: dict[str, Any], expected: tuple[int | None, ...]) -> None:
    async with Wire([httpx.Response(200, json=message(usage=raw))]) as w:
        result = await generate(w.provider, request())
        usage = result.usage
        assert usage is not None and usage.raw == raw
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
        ) == expected


async def test_stream_cumulative_usage_replaces_instead_of_adds() -> None:
    body = Bytes(
        [
            start(
                usage={
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 30,
                }
            ),
            sse("message_delta", delta={}, usage={"input_tokens": 12, "output_tokens": 3}),
            sse(
                "message_delta",
                delta={"stop_reason": "end_turn"},
                usage={
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 32,
                    "cache_read_input_tokens": 20,
                },
            ),
            sse("message_stop"),
        ],
        wait=True,
    )
    async with Wire([streaming(body)]) as w:
        result = await asyncio.wait_for(collect(w.provider), timeout=2)
        assert result.usage is not None
        assert result.usage.input_tokens == 64 and result.usage.output_tokens == 5 and result.usage.total_tokens == 69
        assert result.usage.raw == {
            "input_tokens": 12,
            "output_tokens": 5,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 32,
        }
        assert body.closed == 1 and not body.waiting.is_set()


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "reason,expected",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("tool_use", "tool_call"),
        ("refusal", "content_filter"),
        ("pause_turn", "other"),
        ("future", "other"),
    ],
)
async def test_stop_reasons_are_terminal_without_continuation(streamed: bool, reason: str, expected: str) -> None:
    body = Bytes([
        start(),
        block({"type": "text", "text": "Partial"}),
        stop(),
        sse(
            "message_delta",
            delta={"stop_reason": reason, "stop_sequence": "END" if reason == "stop_sequence" else None},
            usage={"output_tokens": 2},
        ),
        sse("message_stop"),
    ])
    reply = (
        streaming(body)
        if streamed
        else httpx.Response(
            200,
            json=message(
                [{"type": "text", "text": "Partial"}],
                stop_reason=reason,
                stop_sequence="END" if reason == "stop_sequence" else None,
            ),
        )
    )
    async with Wire([reply]) as w:
        result = await collect(w.provider) if streamed else await generate(w.provider, request())
        assert result.finish_reason == expected and result.message.text == "Partial"
        assert result.message.provider_metadata == {
            "anthropic": {"stop_reason": reason, **({"stop_sequence": "END"} if reason == "stop_sequence" else {})}
        }
        assert len(w.requests) == 1


@pytest.mark.parametrize(
    "choice,parallel,expected",
    [
        ("auto", None, {"type": "auto"}),
        ("none", None, {"type": "none"}),
        ("required", True, {"type": "any", "disable_parallel_tool_use": False}),
        (None, False, {"type": "auto", "disable_parallel_tool_use": True}),
        (ToolChoice(name="none"), None, {"type": "tool", "name": "none"}),
    ],
)
async def test_tool_choice_mapping(choice: Any, parallel: bool | None, expected: dict[str, Any]) -> None:
    req = request()
    req.options.tool_choice = choice
    req.options.parallel_tool_calls = parallel
    async with Wire([httpx.Response(200, json=message())]) as w:
        await generate(w.provider, req)
        assert w.payload()["tool_choice"] == expected


@pytest.mark.parametrize("args", ["{bad", "[]", "null", '{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'])
async def test_unrepresentable_tool_history_is_rejected(args: str) -> None:
    req = request()
    req.messages.append(
        Message(role="assistant", parts=[ToolCallPart(tool_call_id="c", tool_name="f", tool_args=args)])
    )
    async with Wire([]) as w:
        with pytest.raises(UnsupportedRequestError, match="tool_args"):
            await generate(w.provider, req)
        assert w.requests == []


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "override"},
        {"messages": []},
        {"system": "override"},
        {"max_tokens": 1},
        {"tools": []},
        {"stream": True},
        {"tool_choice": {"type": "auto"}},
        {"stop_sequences": []},
        {"max_retries": 3},
        {"extra_body": {"model": "override"}},
        {"extra_query": {"stream": True}},
        {"seed": 1},
        {"container": "hosted"},
        {"cache_control": {"type": "permanent"}},
        {"cache_control": {"type": "ephemeral", "ttl": "forever"}},
        {"cache_control": {"type": "ephemeral", "other": True}},
        {"timeout": 0},
        {"extra_headers": {"x": 1}},
    ],
)
async def test_unsupported_options_fail_before_http(extra: dict[str, Any]) -> None:
    req = request()
    req.options.provider_options = extra
    async with Wire([]) as w:
        with pytest.raises(UnsupportedRequestError):
            await generate(w.provider, req)
        assert w.requests == []


@pytest.mark.parametrize(
    "msg",
    [
        Message(role="system", parts=[TextPart(text="late instruction")]),
        Message(role="user", parts=[FilePart(data="https://x.test/image.png", media_type="image/png")]),
        Message(role="assistant", parts=[ReasoningPart(text="unsigned")]),
        Message(
            role="assistant",
            parts=[ReasoningPart(text="visible", provider_metadata={"anthropic": {"redacted_data": "blob"}})],
        ),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(
                    text="thinking",
                    provider_metadata={"anthropic": {"signature": "sig", "cache_control": {"type": "ephemeral"}}},
                )
            ],
        ),
        Message(role="user", parts=[TextPart(text="text", provider_metadata={"openai": {}})]),
        Message(role="user", parts=[ToolCallPart(tool_call_id="c", tool_name="f", tool_args="{}")]),
        Message(role="assistant", parts=[ToolResultPart(tool_call_id="c", tool_name="f", result="result")]),
    ],
)
async def test_invalid_parts_and_metadata_fail_without_history_repair(msg: Message) -> None:
    req = request()
    req.messages.append(msg)
    before = req.model_dump_json()
    async with Wire([]) as w:
        with pytest.raises(UnsupportedRequestError):
            await generate(w.provider, req)
        assert w.requests == [] and req.model_dump_json() == before


async def test_explicit_max_tokens_and_credentials_are_required() -> None:
    with pytest.raises(UnsupportedRequestError, match="api_key"):
        AnthropicMessages()
    async with Wire([]) as w:
        with pytest.raises(UnsupportedRequestError):
            AnthropicMessages(client=w.client, api_key="override")
        with pytest.raises(UnsupportedRequestError):
            AnthropicMessages(client=w.client, base_url="https://other.test")
        req = request()
        req.options.max_output_tokens = None
        with pytest.raises(UnsupportedRequestError, match="max_output_tokens"):
            await generate(w.provider, req)
        req.options.max_output_tokens = 1
        req.options.tool_choice = "none"
        req.options.parallel_tool_calls = False
        with pytest.raises(UnsupportedRequestError, match="parallel"):
            await generate(w.provider, req)
        assert w.requests == []


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
async def test_http_errors_keep_cause_and_do_not_retry(streamed: bool, status: int) -> None:
    reply = httpx.Response(
        status,
        json={"type": "error", "error": {"type": "overloaded_error", "message": "fixture"}},
        headers={"request-id": "request-1"},
    )
    async with Wire([reply]) as w:
        with pytest.raises(ProviderError) as raised:
            if streamed:
                await collect(w.provider)
            else:
                await generate(w.provider, request())
        error = raised.value
        assert error.provider == "anthropic" and error.code == "overloaded_error"
        assert error.status_code == status and error.request_id == "request-1"
        assert isinstance(error.__cause__, anthropic.APIStatusError)
        assert len(w.requests) == 1 and reply.is_closed
        assert w.client.max_retries == 3 and not w.client.is_closed()


@pytest.mark.parametrize("error", [httpx.ConnectError("offline"), httpx.ReadTimeout("timeout")])
async def test_request_transport_failures_do_not_retry(error: httpx.HTTPError) -> None:
    async with Wire([error]) as w:
        with pytest.raises(ProviderError) as raised:
            await generate(w.provider, request())
        assert isinstance(raised.value.__cause__, anthropic.APIConnectionError)
        assert raised.value.__cause__.__cause__ is error and len(w.requests) == 1


async def test_sse_error_retains_partial_and_closes_without_retry() -> None:
    body = Bytes([
        start(),
        block({"type": "text", "text": ""}),
        delta("text_delta", text="Partial"),
        sse("error", error={"type": "overloaded_error", "message": "fixture"}),
    ])
    async with Wire([streaming(body)]) as w:
        async with stream(w.provider, request()) as output:
            with pytest.raises(ProviderError) as raised:
                async for _ in output:
                    pass
        assert output.message.text == "Partial" and output.response is None and output.status == "failed"
        assert raised.value.code == "overloaded_error" and isinstance(raised.value.__cause__, anthropic.APIStatusError)
        assert body.closed == 1 and len(w.requests) == 1


@pytest.mark.parametrize("with_finish", [False, True])
async def test_missing_message_stop_is_incomplete(with_finish: bool) -> None:
    chunks = [start(), block({"type": "text", "text": "Partial"}), stop()]
    if with_finish:
        chunks.append(finish())
    body = Bytes(chunks)
    async with Wire([streaming(body)]) as w:
        async with stream(w.provider, request()) as output:
            with pytest.raises(IncompleteStreamError):
                async for _ in output:
                    pass
        assert output.status == "incomplete" and output.response is None and output.message.text == "Partial"
        assert body.closed == 1


@pytest.mark.parametrize("cancel", [False, True])
async def test_early_close_and_cancel_release_response_not_borrowed_client(cancel: bool) -> None:
    body = Bytes([start(), block({"type": "text", "text": ""}), delta("text_delta", text="Hi")], wait=True)
    async with Wire([streaming(body), httpx.Response(200, json=message())]) as w:
        async with stream(w.provider, request()) as output:
            await anext(output)
            await anext(output)
            if cancel:
                task = asyncio.create_task(anext(output))
                await asyncio.wait_for(body.waiting.wait(), timeout=2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await output.aclose()
        assert output.message.text == "Hi" and output.response is None
        assert output.status == ("cancelled" if cancel else "closed")
        assert body.closed == 1 and not w.client.is_closed()
        await generate(w.provider, request())
        await w.provider.aclose()
        assert not w.client.is_closed()
        with pytest.raises(ProviderError, match="closed"):
            await generate(w.provider, request())


@pytest.mark.parametrize("streamed", [False, True])
async def test_cancellation_during_request_creation(streamed: bool) -> None:
    entered, released = asyncio.Event(), asyncio.Event()

    async def handle(req: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()
        return httpx.Response(200, json=message())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http,
        anthropic.AsyncAnthropic(api_key="fixture", http_client=http) as client,
        AnthropicMessages(client=client) as provider,
    ):
        task = asyncio.create_task(collect(provider) if streamed else generate(provider, request()))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert released.is_set() and not client.is_closed()


@pytest.mark.parametrize("status", [200, 500])
@pytest.mark.parametrize("streamed", [False, True])
async def test_owned_client_context_and_no_retry(monkeypatch: pytest.MonkeyPatch, status: int, streamed: bool) -> None:
    reply = httpx.Response(
        status, json=message() if status == 200 else {"error": {"type": "api_error", "message": "failed"}}
    )
    if status == 200 and streamed:
        reply = streaming(
            Bytes([start(), block({"type": "text", "text": "Hello"}), stop(), finish(), sse("message_stop")])
        )
    async with Wire([reply]) as w:
        real = anthropic.AsyncAnthropic
        clients = []

        def constructor(**kwargs: Any) -> anthropic.AsyncAnthropic:
            client = real(**kwargs, http_client=w.http)
            clients.append(client)
            return client

        monkeypatch.setattr(anthropic, "AsyncAnthropic", constructor)
        async with AnthropicMessages(api_key="owned", base_url="https://owned.test") as provider:
            if status == 500:
                with pytest.raises(ProviderError):
                    await collect(provider) if streamed else await generate(provider, request())
            else:
                await collect(provider) if streamed else await generate(provider, request())
            assert not clients[0].is_closed() and clients[0].max_retries == 0
            assert w.requests[0].headers["x-api-key"] == "owned"
        assert clients[0].is_closed() and reply.is_closed and len(w.requests) == 1
        await provider.aclose()


async def test_remote_read_error_preserves_cause() -> None:
    error = httpx.ReadError("fixture")
    body = Bytes([start(), block({"type": "text", "text": "Partial"})], error=error)
    async with Wire([streaming(body)]) as w:
        with pytest.raises(ProviderError) as raised:
            await collect(w.provider)
        assert raised.value.__cause__ is error and body.closed == 1 and len(w.requests) == 1


@pytest.mark.parametrize(
    "chunks",
    [
        [start(), sse("message_stop")],
        [start(), start()],
        [block({"type": "text", "text": ""})],
        [start(), block({"type": "text", "text": ""}), finish(), sse("message_stop")],
        [start(), block({"type": "text", "text": ""}), stop(), delta("text_delta", text="late")],
        [start(), block({"type": "text", "text": ""}, 2)],
        [start(), block({"type": "text", "text": ""}), delta("signature_delta", signature="wrong kind")],
        [
            start(),
            block({"type": "text", "text": ""}),
            delta("citations_delta", citation={"type": "web_search_result_location"}),
        ],
        [
            start(),
            block({"type": "tool_use", "id": "a", "name": "f", "input": {"n": 1}}),
            delta("input_json_delta", partial_json="{}"),
        ],
        [start(), block({"type": "server_tool_use", "id": "hosted", "name": "web_search", "input": {}})],
        [start(), finish("end_turn"), finish("pause_turn")],
    ],
)
async def test_invalid_stream_order_or_unsupported_blocks_fail(chunks: list[bytes]) -> None:
    body = Bytes(chunks)
    async with Wire([streaming(body)]) as w:
        with pytest.raises(ProviderError):
            await collect(w.provider)
        assert body.closed == 1 and len(w.requests) == 1


@pytest.mark.parametrize("streamed", [False, True])
async def test_malformed_wire_json_keeps_cause(streamed: bool) -> None:
    reply = (
        streaming(Bytes([b"event: message_start\ndata: {broken\n\n"]))
        if streamed
        else httpx.Response(200, content=b"{broken", headers={"content-type": "application/json"})
    )
    async with Wire([reply]) as w:
        with pytest.raises(ProviderError) as raised:
            if streamed:
                await collect(w.provider)
            else:
                await generate(w.provider, request())
        assert isinstance(raised.value.__cause__, ValueError) and reply.is_closed


@pytest.mark.parametrize(
    "raw",
    [
        message([{"type": "server_tool_use", "id": "hosted", "name": "web_search", "input": {}}]),
        message([{"type": "text", "text": "text", "unexpected_field": True}]),
        message([{"type": "thinking", "thinking": "thought", "signature": "sig", "extra": "unsupported"}]),
        message(stop_reason=None),
        message(usage={"input_tokens": -1}),
    ],
)
async def test_invalid_nonstream_output_is_not_silently_dropped(raw: dict[str, Any]) -> None:
    async with Wire([httpx.Response(200, json=raw)]) as w:
        with pytest.raises(ProviderError):
            await generate(w.provider, request())
        assert len(w.requests) == 1
