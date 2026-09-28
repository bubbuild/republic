# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Selected Responses scenarios adapted from ai-python c788059d; see NOTICE.
"""Responses contracts through the real SDK and an offline HTTP/SSE transport."""

import asyncio
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from republic import (
    FilePart,
    IncompleteStreamError,
    Message,
    Provider,
    ProviderError,
    ReasoningPart,
    Request,
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
from republic.providers.openai import OpenAIResponses
from tests.http_fixtures import Bytes, streaming
from tests.openai_fixtures import Wire, request, sse
from tests.responses_fixtures import added, call, delta, event, message, reasoning, response, terminal

pytestmark = pytest.mark.asyncio


def wire(replies: list[httpx.Response | Exception]) -> Wire:
    return Wire(replies, provider_type=OpenAIResponses)


async def collect(provider: Provider, req: Request | None = None) -> Response:
    async with stream(provider, req or request()) as output:
        async for _ in output:
            pass
    assert output.response is not None
    return output.response


async def test_generate_payload_native_items_tools_options_and_identity() -> None:
    req = Request(
        model="alias",
        messages=[
            Message(role="system", parts=[TextPart(text="rules")]),
            Message(role="user", parts=[TextPart(text="Hello")]),
            Message(
                role="assistant",
                parts=[
                    TextPart(text="Checking"),
                    ToolCallPart(tool_call_id="call_old", tool_name="weather", tool_args="{bad"),
                ],
            ),
            Message(
                role="tool", parts=[ToolResultPart(tool_call_id="call_old", tool_name="weather", result={"ok": True})]
            ),
        ],
        tools=[
            Tool(
                name="weather",
                description="Weather",
                parameters={"type": "object"},
                provider_metadata={"openai": {"strict": False}},
            )
        ],
        options=RequestOptions(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=100,
            parallel_tool_calls=True,
            tool_choice=ToolChoice(name="weather"),
            provider_options={
                "reasoning": {"effort": "low", "summary": "auto"},
                "extra_headers": {"x-fixture": "yes"},
                "timeout": 2.0,
            },
        ),
    )
    before = req.model_dump_json()
    raw = response(
        [message(), call()],
        usage={
            "input_tokens": 9,
            "output_tokens": 4,
            "total_tokens": 13,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    )
    async with wire([httpx.Response(200, json=raw)]) as w:
        result = await generate(w.provider, req)
        payload = w.payload()
        assert str(w.requests[0].url) == "https://unit.test/api/v1/responses"
        assert w.requests[0].headers["x-fixture"] == "yes"
        assert payload == {
            "model": "alias",
            "stream": False,
            "store": False,
            "truncation": "disabled",
            "include": ["reasoning.encrypted_content"],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_output_tokens": 100,
            "parallel_tool_calls": True,
            "tool_choice": {"type": "function", "name": "weather"},
            "reasoning": {"effort": "low", "summary": "auto"},
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Weather",
                    "parameters": {"type": "object"},
                    "strict": False,
                }
            ],
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": "rules"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
                {"role": "assistant", "content": "Checking"},
                {"type": "function_call", "call_id": "call_old", "name": "weather", "arguments": "{bad"},
                {"type": "function_call_output", "call_id": "call_old", "output": '{"ok": true}'},
            ],
        }
        assert result.response_id == "resp_1" and result.response_model == "resolved-model"
        assert result.finish_reason == "tool_call" and result.message.text == "Hello"
        assert result.usage is not None and result.usage.total_tokens == 13
        assert result.usage.cache_read_tokens == 2 and result.usage.reasoning_tokens == 1
        assert result.message.tool_calls[0].tool_call_id == "call_1"
        assert result.message.tool_calls[0].provider_metadata == {"openai": {"item_id": "fc_1", "raw_item": call()}}
        assert req.model_dump_json() == before
        assert len(w.requests) == 1


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("summary", [[], [{"type": "summary_text", "text": "Think"}]])
async def test_native_reasoning_round_trip_with_tools_and_full_history(
    streamed: bool, summary: list[dict[str, Any]]
) -> None:
    items = [reasoning(summary=summary), call(1), call(2, args="{broken-json")]
    body = Bytes([sse(terminal(items))])
    first = streaming(body) if streamed else httpx.Response(200, json=response(items))
    async with wire([first, httpx.Response(200, json=response())]) as w:
        result = await collect(w.provider) if streamed else await generate(w.provider, request())
        restored = Response.model_validate_json(result.model_dump_json())
        assert restored == result
        part = restored.message.parts[0]
        assert isinstance(part, ReasoningPart) and part.text == ("Think" if summary else "")
        assert part.provider_metadata == {"openai": {"item_id": "rs_1", "raw_item": items[0]}}
        req = request()
        req.messages += [
            restored.message,
            Message(
                role="tool",
                parts=[
                    ToolResultPart(tool_call_id="call_1", tool_name="weather", result="sunny"),
                    ToolResultPart(tool_call_id="call_2", tool_name="weather", result=None),
                ],
            ),
        ]
        await generate(w.provider, req)
        assert w.payload(1)["input"][1:4] == items
        assert w.payload(1)["input"][4:] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            {"type": "function_call_output", "call_id": "call_2", "output": "null"},
        ]
        assert len(w.requests) == 2  # Only the two explicit caller operations.
        if streamed:
            assert body.closed == 1


async def test_interleaved_calls_and_overlapping_done_snapshots() -> None:
    parts = [call(1, args="{bad"), call(2, args='{"n":2}')]
    chunks = [
        event("created", response=response([], status="in_progress")),
        added(call(1, args="", status="in_progress")),
        added(call(2, args="", status="in_progress"), 1),
        delta("{", item_id="fc_1", kind="function_call_arguments"),
        delta('{"n":', item_id="fc_2", index=1, kind="function_call_arguments"),
        delta("bad", item_id="fc_1", kind="function_call_arguments"),
        event("function_call_arguments.done", output_index=0, item_id="fc_1", arguments="{bad"),
        event("function_call_arguments.done", output_index=1, item_id="fc_2", arguments='{"n":2}'),
        event("output_item.done", output_index=0, item=parts[0]),
        event("output_item.done", output_index=1, item=parts[1]),
        terminal(parts),
    ]
    body = Bytes([sse(chunk) for chunk in chunks], wait=True)
    async with wire([streaming(body)]) as w:
        result = await asyncio.wait_for(collect(w.provider), timeout=2)
        assert [c.tool_args for c in result.message.tool_calls] == ["{bad", '{"n":2}']
        assert [c.tool_call_id for c in result.message.tool_calls] == ["call_1", "call_2"]
        assert body.closed == 1 and not body.waiting.is_set()
        assert w.payload()["stream"] is True
        assert not w.client.is_closed() and w.client.max_retries == 3


async def test_text_reasoning_parts_annotations_and_final_enrichment() -> None:
    raw_reasoning = reasoning(
        summary=[{"type": "summary_text", "text": "Think"}, {"type": "summary_text", "text": " twice"}],
        content=[{"type": "reasoning_text", "text": "native"}],
    )
    annotation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 2,
        "title": "Source",
        "url": "https://example.test",
    }
    raw_message = message(
        "Hi!",
        phase="final_answer",
        content=[
            {"type": "output_text", "text": "Hi!", "annotations": [annotation]},
            {"type": "refusal", "refusal": "No"},
            {"type": "output_text", "text": "Bye", "annotations": []},
        ],
    )
    chunks = [
        added(reasoning(summary=[], status="in_progress", encrypted_content=None)),
        event(
            "reasoning_summary_part.added",
            output_index=0,
            item_id="rs_1",
            summary_index=0,
            part={"type": "summary_text", "text": ""},
        ),
        delta("Thi", item_id="rs_1", kind="reasoning_summary_text", summary_index=0),
        event("reasoning_summary_text.done", output_index=0, item_id="rs_1", summary_index=0, text="Think"),
        event(
            "reasoning_summary_part.done",
            output_index=0,
            item_id="rs_1",
            summary_index=0,
            part={"type": "summary_text", "text": "Think"},
        ),
        delta("native", item_id="rs_1", kind="reasoning_text"),
        event("reasoning_text.done", output_index=0, item_id="rs_1", content_index=0, text="native"),
        added(message(content=[], status="in_progress"), 1),
        event(
            "content_part.added",
            output_index=1,
            item_id="msg_1",
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        ),
        delta("Hi", index=1),
        event(
            "output_text.annotation.added",
            output_index=1,
            item_id="msg_1",
            content_index=0,
            annotation_index=0,
            annotation=annotation,
        ),
        event("output_text.done", output_index=1, item_id="msg_1", content_index=0, text="Hi!"),
        event("content_part.done", output_index=1, item_id="msg_1", content_index=0, part=raw_message["content"][0]),
        delta("N", index=1, kind="refusal", content_index=1),
        event("refusal.done", output_index=1, item_id="msg_1", content_index=1, refusal="No"),
        terminal([raw_reasoning, raw_message], usage={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}),
    ]
    async with wire([streaming(Bytes([sse(chunk) for chunk in chunks]))]) as w:
        result = await collect(w.provider)
        assert result.message.parts[0] == ReasoningPart(
            text="Think twice", provider_metadata={"openai": {"item_id": "rs_1", "raw_item": raw_reasoning}}
        )
        assert result.message.parts[1] == TextPart(
            text="Hi!Bye", provider_metadata={"openai": {"item_id": "msg_1", "raw_item": raw_message}}
        )
        assert result.usage is not None and result.usage.total_tokens == 8


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    count: int


@pytest.mark.parametrize("text,valid", [('{"count":2}', True), ('{"count":"wrong"}', False)])
async def test_structured_output_is_explicit_caller_validation(text: str, valid: bool) -> None:
    req = request()
    format_config = {"type": "json_schema", "name": "Answer", "strict": True, "schema": Answer.model_json_schema()}
    req.options.provider_options = {"text": {"format": format_config}}
    async with wire([httpx.Response(200, json=response([message(text)]))]) as w:
        result = await generate(w.provider, req)
        assert w.payload()["text"] == {"format": format_config}
        if valid:
            assert Answer.model_validate_json(result.message.text).count == 2
        else:
            with pytest.raises(ValidationError):
                Answer.model_validate_json(result.message.text)
        assert len(w.requests) == 1


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "status,details,finish",
    [
        ("completed", {}, "stop"),
        ("incomplete", {"incomplete_details": {"reason": "max_output_tokens"}}, "length"),
        ("incomplete", {"incomplete_details": {"reason": "content_filter"}}, "content_filter"),
        ("incomplete", {"incomplete_details": {"reason": "future"}}, "other"),
        ("failed", {"error": {"code": "server_error", "message": "failure"}}, "error"),
    ],
)
async def test_terminal_outcomes_preserve_partial_data(
    streamed: bool, status: str, details: dict[str, Any], finish: str
) -> None:
    raw = response(
        [message("Partial", status="incomplete" if status != "completed" else "completed")], status=status, **details
    )
    reply = streaming(Bytes([sse(event(status, response=raw))])) if streamed else httpx.Response(200, json=raw)
    async with wire([reply]) as w:
        result = await collect(w.provider) if streamed else await generate(w.provider, request())
        assert result.finish_reason == finish and result.message.text == "Partial"
        assert result.message.provider_metadata is not None
        assert result.message.provider_metadata["openai"] == {
            "response_id": "resp_1",
            "model": "resolved-model",
            "status": status,
            **details,
        }
        assert result.usage is None


@pytest.mark.parametrize("status", ["queued", "in_progress", "cancelled", "future"])
async def test_generate_rejects_nonterminal_response(status: str) -> None:
    async with wire([httpx.Response(200, json=response(status=status))]) as w:
        with pytest.raises(ProviderError, match="nonterminal"):
            await generate(w.provider, request())
        assert len(w.requests) == 1


async def test_missing_terminal_retains_deltas_and_closes() -> None:
    body = Bytes([
        sse(event("queued", response=response([], status="queued"))),
        sse(added(message(content=[], status="in_progress"))),
        sse(delta("Partial")),
        sse("[DONE]"),
    ])
    async with wire([streaming(body)]) as w:
        async with stream(w.provider, request()) as output:
            with pytest.raises(IncompleteStreamError):
                async for _ in output:
                    pass
        assert output.message.text == "Partial" and output.response is None and output.status == "incomplete"
        assert body.closed == 1 and not w.client.is_closed()


@pytest.mark.parametrize(
    "tail",
    [
        event("output_text.done", output_index=0, item_id="msg_1", content_index=0, text="different"),
        event("output_item.done", output_index=0, item=message("different")),
        terminal([message("different")]),
        terminal([]),
        event("completed", response=response(status="in_progress")),
        delta("oops", item_id="other"),
        added({"type": "web_search_call", "id": "ws_1"}, 1),
        event("audio.delta", output_index=0, item_id="msg_1", delta="audio"),
    ],
)
async def test_protocol_conflicts_fail_without_repair(tail: dict[str, Any]) -> None:
    body = Bytes([sse(added(message(content=[], status="in_progress"))), sse(delta("Hi")), sse(tail)])
    async with wire([streaming(body)]) as w:
        async with stream(w.provider, request()) as output:
            with pytest.raises(ProviderError):
                async for _ in output:
                    pass
        assert output.message.text == "Hi" and output.response is None and output.status == "failed"
        assert body.closed == 1 and len(w.requests) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"model": "override"},
        {"input": []},
        {"tools": []},
        {"stream": False},
        {"max_output_tokens": 2},
        {"conversation": "conv_1"},
        {"background": True},
        {"extra_body": {"model": "override"}},
        {"seed": 2},
        {"include": "reasoning.encrypted_content"},
        {"extra_headers": {"x": 1}},
        {"timeout": 0},
    ],
)
async def test_unsupported_options_fail_before_http(options: dict[str, Any]) -> None:
    req = request()
    req.options.provider_options = options
    async with wire([]) as w:
        with pytest.raises(UnsupportedRequestError):
            await generate(w.provider, req)
        assert w.requests == []


@pytest.mark.parametrize(
    "msg",
    [
        Message(role="user", parts=[FilePart(data="https://x.test/audio.wav", media_type="audio/wav")]),
        Message(role="user", parts=[ReasoningPart(text="thought")]),
        Message(role="assistant", parts=[ReasoningPart(text="unbacked")]),
        Message(
            role="assistant",
            parts=[TextPart(text="changed", provider_metadata={"openai": {"item_id": "msg_1", "raw_item": message()}})],
        ),
        Message(role="assistant", parts=[ToolCallPart(tool_call_id="", tool_name="tool", tool_args="{}")]),
        Message(
            role="tool", parts=[ToolResultPart(tool_call_id="call", tool_name="tool", result="failed", is_error=True)]
        ),
        Message(role="user", parts=[TextPart(text="text", provider_metadata={"other": {}})]),
    ],
)
async def test_unsupported_history_fails_before_http(msg: Message) -> None:
    async with wire([]) as w:
        with pytest.raises(UnsupportedRequestError):
            await generate(w.provider, Request(model="model", messages=[msg]))
        assert w.requests == []


@pytest.mark.parametrize("status", [401, 429, 500])
@pytest.mark.parametrize("streamed", [False, True])
async def test_sdk_errors_keep_cause_no_retries(status: int, streamed: bool) -> None:
    reply = httpx.Response(
        status,
        json={"error": {"message": "failed", "type": "api_error", "code": "fixture"}},
        headers={"x-request-id": "request-1"},
    )
    async with wire([reply]) as w:
        with pytest.raises(ProviderError) as raised:
            if streamed:
                await collect(w.provider)
            else:
                await generate(w.provider, request())
        assert isinstance(raised.value.__cause__, openai.APIStatusError)
        assert raised.value.status_code == status and raised.value.request_id == "request-1"
        assert raised.value.code == "fixture" and len(w.requests) == 1 and reply.is_closed


@pytest.mark.parametrize("native", [True, False])
async def test_stream_error_events_release_wire(native: bool) -> None:
    failure = (
        {"type": "error", "message": "failed", "code": "fixture", "sequence_number": 2}
        if native
        else {"error": {"message": "failed", "code": "fixture"}}
    )
    body = Bytes([sse(added(message(content=[], status="in_progress"))), sse(delta("Hi")), sse(failure)])
    async with wire([streaming(body)]) as w:
        async with stream(w.provider, request()) as output:
            with pytest.raises(ProviderError) as raised:
                async for _ in output:
                    pass
        assert output.status == "failed" and output.message.text == "Hi"
        assert raised.value.code == "fixture"
        if not native:
            assert isinstance(raised.value.__cause__, openai.APIError)
        assert body.closed == 1 and not w.client.is_closed()


@pytest.mark.parametrize("cancel", [False, True])
async def test_early_close_and_cancellation_release_only_response(cancel: bool) -> None:
    body = Bytes([sse(added(message(content=[], status="in_progress"))), sse(delta("Hi"))], wait=True)
    async with wire([streaming(body)]) as w:
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
        assert output.status == ("cancelled" if cancel else "closed")
        assert output.message.text == "Hi" and output.response is None
        assert body.closed == 1 and not w.client.is_closed()
        await w.provider.aclose()
        assert not w.client.is_closed()
        with pytest.raises(ProviderError, match="closed"):
            await generate(w.provider, request())


async def test_remote_read_error_keeps_cause_and_releases() -> None:
    error = httpx.ReadError("fixture")
    body = Bytes([sse(added(message(content=[], status="in_progress"))), sse(delta("Hi"))], error=error)
    async with wire([streaming(body)]) as w:
        async with stream(w.provider, request()) as output:
            with pytest.raises(ProviderError) as raised:
                async for _ in output:
                    pass
        assert raised.value.__cause__ is error and output.message.text == "Hi"
        assert body.closed == 1 and len(w.requests) == 1


@pytest.mark.parametrize("streamed", [False, True])
async def test_owned_client_lifecycle(monkeypatch: pytest.MonkeyPatch, streamed: bool) -> None:
    reply = streaming(Bytes([sse(terminal())])) if streamed else httpx.Response(200, json=response())
    async with wire([reply]) as w:
        real = openai.AsyncOpenAI
        clients = []

        def constructor(**kwargs: Any) -> openai.AsyncOpenAI:
            client = real(**kwargs, http_client=w.http)
            clients.append(client)
            return client

        monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
        async with OpenAIResponses(api_key="owned", base_url="https://owned.test/v1") as provider:
            result = await collect(provider) if streamed else await generate(provider, request())
            assert result.message.text == "Hello"
            assert not clients[0].is_closed() and clients[0].max_retries == 0
            assert w.requests[0].headers["authorization"] == "Bearer owned"
            assert str(w.requests[0].url) == "https://owned.test/v1/responses"
        assert clients[0].is_closed() and reply.is_closed
        await provider.aclose()


@pytest.mark.parametrize("streamed", [False, True])
async def test_cancel_during_request_creation(streamed: bool) -> None:
    entered, released = asyncio.Event(), asyncio.Event()

    async def handle(req: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()
        return httpx.Response(200, json=response())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http,
        openai.AsyncOpenAI(api_key="fixture", http_client=http) as client,
        OpenAIResponses(client=client) as provider,
    ):
        operation = collect(provider) if streamed else generate(provider, request())
        task = asyncio.create_task(operation)
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert released.is_set() and not client.is_closed()


@pytest.mark.parametrize("streamed", [False, True])
async def test_malformed_wire_json_preserves_cause(streamed: bool) -> None:
    reply = (
        streaming(Bytes([sse("{bad-json")]))
        if streamed
        else httpx.Response(200, content=b"{bad-json", headers={"content-type": "application/json"})
    )
    async with wire([reply]) as w:
        with pytest.raises(ProviderError) as raised:
            if streamed:
                await collect(w.provider)
            else:
                await generate(w.provider, request())
        assert isinstance(raised.value.__cause__, ValueError)
        assert reply.is_closed and len(w.requests) == 1


async def test_stop_and_conflicting_client_settings_are_rejected() -> None:
    async with wire([]) as w:
        override = OpenAIResponses(client=w.client, api_key="override")
        assert override._client.api_key == "override"
        assert w.client.api_key == "fixture-key"
        await override.aclose()
        req = request()
        req.options.stop = ["END"]
        with pytest.raises(UnsupportedRequestError, match="stop"):
            await generate(w.provider, req)
        assert not w.client.is_closed() and w.requests == []


@pytest.mark.parametrize(
    "tail",
    [
        delta("!"),
        event("output_text.done", output_index=0, item_id="msg_1", content_index=0, text="Hi!"),
        event(
            "content_part.done",
            output_index=0,
            item_id="msg_1",
            content_index=0,
            part={"type": "output_text", "text": "Hi!", "annotations": []},
        ),
        terminal([message("Hi!")]),
    ],
)
async def test_closed_text_cannot_grow(tail: dict[str, Any]) -> None:
    body = Bytes([
        sse(added(message(content=[], status="in_progress"))),
        sse(delta("Hi")),
        sse(event("output_text.done", output_index=0, item_id="msg_1", content_index=0, text="Hi")),
        sse(tail),
    ])
    async with wire([streaming(body)]) as w:
        with pytest.raises(ProviderError):
            await collect(w.provider)
        assert body.closed == 1


async def test_pending_call_identity_and_final_snapshot_keep_order() -> None:
    header = {"type": "function_call", "id": "fc_1", "arguments": "", "status": "in_progress"}
    body = Bytes([
        sse(added(header)),
        sse(delta("{", item_id="fc_1", kind="function_call_arguments")),
        sse(added(message(content=[], status="in_progress"), 1)),
        sse(delta("Hi", index=1)),
        sse(event("output_item.done", output_index=0, item=call(args="{bad"))),
        sse(terminal([call(args="{bad"), message("Hi")])),
    ])
    async with wire([streaming(body)]) as w:
        result = await collect(w.provider)
        assert [part.kind for part in result.message.parts] == ["tool_call", "text"]
        assert result.message.tool_calls[0].tool_call_id == "call_1"
        assert result.message.tool_calls[0].tool_args == "{bad"


async def test_interleaved_content_slots_preserve_native_order() -> None:
    parts = [
        {"type": "output_text", "text": "AC", "annotations": []},
        {"type": "output_text", "text": "B", "annotations": []},
    ]
    body = Bytes([
        sse(added(message(content=[], status="in_progress"))),
        sse(delta("A")),
        sse(delta("B", content_index=1)),
        sse(delta("C")),
        sse(terminal([message(content=parts)])),
    ])
    async with wire([streaming(body)]) as w:
        result = await collect(w.provider)
        assert result.message.text == "ACB"
        assert result.message.parts[0].provider_metadata == {
            "openai": {"item_id": "msg_1", "raw_item": message(content=parts)}
        }


@pytest.mark.parametrize(
    "items",
    [
        [call(call_id="")],
        [call(), call()],
        [message(role="user")],
        [{"type": "image_generation_call", "id": "ig_1", "result": "base64"}],
        [reasoning(summary=[{"type": "future", "text": "opaque"}])],
    ],
)
async def test_invalid_or_unsupported_output_is_explicit(items: list[dict[str, Any]]) -> None:
    async with wire([httpx.Response(200, json=response(items))]) as w:
        with pytest.raises(ProviderError):
            await generate(w.provider, request())
        assert len(w.requests) == 1
