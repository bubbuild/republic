"""Grok proxy protocol through official SDK plus synthetic SSE, never live."""

import asyncio
import traceback
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

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
    ToolChoice,
    ToolResultPart,
    UnsupportedRequestError,
    generate,
    stream,
)
from republic.auth.grok import GrokAuthError
from republic.providers.grok import GrokOAuth
from tests.grok_fixtures import Wire, tokens
from tests.http_fixtures import Bytes, Transport, streaming
from tests.openai_fixtures import request, sse
from tests.responses_fixtures import added, call, delta, event, message, reasoning, terminal

pytestmark = pytest.mark.asyncio


async def test_proxy_payload_headers_and_history_order_with_borrowed_client() -> None:
    req = request()
    req.messages.insert(0, Message(role="system", parts=[TextPart(text="Leading")]))
    req.messages.append(Message(role="system", parts=[TextPart(text="Later")]))
    req.tools = [Tool(name="weather", parameters={"type": "object"})]
    req.options = RequestOptions(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=100,
        tool_choice=ToolChoice(name="weather"),
        provider_options={
            "reasoning": {"effort": "low", "summary": "auto"},
            "prompt_cache_key": "fixture",
            "timeout": 2,
        },
    )
    before = req.model_dump_json()
    body = Bytes([sse(terminal())])
    async with Wire([streaming(body)]) as wire:
        result = await generate(wire.provider, req)
        assert result.response_id == "resp_1" and result.response_model == "resolved-model"
        assert len(wire.requests) == 1 and body.closed == 1
        sent = wire.requests[0]
        assert sent.method == "POST" and str(sent.url) == "https://cli-chat-proxy.grok.com/v1/responses"
        assert sent.headers["authorization"] == f"Bearer {tokens().access_token}"
        assert sent.headers["x-xai-token-auth"] == "xai-grok-cli"
        assert sent.headers["x-authenticateresponse"] == "authenticate-response"
        assert sent.headers["x-grok-client-version"] == "1.0.41"
        assert sent.headers["x-grok-client-mode"] == "headless"
        assert sent.headers["x-grok-client-identifier"] == sent.headers["user-agent"] == "republic"
        assert sent.headers["x-grok-model-override"] == req.model
        assert sent.headers["accept"] == "text/event-stream"
        assert not {"openai-organization", "openai-project", "x-unrelated"} & sent.headers.keys()
        assert wire.payload() == {
            "model": req.model,
            "stream": True,
            "store": False,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": "Leading"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
                {"role": "system", "content": [{"type": "input_text", "text": "Later"}]},
            ],
            "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "weather"},
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 100,
            "reasoning": {"effort": "low", "summary": "auto"},
            "prompt_cache_key": "fixture",
            "include": ["reasoning.encrypted_content"],
        }
        await wire.provider.aclose()
        assert not wire.client.is_closed() and wire.client.max_retries == 4
        assert wire.client.organization == "unrelated-org" and wire.client.project == "unrelated-project"
        assert str(wire.client.base_url) == "https://unrelated.test/v1/" and wire.client.api_key == "unrelated-api-key"
        with pytest.raises(ProviderError, match="closed"):
            await generate(wire.provider, req)
        assert len(wire.requests) == 1
    assert req.model_dump_json() == before


async def test_native_reasoning_parallel_calls_snapshots_and_history_replay() -> None:
    final_items = [reasoning(summary=[]), call(args='{"x":'), message("Hello"), call(2, args="{}")]
    chunks = [
        added(reasoning(summary=[], status="in_progress")),
        added(call(args="", status="in_progress"), 1),
        added(message("", status="in_progress"), 2),
        added(call(2, args="", status="in_progress"), 3),
        delta('{"x":', item_id="fc_1", index=1, kind="function_call_arguments"),
        delta("Hel", item_id="msg_1", index=2),
        delta("{}", item_id="fc_2", index=3, kind="function_call_arguments"),
        delta("lo", item_id="msg_1", index=2),
        event("output_text.done", output_index=2, item_id="msg_1", content_index=0, text="Hello"),
        terminal(
            final_items,
            usage={
                "input_tokens": 20,
                "output_tokens": 9,
                "total_tokens": 29,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        ),
    ]
    body = Bytes([sse(item) for item in chunks])
    async with Wire([streaming(body), streaming(Bytes([sse(terminal())]))]) as wire:
        first = await generate(wire.provider, request())
        assert first.finish_reason == "tool_call" and first.message.text == "Hello"
        assert first.usage is not None and first.usage.total_tokens == 29 and first.usage.reasoning_tokens == 5
        assert isinstance(first.message.parts[0], ReasoningPart)
        assert first.message.parts[0].text == ""
        assert [part.tool_args for part in first.message.tool_calls] == ['{"x":', "{}"]
        restored = Response.model_validate_json(first.model_dump_json())
        req = request()
        req.messages += [
            restored.message,
            Message(
                role="tool",
                parts=[
                    ToolResultPart(tool_call_id="call_1", tool_name="weather", result="caller handled invalid JSON"),
                    ToolResultPart(tool_call_id="call_2", tool_name="weather", result={"ok": True}),
                ],
            ),
        ]
        await generate(wire.provider, req)
        assert wire.payload(1)["input"][1:5] == final_items
        assert wire.payload(1)["input"][5:] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "caller handled invalid JSON"},
            {"type": "function_call_output", "call_id": "call_2", "output": '{"ok": true}'},
        ]
        assert len(wire.requests) == 2 and body.closed == 1


@pytest.mark.parametrize(
    "status,details,finish",
    [
        ("completed", None, "stop"),
        ("incomplete", {"reason": "max_output_tokens"}, "length"),
        ("incomplete", {"reason": "future_reason"}, "other"),
        ("failed", None, "error"),
    ],
)
async def test_terminal_status_and_no_continuation(status: str, details: Any, finish: str) -> None:
    body = Bytes([
        sse(
            terminal(
                status=status, incomplete_details=details, error={"code": "server", "message": tokens().refresh_token}
            )
        )
    ])
    async with Wire([streaming(body)]) as wire:
        async with stream(wire.provider, request()) as output:
            async for _ in output:
                pass
        result = output.response
        assert result is not None and result.finish_reason == finish
        assert len(wire.requests) == 1 and body.closed == 1
        if status == "failed":
            assert "private-refresh" not in result.model_dump_json()


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_http_errors_do_not_refresh_retry_or_leak_secrets(status: int) -> None:
    credentials = tokens()
    async with Wire([
        httpx.Response(status, json={"error": {"message": credentials.access_token, "code": credentials.refresh_token}})
    ]) as wire:
        with pytest.raises(ProviderError) as caught:
            await generate(wire.provider, request())
        assert caught.value.status_code == status and caught.value.provider == "grok"
        assert (
            caught.value.code
            == {401: "unauthorized", 403: "forbidden", 429: "rate_limit", 500: "request_failed"}[status]
        )
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        rendered = "".join(traceback.format_exception(caught.value))
        assert credentials.refresh_token is not None
        assert credentials.access_token not in rendered and credentials.refresh_token not in rendered
        assert len(wire.requests) == 1 and not wire.client.is_closed()


@pytest.mark.parametrize("mode", ["error", "disconnect", "missing", "conflict"])
async def test_stream_failures_close_without_fallback(mode: str) -> None:
    chunks = [added(message("", status="in_progress")), delta("Hi")]
    if mode == "error":
        chunks += [{"type": "error", "code": "fixture", "message": tokens().access_token, "sequence_number": 3}]
    elif mode == "conflict":
        chunks += [terminal([message("Contradiction")])]
    body = Bytes(
        [sse(item) for item in chunks], error=httpx.ReadError("disconnected") if mode == "disconnect" else None
    )
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(IncompleteStreamError if mode == "missing" else ProviderError) as caught:
            await generate(wire.provider, request())
        assert tokens().access_token not in str(caught.value)
        assert len(wire.requests) == 1 and body.closed == 1 and not wire.client.is_closed()


@pytest.mark.parametrize("mode", ["early", "cancel", "generate_cancel"])
async def test_close_and_cancel_leave_client_reusable(mode: str) -> None:
    body = Bytes([sse(added(message("", status="in_progress")))], wait=True)
    async with Wire([streaming(body), streaming(Bytes([sse(terminal())]))]) as wire:
        if mode == "generate_cancel":
            task = asyncio.create_task(generate(wire.provider, request()))
            await asyncio.wait_for(body.waiting.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            async with stream(wire.provider, request()) as output:
                await anext(output)
                if mode == "cancel":
                    task = asyncio.create_task(anext(output))
                    await asyncio.wait_for(body.waiting.wait(), 3)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
            assert output.response is None
            assert output.status == ("cancelled" if mode == "cancel" else "closed")
        assert body.closed == 1 and not wire.client.is_closed()
        await generate(wire.provider, request())
        assert len(wire.requests) == 2


@pytest.mark.parametrize("status", [200, 401])
async def test_owned_client_close(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    reply = streaming(Bytes([sse(terminal())])) if status == 200 else httpx.Response(401, json={"error": "denied"})
    transport = Transport([reply])
    clients = []

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(**kwargs, transport=transport)
        clients.append(client)
        return client

    # Inject only the transport at construction; the SDK request/stream is real.
    monkeypatch.setattr("republic.providers.grok.openai.DefaultAsyncHttpxClient", factory)
    async with GrokOAuth(tokens(), client_version="1.0.41") as provider:
        if status == 200:
            await generate(provider, request())
        else:
            with pytest.raises(ProviderError):
                await generate(provider, request())
        assert not clients[0].is_closed
        assert not clients[0].follow_redirects
    await provider.aclose()
    assert clients[0].is_closed and transport.closed == 1


async def test_redirects_cannot_send_credentials_or_inference_twice() -> None:
    async with Wire([httpx.Response(307, headers={"location": "https://other.test/responses"})]) as wire:
        with pytest.raises(ProviderError):
            await generate(wire.provider, request())
        assert len(wire.requests) == 1
    async with httpx.AsyncClient(follow_redirects=True, transport=Transport([])) as http:
        client = openai.AsyncOpenAI(api_key="fixture-key", http_client=http)
        with pytest.raises(UnsupportedRequestError, match="follow_redirects"):
            GrokOAuth(tokens(), client_version="1.0.41", client=client)
        assert not client.is_closed()


async def test_reasoning_content_and_summary_deltas_cost_usage_and_roundtrip() -> None:
    native = reasoning("Think", content=[{"type": "reasoning_text", "text": "Raw thought"}])
    chunks = [
        added(reasoning("", status="in_progress", content=[])),
        event("reasoning_summary_text.delta", item_id="rs_1", output_index=0, summary_index=0, delta="Thi"),
        event("reasoning_summary_text.delta", item_id="rs_1", output_index=0, summary_index=0, delta="nk"),
        event("reasoning_text.delta", item_id="rs_1", output_index=0, content_index=0, delta="Raw thought"),
        event("output_item.done", output_index=0, item=native),
        terminal(
            [native],
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cost_in_usd_ticks": 42,
                "context_details": {"input_tokens": 60, "output_tokens": 10},
                "input_tokens_details": {"cached_tokens": 50},
            },
        ),
    ]
    async with Wire([streaming(Bytes([sse(item) for item in chunks])), streaming(Bytes([sse(terminal())]))]) as wire:
        result = await generate(wire.provider, request())
        assert result.usage is not None and result.usage.total_tokens == 120
        assert result.usage.raw is not None
        assert result.usage.raw["cost_in_usd_ticks"] == 42 and result.usage.cache_read_tokens == 50
        assert result.usage.raw["context_details"] == {"input_tokens": 60, "output_tokens": 10}
        restored = Response.model_validate_json(result.model_dump_json())
        part = restored.message.parts[0]
        assert isinstance(part, ReasoningPart) and part.text == "Think"
        req = request()
        req.messages.append(restored.message)
        await generate(wire.provider, req)
        assert wire.payload(1)["input"][-1] == native
        assert len(wire.requests) == 2


async def test_structured_output_is_caller_validated_without_repair() -> None:
    class Answer(BaseModel):
        value: int

    req = request()
    req.options = RequestOptions(
        provider_options={
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": Answer.model_json_schema(),
                }
            }
        }
    )
    async with Wire([streaming(Bytes([sse(terminal([message('{"value":"wrong"}')]))]))]) as wire:
        result = await generate(wire.provider, req)
        with pytest.raises(ValidationError):
            Answer.model_validate_json(result.message.text)
        assert wire.payload()["text"] == req.options.provider_options["text"] and len(wire.requests) == 1


async def test_expired_tokens_never_trigger_auth_or_request() -> None:
    async with Wire([], tokens(expires_at=1)) as wire:
        with pytest.raises(GrokAuthError, match="expired"):
            await generate(wire.provider, request())
        assert not wire.requests


@pytest.mark.parametrize(
    "options",
    [
        RequestOptions(stop=[]),
        RequestOptions(parallel_tool_calls=False),
        *[
            RequestOptions(provider_options={key: value})
            for key, value in {
                "store": True,
                "include": [],
                "model": "override",
                "input": [],
                "stream": False,
                "instructions": "override",
                "truncation": "disabled",
                "previous_response_id": "old",
                "extra_headers": {"authorization": "override"},
                "extra_body": {},
                "service_tier": "default",
                "metadata": {},
                "user": "user",
                "tools": [],
            }.items()
        ],
    ],
)
async def test_unsupported_options_are_rejected_without_requests(options: RequestOptions) -> None:
    req = request()
    req.options = options
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests


@pytest.mark.parametrize(
    "part",
    [
        FilePart(data="https://unit.test/image.png", media_type="image/png"),
        TextPart(text="x", provider_metadata={"unsupported": True}),
        ReasoningPart(text="unretained"),
    ],
)
async def test_unsupported_parts_do_not_mutate_or_repair_history(part: Any) -> None:
    req = request()
    req.messages.append(Message(role="user", parts=[part]))
    before = req.model_dump_json()
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests
    assert req.model_dump_json() == before


async def test_model_cannot_inject_routing_header() -> None:
    req = request()
    req.model = "model\nAuthorization: invalid"
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError, match="model"):
            await generate(wire.provider, req)
        assert not wire.requests
