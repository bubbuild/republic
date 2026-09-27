"""Copilot Chat Completions through the real OpenAI SDK; synthetic HTTP only."""

import asyncio
import logging
import traceback
from importlib.metadata import version
from typing import Any, cast

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
    ToolChoice,
    ToolResultPart,
    UnsupportedRequestError,
    generate,
    stream,
)
from republic.auth.github_copilot import CopilotAuthError
from republic.providers.github_copilot import GitHubCopilot
from tests.copilot_fixtures import Wire, login, token
from tests.http_fixtures import Bytes, Transport, streaming
from tests.openai_fixtures import chunk, completion, request, sse

pytestmark = pytest.mark.asyncio


async def test_single_nonstream_http_endpoint_headers_options_and_borrowed_client() -> None:
    req = request()
    req.messages.insert(0, Message(role="system", parts=[TextPart(text="Be brief")]))
    req.tools = [Tool(name="lookup", description="Caller owned", parameters={"type": "object"})]
    req.options = RequestOptions(
        temperature=0.5,
        top_p=0.9,
        max_output_tokens=80,
        stop=["END"],
        tool_choice=ToolChoice(name="lookup"),
        provider_options={
            "timeout": 5,
        },
    )
    before = req.model_dump_json()
    async with Wire([httpx.Response(200, json=completion())]) as wire:
        result = await generate(wire.provider, req)
        assert result.message.text == "Hello" and result.response_id == "chat-1"
        assert result.response_model == "resolved-model" and len(wire.requests) == 1
        sent = wire.requests[0]
        assert str(sent.url) == "https://api.individual.githubcopilot.com/chat/completions"
        assert sent.method == "POST" and sent.headers["authorization"] == "Bearer private-copilot"
        assert sent.headers["copilot-integration-id"] == "fixture-integration"
        assert (
            sent.headers["editor-version"] == sent.headers["editor-plugin-version"] == f"republic/{version('republic')}"
        )
        assert sent.headers["x-github-api-version"] == "2025-10-01"
        assert sent.headers["openai-intent"] == "conversation-panel" and sent.headers["user-agent"] == "republic"
        assert not {"openai-organization", "openai-project", "x-unrelated"} & sent.headers.keys()
        assert wire.payload() == {
            "model": req.model,
            "messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 80,
            "stop": ["END"],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "description": "Caller owned", "parameters": {"type": "object"}},
                }
            ],
        }
        await wire.provider.aclose()
        assert not wire.client.is_closed() and wire.client.max_retries == 4
        assert wire.client.organization == "unrelated-org" and wire.client.api_key == "unrelated-api-key"
        assert str(wire.client.base_url) == "https://unrelated.test/v1/"
        assert wire.client.default_headers["x-unrelated"] == "yes"
        with pytest.raises(ProviderError, match="closed"):
            await generate(wire.provider, req)
    assert req.model_dump_json() == before


async def test_interleaved_tools_metadata_usage_and_explicit_result_history() -> None:
    chunks = [
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"x":'}}]}),
        chunk({
            "tool_calls": [
                {"index": 1, "id": "call_b", "type": "function", "function": {"name": "lookup", "arguments": "{"}}
            ]
        }),
        chunk({"content": "Hi"}, system_fingerprint="fixture-version"),
        chunk({
            "tool_calls": [
                {"index": 0, "id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": "1"}},
                {"index": 1, "function": {"arguments": "}"}},
            ]
        }),
        chunk(finish="tool_calls"),
        chunk(
            choices=[],
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "total_tokens": 21,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        ),
    ]
    body = Bytes([*[sse(item) for item in chunks], sse("[DONE]")])
    async with Wire([streaming(body), httpx.Response(200, json=completion())]) as wire:
        async with stream(wire.provider, request()) as output:
            async for _ in output:
                pass
        result = output.response
        assert result is not None and result.finish_reason == "tool_call" and result.message.text == "Hi"
        assert result.usage is not None and result.usage.total_tokens == 21
        assert result.usage.cache_read_tokens == 3 and result.usage.reasoning_tokens == 2
        assert result.message.provider_metadata == {"openai": {"system_fingerprint": "fixture-version"}}
        assert [call.tool_args for call in result.message.tool_calls] == ['{"x":1', "{}"]  # No JSON repair.
        assert body.closed == 1 and len(wire.requests) == 1
        assert wire.payload()["stream_options"] == {"include_usage": True}
        restored = Response.model_validate_json(result.model_dump_json())
        req = request()
        req.messages += [
            restored.message,
            Message(
                role="tool",
                parts=[
                    ToolResultPart(
                        tool_call_id="call_a", tool_name="lookup", result="caller dealt with malformed JSON"
                    ),
                    ToolResultPart(tool_call_id="call_b", tool_name="lookup", result={"ok": True}),
                ],
            ),
        ]
        await generate(wire.provider, req)
        history = wire.payload(1)["messages"]
        assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"x":1'
        assert history[2:] == [
            {"role": "tool", "tool_call_id": "call_a", "content": "caller dealt with malformed JSON"},
            {"role": "tool", "tool_call_id": "call_b", "content": '{"ok": true}'},
        ]
        assert len(wire.requests) == 2


@pytest.mark.parametrize(
    "reason,expected",
    [("stop", "stop"), ("length", "length"), ("content_filter", "content_filter"), ("future", "other")],
)
async def test_finish_reasons_do_not_continue(reason: str, expected: str) -> None:
    async with Wire([httpx.Response(200, json=completion(finish=reason))]) as wire:
        result = await generate(wire.provider, request())
        assert result.finish_reason == expected and len(wire.requests) == 1
        if reason == "future":
            assert result.message.provider_metadata == {"openai": {"finish_reason": reason}}


@pytest.mark.parametrize("status", [401, 403, 429, 500])
@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_http_errors_no_retry_exchange_redirect_or_secret_leak(
    status: int, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    secret = token().token
    reply = httpx.Response(status, json={"error": {"message": secret, "code": secret}})
    async with Wire([reply]) as wire:
        with pytest.raises(ProviderError) as caught:
            if mode == "generate":
                await generate(wire.provider, request())
            else:
                async with stream(wire.provider, request()) as output:
                    async for _ in output:
                        pass
        assert caught.value.provider == "github-copilot" and caught.value.status_code == status
        assert (
            caught.value.code
            == {401: "unauthorized", 403: "forbidden", 429: "rate_limit", 500: "request_failed"}[status]
        )
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        assert secret not in "".join(traceback.format_exception(caught.value)) + caplog.text
        assert len(wire.requests) == 1 and not wire.client.is_closed()


@pytest.mark.parametrize("mode", ["missing", "error", "disconnect", "invalid"])
async def test_stream_failure_closes_and_preserves_partial_text(mode: str) -> None:
    chunks = [chunk({"content": "partial"})]
    if mode == "error":
        chunks.append({"error": {"message": token().token, "type": "server_error"}})
    elif mode == "invalid":
        chunks.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, finish="tool_calls"))
    body = Bytes([sse(item) for item in chunks], error=httpx.ReadError(token().token) if mode == "disconnect" else None)
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(IncompleteStreamError if mode == "missing" else ProviderError) as caught:
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert output.message.text == "partial" and output.response is None
        assert token().token not in "".join(traceback.format_exception(caught.value))
        assert body.closed == 1 and len(wire.requests) == 1 and not wire.client.is_closed()


@pytest.mark.parametrize("mode", ["early", "cancel"])
async def test_stream_close_cancel_and_client_reuse(mode: str) -> None:
    body = Bytes([sse(chunk({"content": "Hello"}))], wait=True)
    async with Wire([streaming(body), httpx.Response(200, json=completion())]) as wire:
        async with stream(wire.provider, request()) as output:
            await anext(output)  # TextStart.
            await anext(output)  # TextDelta.
            if mode == "cancel":
                task = asyncio.create_task(anext(output))
                await asyncio.wait_for(body.waiting.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert output.status == ("cancelled" if mode == "cancel" else "closed") and output.response is None
        assert body.closed == 1 and not wire.client.is_closed()
        await generate(wire.provider, request())
        assert len(wire.requests) == 2


async def test_generate_cancel_closes_response() -> None:
    body = Bytes([], wait=True)
    async with Wire([httpx.Response(200, stream=body)]) as wire:
        task = asyncio.create_task(generate(wire.provider, request()))
        await asyncio.wait_for(body.waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed == 1 and len(wire.requests) == 1 and not wire.client.is_closed()


@pytest.mark.parametrize("status", [200, 401])
async def test_owned_client_lifetime_and_no_retry(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    transport = Transport([httpx.Response(status, json=completion() if status == 200 else {"error": "denied"})])
    clients = []

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        result = httpx.AsyncClient(**kwargs, transport=transport)
        clients.append(result)
        return result

    monkeypatch.setattr("republic.providers.github_copilot.openai.DefaultAsyncHttpxClient", factory)
    async with GitHubCopilot(token(), integration_id="fixture-integration") as provider:
        if status == 200:
            await generate(provider, request())
        else:
            with pytest.raises(ProviderError):
                await generate(provider, request())
        assert not clients[0].is_closed and not clients[0].follow_redirects
        assert len(transport.requests) == 1
    await provider.aclose()
    assert clients[0].is_closed and transport.closed == 1


async def test_wrong_token_type_expired_token_and_redirects_are_rejected() -> None:
    with pytest.raises(CopilotAuthError, match="inference_token_required"):
        GitHubCopilot(cast(Any, login()), integration_id="fixture")
    async with Wire([], token(expires_at=1)) as wire:
        with pytest.raises(CopilotAuthError, match="copilot_token_expired"):
            await generate(wire.provider, request())
        assert not wire.requests
    async with Wire([httpx.Response(307, headers={"location": "https://attacker.test"})]) as wire:
        with pytest.raises(ProviderError):
            await generate(wire.provider, request())
        assert len(wire.requests) == 1
    async with httpx.AsyncClient(follow_redirects=True, transport=Transport([])) as http:
        client = openai.AsyncOpenAI(api_key="fixture-key", http_client=http)
        with pytest.raises(UnsupportedRequestError, match="follow_redirects"):
            GitHubCopilot(token(), integration_id="fixture", client=client)
        assert not client.is_closed()


@pytest.mark.parametrize(
    "options",
    [
        RequestOptions(parallel_tool_calls=False),
        RequestOptions(tool_choice="required"),
        *[
            RequestOptions(provider_options={key: value})
            for key, value in {
                "model": "override",
                "messages": [],
                "tools": [],
                "stream": False,
                "max_retries": 3,
                "extra_headers": {"Authorization": "override"},
                "extra_body": {},
                "base_url": "https://attacker.test",
                "store": True,
                "previous_response_id": "x",
                "seed": 1,
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            }.items()
        ],
    ],
)
async def test_unsupported_or_managed_options_never_reach_http(options: RequestOptions) -> None:
    req = request()
    req.options = options
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests


@pytest.mark.parametrize(
    "message",
    [
        Message(role="user", parts=[FilePart(data="https://unit.test/image", media_type="image/png")]),
        Message(
            role="assistant",
            parts=[ReasoningPart(text="", provider_metadata={"openai": {"encrypted_content": "opaque"}})],
        ),
        Message(role="user", parts=[TextPart(text="x", provider_metadata={"unsupported": True})]),
        Message(
            role="tool", parts=[ToolResultPart(tool_call_id="call", tool_name="lookup", result="fail", is_error=True)]
        ),
    ],
)
async def test_unsupported_parts_and_metadata_rejected_without_history_repair(message: Message) -> None:
    req = request()
    req.messages.append(message)
    before = req.model_dump_json()
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests
    assert req.model_dump_json() == before


async def test_nonstream_tool_data_and_unrepaired_system_order() -> None:
    call = {"id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": '{"x":'}}
    req = request()
    req.messages.append(Message(role="system", parts=[TextPart(text="Later rules")]))
    reply = completion({"content": None, "tool_calls": [call]}, finish="tool_calls")
    async with Wire([httpx.Response(200, json=reply)]) as wire:
        result = await generate(wire.provider, req)
        assert result.message.tool_calls[0].tool_args == '{"x":' and result.finish_reason == "tool_call"
        assert [m["role"] for m in wire.payload()["messages"]] == ["user", "system"]
        assert len(wire.requests) == 1


async def test_connection_failure_and_unconsumed_stream_send_no_extra_request() -> None:
    async with Wire([httpx.ConnectError(token().token)]) as wire:
        async with stream(wire.provider, request()):
            pass
        assert not wire.requests
        with pytest.raises(ProviderError) as caught:
            await generate(wire.provider, request())
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        assert token().token not in str(caught.value)
        assert len(wire.requests) == 1


@pytest.mark.parametrize("integration", [None, "", "bad\nheader", "a/b"])
async def test_integration_identity_requires_explicit_safe_header(integration: Any) -> None:
    with pytest.raises(UnsupportedRequestError, match="integration_id"):
        GitHubCopilot(token(), integration_id=integration)


@pytest.mark.parametrize("mode", ["generate", "stream"])
@pytest.mark.parametrize("field", ["reasoning_text", "reasoning_opaque", "reasoning_content", "copilot_references"])
async def test_unsupported_native_output_never_silently_dropped(mode: str, field: str) -> None:
    value = {field: "opaque-data"}
    body = Bytes([sse(chunk(value)), sse(chunk(finish="stop"))])
    reply = httpx.Response(200, json=completion(value)) if mode == "generate" else streaming(body)
    async with Wire([reply]) as wire:
        with pytest.raises(ProviderError, match="invalid_response"):
            if mode == "generate":
                await generate(wire.provider, request())
            else:
                async with stream(wire.provider, request()) as output:
                    async for _ in output:
                        pass
        assert len(wire.requests) == 1
        if mode == "stream":
            assert body.closed == 1
