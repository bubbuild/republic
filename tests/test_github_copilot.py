from __future__ import annotations

import json
import subprocess
import time
import traceback

import httpx2
import pytest

import republic
from republic.providers import CopilotAuth, GitHubCLIAuth
from tests.conftest import FakeService


async def test_messages_stops_at_copilot_terminal_marker(service: FakeService) -> None:
    text = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
    service.reply_events([text, "[DONE]", text])
    model = republic.get_model(
        "github-copilot:test", api_format="messages", api_key="test-key", http_client=service.client()
    )

    async with model.stream("Hi") as stream:
        events = [event async for event in stream]

    assert stream.text == "Hello"
    assert "".join(event.chunk for event in events if isinstance(event, republic.events.TextDelta)) == "Hello"
    assert isinstance(events[-1], republic.events.Completed)


@pytest.mark.parametrize("api_format", ["responses", "messages"])
async def test_malformed_copilot_events_still_raise(service: FakeService, api_format: str) -> None:
    service.reply_events(["not-json", "[DONE]"])
    model = republic.get_model(
        "github-copilot:test", api_format=api_format, api_key="test-key", http_client=service.client()
    )

    with pytest.raises(json.JSONDecodeError):
        async with model.stream("Hi") as stream:
            async for _ in stream:
                pass


@pytest.mark.parametrize("exchange", [False, True])
async def test_responses_keeps_parallel_calls_with_changing_item_ids(service: FakeService, exchange: bool) -> None:
    if exchange:
        service.reply_json({
            "token": "copilot-token",
            "refresh_in": 1800,
            "endpoints": {"api": "https://api.individual.githubcopilot.com"},
        })
    first = {"type": "function_call", "id": "encoded-add-0", "call_id": "call-0", "name": "add", "arguments": ""}
    second = {"type": "function_call", "id": "encoded-add-1", "call_id": "call-1", "name": "multiply", "arguments": ""}
    reasoning = {"type": "reasoning", "id": "original-reasoning-id", "encrypted_content": "opaque", "summary": []}
    service.reply_events([
        {"type": "response.output_item.added", "output_index": 0, "item": first},
        {"type": "response.output_item.added", "output_index": 1, "item": second},
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "encoded-delta-0a",
            "delta": '{"a":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "encoded-delta-1a",
            "delta": '{"b":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "encoded-delta-0b",
            "delta": "2}",
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "encoded-delta-1b",
            "delta": "3}",
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {**second, "id": "encoded-done-1", "arguments": '{"b":3}'},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {**first, "id": "encoded-done-0", "arguments": '{"a":2}'},
        },
        {"type": "response.output_item.done", "output_index": 2, "item": reasoning},
        "[DONE]",
    ])
    service.reply_json({"output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]})
    model = republic.get_model(
        "github-copilot:test",
        api_format="responses",
        api_key="test-key",
        auth=CopilotAuth("plugin-token") if exchange else None,
        http_client=service.client(),
    )

    async with model.stream("Calculate") as stream:
        events = [event async for event in stream]
    calls = stream.tool_calls
    assert [(call.id, call.name, call.args) for call in calls] == [
        ("call-0", "add", {"a": 2}),
        ("call-1", "multiply", {"b": 3}),
    ]
    ready = [event.call for event in events if isinstance(event, republic.events.ToolCallReady)]
    assert ready == [calls[1], calls[0]]
    assert stream.response.finish_reason == "tool_calls"

    results = [republic.tool_result(call, output) for call, output in zip(calls, ["5", "6"], strict=True)]
    response = await model.chat(["Calculate", stream.response.message, republic.assistant(tool_results=results)])

    assert response.text == "done"
    assert service.body()["input"][1:] == [
        reasoning,
        {"type": "function_call", "call_id": "call-0", "name": "add", "arguments": '{"a":2}'},
        {"type": "function_call", "call_id": "call-1", "name": "multiply", "arguments": '{"b":3}'},
        {"type": "function_call_output", "call_id": "call-0", "output": "5"},
        {"type": "function_call_output", "call_id": "call-1", "output": "6"},
    ]


@pytest.mark.parametrize("explicit_auth", [False, True])
@pytest.mark.parametrize("integration", [None, "issuing-app"])
async def test_github_uses_selected_cli_login_directly(
    monkeypatch: pytest.MonkeyPatch, service: FakeService, explicit_auth: bool, integration: str | None
) -> None:
    commands = []
    token = "ghs_installation-token" if integration else "cli-token"

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{token}\n")

    monkeypatch.setattr(subprocess, "run", run)
    service.reply_json({"choices": [{"message": {"content": "hi"}}]})
    monkeypatch.delenv("REPUBLIC_GITHUB_COPILOT_API_KEY", raising=False)
    headers = {"copilot-integration-id": integration} if integration else {}
    if explicit_auth:
        model = republic.get_model(
            "github-copilot:test",
            auth=GitHubCLIAuth(executable="/usr/bin/gh"),
            headers=headers,
            http_client=service.client(),
        )
    else:
        model = republic.get_model("github-copilot:test", headers=headers, http_client=service.client())

    response = await model.chat("hello")

    assert commands == [["/usr/bin/gh" if explicit_auth else "gh", "auth", "token", "--hostname", "github.com"]]
    assert len(service.requests) == 1
    assert service.requests[0].headers["authorization"] == f"Bearer {token}"
    assert str(service.requests[0].url) == "https://api.githubcopilot.com/chat/completions"
    assert service.requests[0].headers.get_list("copilot-integration-id") == ([integration] if integration else [])
    assert "editor-version" not in service.requests[0].headers
    assert response.text == "hi"


@pytest.mark.parametrize("override_headers", [False, True])
async def test_copilot_exchanges_explicit_credentials(service: FakeService, override_headers: bool) -> None:
    service.reply_json({
        "token": "copilot-token",
        "expires_at": int(time.time()) + 1800,
        "endpoints": {"api": "https://api.individual.githubcopilot.com"},
    })
    service.reply_json({"choices": [{"message": {"content": "hi"}}]})
    headers = {
        "copilot-integration-id": "custom-app",
        "EDITOR-VERSION": "my-editor/9",
        "Editor-Plugin-Version": "my-plugin/1",
        "x-github-api-version": "2026-06-01",
    }
    model = republic.get_model(
        "github-copilot:test",
        auth=CopilotAuth("plugin-token"),
        headers=headers if override_headers else None,
        http_client=service.client(),
    )

    response = await model.chat("hello")

    exchange, inference = service.requests
    assert exchange.method == "GET"
    assert str(exchange.url) == "https://api.github.com/copilot_internal/v2/token"
    assert exchange.headers["authorization"] == "token plugin-token"
    assert exchange.headers["x-github-api-version"] == "2025-04-01"
    assert inference.method == "POST"
    assert str(inference.url) == "https://api.individual.githubcopilot.com/chat/completions"
    assert inference.headers["host"] == "api.individual.githubcopilot.com"
    assert inference.headers["authorization"] == "Bearer copilot-token"
    if override_headers:
        for name, value in headers.items():
            assert inference.headers.get_list(name) == [value]
    else:
        assert inference.headers["copilot-integration-id"] == "vscode-chat"
        assert inference.headers["editor-version"] == "vscode/1.107.0"
        assert inference.headers["editor-plugin-version"] == "copilot-chat/0.35.0"
        assert inference.headers["x-github-api-version"] == "2025-10-01"
    assert service.body()["messages"] == [{"role": "user", "content": "hello"}]
    assert response.text == "hi"


@pytest.mark.parametrize(
    "expiry",
    [
        {"refresh_in": 300},
        {"expires_at": 1300},
        {"refresh_in": 300, "expires_at": 4600},
        {"refresh_in": 3600, "expires_at": 1300},
    ],
)
async def test_copilot_reuses_the_token_then_renews_it(
    monkeypatch: pytest.MonkeyPatch, service: FakeService, expiry: dict[str, int]
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1000)
    service.reply_json({
        "token": "first-token",
        **expiry,
        "endpoints": {"api": "https://api.githubcopilot.com"},
    })
    service.reply_json({"choices": [{"message": {"content": "one"}}]})
    service.reply_json({"choices": [{"message": {"content": "two"}}]})
    service.reply_json({
        "token": "second-token",
        "refresh_in": 1800,
        "endpoints": {"api": "https://api.individual.githubcopilot.com"},
    })
    service.reply_json({"choices": [{"message": {"content": "three"}}]})
    model = republic.get_model("github-copilot:test", auth=CopilotAuth("plugin-token"), http_client=service.client())

    assert (await model.chat("one")).text == "one"
    assert (await model.chat("two")).text == "two"
    assert [request.headers.get("authorization") for request in service.requests] == [
        "token plugin-token",
        "Bearer first-token",
        "Bearer first-token",
    ]

    monkeypatch.setattr(time, "time", lambda: 1241)
    assert (await model.chat("three")).text == "three"
    assert [request.headers.get("authorization") for request in service.requests[3:]] == [
        "token plugin-token",
        "Bearer second-token",
    ]
    assert str(service.requests[-1].url) == "https://api.individual.githubcopilot.com/chat/completions"
    assert service.requests[-1].headers["host"] == "api.individual.githubcopilot.com"


@pytest.mark.parametrize("renewal", [False, True])
async def test_copilot_exchange_failure_stops_before_inference(
    monkeypatch: pytest.MonkeyPatch, service: FakeService, renewal: bool
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not fall back to gh"))
    monkeypatch.setattr(time, "time", lambda: 1000)
    if renewal:
        service.reply_json({
            "token": "copilot-token",
            "refresh_in": 300,
            "endpoints": {"api": "https://api.individual.githubcopilot.com"},
        })
        service.reply_json({"choices": [{"message": {"content": "hi"}}]})
    service.reply_json({"message": "private-plugin-token is not authorized"}, status_code=403)
    model = republic.get_model(
        "github-copilot:test", auth=CopilotAuth("private-plugin-token"), http_client=service.client()
    )

    if renewal:
        assert (await model.chat("hello")).text == "hi"
        monkeypatch.setattr(time, "time", lambda: 1241)

    with pytest.raises(republic.AuthenticationError) as error:
        await model.chat("hello")

    assert len(service.requests) == (3 if renewal else 1)
    assert str(service.requests[-1].url) == "https://api.github.com/copilot_internal/v2/token"
    assert "private-plugin-token" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"token": ""},
        {"token": "copilot-token", "refresh_in": 1800},
        {"token": "copilot-token", "endpoints": {"api": "https://api.githubcopilot.com"}},
        {"token": "copilot-token", "refresh_in": True, "endpoints": {"api": "https://api.githubcopilot.com"}},
        {"token": "copilot-token", "refresh_in": 1800, "endpoints": {"api": "http://api.githubcopilot.com"}},
    ],
)
async def test_copilot_invalid_exchange_stops_before_inference(service: FakeService, payload: object) -> None:
    service.reply_json(payload)
    model = republic.get_model("github-copilot:test", auth=CopilotAuth("plugin-token"), http_client=service.client())

    with pytest.raises(republic.AuthenticationError):
        await model.chat("hello")

    assert len(service.requests) == 1
    assert service.requests[0].url.path == "/copilot_internal/v2/token"


async def test_copilot_streams_after_reading_the_exchange() -> None:
    received_text = False

    class ChatStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            assert received_text, "inference must stream without buffering the whole response"
            yield b"data: [DONE]\n\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "api.github.com":
            return httpx2.Response(
                200,
                stream=httpx2.ByteStream(
                    b'{"token":"copilot-token","refresh_in":1800,"endpoints":{"api":"https://api.githubcopilot.com"}}'
                ),
            )
        assert request.headers["authorization"] == "Bearer copilot-token"
        return httpx2.Response(200, stream=ChatStream())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = republic.get_model("github-copilot:test", auth=CopilotAuth("plugin-token"), http_client=client)
        async with model.stream("hello") as stream:
            async for event in stream:
                if isinstance(event, republic.events.TextDelta):
                    assert event.chunk == "hi"
                    received_text = True

    assert stream.text == "hi"


def test_copilot_auth_reads_sync_exchange_responses() -> None:
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.host == "api.github.com":
            return httpx2.Response(
                200,
                stream=httpx2.ByteStream(
                    b'{"token":"copilot-token","refresh_in":1800,"endpoints":{"api":"https://api.individual.githubcopilot.com"}}'
                ),
            )
        return httpx2.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        response = client.post(
            "https://api.githubcopilot.com/chat/completions?trace=1",
            json={"input": "hello"},
            auth=CopilotAuth("plugin-token"),
        )

    assert [str(request.url) for request in requests] == [
        "https://api.github.com/copilot_internal/v2/token",
        "https://api.individual.githubcopilot.com/chat/completions?trace=1",
    ]
    assert requests[-1].headers["authorization"] == "Bearer copilot-token"
    assert requests[-1].headers["host"] == "api.individual.githubcopilot.com"
    assert json.loads(requests[-1].content) == {"input": "hello"}
    assert response.json()["choices"][0]["message"]["content"] == "hi"


async def test_github_failure_does_not_fall_back_or_expose_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "unwanted-fallback")
    monkeypatch.delenv("REPUBLIC_GITHUB_COPILOT_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "sensitive stderr"))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: pytest.fail("must not send inference"))
    ) as client:
        model = republic.get_model("github-copilot:test", http_client=client)
        with pytest.raises(republic.AuthenticationError) as error:
            await model.chat("hello")
    assert "sensitive" not in str(error.value)


@pytest.mark.parametrize("api_format", ["chat", "responses", "messages"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exchange", [False, True])
async def test_copilot_uses_each_native_endpoint(
    service: FakeService, api_format: str, streaming: bool, exchange: bool
) -> None:
    if exchange:
        service.reply_json({
            "token": "copilot-token",
            "refresh_in": 1800,
            "endpoints": {"api": "https://api.individual.githubcopilot.com"},
        })
    if streaming:
        event = {
            "chat": {"choices": [{"delta": {"content": "hi"}}]},
            "responses": {"type": "response.output_text.delta", "delta": "hi"},
            "messages": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        }[api_format]
        service.reply_events([event])
    else:
        body = {
            "chat": {"choices": [{"message": {"content": "hi"}}]},
            "responses": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]},
            "messages": {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}},
        }[api_format]
        service.reply_json(body)
    model = republic.get_model(
        "github-copilot:test",
        api_format=api_format,
        api_key="test-key",
        auth=CopilotAuth("plugin-token") if exchange else None,
        http_client=service.client(),
    )

    if streaming:
        async with model.stream("hello") as stream:
            async for _ in stream:
                pass
        response = stream.response
    else:
        response = await model.chat("hello")

    assert response.text == "hi"
    assert (
        service.requests[-1].url.path
        == {"chat": "/chat/completions", "responses": "/responses", "messages": "/v1/messages"}[api_format]
    )
