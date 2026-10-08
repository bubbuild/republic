from __future__ import annotations

import json
import subprocess
import time

import httpx2
import pytest

import republic
from republic.providers import GitHubCLIAuth
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


async def test_responses_keeps_parallel_calls_with_changing_item_ids(service: FakeService) -> None:
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
        "github-copilot:test", api_format="responses", api_key="test-key", http_client=service.client()
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
async def test_github_exchanges_the_cli_login_for_copilot_tokens(
    monkeypatch: pytest.MonkeyPatch, service: FakeService, explicit_auth: bool
) -> None:
    commands = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cli-token\n")

    monkeypatch.setattr(subprocess, "run", run)
    service.reply_json({
        "token": "copilot-token",
        "expires_at": int(time.time()) + 1800,
        "endpoints": {"api": "https://api.individual.githubcopilot.com"},
    })
    service.reply_json({"choices": [{"message": {"content": "hi"}}]})
    monkeypatch.delenv("REPUBLIC_GITHUB-COPILOT_API_KEY", raising=False)
    if explicit_auth:
        model = republic.get_model(
            "github-copilot:test", auth=GitHubCLIAuth(executable="/usr/bin/gh"), http_client=service.client()
        )
    else:
        model = republic.get_model("github-copilot:test", http_client=service.client())

    response = await model.chat("hello")

    assert commands == [["/usr/bin/gh" if explicit_auth else "gh", "auth", "token", "--hostname", "github.com"]]
    exchange, inference = service.requests
    assert str(exchange.url) == "https://api.github.com/copilot_internal/v2/token"
    assert exchange.headers["authorization"] == "token cli-token"
    assert exchange.headers["x-github-api-version"] == "2025-04-01"
    # The login alone is rejected by inference, and the exchange names the origin to use.
    assert str(inference.url) == "https://api.individual.githubcopilot.com/chat/completions"
    assert inference.headers["authorization"] == "Bearer copilot-token"
    assert inference.headers["copilot-integration-id"] == "vscode-chat"
    assert inference.headers["editor-version"] == "vscode/1.95.0"
    assert inference.headers["editor-plugin-version"] == "copilot-chat/0.26.7"
    assert inference.headers["x-github-api-version"] == "2025-10-01"
    assert response.text == "hi"


async def test_copilot_reuses_the_token_then_renews_it(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
    commands = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cli-token\n")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv("REPUBLIC_GITHUB-COPILOT_API_KEY", raising=False)
    issued = time.time()
    service.reply_json({
        "token": "first-token",
        "refresh_in": 1800,
        "endpoints": {"api": "https://api.githubcopilot.com"},
    })
    service.reply_json({"choices": [{"message": {"content": "one"}}]})
    service.reply_json({"choices": [{"message": {"content": "two"}}]})
    service.reply_json({
        "token": "second-token",
        "refresh_in": 1800,
        "endpoints": {"api": "https://api.githubcopilot.com"},
    })
    service.reply_json({"choices": [{"message": {"content": "three"}}]})
    model = republic.get_model("github-copilot:test", http_client=service.client())

    assert (await model.chat("one")).text == "one"
    assert (await model.chat("two")).text == "two"
    assert len(commands) == 1
    assert [request.headers.get("authorization") for request in service.requests] == [
        "token cli-token",
        "Bearer first-token",
        "Bearer first-token",
    ]

    monkeypatch.setattr(time, "time", lambda: issued + 1900)
    assert (await model.chat("three")).text == "three"
    assert len(commands) == 2
    assert [request.headers.get("authorization") for request in service.requests[3:]] == [
        "token cli-token",
        "Bearer second-token",
    ]


async def test_copilot_exchange_failure_stops_before_inference(
    monkeypatch: pytest.MonkeyPatch, service: FakeService
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="cli-token\n")
    )
    monkeypatch.delenv("REPUBLIC_GITHUB-COPILOT_API_KEY", raising=False)
    service.reply_json({"message": "the account has no Copilot access"}, status_code=403)
    model = republic.get_model("github-copilot:test", http_client=service.client())

    with pytest.raises(republic.AuthenticationError) as error:
        await model.chat("hello")

    assert len(service.requests) == 1
    assert str(service.requests[0].url) == "https://api.github.com/copilot_internal/v2/token"
    assert "no Copilot access" not in str(error.value)


async def test_copilot_streams_after_the_exchange(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="cli-token\n")
    )
    monkeypatch.delenv("REPUBLIC_GITHUB-COPILOT_API_KEY", raising=False)
    service.reply_json({
        "token": "copilot-token",
        "refresh_in": 1800,
        "endpoints": {"api": "https://api.githubcopilot.com"},
    })
    service.reply_events([{"choices": [{"delta": {"content": "hi"}}]}])
    model = republic.get_model("github-copilot:test", http_client=service.client())

    async with model.stream("hello") as stream:
        async for _ in stream:
            pass

    assert stream.text == "hi"
    assert [str(request.url) for request in service.requests] == [
        "https://api.github.com/copilot_internal/v2/token",
        "https://api.githubcopilot.com/chat/completions",
    ]


def test_github_auth_also_signs_sync_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="cli-token\n")
    )
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.host == "api.github.com":
            return httpx2.Response(200, json={"token": "copilot-token", "refresh_in": 1800})
        return httpx2.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        response = client.post("https://api.githubcopilot.com/chat/completions", json={}, auth=GitHubCLIAuth())

    assert [str(request.url) for request in requests] == [
        "https://api.github.com/copilot_internal/v2/token",
        "https://api.githubcopilot.com/chat/completions",
    ]
    assert requests[-1].headers["authorization"] == "Bearer copilot-token"
    assert response.json()["choices"][0]["message"]["content"] == "hi"


async def test_github_failure_does_not_fall_back_or_expose_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "unwanted-fallback")
    monkeypatch.delenv("REPUBLIC_GITHUB-COPILOT_API_KEY", raising=False)
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
async def test_copilot_uses_each_native_endpoint(service: FakeService, api_format: str, streaming: bool) -> None:
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
        "github-copilot:test", api_format=api_format, api_key="test-key", http_client=service.client()
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
        service.requests[0].url.path
        == {"chat": "/chat/completions", "responses": "/responses", "messages": "/v1/messages"}[api_format]
    )
