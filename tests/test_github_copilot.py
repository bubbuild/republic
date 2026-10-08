from __future__ import annotations

import json
import subprocess

import httpx2
import pytest

import republic
from republic.providers import GitHubCLIAuth
from tests.conftest import FakeService


async def test_messages_stops_at_copilot_terminal_marker(service: FakeService) -> None:
    text = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
    service.reply_events([text, "[DONE]", text])
    model = republic.get_model("github-copilot:test", api_format="messages", http_client=service.client())

    async with model.stream("Hi") as stream:
        events = [event async for event in stream]

    assert stream.text == "Hello"
    assert "".join(event.chunk for event in events if isinstance(event, republic.events.TextDelta)) == "Hello"
    assert isinstance(events[-1], republic.events.Completed)


@pytest.mark.parametrize("api_format", ["responses", "messages"])
async def test_malformed_copilot_events_still_raise(service: FakeService, api_format: str) -> None:
    service.reply_events(["not-json", "[DONE]"])
    model = republic.get_model("github-copilot:test", api_format=api_format, http_client=service.client())

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
    model = republic.get_model("github-copilot:test", api_format="responses", http_client=service.client())

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


async def test_github_uses_selected_cli_login(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
    commands = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cli-token\n")

    monkeypatch.setattr(subprocess, "run", run)
    service.reply_json({"choices": [{"message": {"content": "hi"}}]})
    model = republic.get_model(
        "github-copilot:test", auth=GitHubCLIAuth(executable="/usr/bin/gh"), http_client=service.client()
    )

    response = await model.chat("hello")

    assert commands == [["/usr/bin/gh", "auth", "token", "--hostname", "github.com"]]
    assert service.requests[0].headers["authorization"] == "Bearer cli-token"
    assert str(service.requests[0].url) == "https://api.githubcopilot.com/chat/completions"
    assert response.text == "hi"


async def test_github_failure_does_not_fall_back_or_expose_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "unwanted-fallback")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "sensitive stderr"))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: pytest.fail("must not send inference"))
    ) as client:
        with pytest.raises(republic.AuthenticationError) as error:
            await client.post("https://example.test", json={}, auth=GitHubCLIAuth())
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
    model = republic.get_model("github-copilot:test", api_format=api_format, http_client=service.client())

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
