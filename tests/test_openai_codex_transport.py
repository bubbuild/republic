from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

import republic.core.execution as execution
from republic import LLM, tool
from republic.auth.openai_codex import extract_openai_codex_account_id

JWT_PAYLOAD = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}).encode("utf-8")
TOKEN = "aaa." + base64.urlsafe_b64encode(JWT_PAYLOAD).decode("ascii").rstrip("=") + ".bbb"


class FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_extract_openai_codex_account_id_reads_jwt_claim() -> None:
    assert extract_openai_codex_account_id(TOKEN) == "acct-test"
    assert extract_openai_codex_account_id("sk-test") is None


def test_openai_oauth_token_uses_codex_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _unexpected_create(*args, **kwargs):
        raise AssertionError

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        sse = "\n\n".join(
            [
                'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "hello"}),
                'data: ' + json.dumps({"type": "response.output_text.delta", "delta": " world"}),
                'data: ' + json.dumps(
                    {
                        "type": "response.completed",
                        "response": {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
                    }
                ),
            ]
        ) + "\n\n"
        return FakeHTTPResponse(sse)

    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    monkeypatch.setattr("republic.clients.openai_codex.urllib.request.urlopen", _urlopen)

    llm = LLM(
        model="openai:gpt-5.3-codex",
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )

    assert llm.chat("Say hello") == "hello world"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    headers = {str(k).lower(): str(v) for k, v in dict(captured["headers"]).items()}
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert headers["chatgpt-account-id"] == "acct-test"
    assert headers["openai-beta"] == "responses=experimental"
    body = captured["body"]
    assert body["model"] == "gpt-5.3-codex"
    assert body["stream"] is True
    assert body["input"][0]["role"] == "user"


def test_openai_oauth_tool_calls_are_parsed_and_tools_are_converted(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _unexpected_create(*args, **kwargs):
        raise AssertionError

    def _urlopen(request, timeout=0):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        sse = "\n\n".join(
            [
                'data: '
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": json.dumps({"message": "hello"}),
                            "status": "completed",
                        },
                    }
                ),
                'data: ' + json.dumps({"type": "response.completed", "response": {"status": "completed"}}),
            ]
        ) + "\n\n"
        return FakeHTTPResponse(sse)

    @tool
    def echo(message: str) -> str:
        return message

    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    monkeypatch.setattr("republic.clients.openai_codex.urllib.request.urlopen", _urlopen)

    llm = LLM(
        model="openai:gpt-5-codex",
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )

    calls = llm.tool_calls("Use echo", tools=[echo])
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "echo",
                "arguments": json.dumps({"message": "hello"}),
            },
        }
    ]
    body = captured["body"]
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["name"] == "echo"
    assert body["tools"][0]["description"] == ""
    assert body["tools"][0]["parameters"]["type"] == "object"
    assert body["tools"][0]["parameters"]["properties"]["message"]["type"] == "string"
    assert body["tools"][0]["parameters"]["required"] == ["message"]


def test_openai_oauth_run_tools_executes_tool(monkeypatch) -> None:
    def _unexpected_create(*args, **kwargs):
        raise AssertionError

    def _urlopen(request, timeout=0):
        sse = "\n\n".join(
            [
                'data: '
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": json.dumps({"message": "hello"}),
                            "status": "completed",
                        },
                    }
                ),
                'data: ' + json.dumps({"type": "response.completed", "response": {"status": "completed"}}),
            ]
        ) + "\n\n"
        return FakeHTTPResponse(sse)

    @tool
    def echo(message: str) -> str:
        return f"echo:{message}"

    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    monkeypatch.setattr("republic.clients.openai_codex.urllib.request.urlopen", _urlopen)

    llm = LLM(
        model="openai:gpt-5-codex",
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )

    result = llm.run_tools("Use echo", tools=[echo])
    assert result.kind == "tools"
    assert result.tool_calls[0]["function"]["name"] == "echo"
    assert result.tool_results == ["echo:hello"]


def test_regular_openai_key_still_uses_anyllm(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []

    class StubClient:
        def completion(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))], usage=None)

        async def acompletion(self, **kwargs):
            return self.completion(**kwargs)

    def _create(provider: str, **kwargs: object):
        created.append((provider, dict(kwargs)))
        return StubClient()

    monkeypatch.setattr(execution.AnyLLM, "create", _create)

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="sk-test",
    )

    assert llm.chat("Say hello") == "ok"
    assert created[0][0] == "openai"
    assert created[0][1]["api_key"] == "sk-test"



def test_openai_oauth_stream_yields_text_and_usage(monkeypatch) -> None:
    def _unexpected_create(*args, **kwargs):
        raise AssertionError

    def _urlopen(request, timeout=0):
        sse = "\n\n".join(
            [
                'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "Checking "}),
                'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "tools"}),
                'data: ' + json.dumps(
                    {
                        "type": "response.completed",
                        "response": {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}},
                    }
                ),
            ]
        ) + "\n\n"
        return FakeHTTPResponse(sse)

    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    monkeypatch.setattr("republic.clients.openai_codex.urllib.request.urlopen", _urlopen)

    llm = LLM(
        model="openai:gpt-5-codex",
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )

    stream = llm.stream("Check tools")
    assert list(stream) == ["Checking ", "tools"]
    assert stream.error is None
    assert stream.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_openai_oauth_stream_events_carries_tools_usage_and_final(monkeypatch) -> None:
    def _unexpected_create(*args, **kwargs):
        raise AssertionError

    def _urlopen(request, timeout=0):
        sse = "\n\n".join(
            [
                'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "Checking "}),
                'data: '
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": json.dumps({"message": "tokyo"}),
                            "status": "completed",
                        },
                    }
                ),
                'data: ' + json.dumps(
                    {
                        "type": "response.completed",
                        "response": {"usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}},
                    }
                ),
            ]
        ) + "\n\n"
        return FakeHTTPResponse(sse)

    @tool
    def echo(message: str) -> str:
        return message.upper()

    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    monkeypatch.setattr("republic.clients.openai_codex.urllib.request.urlopen", _urlopen)

    llm = LLM(
        model="openai:gpt-5-codex",
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )

    stream = llm.stream_events("Call echo for tokyo", tools=[echo])
    events = list(stream)
    kinds = [event.kind for event in events]

    assert "text" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "usage" in kinds
    assert kinds[-1] == "final"

    tool_result = next(event for event in events if event.kind == "tool_result")
    assert tool_result.data["result"] == "TOKYO"
    assert stream.error is None
    assert stream.usage == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


@pytest.mark.asyncio
async def test_openai_oauth_stream_events_async_executes_tool_handler(monkeypatch) -> None:
    def _unexpected_create(*args, **kwargs):
        raise AssertionError

    def _urlopen(request, timeout=0):
        sse = "\n\n".join(
            [
                'data: '
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": json.dumps({"message": "tokyo"}),
                            "status": "completed",
                        },
                    }
                ),
                'data: ' + json.dumps({"type": "response.completed", "response": {"status": "completed"}}),
            ]
        ) + "\n\n"
        return FakeHTTPResponse(sse)

    @tool
    async def echo(message: str) -> str:
        return message.upper()

    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    monkeypatch.setattr("republic.clients.openai_codex.urllib.request.urlopen", _urlopen)

    llm = LLM(
        model="openai:gpt-5-codex",
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )

    stream = await llm.stream_events_async("Call echo for tokyo", tools=[echo])
    events = [event async for event in stream]
    tool_results = [event for event in events if event.kind == "tool_result"]

    assert len(tool_results) == 1
    assert tool_results[0].data["result"] == "TOKYO"
    assert stream.error is None
