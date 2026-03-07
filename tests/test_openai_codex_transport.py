from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import republic.core.execution as execution
from republic import LLM, tool
from republic.auth.openai_codex import extract_openai_codex_account_id

JWT_PAYLOAD = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}).encode("utf-8")
TOKEN = "aaa." + base64.urlsafe_b64encode(JWT_PAYLOAD).decode("ascii").rstrip("=") + ".bbb"


class FakeHTTPBodyResponse:
    def __init__(self, *, status_code: int = 200, body: str = "", content_type: str = "text/event-stream") -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {"content-type": content_type}
        self.encoding = "utf-8"
        self.text = body

    def read(self) -> bytes:
        return self._body.encode("utf-8")


class FakeHTTPStreamResponse(FakeHTTPBodyResponse):
    def __enter__(self) -> FakeHTTPStreamResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_lines(self):
        yield from self._body.split("\n")


class FakeHTTPClient:
    def __init__(
        self,
        *,
        sse: str,
        captured: dict[str, Any] | None,
        status_code: int = 200,
        content_type: str = "text/event-stream",
    ) -> None:
        self._sse = sse
        self._captured = captured
        self._status_code = status_code
        self._content_type = content_type

    def __enter__(self) -> FakeHTTPClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url, *, headers=None, json=None):
        if self._captured is not None:
            self._captured["method"] = "POST"
            self._captured["url"] = url
            self._captured["headers"] = headers or {}
            self._captured["body"] = json
        return FakeHTTPBodyResponse(status_code=self._status_code, body=self._sse, content_type=self._content_type)

    def stream(self, method, url, *, headers=None, json=None):
        if self._captured is not None:
            self._captured["method"] = method
            self._captured["url"] = url
            self._captured["headers"] = headers or {}
            self._captured["body"] = json
        return FakeHTTPStreamResponse(status_code=self._status_code, body=self._sse, content_type=self._content_type)


def _patch_codex_stream(
    monkeypatch,
    *,
    sse: str,
    captured: dict[str, Any] | None = None,
    status_code: int = 200,
    content_type: str = "text/event-stream",
) -> None:
    monkeypatch.setattr(
        "republic.clients.openai_codex.httpx.Client",
        lambda *args, **kwargs: FakeHTTPClient(
            sse=sse,
            captured=captured,
            status_code=status_code,
            content_type=content_type,
        ),
    )


def _unexpected_create(*args, **kwargs):
    raise AssertionError


def _sse(*events: dict[str, object]) -> str:
    return "\n\n".join("data: " + json.dumps(event) for event in events) + "\n\n"


def _build_codex_oauth_llm(
    monkeypatch,
    *,
    sse: str,
    model: str = "openai:gpt-5-codex",
    capture_request: bool = False,
) -> tuple[LLM, dict[str, Any] | None]:
    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    captured: dict[str, Any] | None = {} if capture_request else None
    _patch_codex_stream(monkeypatch, sse=sse, captured=captured)
    llm = LLM(
        model=model,
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )
    return llm, captured


def test_extract_openai_codex_account_id_reads_jwt_claim() -> None:
    assert extract_openai_codex_account_id(TOKEN) == "acct-test"
    assert extract_openai_codex_account_id("sk-test") is None


def test_openai_oauth_token_uses_codex_backend(monkeypatch) -> None:
    llm, captured = _build_codex_oauth_llm(
        monkeypatch,
        sse=_sse(
            {"type": "response.output_text.delta", "delta": "hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
            },
        ),
        model="openai:gpt-5.3-codex",
        capture_request=True,
    )

    assert llm.chat("Say hello") == "hello world"
    assert isinstance(captured, dict)
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    raw_headers = cast(dict[str, Any], captured["headers"])
    headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert headers["chatgpt-account-id"] == "acct-test"
    assert headers["openai-beta"] == "responses=experimental"
    body = cast(dict[str, Any], captured["body"])
    assert body["model"] == "gpt-5.3-codex"
    assert body["stream"] is True
    assert body["input"][0]["role"] == "user"


def test_openai_oauth_tool_calls_are_parsed_and_tools_are_converted(monkeypatch) -> None:
    sse = _sse(
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
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    )

    @tool
    def echo(message: str) -> str:
        return message

    llm, captured = _build_codex_oauth_llm(monkeypatch, sse=sse, capture_request=True)

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
    assert isinstance(captured, dict)
    body = cast(dict[str, Any], captured["body"])
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["name"] == "echo"
    assert body["tools"][0]["description"] == ""
    assert body["tools"][0]["parameters"]["type"] == "object"
    assert body["tools"][0]["parameters"]["properties"]["message"]["type"] == "string"
    assert body["tools"][0]["parameters"]["required"] == ["message"]


def test_openai_oauth_run_tools_executes_tool(monkeypatch) -> None:
    sse = _sse(
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
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    )

    @tool
    def echo(message: str) -> str:
        return f"echo:{message}"

    llm, _ = _build_codex_oauth_llm(monkeypatch, sse=sse)

    result = llm.run_tools("Use echo", tools=[echo])
    assert result.kind == "tools"
    assert result.tool_calls[0]["function"]["name"] == "echo"
    assert result.tool_results == ["echo:hello"]


def test_regular_openai_key_still_uses_anyllm(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []

    class StubClient:
        def completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))], usage=None
            )

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
    llm, _ = _build_codex_oauth_llm(
        monkeypatch,
        sse=_sse(
            {"type": "response.output_text.delta", "delta": "Checking "},
            {"type": "response.output_text.delta", "delta": "tools"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}},
            },
        ),
    )

    stream = llm.stream("Check tools")
    assert list(stream) == ["Checking ", "tools"]
    assert stream.error is None
    assert stream.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_openai_oauth_stream_events_carries_tools_usage_and_final(monkeypatch) -> None:
    sse = _sse(
        {"type": "response.output_text.delta", "delta": "Checking "},
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
        },
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}},
        },
    )

    @tool
    def echo(message: str) -> str:
        return message.upper()

    llm, _ = _build_codex_oauth_llm(monkeypatch, sse=sse)

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
    sse = _sse(
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
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    )

    @tool
    async def echo(message: str) -> str:
        return message.upper()

    llm, _ = _build_codex_oauth_llm(monkeypatch, sse=sse)

    stream = await llm.stream_events_async("Call echo for tokyo", tools=[echo])
    events = [event async for event in stream]
    tool_results = [event for event in events if event.kind == "tool_result"]

    assert len(tool_results) == 1
    assert tool_results[0].data["result"] == "TOKYO"
    assert stream.error is None
