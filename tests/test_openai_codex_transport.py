from __future__ import annotations

import base64
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import republic.core.execution as execution
from republic import LLM, tool
from republic.auth.openai_codex import extract_openai_codex_account_id

from .fakes import (
    make_responses_completed,
    make_responses_function_call,
    make_responses_response,
    make_responses_text_delta,
)

JWT_PAYLOAD = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}).encode("utf-8")
TOKEN = "aaa." + base64.urlsafe_b64encode(JWT_PAYLOAD).decode("ascii").rstrip("=") + ".bbb"


class _AsyncResponsesApi:
    def __init__(self, queue: deque[Any], calls: list[dict[str, Any]]) -> None:
        self._queue = queue
        self.calls = calls

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if not self._queue:
            raise AssertionError("No queued Codex response")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class _CodexAnyLLMClient:
    SUPPORTS_RESPONSES = True

    def __init__(self, queue: deque[Any], calls: list[dict[str, Any]]) -> None:
        self.client = SimpleNamespace(responses=_AsyncResponsesApi(queue, calls))

    def completion(self, **_: Any) -> Any:
        raise AssertionError("Codex path should not use completion transport")

    async def acompletion(self, **_: Any) -> Any:
        raise AssertionError("Codex path should not use completion transport")

    def responses(self, **_: Any) -> Any:
        raise AssertionError("Codex path should bypass any-llm responses wrapper")

    async def aresponses(self, **_: Any) -> Any:
        raise AssertionError("Codex path should bypass any-llm responses wrapper")


def _async_items(*items: Any):
    async def _iterator():
        for item in items:
            yield item

    return _iterator()


def _build_codex_llm(
    monkeypatch,
    *queued_responses: Any,
    model: str = "openai:gpt-5-codex",
) -> tuple[LLM, list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    create_calls: list[tuple[str, dict[str, Any]]] = []
    api_calls: list[dict[str, Any]] = []
    queue = deque(queued_responses)

    def _create(provider: str, **kwargs: Any) -> _CodexAnyLLMClient:
        create_calls.append((provider, dict(kwargs)))
        return _CodexAnyLLMClient(queue, api_calls)

    monkeypatch.setattr(execution.AnyLLM, "create", _create)
    llm = LLM(
        model=model,
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )
    return llm, create_calls, api_calls


def test_extract_openai_codex_account_id_reads_jwt_claim() -> None:
    assert extract_openai_codex_account_id(TOKEN) == "acct-test"
    assert extract_openai_codex_account_id("sk-test") is None


def test_openai_oauth_token_uses_anyllm_openai_client(monkeypatch) -> None:
    llm, create_calls, api_calls = _build_codex_llm(
        monkeypatch,
        make_responses_response(text="hello world"),
        model="openai:gpt-5.3-codex",
    )

    assert llm.chat("Say hello") == "hello world"
    assert create_calls == [
        (
            "openai",
            {
                "api_key": TOKEN,
                "api_base": "https://chatgpt.com/backend-api/codex",
                "default_headers": {
                    "chatgpt-account-id": "acct-test",
                    "OpenAI-Beta": "responses=experimental",
                    "originator": "republic",
                },
            },
        )
    ]
    assert api_calls[0]["model"] == "gpt-5.3-codex"
    assert api_calls[0]["input"][0]["role"] == "user"
    assert api_calls[0]["instructions"] == "You are Codex."
    assert api_calls[0]["include"] == ["reasoning.encrypted_content"]
    assert api_calls[0]["store"] is False
    assert api_calls[0]["text"] == {"verbosity": "medium"}
    assert api_calls[0]["stream"] is True


def test_openai_oauth_responses_preserves_extra_headers(monkeypatch) -> None:
    llm, _, api_calls = _build_codex_llm(
        monkeypatch,
        make_responses_response(text="hello"),
    )

    assert llm.chat("Say hello", extra_headers={"X-Title": "Republic"}) == "hello"
    assert api_calls[0]["extra_headers"] == {"X-Title": "Republic"}


def test_openai_oauth_non_stream_collects_stream_events(monkeypatch) -> None:
    llm, _, api_calls = _build_codex_llm(
        monkeypatch,
        _async_items(
            make_responses_text_delta("codex"),
            make_responses_text_delta("-ok"),
            make_responses_completed({"total_tokens": 4}),
        ),
    )

    assert llm.chat("Reply with exactly: codex-ok") == "codex-ok"
    assert api_calls[0]["stream"] is True


def test_openai_oauth_tool_calls_use_responses_transport(monkeypatch) -> None:
    @tool
    def echo(message: str) -> str:
        return message

    llm, _, api_calls = _build_codex_llm(
        monkeypatch,
        make_responses_response(tool_calls=[make_responses_function_call("echo", '{"message":"hello"}')]),
    )

    calls = llm.tool_calls("Use echo", tools=[echo])
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "echo",
                "arguments": '{"message":"hello"}',
            },
        }
    ]
    assert api_calls[0]["tools"][0]["type"] == "function"
    assert api_calls[0]["tools"][0]["name"] == "echo"


def test_openai_oauth_stream_yields_text_and_usage(monkeypatch) -> None:
    llm, _, _ = _build_codex_llm(
        monkeypatch,
        _async_items(
            make_responses_text_delta("Checking "),
            make_responses_text_delta("tools"),
            make_responses_completed({"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
        ),
    )

    stream = llm.stream("Check tools")
    assert list(stream) == ["Checking ", "tools"]
    assert stream.error is None
    assert stream.usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }


def test_regular_openai_key_still_uses_anyllm(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []

    class StubClient:
        def completion(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
                usage=None,
            )

        async def acompletion(self, **kwargs: Any) -> Any:
            return self.completion(**kwargs)

    def _create(provider: str, **kwargs: object) -> StubClient:
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
