from __future__ import annotations

import base64
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

from republic import LLM, tool
from republic.auth.openai_codex import extract_openai_codex_account_id
from republic.providers import openai as openai_provider_module

from .fakes import (
    install_fake_provider_runtime,
    make_response,
    make_responses_completed,
    make_responses_function_done,
    make_responses_text_delta,
)

JWT_PAYLOAD = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}).encode("utf-8")
TOKEN = "aaa." + base64.urlsafe_b64encode(JWT_PAYLOAD).decode("ascii").rstrip("=") + ".bbb"


class _NoQueuedCodexResponseError(AssertionError):
    def __init__(self) -> None:
        super().__init__("No queued Codex response")


class _ResponsesApi:
    def __init__(self, queue: deque[Any], calls: list[dict[str, Any]]) -> None:
        self._queue = queue
        self.calls = calls

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if not self._queue:
            raise _NoQueuedCodexResponseError
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        if hasattr(item, "__aiter__"):
            return _async_iter_to_sync_iter(item)
        return item


def _async_items(*items: Any):
    class _AsyncIterator:
        def __init__(self, values: tuple[Any, ...]) -> None:
            self._values = deque(values)

        def __aiter__(self):
            return self

        async def __anext__(self) -> Any:
            if not self._values:
                raise StopAsyncIteration
            return self._values.popleft()

    return _AsyncIterator(tuple(items))


def _async_iter_to_sync_iter(async_iter):
    from .fakes import FakeProviderBackend

    return FakeProviderBackend._async_iter_to_sync_iter(async_iter)


def _build_codex_llm(
    monkeypatch,
    *queued_responses: Any,
    model: str = "openai:gpt-5-codex",
) -> tuple[LLM, list[dict[str, Any]], list[dict[str, Any]]]:
    init_calls: list[dict[str, Any]] = []
    api_calls: list[dict[str, Any]] = []
    queue = deque(queued_responses)

    def _sync(self):
        if getattr(self, "_sync_client", None) is None:
            init_calls.append({
                "api_key": self._context.api_key,
                "api_base": openai_provider_module.resolve_openai_codex_api_base(self._context.api_base),
                "default_headers": self._default_headers(),
            })
            self._sync_client = SimpleNamespace(responses=_ResponsesApi(queue, api_calls))
        return self._sync_client

    monkeypatch.setattr(openai_provider_module.OpenAICodexBackend, "_sync", _sync)
    llm = LLM(
        model=model,
        api_key_resolver=lambda provider: TOKEN if provider == "openai" else None,
    )
    return llm, init_calls, api_calls


def test_extract_openai_codex_account_id_reads_jwt_claim() -> None:
    assert extract_openai_codex_account_id(TOKEN) == "acct-test"
    assert extract_openai_codex_account_id("sk-test") is None


def test_openai_oauth_token_uses_codex_special_client(monkeypatch) -> None:
    llm, init_calls, api_calls = _build_codex_llm(
        monkeypatch,
        _async_items(
            make_responses_text_delta("hello "),
            make_responses_text_delta("world"),
            make_responses_completed({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
        ),
        model="openai:gpt-5.3-codex",
    )

    assert llm.chat("Say hello") == "hello world"
    assert init_calls == [
        {
            "api_key": TOKEN,
            "api_base": "https://chatgpt.com/backend-api/codex",
            "default_headers": {
                "chatgpt-account-id": "acct-test",
                "OpenAI-Beta": "responses=experimental",
                "originator": "republic",
            },
        }
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
        _async_items(
            make_responses_text_delta("hello"),
            make_responses_completed({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        ),
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


def test_openai_oauth_drops_response_token_limit_parameters(monkeypatch) -> None:
    llm, _, api_calls = _build_codex_llm(
        monkeypatch,
        _async_items(
            make_responses_text_delta("limited"),
            make_responses_completed({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        ),
    )

    assert llm.chat("Reply with exactly: limited", max_tokens=12) == "limited"
    assert "max_tokens" not in api_calls[0]
    assert "max_output_tokens" not in api_calls[0]


def test_openai_oauth_tool_calls_use_responses_transport(monkeypatch) -> None:
    @tool
    def echo(message: str) -> str:
        return message

    llm, _, api_calls = _build_codex_llm(
        monkeypatch,
        _async_items(
            make_responses_function_done("echo", '{"message":"hello"}', item_id="call_1"),
            make_responses_completed({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        ),
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


def test_regular_openai_key_still_uses_builtin_openai_backend(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []
    factory = install_fake_provider_runtime(monkeypatch, created)
    factory.ensure("openai").queue_completion(make_response(text="ok"))

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="sk-test",
    )

    assert llm.chat("Say hello") == "ok"
    assert created[0][0] == "openai"
    assert created[0][1]["api_key"] == "sk-test"
