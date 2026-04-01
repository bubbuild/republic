from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from republic import LLM, tool
from republic.conversation import conversation_from_messages
from republic.core.results import RepublicError
from republic.providers import github_copilot as github_copilot_provider_module
from republic.providers.codecs import conversation_to_completion_messages
from republic.providers.github_copilot import GitHubCopilotBackend
from republic.providers.types import ChatRequest, ProviderContext

_TEST_GITHUB_TOKEN = "gho_token"  # noqa: S105


class _NoQueuedGitHubResponseError(AssertionError):
    def __init__(self) -> None:
        super().__init__("No queued GitHub response")


def _make_completion_response(
    *,
    text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> Any:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4.1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": _normalize_tool_calls(tool_calls),
                },
            }
        ],
        "usage": _normalize_usage(usage),
    }


def _make_completion_chunk(
    *,
    text: str = "",
    tool_calls: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> Any:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": text or None,
                    "tool_calls": _normalize_tool_calls(tool_calls),
                },
                "finish_reason": None,
            }
        ],
        "usage": _normalize_usage(usage),
    }


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    payload = dict(usage)
    if "prompt_tokens" not in payload and "input_tokens" in payload:
        payload["prompt_tokens"] = payload["input_tokens"]
    if "completion_tokens" not in payload and "output_tokens" in payload:
        payload["completion_tokens"] = payload["output_tokens"]
    return payload


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


def _normalize_tool_calls(tool_calls: list[Any] | None) -> list[Any]:
    normalized: list[Any] = []
    for tool_call in tool_calls or []:
        if isinstance(tool_call, SimpleNamespace):
            function = getattr(tool_call, "function", None)
            normalized.append({
                "id": getattr(tool_call, "id", None),
                "index": getattr(tool_call, "index", None),
                "type": getattr(tool_call, "type", None),
                "function": {
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", None),
                },
            })
        else:
            normalized.append(tool_call)
    return normalized


def _build_copilot_llm(
    monkeypatch,
    *queued_responses: Any,
    model: str = "github-copilot:gpt-4.1",
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
) -> tuple[LLM, list[dict[str, Any]], list[dict[str, Any]]]:
    init_calls: list[dict[str, Any]] = []
    completion_calls: list[dict[str, Any]] = []
    queue = deque(queued_responses)

    def _record_context(self) -> None:
        if init_calls:
            return
        headers = dict(self._headers())
        headers.pop("Authorization", None)
        init_calls.append({
            "api_key": self._context.api_key,
            "api_base": github_copilot_provider_module.resolve_github_copilot_api_base(self._context.api_base),
            "default_headers": headers,
        })

    def _post_chat(self, request):
        _record_context(self)
        completion_calls.append({
            "model": request.model,
            "messages": conversation_to_completion_messages(request.conversation),
            "tools": request.tools,
            "stream": request.stream,
            **dict(request.kwargs),
        })
        if not queue:
            raise _NoQueuedGitHubResponseError
        return queue.popleft()

    def _stream_chat(self, request):
        _record_context(self)
        completion_calls.append({
            "model": request.model,
            "messages": conversation_to_completion_messages(request.conversation),
            "tools": request.tools,
            "stream": request.stream,
            **dict(request.kwargs),
        })
        if not queue:
            raise _NoQueuedGitHubResponseError
        item = queue.popleft()
        if hasattr(item, "__aiter__"):
            return _async_iter_to_sync_iter(item)
        return item

    async def _astream_chat(self, request):
        _record_context(self)
        completion_calls.append({
            "model": request.model,
            "messages": conversation_to_completion_messages(request.conversation),
            "tools": request.tools,
            "stream": request.stream,
            **dict(request.kwargs),
        })
        if not queue:
            raise _NoQueuedGitHubResponseError
        return queue.popleft()

    monkeypatch.setattr(github_copilot_provider_module.GitHubCopilotBackend, "_post_chat", _post_chat)
    monkeypatch.setattr(github_copilot_provider_module.GitHubCopilotBackend, "_stream_chat", _stream_chat)
    monkeypatch.setattr(github_copilot_provider_module.GitHubCopilotBackend, "_astream_chat", _astream_chat)
    llm = LLM(
        model=model,
        api_key_resolver=lambda provider: _TEST_GITHUB_TOKEN if provider == "github-copilot" else None,
        api_base=api_base,
        client_args=client_args,
    )
    return llm, init_calls, completion_calls


def test_github_copilot_uses_special_client(monkeypatch) -> None:
    llm, init_calls, completion_calls = _build_copilot_llm(
        monkeypatch,
        _make_completion_response(text="hello from github"),
    )

    assert llm.chat("Say hello") == "hello from github"
    assert init_calls == [
        {
            "api_key": _TEST_GITHUB_TOKEN,
            "api_base": "https://models.github.ai/inference",
            "default_headers": {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        }
    ]
    assert completion_calls[0]["model"] == "gpt-4.1"
    assert completion_calls[0]["messages"][0]["role"] == "user"


def test_github_copilot_normalizes_api_base_and_merges_headers(monkeypatch) -> None:
    llm, init_calls, _ = _build_copilot_llm(
        monkeypatch,
        _make_completion_response(text="ok"),
        api_base="https://models.github.ai",
        client_args={
            "default_headers": {
                "Accept": "application/json",
                "X-Test": "1",
            },
            "session_timeout": 90,
            "cli_path": "var/copilot",
        },
    )

    assert llm.chat("Say hello") == "ok"
    assert init_calls[0] == {
        "api_key": _TEST_GITHUB_TOKEN,
        "api_base": "https://models.github.ai/inference",
        "default_headers": {
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "X-Test": "1",
        },
    }


def test_github_copilot_stream_yields_text_and_usage(monkeypatch) -> None:
    llm, _, _ = _build_copilot_llm(
        monkeypatch,
        _async_items(
            _make_completion_chunk(text="Checking "),
            _make_completion_chunk(text="tools"),
            _make_completion_chunk(usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
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


@pytest.mark.asyncio
async def test_github_copilot_stream_async_yields_text_and_usage(monkeypatch) -> None:
    llm, _, _ = _build_copilot_llm(
        monkeypatch,
        _async_items(
            _make_completion_chunk(text="A"),
            _make_completion_chunk(text="B"),
            _make_completion_chunk(usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        ),
    )

    stream = await llm.stream_async("Say AB")
    items = [item async for item in stream]
    assert items == ["A", "B"]
    assert stream.usage == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }


def test_github_copilot_tool_calls_are_parsed(monkeypatch) -> None:
    llm, _, _ = _build_copilot_llm(
        monkeypatch,
        _make_completion_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"message":"hello"}',
                    },
                }
            ]
        ),
    )

    @tool
    def echo(message: str) -> str:
        return message

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


def test_github_copilot_run_tools_executes_tool(monkeypatch) -> None:
    llm, _, _ = _build_copilot_llm(
        monkeypatch,
        _make_completion_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"message":"tokyo"}',
                    },
                }
            ]
        ),
    )

    @tool
    def echo(message: str) -> str:
        return message.upper()

    result = llm.run_tools("Call echo", tools=[echo])
    assert result.kind == "tools"
    assert result.tool_calls[0]["function"]["name"] == "echo"
    assert result.tool_results == ["TOKYO"]


def test_github_copilot_stream_events_carries_tools_usage_and_final(monkeypatch) -> None:
    llm, _, _ = _build_copilot_llm(
        monkeypatch,
        _async_items(
            _make_completion_chunk(text="Checking "),
            _make_completion_chunk(text="tools"),
            _make_completion_chunk(
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        index=0,
                        type="function",
                        function=SimpleNamespace(
                            name="echo",
                            arguments='{"message":"tokyo"}',
                        ),
                    )
                ]
            ),
            _make_completion_chunk(usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}),
        ),
    )

    @tool
    def echo(message: str) -> str:
        return message.upper()

    stream = llm.stream_events("Call echo", tools=[echo])
    events = list(stream)
    kinds = [event.kind for event in events]

    assert "text" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "usage" in kinds
    assert kinds[-1] == "final"


def test_github_copilot_responses_format_is_rejected(monkeypatch) -> None:
    llm = LLM(
        model="github-copilot:gpt-4.1",
        api_key="gho_token",
        api_format="responses",
    )

    with pytest.raises(RepublicError, match="requested transport 'responses' is not supported"):
        llm.chat("Say hello")


def test_github_copilot_reuses_httpx_client_and_preserves_client_args(monkeypatch) -> None:
    created_clients: list[dict[str, Any]] = []

    class _DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return _make_completion_response(text="hello")

    class _DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            created_clients.append(dict(kwargs))

        def post(self, *args: Any, **kwargs: Any) -> _DummyResponse:
            return _DummyResponse()

    monkeypatch.setattr(httpx, "Client", _DummyClient)

    backend = GitHubCopilotBackend(
        ProviderContext(
            provider="github-copilot",
            api_key=_TEST_GITHUB_TOKEN,
            api_base=None,
            client_args={
                "timeout": 7,
                "verify": False,
                "limits": "custom",
                "default_headers": {"X-Test": "1"},
            },
        )
    )
    request = ChatRequest(
        transport="completion",
        model="gpt-4.1",
        conversation=conversation_from_messages([{"role": "user", "content": "hi"}]),
        stream=False,
        reasoning_effort=None,
        kwargs={},
    )

    first = backend.chat(request)
    second = backend.chat(request)

    assert first.payload["choices"][0]["message"]["content"] == "hello"
    assert second.payload["choices"][0]["message"]["content"] == "hello"
    assert created_clients == [{"timeout": 7, "verify": False, "limits": "custom"}]
