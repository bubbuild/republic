from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import republic.core.execution as execution
from republic import LLM, tool

_TEST_GITHUB_TOKEN = "gho_token"  # noqa: S105


class FakeSessionEventType:
    ASSISTANT_MESSAGE_DELTA = "assistant.message.delta"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_USAGE = "assistant.usage"
    SESSION_ERROR = "session.error"
    SESSION_IDLE = "session.idle"
    EXTERNAL_TOOL_REQUESTED = "external_tool.requested"


class FakePermissionRequestResult:
    def __init__(self, kind: str = "denied-no-approval-rule-and-could-not-request-from-user") -> None:
        self.kind = kind


class FakeCopilotSession:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self._handler = None
        self.sent: list[dict[str, Any]] = []
        self.destroyed = False
        self.aborted = False

    def on(self, handler):
        self._handler = handler

        def _unsubscribe():
            self._handler = None

        return _unsubscribe

    async def send_and_wait(self, options: dict[str, Any], timeout: float | None = None):
        self.sent.append({"options": dict(options), "timeout": timeout})
        final_event = None
        for event in self._events:
            if self._handler is not None:
                self._handler(event)
            if event.type == FakeSessionEventType.ASSISTANT_MESSAGE:
                final_event = event
        return final_event

    async def destroy(self) -> None:
        self.destroyed = True

    async def send(self, options: dict[str, Any]) -> str:
        self.sent.append({"options": dict(options)})
        for event in self._events:
            if self._handler is not None:
                self._handler(event)
        return "msg_1"

    async def abort(self) -> None:
        self.aborted = True


class FakeCopilotClient:
    instances: ClassVar[list[FakeCopilotClient]] = []
    next_events: ClassVar[list[Any]] = []

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = dict(options)
        self.started = False
        self.stopped = False
        self.created_sessions: list[dict[str, Any]] = []
        FakeCopilotClient.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def create_session(self, config: dict[str, Any]) -> FakeCopilotSession:
        self.created_sessions.append(dict(config))
        session = FakeCopilotSession(list(self.next_events))
        self.session = session
        return session


def _unexpected_create(*args, **kwargs):
    raise AssertionError


def _event(event_type: str, **data: Any) -> Any:
    return SimpleNamespace(type=event_type, data=SimpleNamespace(**data))


def _build_copilot_llm(monkeypatch, *, events: list[Any], model: str = "github-copilot:gpt-4.1") -> LLM:
    monkeypatch.setattr(execution.AnyLLM, "create", _unexpected_create)
    FakeCopilotClient.instances = []
    FakeCopilotClient.next_events = list(events)
    monkeypatch.setattr(
        "republic.clients.github_copilot._load_copilot_sdk",
        lambda: SimpleNamespace(
            client_type=FakeCopilotClient,
            permission_result_type=FakePermissionRequestResult,
            event_type=FakeSessionEventType,
            tool_type=SimpleNamespace,
        ),
    )
    return LLM(
        model=model,
        api_key_resolver=lambda provider: _TEST_GITHUB_TOKEN if provider == "github-copilot" else None,
    )


def test_github_copilot_oauth_uses_custom_backend(monkeypatch) -> None:
    llm = _build_copilot_llm(
        monkeypatch,
        events=[
            _event(FakeSessionEventType.ASSISTANT_MESSAGE, content="hello from copilot"),
            _event(FakeSessionEventType.ASSISTANT_USAGE, input_tokens=2, output_tokens=3),
            _event(FakeSessionEventType.SESSION_IDLE),
        ],
    )

    assert llm.chat("Say hello") == "hello from copilot"
    client = FakeCopilotClient.instances[0]
    assert client.options["env"]["GH_TOKEN"] == _TEST_GITHUB_TOKEN
    assert client.options["env"]["GITHUB_TOKEN"] == _TEST_GITHUB_TOKEN
    session_config = client.created_sessions[0]
    assert session_config["model"] == "gpt-4.1"
    assert session_config["streaming"] is True


def test_github_copilot_stream_yields_text_and_usage(monkeypatch) -> None:
    llm = _build_copilot_llm(
        monkeypatch,
        events=[
            _event(FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="Checking "),
            _event(FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="tools"),
            _event(FakeSessionEventType.ASSISTANT_USAGE, input_tokens=3, output_tokens=2),
            _event(FakeSessionEventType.ASSISTANT_MESSAGE, content="Checking tools"),
            _event(FakeSessionEventType.SESSION_IDLE),
        ],
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
    llm = _build_copilot_llm(
        monkeypatch,
        events=[
            _event(FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="A"),
            _event(FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="B"),
            _event(FakeSessionEventType.ASSISTANT_USAGE, input_tokens=1, output_tokens=1),
            _event(FakeSessionEventType.ASSISTANT_MESSAGE, content="AB"),
            _event(FakeSessionEventType.SESSION_IDLE),
        ],
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
    llm = _build_copilot_llm(
        monkeypatch,
        events=[
            _event(
                FakeSessionEventType.EXTERNAL_TOOL_REQUESTED,
                tool_name="echo",
                tool_call_id="call_1",
                arguments={"message": "hello"},
            )
        ],
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
    client = FakeCopilotClient.instances[0]
    assert client.session.aborted is True


def test_github_copilot_permission_handler_allows_only_custom_tools() -> None:
    from republic.clients.github_copilot import GitHubCopilotClient

    approved = GitHubCopilotClient._handle_permission_request(
        request=SimpleNamespace(kind="custom-tool"),
        invocation={"session_id": "session_1"},
        permission_result_type=FakePermissionRequestResult,
    )
    denied = GitHubCopilotClient._handle_permission_request(
        request=SimpleNamespace(kind="shell"),
        invocation={"session_id": "session_1"},
        permission_result_type=FakePermissionRequestResult,
    )

    assert approved.kind == "approved"
    assert denied.kind == "denied-no-approval-rule-and-could-not-request-from-user"


def test_github_copilot_run_tools_executes_tool(monkeypatch) -> None:
    llm = _build_copilot_llm(
        monkeypatch,
        events=[
            _event(
                FakeSessionEventType.EXTERNAL_TOOL_REQUESTED,
                tool_name="echo",
                tool_call_id="call_1",
                arguments={"message": "tokyo"},
            )
        ],
    )

    @tool
    def echo(message: str) -> str:
        return message.upper()

    result = llm.run_tools("Call echo", tools=[echo])
    assert result.kind == "tools"
    assert result.tool_calls[0]["function"]["name"] == "echo"
    assert result.tool_results == ["TOKYO"]


def test_github_copilot_stream_events_carries_tools_usage_and_final(monkeypatch) -> None:
    llm = _build_copilot_llm(
        monkeypatch,
        events=[
            _event(FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content="Checking "),
            _event(
                FakeSessionEventType.EXTERNAL_TOOL_REQUESTED,
                tool_name="echo",
                tool_call_id="call_1",
                arguments={"message": "tokyo"},
            ),
            _event(FakeSessionEventType.ASSISTANT_USAGE, input_tokens=5, output_tokens=3),
        ],
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


def test_github_copilot_provider_does_not_fall_back_to_anyllm(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []

    def _create(provider: str, **kwargs: object):
        created.append((provider, dict(kwargs)))
        raise AssertionError

    monkeypatch.setattr(execution.AnyLLM, "create", _create)
    FakeCopilotClient.instances = []
    FakeCopilotClient.next_events = [
        _event(FakeSessionEventType.ASSISTANT_MESSAGE, content="ok"),
        _event(FakeSessionEventType.SESSION_IDLE),
    ]
    monkeypatch.setattr(
        "republic.clients.github_copilot._load_copilot_sdk",
        lambda: SimpleNamespace(
            client_type=FakeCopilotClient,
            permission_result_type=FakePermissionRequestResult,
            event_type=FakeSessionEventType,
            tool_type=SimpleNamespace,
        ),
    )

    llm = LLM(model="github-copilot:gpt-4.1", api_key="gho_token")

    assert llm.chat("Say hello") == "ok"
    assert created == []
