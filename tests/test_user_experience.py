from __future__ import annotations

from types import SimpleNamespace

import pytest

from republic import LLM, TapeContext, tool
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.store import AsyncTapeStoreAdapter, InMemoryTapeStore

from .fakes import make_chunk, make_response, make_tool_call


def test_chat_retries_and_returns_text(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(RuntimeError("temporary failure"), make_response(text="ready"))

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="dummy",
        max_retries=2,
        error_classifier=lambda _: ErrorKind.TEMPORARY,
    )

    out = llm.chat("Reply with ready", max_tokens=8)
    assert out == "ready"
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("max_retries", "expected_calls"),
    [
        (0, 1),
        (1, 2),
        (3, 4),
    ],
)
def test_chat_retry_budget_is_one_plus_max_retries(fake_anyllm, max_retries: int, expected_calls: int) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(*(RuntimeError("temporary failure") for _ in range(expected_calls)))

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="dummy",
        max_retries=max_retries,
        error_classifier=lambda _: ErrorKind.TEMPORARY,
    )

    with pytest.raises(RepublicError) as exc_info:
        llm.chat("Reply with ready", max_tokens=8)
    assert exc_info.value.kind == ErrorKind.TEMPORARY
    assert len(client.calls) == expected_calls


def test_chat_uses_fallback_model(fake_anyllm) -> None:
    primary = fake_anyllm.ensure("openai")
    fallback = fake_anyllm.ensure("anthropic")

    primary.queue_completion(RuntimeError("primary down"))
    fallback.queue_completion(make_response(text="fallback ok"))

    llm = LLM(
        model="openai:gpt-4o-mini",
        fallback_models=["anthropic:claude-3-5-sonnet-latest"],
        max_retries=0,
        api_key={"openai": "x", "anthropic": "y"},
        error_classifier=lambda _: ErrorKind.TEMPORARY,
    )

    out = llm.chat("Ping")
    assert out == "fallback ok"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_chat_fallbacks_on_auth_error(fake_anyllm) -> None:
    class FakeAuthError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.status_code = 401

    primary = fake_anyllm.ensure("openai")
    fallback = fake_anyllm.ensure("openrouter")

    primary.queue_completion(FakeAuthError("invalid api key"))
    fallback.queue_completion(make_response(text="fallback ok"))

    llm = LLM(
        model="openai:gpt-4o-mini",
        fallback_models=["openrouter:openrouter/free"],
        max_retries=2,
        api_key={"openai": "bad", "openrouter": "ok"},
    )

    out = llm.chat("Ping")
    assert out == "fallback ok"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_chat_fallbacks_on_rate_limit_like_error(fake_anyllm) -> None:
    class FakeRateLimitError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.status_code = 429

    primary = fake_anyllm.ensure("openai")
    fallback = fake_anyllm.ensure("openrouter")

    primary.queue_completion(FakeRateLimitError("too many requests"))
    fallback.queue_completion(make_response(text="fallback ok"))

    llm = LLM(
        model="openai:gpt-4o-mini",
        fallback_models=["openrouter:openrouter/free"],
        max_retries=0,
        api_key={"openai": "x", "openrouter": "y"},
    )

    out = llm.chat("Ping")
    assert out == "fallback ok"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_tape_requires_anchor_then_records_full_run(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="step one"), make_response(text="step two"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    tape = llm.tape("ops")

    with pytest.raises(RepublicError) as exc_info:
        llm.chat("Investigate DB timeout", tape="ops")
    assert exc_info.value.kind == ErrorKind.NOT_FOUND
    assert len(client.calls) == 0

    tape.handoff("incident_42", state={"owner": "tier1"})
    first = llm.chat("Investigate DB timeout", tape="ops")
    second = llm.chat("Include rollback criteria", tape="ops")
    assert first == "step one"
    assert second == "step two"

    second_messages = client.calls[-1]["messages"]
    assert [message["role"] for message in second_messages] == ["user", "assistant", "user"]

    entries = list(tape.query.all())
    kinds = [entry.kind for entry in entries]
    assert kinds[0] == "error"
    assert entries[0].payload["kind"] == ErrorKind.NOT_FOUND.value
    assert "anchor" in kinds
    assert kinds[-1] == "event"

    run_event = entries[-1]
    assert run_event.payload["name"] == "run"
    assert run_event.payload["data"]["status"] == "ok"


def test_tape_chat_shortcuts_bind_current_tape(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="step one"), make_response(text="step two"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    tape = llm.tape("ops")
    tape.handoff("incident_42")

    first = tape.chat("Investigate DB timeout")
    second = tape.chat("Include rollback criteria")
    assert first == "step one"
    assert second == "step two"

    second_messages = client.calls[-1]["messages"]
    assert [message["role"] for message in second_messages] == ["user", "assistant", "user"]


@tool
def echo(text: str) -> str:
    return text.upper()


@tool
async def async_echo(text: str) -> str:
    return text.upper()


def test_stream_events_carries_text_tools_usage_and_final(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        iter([
            make_chunk(text="Checking "),
            make_chunk(tool_calls=[make_tool_call("echo", '{"text":"to', call_id="call_1")]),
            make_chunk(
                tool_calls=[make_tool_call("echo", 'kyo"}', call_id="call_1")],
                usage={"total_tokens": 12},
            ),
        ])
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
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
    assert stream.usage == {"total_tokens": 12}


def test_stream_events_merges_tool_deltas_without_id_or_index(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        iter([
            make_chunk(tool_calls=[make_tool_call("echo", '{"text":"to', call_id="call_1")]),
            make_chunk(
                tool_calls=[
                    SimpleNamespace(
                        type="function",
                        function=SimpleNamespace(name="", arguments='kyo"}'),
                    )
                ],
                usage={"total_tokens": 9},
            ),
        ])
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    stream = llm.stream_events("Call echo for tokyo", tools=[echo])
    events = list(stream)

    tool_calls = [event for event in events if event.kind == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].data["call"]["function"]["name"] == "echo"
    assert tool_calls[0].data["call"]["function"]["arguments"] == '{"text":"tokyo"}'

    tool_results = [event for event in events if event.kind == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].data["result"] == "TOKYO"
    assert stream.error is None
    assert stream.usage == {"total_tokens": 9}


@pytest.mark.asyncio
async def test_run_tools_async_executes_async_tool_handler(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(tool_calls=[make_tool_call("async_echo", '{"text":"tokyo"}')]))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    result = await llm.run_tools_async("Call async echo for tokyo", tools=[async_echo])

    assert result.kind == "tools"
    assert result.tool_results == ["TOKYO"]
    assert result.error is None


@pytest.mark.asyncio
async def test_chat_async_with_async_tape_store_uses_async_tape_manager(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="step one"), make_response(text="step two"))

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="dummy",
        tape_store=AsyncTapeStoreAdapter(InMemoryTapeStore()),
        context=TapeContext(anchor=None),
    )

    first = await llm.chat_async("Investigate DB timeout", tape="ops")
    second = await llm.chat_async("Include rollback criteria", tape="ops")

    assert first == "step one"
    assert second == "step two"
    second_messages = client.calls[-1]["messages"]
    assert [message["role"] for message in second_messages] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_stream_async_with_async_tape_store_persists_history(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        make_response(text="step one"),
        make_response(text="step two"),
    )

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="dummy",
        tape_store=AsyncTapeStoreAdapter(InMemoryTapeStore()),
        context=TapeContext(anchor=None),
    )

    stream = await llm.stream_async("Investigate DB timeout", tape="ops")
    first = "".join([part async for part in stream])
    second = await llm.chat_async("Include rollback criteria", tape="ops")

    assert first == "step one"
    assert second == "step two"
    second_messages = client.calls[-1]["messages"]
    assert [message["role"] for message in second_messages] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_tool_calls_async_with_async_tape_store_keeps_user_history(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]),
        make_response(tool_calls=[make_tool_call("echo", '{"text":"osaka"}')]),
    )

    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="dummy",
        tape_store=AsyncTapeStoreAdapter(InMemoryTapeStore()),
        context=TapeContext(anchor=None),
    )

    first = await llm.tool_calls_async("Call echo for tokyo", tape="ops", tools=[echo])
    second = await llm.tool_calls_async("Call echo for osaka", tape="ops", tools=[echo])

    assert first[0]["function"]["name"] == "echo"
    assert second[0]["function"]["name"] == "echo"
    second_messages = client.calls[-1]["messages"]
    assert [message["role"] for message in second_messages] == ["user", "user"]


def test_sync_chat_with_async_tape_store_is_rejected(fake_anyllm) -> None:
    llm = LLM(
        model="openai:gpt-4o-mini",
        api_key="dummy",
        tape_store=AsyncTapeStoreAdapter(InMemoryTapeStore()),
        context=TapeContext(anchor=None),
    )

    with pytest.raises(RepublicError) as exc_info:
        llm.chat("Ping", tape="ops")
    assert exc_info.value.kind == ErrorKind.INVALID_INPUT
    assert "Sync tape APIs are unavailable" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_events_async_executes_async_tool_handler(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(tool_calls=[make_tool_call("async_echo", '{"text":"tokyo"}')]))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    stream = await llm.stream_events_async("Call async echo for tokyo", tools=[async_echo])
    events = [event async for event in stream]
    tool_results = [event for event in events if event.kind == "tool_result"]

    assert len(tool_results) == 1
    assert tool_results[0].data["result"] == "TOKYO"
    assert stream.error is None


def test_text_shortcuts_and_embeddings_share_the_same_facade(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        make_response(tool_calls=[make_tool_call("if_decision", '{"value": true}')]),
        make_response(tool_calls=[make_tool_call("classify_decision", '{"label": "support"}')]),
        make_response(tool_calls=[make_tool_call("classify_decision", '{"label": "other"}')]),
    )
    client.queue_embedding({"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")

    decision = llm.if_("The service is down", "Should we page on-call?")
    assert decision is True

    label = llm.classify("Need invoice support", ["support", "sales"])
    assert label == "support"

    with pytest.raises(RepublicError) as exc_info:
        llm.classify("Unknown intent", ["support", "sales"])
    assert exc_info.value.kind == ErrorKind.INVALID_INPUT

    embedding = llm.embed("incident summary")
    assert embedding == {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
