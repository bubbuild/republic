# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified/extended for Republic's single-call contract; see NOTICE.
import asyncio
from collections.abc import AsyncGenerator

import pytest
from pydantic import TypeAdapter

from republic import (
    FilePart,
    IncompleteStreamError,
    Message,
    ProviderError,
    ReasoningPart,
    Request,
    Response,
    StreamProtocolError,
    Tool,
    ToolCallPart,
    ToolResultPart,
    Usage,
    events,
    generate,
    stream,
)
from tests.fakes import FakeProvider, request

pytestmark = pytest.mark.asyncio


async def test_interleaved_tools_text_reasoning_and_identity() -> None:
    script: list[events.Event] = [
        events.ReasoningStart(block_id="r", provider_metadata={"p": {"item_id": "r-1", "signature": "a"}}),
        events.ToolStart(tool_call_id="a", tool_name="weather"),
        events.ToolDelta(tool_call_id="a", chunk='{"city":'),
        events.TextStart(block_id="t"),
        events.TextDelta(block_id="t", chunk="Checking "),
        events.ToolStart(tool_call_id="b", tool_name="time"),
        events.ToolDelta(tool_call_id="b", chunk='{"zone":"'),
        events.ReasoningDelta(block_id="r", chunk="consider", provider_metadata={"p": {"signature": "ab"}}),
        events.ToolDelta(tool_call_id="a", chunk='"上海"}'),
        events.ToolEnd(tool_call_id="a", provider_metadata={"p": {"item_id": "tool-a"}}),
        events.ReasoningDelta(block_id="r", chunk=" options", provider_metadata={"other": {"flag": True}}),
        events.ToolDelta(tool_call_id="b", chunk='UTC"}'),
        events.ToolEnd(tool_call_id="b"),
        events.TextDelta(block_id="t", chunk="now"),
        events.TextEnd(block_id="t", provider_metadata={"p": {"citation": [1]}}),
        events.ReasoningEnd(block_id="r", provider_metadata={"p": {"signature": "abc", "encrypted": "opaque"}}),
        events.FileEvent(part=FilePart.from_bytes(b"image", media_type="image/png")),
        events.StreamEnd(
            response_id="res-1",
            response_model="actual-model",
            finish_reason="tool_call",
            usage=Usage(input_tokens=12, output_tokens=7, reasoning_tokens=2, cache_read_tokens=0),
            provider_metadata={"p": {"response_data": "kept"}},
        ),
    ]
    # Events can cross a JSON boundary independently of the runtime iterator.
    adapter = TypeAdapter[events.Event](events.Event)
    for event in script:
        assert adapter.validate_json(adapter.dump_json(event)) == event
    fake = FakeProvider(script)
    async with stream(fake, request()) as output:
        received = [event async for event in output]
        assert output.status == "completed"
        assert fake.closed == 1
        result = output.response
    assert received == script
    assert result is not None
    assert result.response_id == "res-1"
    assert result.response_model == "actual-model"
    assert result.finish_reason == "tool_call"
    terminal = script[-1]
    assert isinstance(terminal, events.StreamEnd)
    assert result.usage == terminal.usage
    assert result.message.text == "Checking now"
    assert [call.tool_args for call in result.message.tool_calls] == ['{"city":"上海"}', '{"zone":"UTC"}']
    assert [part.kind for part in result.message.parts] == ["reasoning", "tool_call", "text", "tool_call", "file"]
    reasoning = result.message.parts[0]
    assert isinstance(reasoning, ReasoningPart)
    assert reasoning.text == "consider options"
    assert reasoning.provider_metadata == {
        "p": {"item_id": "r-1", "signature": "abc", "encrypted": "opaque"},
        "other": {"flag": True},
    }
    assert result.message.provider_metadata == {"p": {"response_data": "kept"}}
    assert Response.model_validate_json(result.model_dump_json()) == result
    assert len(fake.calls) == fake.opened == fake.closed == 1
    # A caller can reuse restored metadata as history without SDK repair.
    next_request = request()
    next_request.messages.append(Message.model_validate_json(result.message.model_dump_json()))
    await generate(fake, next_request)
    assert fake.calls[-1][1] == next_request
    assert len(fake.calls) == 2


async def test_snapshots_and_source_events_do_not_alias_aggregation() -> None:
    start = events.ReasoningStart(block_id="r", provider_metadata={"p": {"id": "original"}})
    fake = FakeProvider([
        start,
        events.ReasoningDelta(block_id="r", chunk="private"),
        events.ReasoningEnd(block_id="r"),
        events.StreamEnd(),
    ])
    async with stream(fake, request()) as output:
        event = await anext(output)
        event.provider_metadata = {"p": {"id": "mutated event"}}
        snapshot = output.message
        snapshot.parts.clear()
        assert len(output.message.parts) == 1
        async for _ in output:
            pass
        response = output.response
        assert response is not None
        assert response.message.parts[0].provider_metadata == {"p": {"id": "original"}}
        response.message.parts.clear()
        assert len(output.message.parts) == 1
        assert output.response is not None
        assert len(output.response.message.parts) == 1


@pytest.mark.parametrize(
    "script", [[], [events.TextStart(block_id="t"), events.TextDelta(block_id="t", chunk="partial")]]
)
async def test_exhaustion_without_terminal_event_is_incomplete(script: list[events.Event]) -> None:
    fake = FakeProvider(script)
    async with stream(fake, request()) as output:
        with pytest.raises(IncompleteStreamError, match="without StreamEnd"):
            async for _ in output:
                pass
        assert output.status == "incomplete"
        assert output.response is None
        assert output.message.text == ("partial" if script else "")
        assert fake.closed == 1
    assert len(fake.calls) == fake.closed == 1


@pytest.mark.parametrize(
    "script",
    [
        [events.ToolDelta(tool_call_id="missing", chunk="{}")],
        [events.TextStart(block_id="t"), events.ReasoningDelta(block_id="t", chunk="wrong kind")],
        [events.TextStart(block_id="t"), events.TextStart(block_id="t")],
        [events.TextStart(block_id="t"), events.TextEnd(block_id="t"), events.TextStart(block_id="t")],
        [events.TextStart(block_id="t"), events.TextEnd(block_id="t"), events.TextDelta(block_id="t", chunk="late")],
        [events.ToolStart(tool_call_id="c", tool_name="f"), events.StreamEnd()],
        [events.TextStart(block_id="")],
    ],
)
async def test_invalid_event_order_is_visible_and_closes(script: list[events.Event]) -> None:
    fake = FakeProvider(script)
    with pytest.raises(StreamProtocolError):
        async with stream(fake, request()) as output:
            async for _ in output:
                pass
    assert output.status == "failed"
    assert output.response is None
    assert fake.closed == 1


async def test_terminal_length_is_a_response_with_verbatim_truncated_arguments() -> None:
    fake = FakeProvider([
        events.ToolStart(tool_call_id="c", tool_name="f"),
        events.ToolDelta(tool_call_id="c", chunk='{"x":'),
        events.ToolEnd(tool_call_id="c"),
        events.StreamEnd(finish_reason="length"),
    ])
    async with stream(fake, request()) as output:
        async for _ in output:
            pass
    assert output.status == "completed"
    assert output.response is not None
    assert output.response.finish_reason == "length"
    assert output.message.tool_calls[0].tool_args == '{"x":'


async def test_early_break_preserves_partial_output_without_draining() -> None:
    fake = FakeProvider([
        events.ToolStart(tool_call_id="c", tool_name="f"),
        events.ToolDelta(tool_call_id="c", chunk='{"x":'),
        events.ToolDelta(tool_call_id="c", chunk="1}"),
        events.ToolEnd(tool_call_id="c"),
        events.StreamEnd(),
    ])
    async with stream(fake, request()) as output:
        async for event in output:
            if isinstance(event, events.ToolDelta):
                break
    assert output.status == "closed"
    assert output.response is None
    assert output.message.tool_calls[0].tool_args == '{"x":'
    assert fake.emitted == 2
    assert fake.closed == len(fake.calls) == 1


async def test_explicit_close_is_idempotent_and_stops_iteration() -> None:
    fake = FakeProvider([events.TextStart(block_id="t"), events.StreamEnd()])
    async with stream(fake, request()) as output:
        await anext(output)
        await output.aclose()
        await output.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(output)
    assert fake.closed == fake.emitted == 1
    assert output.status == "closed"


async def test_unconsumed_stream_does_not_open_a_request() -> None:
    fake = FakeProvider([events.StreamEnd()])
    async with stream(fake, request()) as output:
        assert output.response is None
    assert output.status == "closed"
    assert fake.calls == []
    assert fake.opened == fake.closed == 0


async def test_cancellation_while_waiting_for_provider_releases_resources() -> None:
    fake = FakeProvider(
        [events.TextStart(block_id="t"), events.TextDelta(block_id="t", chunk="partial")], wait=asyncio.Event()
    )
    async with stream(fake, request()) as output:

        async def consume() -> None:
            async for _ in output:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(fake.waiting.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake.closed == 1
        assert output.status == "cancelled"
        assert output.response is None
        assert output.message.text == "partial"
    assert len(fake.calls) == fake.closed == 1


async def test_cancellation_in_consumer_body_closes_suspended_provider() -> None:
    fake = FakeProvider([events.TextStart(block_id="t"), events.StreamEnd()])
    ready = asyncio.Event()
    outputs = []

    async def consume() -> None:
        async with stream(fake, request()) as output:
            outputs.append(output)
            await anext(output)
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert outputs[0].status == "cancelled"
    assert fake.closed == fake.emitted == 1


async def test_cancellation_during_close_waits_for_cleanup() -> None:
    gate = asyncio.Event()
    fake = FakeProvider([events.TextStart(block_id="t")], close_gate=gate)
    async with stream(fake, request()) as output:
        await anext(output)
        closing = asyncio.create_task(output.aclose())
        await asyncio.wait_for(fake.closing.wait(), timeout=1)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        assert fake.closed == 0
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert fake.closed == 1
        assert output.status == "cancelled"
        await output.aclose()
    assert fake.closed == 1


async def test_caller_exception_closes_without_masking_it() -> None:
    fake = FakeProvider([events.TextStart(block_id="t")])
    error = ValueError("consumer failed")
    with pytest.raises(ValueError, match="consumer failed") as raised:
        async with stream(fake, request()) as output:
            await anext(output)
            raise error
    assert raised.value is error
    assert fake.closed == 1


async def test_provider_failure_preserves_error_without_retry_or_generate_fallback() -> None:
    error = ProviderError("unavailable", provider="fake", status_code=503)
    fake = FakeProvider([events.TextStart(block_id="t")], error=error)
    with pytest.raises(ProviderError) as raised:
        async with stream(fake, request()) as output:
            async for _ in output:
                pass
    assert raised.value is error
    assert output.status == "failed"
    assert fake.calls[0][0] == "stream"
    assert len(fake.calls) == fake.closed == 1
    with pytest.raises(ProviderError) as raised:
        await generate(fake, request())
    assert raised.value is error
    assert [kind for kind, _ in fake.calls] == ["stream", "generate"]
    assert fake.closed == 2


async def test_generate_cancellation_propagates_without_fallback() -> None:
    fake = FakeProvider([], wait=asyncio.Event())
    task = asyncio.create_task(generate(fake, request()))
    await asyncio.wait_for(fake.waiting.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.closed == len(fake.calls) == 1


async def test_both_entrypoints_preserve_history_and_do_not_execute_tools() -> None:
    original = request()
    # Deliberately unmatched result and unresolved call: no insertion, dropping,
    # execution or synthetic response is permitted at the SDK boundary.
    original.messages.extend([
        Message(role="tool", parts=[ToolResultPart(tool_call_id="orphan", tool_name="f", result=None)]),
        Message(role="assistant", parts=[ToolCallPart(tool_call_id="pending", tool_name="f", tool_args="{}")]),
    ])
    original.tools = [Tool(name="f", parameters={"type": "object"})]
    serialized = original.model_dump_json()

    class MutatingProvider(FakeProvider):
        async def generate(self, request: Request) -> Response:
            response = await super().generate(request)
            request.messages[0].parts.clear()
            request.tools[0].parameters.clear()
            return response

        async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
            self.calls.append(("stream", request.model_copy(deep=True)))
            request.messages[0].parts.clear()
            request.tools.clear()
            yield events.ToolStart(tool_call_id="new", tool_name="f")
            yield events.ToolDelta(tool_call_id="new", chunk="{}")
            yield events.ToolEnd(tool_call_id="new")
            yield events.StreamEnd(finish_reason="tool_call")

    fake = MutatingProvider([])
    fake.result = Response(message=original.messages[-1].model_copy(deep=True), finish_reason="tool_call")
    response = await generate(fake, original)
    assert response.message.tool_calls[0].tool_call_id == "pending"
    async with stream(fake, original) as output:
        async for _ in output:
            pass
    assert original.model_dump_json() == serialized
    assert [kind for kind, _ in fake.calls] == ["generate", "stream"]
    assert all(recorded == original for _, recorded in fake.calls)
    assert output.message.tool_calls[0].tool_call_id == "new"
