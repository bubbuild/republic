from __future__ import annotations

from typing import Any

import pytest

from republic import LLM, tool
from republic.clients.chat import ChatClient
from republic.core.execution import LLMCore
from republic.core.results import ErrorPayload

from .fakes import (
    make_chunk,
    make_response,
    make_responses_completed,
    make_responses_function_call,
    make_responses_function_delta,
    make_responses_function_done,
    make_responses_output_item_added,
    make_responses_output_item_done,
    make_responses_response,
    make_responses_text_delta,
    make_tool_call,
)


@tool
def echo(text: str) -> str:
    return text.upper()


def _compact_stream_events(events: list[Any]) -> list[tuple[str, Any]]:
    compact: list[tuple[str, Any]] = []
    for event in events:
        if event.kind == "text":
            compact.append(("text", event.data["delta"]))
        elif event.kind == "tool_call":
            call = event.data["call"]
            compact.append(("tool_call", (call.get("id"), call["function"]["name"], call["function"]["arguments"])))
        elif event.kind == "tool_result":
            compact.append(("tool_result", event.data["result"]))
        elif event.kind == "usage":
            compact.append(("usage", event.data))
        elif event.kind == "error":
            compact.append(("error", event.data))
        elif event.kind == "final":
            final = event.data
            compact.append((
                "final",
                {
                    "text": final.get("text"),
                    "tool_calls": [
                        (call.get("id"), call["function"]["name"], call["function"]["arguments"])
                        for call in (final.get("tool_calls") or [])
                    ],
                    "tool_results": final.get("tool_results"),
                    "usage": final.get("usage"),
                    "error": final.get("error"),
                },
            ))
    return compact


def _as_async_iter(items: list[Any]) -> Any:
    async def _generator() -> Any:
        for item in items:
            yield item

    return _generator()


def _responses_stream_text_items() -> list[Any]:
    return [
        make_responses_text_delta("hello"),
        make_responses_text_delta(" world"),
        make_responses_completed({"total_tokens": 7}),
    ]


def _responses_stream_event_items() -> list[Any]:
    return [
        make_responses_text_delta("Checking "),
        make_responses_output_item_added(item_id="fc_1", call_id="call_1", name="echo", arguments=""),
        make_responses_function_delta('{"text":"to', item_id="fc_1"),
        make_responses_function_delta('kyo"}', item_id="fc_1"),
        make_responses_output_item_done(
            item_id="fc_1",
            call_id="call_1",
            name="echo",
            arguments='{"text":"tokyo"}',
        ),
        make_responses_completed({"total_tokens": 12}),
    ]


def _completion_stream_text_items() -> list[Any]:
    return [
        make_chunk(text="hello"),
        make_chunk(text=" world", usage={"total_tokens": 7}),
    ]


def _completion_stream_event_items() -> list[Any]:
    return [
        make_chunk(text="Checking "),
        make_chunk(tool_calls=[make_tool_call("echo", '{"text":"to', call_id="call_1")]),
        make_chunk(
            tool_calls=[make_tool_call("echo", 'kyo"}', call_id="call_1")],
            usage={"total_tokens": 12},
        ),
    ]


def _main_path_payloads(*, async_mode: bool) -> list[Any]:
    wrap_stream = _as_async_iter if async_mode else iter
    return [
        make_response(text="ready"),
        make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]),
        make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]),
        wrap_stream(_completion_stream_text_items()),
        wrap_stream(_completion_stream_event_items()),
    ]


def _queue_main_path_fixtures(client: Any, *, async_mode: bool) -> None:
    payloads = _main_path_payloads(async_mode=async_mode)
    queue = client.queue_acompletion if async_mode else client.queue_completion
    queue(*payloads)


def test_default_api_format_uses_completion(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="hello"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    result = llm.chat("hi")

    assert result == "hello"
    assert client.calls[-1].get("responses") is None


def test_responses_api_format_uses_responses(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_responses(make_responses_response(text="hello"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy", api_format="responses")
    result = llm.chat("hi")

    assert result == "hello"
    assert client.calls[-1].get("responses") is True
    assert client.calls[-1]["input_data"][0]["role"] == "user"


def test_openrouter_responses_works_even_if_provider_flag_is_false(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.SUPPORTS_RESPONSES = False
    client.queue_responses(make_responses_response(text="hello"))

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    result = llm.chat("hi")

    assert result == "hello"
    assert client.calls[-1].get("responses") is True


def test_openrouter_anthropic_tools_rejects_responses_format(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.SUPPORTS_RESPONSES = False
    llm = LLM(model="openrouter:anthropic/claude-3.5-haiku", api_key="dummy", api_format="responses")

    with pytest.raises(ErrorPayload) as exc_info:
        llm.tool_calls(
            "Call echo for tokyo",
            tools=[echo],
            tool_choice={"type": "function", "function": {"name": "echo"}},
        )
    assert exc_info.value.kind == "invalid_input"


def test_openrouter_anthropic_tools_work_with_completion_format(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_completion(make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]))

    llm = LLM(model="openrouter:anthropic/claude-3.5-haiku", api_key="dummy")
    calls = llm.tool_calls("Call echo for tokyo", tools=[echo])

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    assert client.calls[-1].get("responses") is None


def test_anthropic_messages_format_maps_to_completion(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_completion(make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]))

    llm = LLM(model="openrouter:anthropic/claude-3.5-haiku", api_key="dummy", api_format="anthropic_messages")
    calls = llm.tool_calls("Call echo for tokyo", tools=[echo])

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    assert client.calls[-1].get("responses") is None


def test_anthropic_messages_format_rejects_non_anthropic_model(fake_anyllm) -> None:
    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy", api_format="anthropic_messages")
    with pytest.raises(ErrorPayload) as exc_info:
        llm.chat("hi")
    assert exc_info.value.kind == "invalid_input"


def test_responses_tool_choice_accepts_completion_function_shape(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_responses(
        make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy", api_format="responses")
    calls = llm.tool_calls(
        "Call echo for tokyo",
        tools=[echo],
        tool_choice={"type": "function", "function": {"name": "echo"}},
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    assert client.calls[-1]["tool_choice"] == {"type": "function", "name": "echo"}


def test_extract_tool_calls_from_responses() -> None:
    response = make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"hi"}')])

    calls = ChatClient._extract_tool_calls(response)

    assert calls == [
        {
            "function": {"name": "echo", "arguments": '{"text":"hi"}'},
            "id": "call_1",
            "type": "function",
        }
    ]


def test_non_stream_completion_splits_concatenated_tool_arguments(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        make_response(
            tool_calls=[
                make_tool_call(
                    "echo",
                    '{"text":"tokyo"}{"text":"osaka"}',
                    call_id="call_1",
                )
            ]
        )
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    result = llm.run_tools("Call echo twice", tools=[echo])

    assert result.kind == "tools"
    assert [call["id"] for call in result.tool_calls] == ["call_1", "call_1__2"]
    assert result.tool_results == ["TOKYO", "OSAKA"]


def test_stream_events_splits_concatenated_tool_arguments(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        iter([
            make_chunk(
                tool_calls=[
                    make_tool_call(
                        "echo",
                        '{"text":"tokyo"}{"text":"osaka"}',
                        call_id="call_1",
                    )
                ],
                usage={"total_tokens": 8},
            )
        ])
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    events = list(llm.stream_events("Call echo twice", tools=[echo]))

    tool_calls = [event.data["call"] for event in events if event.kind == "tool_call"]
    assert [call["id"] for call in tool_calls] == ["call_1", "call_1__2"]
    assert [call["function"]["arguments"] for call in tool_calls] == ['{"text":"tokyo"}', '{"text":"osaka"}']
    tool_results = [event.data["result"] for event in events if event.kind == "tool_result"]
    assert tool_results == ["TOKYO", "OSAKA"]


def test_split_messages_for_responses() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
    ]

    instructions, input_items = LLMCore._split_messages_for_responses(messages)

    assert instructions == "sys"
    assert input_items == [
        {"role": "user", "content": "hi", "type": "message"},
        {"type": "function_call", "name": "echo", "arguments": '{"text":"hi"}', "call_id": "call_1"},
        {"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'},
    ]


def test_stream_uses_responses_and_collects_usage(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        iter([
            make_responses_text_delta("Hello"),
            make_responses_text_delta(" world"),
            make_responses_completed({"total_tokens": 7}),
        ])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    stream = llm.stream("Say hello")
    text = "".join(list(stream))

    assert text == "Hello world"
    assert stream.error is None
    assert stream.usage == {"total_tokens": 7}
    assert client.calls[-1]["responses"] is True
    assert client.calls[-1]["stream"] is True


def test_stream_events_supports_responses_tool_events(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        iter([
            make_responses_text_delta("Checking "),
            make_responses_function_delta('{"text":"to', item_id="call_rsp_1"),
            make_responses_function_delta('kyo"}', item_id="call_rsp_1"),
            make_responses_function_done("echo", '{"text":"tokyo"}', item_id="call_rsp_1"),
            make_responses_completed({"total_tokens": 12}),
        ])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    stream = llm.stream_events("Call echo for tokyo", tools=[echo])
    events = list(stream)

    kinds = [event.kind for event in events]
    assert "text" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "usage" in kinds
    assert kinds[-1] == "final"

    tool_calls = [event for event in events if event.kind == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].data["call"]["function"]["name"] == "echo"
    assert tool_calls[0].data["call"]["function"]["arguments"] == '{"text":"tokyo"}'

    tool_results = [event for event in events if event.kind == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].data["result"] == "TOKYO"
    assert stream.error is None
    assert stream.usage == {"total_tokens": 12}


def test_stream_and_events_support_responses_dict_events(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        iter([
            {"type": "response.output_text.delta", "delta": "Checking "},
            {"type": "response.function_call_arguments.delta", "item_id": "call_d1", "delta": '{"text":"to'},
            {
                "type": "response.function_call_arguments.done",
                "item_id": "call_d1",
                "name": "echo",
                "arguments": '{"text":"tokyo"}',
            },
            {"type": "response.completed", "response": {"usage": {"total_tokens": 5}}},
        ])
    )
    client.queue_responses(make_responses_response(text="ready", usage={"total_tokens": 3}))

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    events_stream = llm.stream_events("Call echo for tokyo", tools=[echo])
    events = list(events_stream)
    tool_call = next(event for event in events if event.kind == "tool_call")
    assert tool_call.data["call"]["function"]["arguments"] == '{"text":"tokyo"}'
    assert events_stream.usage == {"total_tokens": 5}

    text_stream = llm.stream("Reply with ready")
    text = "".join(list(text_stream))
    assert text == "ready"
    assert text_stream.usage == {"total_tokens": 3}


def test_stream_events_parity_between_completion_and_responses(fake_anyllm) -> None:
    completion_client = fake_anyllm.ensure("openai")
    completion_client.queue_completion(
        iter([
            make_chunk(text="Checking "),
            make_chunk(tool_calls=[make_tool_call("echo", '{"text":"to', call_id="call_1")]),
            make_chunk(
                tool_calls=[make_tool_call("echo", 'kyo"}', call_id="call_1")],
                usage={"total_tokens": 12},
            ),
        ])
    )

    responses_client = fake_anyllm.ensure("openrouter")
    responses_client.queue_responses(
        iter([
            make_responses_text_delta("Checking "),
            make_responses_output_item_added(item_id="fc_1", call_id="call_1", name="echo", arguments=""),
            make_responses_function_delta('{"text":"to', item_id="fc_1"),
            make_responses_function_delta('kyo"}', item_id="fc_1"),
            make_responses_function_done("echo", '{"text":"tokyo"}', item_id="fc_1"),
            make_responses_completed({"total_tokens": 12}),
        ])
    )

    completion_llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    responses_llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")

    completion_events = list(completion_llm.stream_events("Call echo for tokyo", tools=[echo]))
    responses_events = list(responses_llm.stream_events("Call echo for tokyo", tools=[echo]))
    assert completion_client.calls[-1].get("responses") is None
    assert responses_client.calls[-1].get("responses") is True

    assert _compact_stream_events(completion_events) == _compact_stream_events(responses_events)


def test_stream_events_responses_output_item_events_keep_call_id(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        iter([
            make_responses_output_item_added(item_id="fc_123", call_id="call_abc", name="echo"),
            make_responses_function_delta('{"text":"to', item_id="fc_123"),
            make_responses_function_delta('kyo"}', item_id="fc_123"),
            make_responses_output_item_done(
                item_id="fc_123",
                call_id="call_abc",
                name="echo",
                arguments='{"text":"tokyo"}',
            ),
            make_responses_completed({"total_tokens": 6}),
        ])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    events = list(llm.stream_events("Call echo for tokyo", tools=[echo]))

    tool_call = next(event for event in events if event.kind == "tool_call").data["call"]
    assert tool_call["id"] == "call_abc"
    assert tool_call["function"]["name"] == "echo"
    assert tool_call["function"]["arguments"] == '{"text":"tokyo"}'


def test_stream_usage_accepts_responses_in_progress_usage(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        iter([
            make_responses_text_delta("ok"),
            {"type": "response.in_progress", "response": {"usage": {"total_tokens": 2}}},
        ])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    stream = llm.stream("Reply with ok")
    assert "".join(list(stream)) == "ok"
    assert stream.usage == {"total_tokens": 2}


def test_non_stream_responses_tool_calls_converts_tools_payload(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    calls = llm.tool_calls("Call echo for tokyo", tools=[echo])

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    sent_tools = client.calls[-1]["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["name"] == "echo"
    assert sent_tools[0]["description"] == ""
    assert sent_tools[0]["parameters"]["type"] == "object"
    assert "function" not in sent_tools[0]


def test_non_stream_responses_run_tools_uses_converted_tools(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    result = llm.run_tools("Call echo for tokyo", tools=[echo])

    assert result.kind == "tools"
    assert result.tool_results == ["TOKYO"]
    sent_tools = client.calls[-1]["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["name"] == "echo"
    assert "function" not in sent_tools[0]


def test_chat_reasoning_effort_for_completion_is_forwarded(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="ready"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    assert llm.chat("Reply with ready", reasoning_effort="high") == "ready"

    call = client.calls[-1]
    assert call.get("reasoning_effort") == "high"


def test_chat_reasoning_effort_for_responses_is_mapped(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_response(text="ready"))

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    assert llm.chat("Reply with ready", reasoning_effort="low") == "ready"

    call = client.calls[-1]
    assert call["responses"] is True
    assert call.get("reasoning") == {"effort": "low"}
    assert "reasoning_effort" not in call


def test_chat_reasoning_kwarg_has_priority_over_reasoning_effort(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_response(text="ready"))

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    assert llm.chat("Reply with ready", reasoning_effort="low", reasoning={"effort": "high"}) == "ready"

    call = client.calls[-1]
    assert call["responses"] is True
    assert call.get("reasoning") == {"effort": "high"}


def test_stream_completion_defaults_include_usage(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        iter([
            make_chunk(text="hello"),
            make_chunk(text=" world", usage={"total_tokens": 7}),
        ])
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    stream = llm.stream("Say hello")
    assert "".join(list(stream)) == "hello world"
    assert stream.usage == {"total_tokens": 7}

    assert client.calls[-1].get("stream_options") == {"include_usage": True}


def test_stream_completion_preserves_user_stream_options(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(
        iter([
            make_chunk(text="hello"),
            make_chunk(text=" world", usage={"total_tokens": 7}),
        ])
    )

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    stream = llm.stream("Say hello", stream_options={"include_usage": False, "custom": True})
    assert "".join(list(stream)) == "hello world"
    assert stream.usage == {"total_tokens": 7}

    assert client.calls[-1].get("stream_options") == {"include_usage": False, "custom": True}


def test_openai_completion_uses_max_completion_tokens(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="hello"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    assert llm.chat("Say hello", max_tokens=11) == "hello"

    call = client.calls[-1]
    assert call.get("max_completion_tokens") == 11
    assert "max_tokens" not in call


def test_non_openai_completion_uses_max_tokens(fake_anyllm) -> None:
    client = fake_anyllm.ensure("anthropic")
    client.queue_completion(make_response(text="hello"))

    llm = LLM(model="anthropic:claude-3-5-haiku-latest", api_key="dummy")
    assert llm.chat("Say hello", max_tokens=11) == "hello"

    call = client.calls[-1]
    assert call.get("max_tokens") == 11
    assert "max_completion_tokens" not in call


def test_non_stream_chat_parity_between_completion_and_responses(fake_anyllm) -> None:
    completion_client = fake_anyllm.ensure("openai")
    completion_client.queue_completion(make_response(text="ready"))

    responses_client = fake_anyllm.ensure("openrouter")
    responses_client.queue_responses(make_responses_response(text="ready"))

    completion_llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    responses_llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")

    assert completion_llm.chat("Reply with ready") == "ready"
    assert responses_llm.chat("Reply with ready") == "ready"


def test_non_stream_run_tools_parity_between_completion_and_responses(fake_anyllm) -> None:
    completion_client = fake_anyllm.ensure("openai")
    completion_client.queue_completion(make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]))

    responses_client = fake_anyllm.ensure("openrouter")
    responses_client.queue_responses(
        make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    )

    completion_llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    responses_llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")

    completion_result = completion_llm.run_tools("Call echo for tokyo", tools=[echo])
    responses_result = responses_llm.run_tools("Call echo for tokyo", tools=[echo])

    assert completion_result.kind == responses_result.kind == "tools"
    assert completion_result.tool_results == responses_result.tool_results == ["TOKYO"]


def test_sync_main_paths(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    _queue_main_path_fixtures(client, async_mode=False)
    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")

    assert llm.chat("Reply with ready") == "ready"

    calls = llm.tool_calls("Call echo for tokyo", tools=[echo])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"

    run_result = llm.run_tools("Call echo for tokyo", tools=[echo])
    assert run_result.kind == "tools"
    assert run_result.tool_results == ["TOKYO"]

    text_stream = llm.stream("Say hello")
    assert "".join(list(text_stream)) == "hello world"
    assert text_stream.usage == {"total_tokens": 7}

    event_stream = llm.stream_events("Call echo for tokyo", tools=[echo])
    events = list(event_stream)
    compact = _compact_stream_events(events)
    assert compact[0] == ("text", "Checking ")
    assert compact[1][0] == "tool_call"
    assert compact[2] == ("tool_result", "TOKYO")
    assert compact[3] == ("usage", {"total_tokens": 12})
    assert compact[-1][0] == "final"
    assert event_stream.usage == {"total_tokens": 12}

    assert all(call.get("responses") is None for call in client.calls)


@pytest.mark.asyncio
async def test_async_main_paths(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    _queue_main_path_fixtures(client, async_mode=True)
    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")

    assert await llm.chat_async("Reply with ready") == "ready"

    calls = await llm.tool_calls_async("Call echo for tokyo", tools=[echo])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"

    run_result = await llm.run_tools_async("Call echo for tokyo", tools=[echo])
    assert run_result.kind == "tools"
    assert run_result.tool_results == ["TOKYO"]

    text_stream = await llm.stream_async("Say hello")
    assert "".join([part async for part in text_stream]) == "hello world"
    assert text_stream.usage == {"total_tokens": 7}

    event_stream = await llm.stream_events_async("Call echo for tokyo", tools=[echo])
    events = [event async for event in event_stream]
    compact = _compact_stream_events(events)
    assert compact[0] == ("text", "Checking ")
    assert compact[1][0] == "tool_call"
    assert compact[2] == ("tool_result", "TOKYO")
    assert compact[3] == ("usage", {"total_tokens": 12})
    assert compact[-1][0] == "final"
    assert event_stream.usage == {"total_tokens": 12}

    assert all(call.get("responses") is None for call in client.calls)
