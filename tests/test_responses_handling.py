from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from republic import LLM, tool
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
    make_responses_reasoning_item_added,
    make_responses_reasoning_item_done,
    make_responses_reasoning_response,
    make_responses_response,
    make_responses_text_delta,
    make_tool_call,
)


@tool
def echo(text: str) -> str:
    return text.upper()


class UnexpectedAsyncResponsesCall(AssertionError):
    def __init__(self) -> None:
        super().__init__("sync path should go through client.responses")


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


async def _async_items(*items: Any) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _completion_stream_event_items() -> list[Any]:
    return [
        make_chunk(text="Checking "),
        make_chunk(tool_calls=[make_tool_call("echo", '{"text":"to', call_id="call_1")]),
        make_chunk(
            tool_calls=[make_tool_call("echo", 'kyo"}', call_id="call_1")],
            usage={"total_tokens": 12},
        ),
    ]


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


def test_messages_format_maps_to_completion(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_completion(make_response(tool_calls=[make_tool_call("echo", '{"text":"tokyo"}')]))

    llm = LLM(model="openrouter:anthropic/claude-3.5-haiku", api_key="dummy", api_format="messages")
    calls = llm.tool_calls("Call echo for tokyo", tools=[echo])

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    assert client.calls[-1].get("responses") is None


def test_messages_format_rejects_non_anthropic_model(fake_anyllm) -> None:
    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy", api_format="messages")
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


def test_sync_stream_uses_client_level_async_responses_bridge(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")

    async def _unexpected_aresponses(**_: Any) -> Any:
        raise UnexpectedAsyncResponsesCall

    client.aresponses = _unexpected_aresponses
    client.queue_aresponses(
        _async_items(
            make_responses_text_delta("Hello"),
            make_responses_text_delta(" world"),
            make_responses_completed({"total_tokens": 7}),
        )
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    stream = llm.stream("Say hello")

    assert "".join(list(stream)) == "Hello world"
    assert stream.error is None
    assert stream.usage == {"total_tokens": 7}


def test_stream_treats_completed_reasoning_only_response_as_success(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_reasoning_response())

    llm = LLM(model="openrouter:openai/gpt-5.4-pro", api_key="dummy", api_format="responses", max_retries=0)
    call_id = "call_test_telegram_1"
    messages = [
        {
            "role": "user",
            "content": (
                "A Telegram user sent the message `h`. "
                "You already used the send_telegram tool to send `Hello`. "
                "Now continue the assistant turn based on the tool result."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "send_telegram", "arguments": '{"message":"Hello"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": '{"status":"sent","message":"Hello","chat_id":"test-chat"}',
        },
    ]

    stream = llm.stream(messages=messages, max_tokens=128)
    assert list(stream) == []
    assert stream.error is None
    assert stream.usage == {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129}


def test_stream_treats_completed_reasoning_only_stream_as_success(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_aresponses(
        _async_items(
            make_responses_reasoning_item_added(),
            make_responses_reasoning_item_done(),
            make_responses_completed({"input_tokens": 1, "output_tokens": 128, "total_tokens": 129}),
        )
    )

    llm = LLM(model="openrouter:openai/gpt-5.4-pro", api_key="dummy", api_format="responses", max_retries=0)
    stream = llm.stream("Continue after a completed tool result.", max_tokens=128)

    assert list(stream) == []
    assert stream.error is None
    assert stream.usage == {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129}


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
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "usage" in kinds
    assert kinds[-1] == "final"


def test_stream_events_treat_completed_reasoning_only_response_as_success(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_reasoning_response())

    llm = LLM(model="openrouter:openai/gpt-5.4-pro", api_key="dummy", api_format="responses", max_retries=0)
    call_id = "call_test_telegram_1"
    messages = [
        {
            "role": "user",
            "content": (
                "A Telegram user sent the message `h`. "
                "You already used the send_telegram tool to send `Hello`. "
                "Now continue the assistant turn based on the tool result."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "send_telegram", "arguments": '{"message":"Hello"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": '{"status":"sent","message":"Hello","chat_id":"test-chat"}',
        },
    ]

    events = list(llm.stream_events(messages=messages, max_tokens=128))
    assert _compact_stream_events(events) == [
        ("usage", {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129}),
        (
            "final",
            {
                "text": None,
                "tool_calls": [],
                "tool_results": [],
                "usage": {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129},
                "error": None,
            },
        ),
    ]


def test_stream_events_treat_completed_reasoning_only_stream_as_success(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_aresponses(
        _async_items(
            make_responses_reasoning_item_added(),
            make_responses_reasoning_item_done(),
            make_responses_completed({"input_tokens": 1, "output_tokens": 128, "total_tokens": 129}),
        )
    )

    llm = LLM(model="openrouter:openai/gpt-5.4-pro", api_key="dummy", api_format="responses", max_retries=0)
    events = list(llm.stream_events("Continue after a completed tool result.", max_tokens=128))

    assert _compact_stream_events(events) == [
        ("usage", {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129}),
        (
            "final",
            {
                "text": None,
                "tool_calls": [],
                "tool_results": [],
                "usage": {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129},
                "error": None,
            },
        ),
    ]


def test_sync_stream_events_use_client_level_async_responses_bridge(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")

    async def _unexpected_aresponses(**_: Any) -> Any:
        raise UnexpectedAsyncResponsesCall

    client.aresponses = _unexpected_aresponses
    client.queue_aresponses(
        _async_items(
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
        )
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    events = list(llm.stream_events("Call echo for tokyo", tools=[echo]))

    assert _compact_stream_events(events) == [
        ("text", "Checking "),
        ("tool_call", ("call_1", "echo", '{"text":"tokyo"}')),
        ("tool_result", "TOKYO"),
        ("usage", {"total_tokens": 12}),
        (
            "final",
            {
                "text": "Checking ",
                "tool_calls": [("call_1", "echo", '{"text":"tokyo"}')],
                "tool_results": ["TOKYO"],
                "usage": {"total_tokens": 12},
                "error": None,
            },
        ),
    ]


def test_stream_events_parity_between_completion_and_responses(fake_anyllm) -> None:
    completion_client = fake_anyllm.ensure("openai")
    completion_client.queue_completion(iter(_completion_stream_event_items()))

    responses_client = fake_anyllm.ensure("openrouter")
    responses_client.queue_responses(
        iter([
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
        ])
    )

    completion_llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    responses_llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")

    completion_events = list(completion_llm.stream_events("Call echo for tokyo", tools=[echo]))
    responses_events = list(responses_llm.stream_events("Call echo for tokyo", tools=[echo]))

    assert _compact_stream_events(completion_events) == _compact_stream_events(responses_events)


def test_stream_events_parity_between_completion_and_messages(fake_anyllm) -> None:
    completion_client = fake_anyllm.ensure("openai")
    completion_client.queue_completion(iter(_completion_stream_event_items()))

    messages_client = fake_anyllm.ensure("openrouter")
    messages_client.queue_completion(iter(_completion_stream_event_items()))

    completion_llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    messages_llm = LLM(model="openrouter:anthropic/claude-3.5-haiku", api_key="dummy", api_format="messages")

    completion_events = list(completion_llm.stream_events("Call echo for tokyo", tools=[echo]))
    messages_events = list(messages_llm.stream_events("Call echo for tokyo", tools=[echo]))

    assert _compact_stream_events(completion_events) == _compact_stream_events(messages_events)


def test_non_stream_responses_tool_calls_converts_tools_payload(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(
        make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    )

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    calls = llm.tool_calls("Call echo for tokyo", tools=[echo])

    assert len(calls) == 1
    sent_tools = client.calls[-1]["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["name"] == "echo"
    assert "function" not in sent_tools[0]


def test_chat_reasoning_effort_for_responses_is_mapped(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_response(text="ready"))

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    assert llm.chat("Reply with ready", reasoning_effort="low") == "ready"

    call = client.calls[-1]
    assert call["responses"] is True
    assert call.get("reasoning") == {"effort": "low"}
    assert "reasoning_effort" not in call


def test_completed_reasoning_only_response_after_tool_result_is_not_retried(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_reasoning_response())

    llm = LLM(model="openrouter:openai/gpt-5.4-pro", api_key="dummy", api_format="responses", max_retries=0)
    call_id = "call_test_telegram_1"
    messages = [
        {
            "role": "user",
            "content": (
                "A Telegram user sent the message `h`. "
                "You already used the send_telegram tool to send `Hello`. "
                "Now continue the assistant turn based on the tool result."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "send_telegram", "arguments": '{"message":"Hello"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": '{"status":"sent","message":"Hello","chat_id":"test-chat"}',
        },
    ]

    assert llm.chat(messages=messages, max_tokens=128) == ""


def test_run_tools_treats_completed_reasoning_only_response_as_success(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_reasoning_response())

    llm = LLM(model="openrouter:openai/gpt-5.4-pro", api_key="dummy", api_format="responses", max_retries=0)
    result = llm.run_tools("Reply after a tool has already completed.", tools=[echo], max_tokens=128)

    assert result.kind == "text"
    assert result.text == ""
    assert result.error is None


def test_completion_preserves_extra_headers(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openai")
    client.queue_completion(make_response(text="hello"))

    llm = LLM(model="openai:gpt-4o-mini", api_key="dummy")
    assert llm.chat("Say hello", extra_headers={"X-Title": "Republic"}) == "hello"

    call = client.calls[-1]
    assert call.get("extra_headers") == {"X-Title": "Republic"}


def test_messages_preserves_extra_headers(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_completion(make_response(text="hello"))

    llm = LLM(model="openrouter:anthropic/claude-3.5-haiku", api_key="dummy", api_format="messages")
    assert llm.chat("Say hello", extra_headers={"X-Title": "Republic"}) == "hello"

    call = client.calls[-1]
    assert call.get("extra_headers") == {"X-Title": "Republic"}


def test_responses_drops_extra_headers(fake_anyllm) -> None:
    client = fake_anyllm.ensure("openrouter")
    client.queue_responses(make_responses_response(text="hello"))

    llm = LLM(model="openrouter:openrouter/free", api_key="dummy", api_format="responses")
    assert llm.chat("Say hello", extra_headers={"X-Title": "Republic"}) == "hello"

    call = client.calls[-1]
    assert "extra_headers" not in call


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
