from __future__ import annotations

from republic.conversation import conversation_from_messages
from republic.providers.codecs import (
    conversation_to_anthropic_messages,
    conversation_to_completion_messages,
    conversation_to_openai_responses_input,
)


def test_openai_responses_serializes_structured_tool_output() -> None:
    conversation = conversation_from_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": {"text": "hi"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": {"ok": True, "items": [1, 2]}},
    ])

    instructions, input_items = conversation_to_openai_responses_input(conversation)

    assert instructions == "Be concise."
    assert input_items[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok":true,"items":[1,2]}',
    }


def test_openai_responses_preserves_typed_tool_output_items() -> None:
    conversation = conversation_from_messages([
        {"role": "user", "content": "show files"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "input_text", "text": "done"},
                {"type": "input_file", "file_id": "file_123"},
            ],
        },
    ])

    _, input_items = conversation_to_openai_responses_input(conversation)

    assert input_items[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "done"},
            {"type": "input_file", "file_id": "file_123"},
        ],
    }


def test_anthropic_codec_merges_consecutive_tool_results_into_one_user_turn() -> None:
    conversation = conversation_from_messages([
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city":"Macau"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lookup_time", "arguments": '{"timezone":"Asia/Macau"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "10:00"},
    ])

    _, messages = conversation_to_anthropic_messages(conversation)

    assert messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "lookup_weather", "input": {"city": "Macau"}},
                {"type": "tool_use", "id": "call_2", "name": "lookup_time", "input": {"timezone": "Asia/Macau"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "10:00"},
            ],
        },
    ]


def test_completion_codec_preserves_tool_call_metadata() -> None:
    conversation = conversation_from_messages([
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"tokyo"}'},
                    "thought_signature": "sig_123",
                }
            ],
        }
    ])

    messages = conversation_to_completion_messages(conversation)

    assert messages == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"tokyo"}'},
                    "thought_signature": "sig_123",
                }
            ],
        }
    ]
