from __future__ import annotations

import json

import pytest

import republic
from republic import Image, Message, ProviderData, Reasoning, Text, ToolCall, Video


def test_single_text_message_uses_plain_content() -> None:
    message = Message("user", (Text("hello"),))

    data = message.to_dict()

    assert data == {"role": "user", "content": "hello"}
    assert Message.from_dict(data) == message


def test_multiple_parts_use_typed_content_list() -> None:
    message = Message(
        "assistant",
        (
            Reasoning("thinking"),
            Text("look"),
            Text("again"),
            Image("image/png", data=b"\x89PNG"),
            Video("video/mp4", url="https://example.com/a.mp4"),
            ProviderData("responses", {"type": "reasoning", "id": "rs_1"}),
        ),
    )

    data = message.to_dict()

    assert data["content"] == [
        {"type": "reasoning", "text": "thinking"},
        {"type": "text", "text": "look"},
        {"type": "text", "text": "again"},
        {"type": "image", "media_type": "image/png", "data": "iVBORw=="},
        {"type": "video", "media_type": "video/mp4", "url": "https://example.com/a.mp4"},
        {"type": "provider_data", "api_format": "responses", "payload": {"type": "reasoning", "id": "rs_1"}},
    ]
    assert Message.from_dict(json.loads(json.dumps(data))) == message


def test_single_non_text_part_uses_content_list() -> None:
    message = Message("user", (Image("image/png", url="https://example.com/a.png"),))

    data = message.to_dict()

    assert data["content"] == [{"type": "image", "media_type": "image/png", "url": "https://example.com/a.png"}]
    assert Message.from_dict(data) == message


def test_tool_calls_round_trip() -> None:
    call = ToolCall("call_1", "lookup", '{"q": "x"}', metadata={"thought_signature": "sig"})
    message = republic.assistant(tool_calls=[call])

    data = message.to_dict()

    assert data == {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "name": "lookup", "arguments": '{"q": "x"}', "metadata": {"thought_signature": "sig"}}
        ],
    }
    restored = Message.from_dict(json.loads(json.dumps(data)))
    assert restored == message
    assert restored.tool_calls[0].metadata == {"thought_signature": "sig"}


def test_tool_message_with_text_output() -> None:
    message = republic.tool(ToolCall("call_1", "lookup", "{}"), "done", is_error=True)

    data = message.to_dict()

    assert data == {
        "role": "tool",
        "content": "done",
        "tool_call": {"id": "call_1", "name": "lookup", "arguments": "{}"},
        "is_error": True,
    }
    assert Message.from_dict(data) == message


def test_tool_message_with_multiple_parts() -> None:
    message = republic.tool(ToolCall("call_1", "screenshot", "{}"), "captured", Image("image/png", data=b"png"))

    data = message.to_dict()

    assert data["role"] == "tool"
    assert data["content"] == [
        {"type": "text", "text": "captured"},
        {"type": "image", "media_type": "image/png", "data": "cG5n"},
    ]
    assert "is_error" not in data
    assert Message.from_dict(json.loads(json.dumps(data))) == message


def test_from_dict_rejects_unknown_part_type() -> None:
    with pytest.raises(ValueError, match="Unknown message part type"):
        Message.from_dict({"role": "user", "content": [{"type": "audio"}]})
