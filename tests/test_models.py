from __future__ import annotations

import json
from pathlib import Path

import pytest

import republic
from republic.history import InMemoryHistory
from tests.conftest import FakeService


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("api_format", ["chat", "responses"])
async def test_openai_stream_stops_at_done(service: FakeService, provider: str, api_format: str) -> None:
    text_event = (
        {"choices": [{"delta": {"content": "Hello"}}]}
        if api_format == "chat"
        else {"type": "response.output_text.delta", "delta": "Hello"}
    )
    service.reply_events([text_event, "[DONE]", text_event])
    model = republic.get_model(f"{provider}:test-model", api_format=api_format, http_client=service.client())

    async with model.stream("Hi") as stream:
        events = [event async for event in stream]

    assert stream.text == "Hello"
    assert len(events) == 2
    assert isinstance(events[-1], republic.events.Completed)


@pytest.mark.parametrize(
    ("provider", "api_format", "payload"),
    [
        ("openai", "chat", "not-json"),
        ("openai", "responses", "not-json"),
        ("anthropic", "messages", "[DONE]"),
        ("google", "gemini", "[DONE]"),
    ],
)
async def test_invalid_stream_json_raises(service: FakeService, provider: str, api_format: str, payload: str) -> None:
    service.reply_events([payload, "[DONE]"])
    model = republic.get_model(f"{provider}:test-model", api_format=api_format, http_client=service.client())

    with pytest.raises(json.JSONDecodeError):
        async with model.stream("Hi") as stream:
            async for _ in stream:
                pass


def chat_reply(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class TestHistory:
    async def test_previous_turns_are_sent_with_new_input(self, service: FakeService) -> None:
        service.reply_json(chat_reply("I'm fine."))
        service.reply_json(chat_reply("Great."))
        model = republic.get_model(
            "openrouter:vendor/model",
            api_format="chat",
            history=InMemoryHistory(max_entries=10),
            http_client=service.client(),
        )

        await model.chat("Hello, how are you?")
        await model.chat("How about you?")

        assert service.body()["messages"] == [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine."},
            {"role": "user", "content": "How about you?"},
        ]

    async def test_trimming_restarts_at_a_user_message(self) -> None:
        history = InMemoryHistory(max_entries=3)
        call = republic.ToolCall("call_1", "lookup", "{}")

        await history.write([
            republic.user("first"),
            republic.assistant(tool_calls=[call]),
            republic.tool(call, "done"),
            republic.user("second"),
        ])

        assert await history.read() == [republic.user("second")]

    def test_rejects_non_positive_limit(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            InMemoryHistory(max_entries=0)


class TestContent:
    def test_image_from_path_guesses_media_type(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        path.write_bytes(b"jpeg")

        assert republic.image(path) == republic.Image("image/jpeg", data=b"jpeg")

    def test_image_from_data_url(self) -> None:
        assert republic.image("data:image/png;base64,cG5n") == republic.Image("image/png", data=b"png")

    def test_raw_bytes_need_a_media_type(self) -> None:
        with pytest.raises(ValueError, match="media_type"):
            republic.video(b"mp4")

    def test_tool_results_announce_calls_once(self) -> None:
        from republic.formats._base import normalize

        call = republic.ToolCall("call_1", "lookup", "{}")
        result = republic.tool(call, "done")

        assert normalize([republic.assistant(tool_calls=[call]), result]) == [
            republic.assistant(tool_calls=[call]),
            result,
        ]

    def test_unannounced_calls_join_the_assistant_turn_before_their_results(self) -> None:
        from republic.formats._base import normalize

        first = republic.ToolCall("call_1", "lookup", "{}")
        second = republic.ToolCall("call_2", "lookup", "{}")
        results = [republic.tool(first, "one"), republic.tool(second, "two")]

        assert normalize([republic.user("hi"), *results]) == [
            republic.user("hi"),
            republic.assistant(tool_calls=[first, second]),
            *results,
        ]
        assert normalize([republic.assistant("Checking."), *results]) == [
            republic.assistant("Checking.", tool_calls=[first, second]),
            *results,
        ]

    def test_tool_call_belongs_only_to_tool_messages(self) -> None:
        call = republic.ToolCall("call_1", "lookup", "{}")

        with pytest.raises(ValueError, match="tool_call"):
            republic.Message("tool")
        with pytest.raises(ValueError, match="tool_call"):
            republic.Message("assistant", tool_call=call)
