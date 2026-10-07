from __future__ import annotations

from pathlib import Path

import pytest

import republic
from republic.history import InMemoryHistory
from tests.conftest import FakeService


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
            republic.assistant(tool_results=[republic.tool_result(call, "done")]),
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

    def test_assistant_tool_results_announce_calls_once(self) -> None:
        from republic._formats.base import normalize

        call = republic.ToolCall("call_1", "lookup", "{}")
        result = republic.tool_result(call, "done")

        messages = normalize([republic.assistant(tool_calls=[call]), republic.assistant(tool_results=[result])])

        assert messages == [
            republic.Message("assistant", tool_calls=(call,)),
            republic.Message("tool", tool_results=(result,)),
        ]
