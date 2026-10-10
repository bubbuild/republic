from __future__ import annotations

from pathlib import Path

import pytest

import republic
from republic.errors import UnsupportedFeatureError
from tests.conftest import FakeService


@pytest.mark.parametrize("source_kind", ["path", "data-url", "bytes"])
async def test_audio_sources_reach_the_provider(service: FakeService, tmp_path: Path, source_kind: str) -> None:
    path = tmp_path / "voice.wav"
    path.write_bytes(b"voice")
    sources = {
        "path": path,
        "data-url": "data:audio/wav;base64,dm9pY2U=",
        "bytes": b"voice",
    }
    service.reply_json({"choices": [{"message": {"content": "heard"}}]})
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client())
    response = await model.chat(
        republic.user(
            republic.audio(sources[source_kind], **({"media_type": "audio/wav"} if source_kind == "bytes" else {}))
        )
    )
    assert response.text == "heard"
    assert service.body()["messages"][0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": "dm9pY2U=", "format": "wav"}}
    ]


def test_audio_bytes_require_a_media_type() -> None:
    with pytest.raises(ValueError, match="media_type"):
        republic.audio(b"voice")


@pytest.mark.parametrize("source", [b"voice", "https://example.test/voice.ogg"])
async def test_gemini_encodes_audio(service: FakeService, source: bytes | str) -> None:
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "heard"}]}, "finishReason": "STOP"}]})
    audio = republic.audio(source, media_type="audio/ogg")
    response = await republic.get_model("google:test", http_client=service.client()).chat(
        republic.user("Listen", audio)
    )
    expected = (
        {"inlineData": {"mimeType": "audio/ogg", "data": "dm9pY2U="}}
        if isinstance(source, bytes)
        else {"fileData": {"mimeType": "audio/ogg", "fileUri": source}}
    )
    assert service.body()["contents"][0]["parts"] == [{"text": "Listen"}, expected]
    assert response.text == "heard"


@pytest.mark.parametrize(
    "suffix,alias,canonical", [("wav", "audio/x-wav", "audio/wav"), ("aiff", "audio/x-aiff", "audio/aiff")]
)
@pytest.mark.parametrize("source_kind", ["path", "url", "data-url"])
async def test_gemini_encodes_wav_and_aiff_with_the_canonical_mime_type(
    service: FakeService,
    tmp_path: Path,
    source_kind: str,
    suffix: str,
    alias: str,
    canonical: str,
) -> None:
    """Google accepts only the canonical names, not the aliases the standard library infers for these suffixes."""
    path = tmp_path / f"voice.{suffix}"
    path.write_bytes(b"voice")
    sources = {
        "path": path,
        "url": f"https://media.example.test/voice.{suffix}",
        "data-url": f"data:{alias};base64,dm9pY2U=",
    }
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "heard"}]}, "finishReason": "STOP"}]})
    response = await republic.get_model("google:test", http_client=service.client()).chat(
        republic.user("Listen", republic.audio(sources[source_kind]))
    )
    expected = (
        {"fileData": {"mimeType": canonical, "fileUri": sources["url"]}}
        if source_kind == "url"
        else {"inlineData": {"mimeType": canonical, "data": "dm9pY2U="}}
    )
    assert service.body()["contents"][0]["parts"] == [{"text": "Listen"}, expected]
    assert response.text == "heard"


@pytest.mark.parametrize("mime,wire_format", [("audio/wav", "wav"), ("audio/x-wav", "wav"), ("audio/mpeg", "mp3")])
async def test_chat_encodes_audio(service: FakeService, mime: str, wire_format: str) -> None:
    service.reply_json({"choices": [{"message": {"content": "heard"}}]})
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client())
    await model.chat(republic.user(republic.audio(b"voice", media_type=mime)))
    assert service.body()["messages"][0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": "dm9pY2U=", "format": wire_format}}
    ]


@pytest.mark.parametrize(
    "provider,api_format", [("openai", "responses"), ("anthropic", "messages"), ("openai", "chat")]
)
async def test_unsupported_audio_fails_before_request(service: FakeService, provider: str, api_format: str) -> None:
    model = republic.get_model(f"{provider}:test", api_format=api_format, http_client=service.client())
    with pytest.raises(UnsupportedFeatureError):
        await model.chat(republic.user(republic.audio("https://example.test/voice.ogg", media_type="audio/ogg")))
    assert not service.requests


async def test_openrouter_encodes_ogg_without_transcoding(service: FakeService) -> None:
    service.reply_json({"choices": [{"message": {"content": "heard"}}]})
    model = republic.get_model("openrouter:test", api_format="chat", http_client=service.client())
    await model.chat(republic.user(republic.audio(b"voice", media_type="audio/ogg")))
    assert service.body()["messages"][0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": "dm9pY2U=", "format": "ogg"}}
    ]


async def test_openai_rejects_ogg_before_sending(service: FakeService) -> None:
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client())
    with pytest.raises(UnsupportedFeatureError, match="audio/ogg"):
        await model.chat(republic.user(republic.audio(b"voice", media_type="audio/ogg")))
    assert not service.requests


@pytest.mark.parametrize(
    "loader,mime,suffix",
    [(republic.image, "image/png", "png"), (republic.audio, "audio/ogg", "ogg"), (republic.video, "video/mp4", "mp4")],
)
async def test_signed_media_urls_reach_the_provider(service: FakeService, loader, mime: str, suffix: str) -> None:
    url = f"https://media.example.test/file.{suffix}?signature=opaque#fragment"
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "received"}]}, "finishReason": "STOP"}]})
    model = republic.get_model("google:test", http_client=service.client())
    response = await model.chat(republic.user("Inspect", loader(url)))
    assert response.text == "received"
    assert service.body()["contents"][0]["parts"] == [
        {"text": "Inspect"},
        {"fileData": {"mimeType": mime, "fileUri": url}},
    ]


@pytest.mark.parametrize(
    "provider,api_format",
    [
        ("openai", "responses"),
        ("codex", "responses"),
        ("anthropic", "messages"),
        ("github-copilot", "messages"),
    ],
)
async def test_unsupported_inline_audio_is_a_format_error(
    service: FakeService,
    provider: str,
    api_format: str,
) -> None:
    model = republic.get_model(f"{provider}:test", api_format=api_format, api_key="test", http_client=service.client())
    with pytest.raises(UnsupportedFeatureError, match="audio"):
        await model.chat(republic.user("Listen", republic.Audio("audio/wav", data=b"voice")))
    assert not service.requests


async def test_streaming_sends_audio_and_returns_text(service: FakeService) -> None:
    service.reply_events([
        {"choices": [{"delta": {"content": "heard"}, "finish_reason": "stop"}]},
        "[DONE]",
    ])
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client())
    audio = republic.Audio("audio/wav", data=b"voice")
    async with model.stream(republic.user("Listen", audio)) as stream:
        _ = [event async for event in stream]
    assert stream.response.text == "heard"
    assert service.body()["messages"][0]["content"] == [
        {"type": "text", "text": "Listen"},
        {"type": "input_audio", "input_audio": {"data": "dm9pY2U=", "format": "wav"}},
    ]


async def test_follow_up_request_replays_audio(service: FakeService) -> None:
    from republic.history import InMemoryHistory

    for _ in range(2):
        service.reply_json({"choices": [{"message": {"content": "heard"}}]})
    history = InMemoryHistory()
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client(), history=history)
    await model.chat(republic.user("Listen", republic.audio(b"voice", media_type="audio/wav")))
    await model.chat("What did you hear?")
    assert service.body()["messages"][0]["content"] == [
        {"type": "text", "text": "Listen"},
        {"type": "input_audio", "input_audio": {"data": "dm9pY2U=", "format": "wav"}},
    ]
    assert service.body()["messages"][1:] == [
        {"role": "assistant", "content": "heard"},
        {"role": "user", "content": "What did you hear?"},
    ]


@pytest.mark.parametrize("mime", ["audio/pcm16", "audio/pcm24", "audio/webm"])
async def test_openrouter_rejects_unsupported_audio_mime_types(service: FakeService, mime: str) -> None:
    model = republic.get_model("openrouter:test", api_format="chat", http_client=service.client())
    with pytest.raises(UnsupportedFeatureError):
        await model.chat(republic.user(republic.audio(b"voice", media_type=mime)))
    assert not service.requests
