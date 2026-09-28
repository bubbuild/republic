"""Codex's sourced input_audio/audio_url wire through the real OpenAI client."""

import asyncio

import pytest

from republic import (
    FilePart,
    Message,
    Request,
    Response,
    TextPart,
    ToolResultPart,
    UnsupportedRequestError,
    generate,
    stream,
)
from republic.providers.openai import OpenAIResponses
from tests.codex_fixtures import Wire
from tests.http_fixtures import Bytes, streaming
from tests.openai_fixtures import Wire as OpenAIWire
from tests.openai_fixtures import request, sse
from tests.responses_fixtures import added, call, message, reasoning, terminal

pytestmark = pytest.mark.asyncio

AUDIO = [
    (FilePart(data="https://assets.test/audio.wav", media_type="audio/wav"), "https://assets.test/audio.wav"),
    (FilePart.from_bytes(b"audio", media_type="audio/mpeg"), "data:audio/mpeg;base64,YXVkaW8="),
    (FilePart(data="data:audio/ogg;base64,YXVkaW8=", media_type="audio/ogg"), "data:audio/ogg;base64,YXVkaW8="),
]


@pytest.mark.parametrize("part,url", AUDIO)
@pytest.mark.parametrize("use_stream", [False, True])
async def test_audio_image_order_and_complete_native_history(part, url, use_stream):
    req = request()
    req.messages[0].parts = [
        TextPart(text="before"),
        part,
        FilePart(data="https://assets.test/image", media_type="image/png"),
        TextPart(text="after"),
    ]
    before = req.model_dump_json()
    req = Request.model_validate_json(before)
    native = [reasoning(summary=[]), call()]
    bodies = [Bytes([sse(terminal(native))]), Bytes([sse(terminal())])]
    async with Wire([streaming(body) for body in bodies]) as wire:
        if use_stream:
            async with stream(wire.provider, req) as output:
                async for _ in output:
                    pass
            result = output.response
            assert result is not None
        else:
            result = await generate(wire.provider, req)
        assert req.model_dump_json() == before
        req.messages.extend([
            Response.model_validate_json(result.model_dump_json()).message,
            Message(role="tool", parts=[ToolResultPart(tool_call_id="call_1", tool_name="weather", result="done")]),
        ])
        await generate(wire.provider, Request.model_validate_json(req.model_dump_json()))
        expected = [
            {"type": "input_text", "text": "before"},
            {"type": "input_audio", "audio_url": url},
            {"type": "input_image", "image_url": "https://assets.test/image"},
            {"type": "input_text", "text": "after"},
        ]
        for index in (0, 1):
            assert wire.payload(index)["input"][0]["content"] == expected
            assert wire.payload(index)["stream"] is True
            assert wire.requests[index].url.path == "/backend-api/codex/responses"
        assert wire.payload(1)["input"][1:3] == native
        assert wire.payload(1)["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": "done"}
        assert len(wire.requests) == 2 and all(body.closed == 1 for body in bodies)
        assert not wire.client.is_closed()


@pytest.mark.parametrize(
    "part",
    [
        FilePart(data="file-audio", media_type="audio/wav", encoding="file_id"),
        FilePart(data="/private/audio.wav", media_type="audio/wav"),
        FilePart(data="file:///private/audio.wav", media_type="audio/wav"),
        FilePart(data="bad!", media_type="audio/wav", encoding="base64"),
        FilePart(data="data:audio/ogg;base64,YQ==", media_type="audio/wav"),
        FilePart(data="https://assets.test/audio.wav", media_type="audio/wav", filename="audio.wav"),
        FilePart(
            data="https://assets.test/audio.wav",
            media_type="audio/wav",
            provider_metadata={"openai": {"format": "wav"}},
        ),
    ],
)
async def test_unrepresentable_audio_fails_without_http_or_fetch(part):
    req = request()
    req.messages[0].parts = [part]
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests


@pytest.mark.parametrize("role", ["system", "assistant"])
async def test_audio_is_user_content_only(role):
    req = request()
    req.messages = [Message(role=role, parts=[AUDIO[0][0]])]
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests


async def test_codex_audio_wire_does_not_extend_standard_responses():
    req = request()
    req.messages[0].parts = [AUDIO[0][0]]
    async with OpenAIWire([], provider_type=OpenAIResponses) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests


@pytest.mark.parametrize("mode", ["early", "cancel", "generate_cancel"])
async def test_audio_close_cancel_and_client_reuse(mode):
    req = request()
    req.messages[0].parts = [AUDIO[0][0]]
    body = Bytes([sse(added(message("", status="in_progress")))], wait=True)
    async with Wire([streaming(body), streaming(Bytes([sse(terminal())]))]) as wire:
        if mode == "generate_cancel":
            task = asyncio.create_task(generate(wire.provider, req))
            await asyncio.wait_for(body.waiting.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            async with stream(wire.provider, req) as output:
                await anext(output)
                if mode == "cancel":
                    task = asyncio.create_task(anext(output))
                    await asyncio.wait_for(body.waiting.wait(), 3)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
            assert output.response is None
            assert output.status == ("cancelled" if mode == "cancel" else "closed")
        assert body.closed == 1 and not wire.client.is_closed()
        await generate(wire.provider, req)
        assert len(wire.requests) == 2
