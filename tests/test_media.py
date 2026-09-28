# Copyright 2026 Vercel, Inc. Licensed under Apache-2.0.
# Media conversion scenarios adapted from ai-python c788059dd1; see NOTICE.
"""Messages reach the actual official SDK with ordered, persistable media."""

import httpx
import pytest

from republic import (
    FilePart,
    ProviderError,
    Request,
    Response,
    TextPart,
    UnsupportedRequestError,
    generate,
    stream,
)
from republic.providers.openai import OpenAIResponses
from tests import anthropic_fixtures as anthropic
from tests import codex_fixtures as codex
from tests.http_fixtures import Bytes, streaming
from tests.openai_fixtures import Wire, chunk, completion, request, sse
from tests.responses_fixtures import message, reasoning, response, terminal


def frames(*records):
    return b"".join(sse(record) for record in records)


pytestmark = pytest.mark.asyncio

CHAT = [
    (
        FilePart(
            data="https://assets.test/image.png",
            media_type="image/png",
            provider_metadata={"openai": {"detail": "low"}},
        ),
        {"type": "image_url", "image_url": {"url": "https://assets.test/image.png", "detail": "low"}},
    ),
    (
        FilePart.from_bytes(b"image", media_type="image/png"),
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
    ),
    (
        FilePart.from_bytes(b"audio", media_type="audio/mpeg"),
        {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "mp3"}},
    ),
    (
        FilePart.from_bytes(b"audio", media_type="audio/ogg"),
        {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "ogg"}},
    ),
    (
        FilePart(data="data:audio/wav;base64,YXVkaW8=", media_type="audio/wav"),
        {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "wav"}},
    ),
    (
        FilePart(data="https://assets.test/video.mp4", media_type="video/mp4"),
        {"type": "video_url", "video_url": {"url": "https://assets.test/video.mp4"}},
    ),
    (
        FilePart.from_bytes(b"video", media_type="video/mp4"),
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,dmlkZW8="}},
    ),
    (
        FilePart.from_bytes(b"pdf", media_type="application/pdf", filename="input.pdf"),
        {"type": "file", "file": {"file_data": "data:application/pdf;base64,cGRm", "filename": "input.pdf"}},
    ),
    (
        FilePart(data="file-pdf", encoding="file_id", media_type="application/pdf"),
        {"type": "file", "file": {"file_id": "file-pdf"}},
    ),
    (FilePart.from_bytes(b"text", media_type="text/plain"), {"type": "text", "text": "text"}),
]


@pytest.mark.parametrize("part,wire_part", CHAT)
@pytest.mark.parametrize("use_stream", [False, True])
async def test_chat_media_order_serde_and_no_fetch(part, wire_part, use_stream):
    req = request()
    req.messages[0].parts = [TextPart(text="before"), part, TextPart(text="after")]
    req = Request.model_validate_json(req.model_dump_json())
    body = Bytes([frames(chunk({"content": "seen"}), chunk({}, finish="stop")), b"data: [DONE]\n\n"])
    reply = streaming(body) if use_stream else httpx.Response(200, json=completion())
    async with Wire([reply]) as wire:
        if use_stream:
            async with stream(wire.provider, req) as output:
                async for _ in output:
                    pass
            assert output.response is not None and body.closed == 1
        else:
            await generate(wire.provider, req)
        content = [{"type": "text", "text": "before"}, wire_part, {"type": "text", "text": "after"}]
        # Text files follow upstream's decoded-text conversion.
        expected = "beforetextafter" if wire_part["type"] == "text" else content
        assert wire.payload()["messages"][0]["content"] == expected
        assert len(wire.requests) == 1


RESPONSES = [
    (
        FilePart(data="https://assets.test/image.png", media_type="image/png"),
        {"type": "input_image", "image_url": "https://assets.test/image.png"},
    ),
    (
        FilePart.from_bytes(b"image", media_type="image/png"),
        {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
    ),
    (
        FilePart(data="file-image", encoding="file_id", media_type="image/png"),
        {"type": "input_image", "file_id": "file-image"},
    ),
    (
        FilePart(data="https://assets.test/doc.pdf", media_type="application/pdf"),
        {"type": "input_file", "file_url": "https://assets.test/doc.pdf"},
    ),
    (
        FilePart.from_bytes(b"pdf", media_type="application/pdf"),
        {"type": "input_file", "file_data": "data:application/pdf;base64,cGRm", "filename": "document.pdf"},
    ),
    (
        FilePart(data="file-pdf", encoding="file_id", media_type="application/pdf"),
        {"type": "input_file", "file_id": "file-pdf"},
    ),
    (FilePart.from_bytes(b"text", media_type="text/plain"), {"type": "input_text", "text": "text"}),
]


@pytest.mark.parametrize("part,wire_part", RESPONSES)
async def test_responses_media_and_encrypted_history(part, wire_part):
    req = request()
    req.messages[0].parts = [TextPart(text="before"), part, TextPart(text="after")]
    opaque = reasoning("", summary=[])
    async with Wire(
        [httpx.Response(200, json=response([opaque, message()])), httpx.Response(200, json=response())],
        provider_type=OpenAIResponses,
    ) as wire:
        first = await generate(wire.provider, req)
        req.messages.append(Response.model_validate_json(first.model_dump_json()).message)
        await generate(wire.provider, Request.model_validate_json(req.model_dump_json()))
        for index in (0, 1):
            payload = wire.payload(index)
            assert payload["input"][0]["content"] == [
                {"type": "input_text", "text": "before"},
                wire_part,
                {"type": "input_text", "text": "after"},
            ]
        assert wire.payload(1)["input"][1] == opaque
        assert len(wire.requests) == 2


@pytest.mark.parametrize("part,wire_part", RESPONSES[:3])
async def test_codex_images_on_single_sse_request(part, wire_part):
    req = request()
    req.messages[0].parts = [part]
    body = Bytes([frames(terminal())])
    async with codex.Wire([streaming(body)]) as wire:
        await generate(wire.provider, req)
        assert wire.payload()["input"][0]["content"] == [wire_part]
        assert len(wire.requests) == body.closed == 1


@pytest.mark.parametrize(
    "media_type,kind", [("image/png", "image"), ("application/pdf", "document"), ("text/plain", "document")]
)
@pytest.mark.parametrize("encoding", ["url", "base64", "file_id"])
async def test_anthropic_media_cache_and_signed_history(media_type, kind, encoding):
    data = {"url": "https://assets.test/file", "base64": "Ynl0ZXM=", "file_id": "file-fixture"}[encoding]
    part = FilePart(
        data=data,
        media_type=media_type,
        encoding=encoding,
        provider_metadata={"anthropic": {"cache_control": {"type": "ephemeral"}}},
    )
    req = anthropic.request()
    req.messages[0].parts = [TextPart(text="before"), part]
    thinking = {"type": "thinking", "thinking": "consider", "signature": "original"}
    async with anthropic.Wire([
        httpx.Response(200, json=anthropic.message([thinking])),
        httpx.Response(200, json=anthropic.message()),
    ]) as wire:
        first = await generate(wire.provider, req)
        req.messages.append(Response.model_validate_json(first.model_dump_json()).message)
        await generate(wire.provider, Request.model_validate_json(req.model_dump_json()))
        source = (
            {"type": "url", "url": data}
            if encoding == "url"
            else (
                {"type": "file", "file_id": data}
                if encoding == "file_id"
                else {"type": "base64", "media_type": media_type, "data": data}
            )
        )
        if encoding == "base64" and media_type == "text/plain":
            source = {"type": "text", "media_type": "text/plain", "data": "bytes"}
        expected = [
            {"type": "text", "text": "before"},
            {"type": kind, "source": source, "cache_control": {"type": "ephemeral"}},
        ]
        assert wire.payload()["messages"][0]["content"] == expected
        assert wire.payload(1)["messages"][0]["content"] == expected
        assert wire.payload(1)["messages"][1]["content"] == [thinking]
        assert len(wire.requests) == 2


@pytest.mark.parametrize(
    "part",
    [
        FilePart(data="file:///private", media_type="image/png"),
        FilePart(data="https://assets.test/audio.wav", media_type="audio/wav"),
        FilePart(data="bad!", encoding="base64", media_type="image/png"),
        FilePart(data="data:audio/wav;base64,YQ==", media_type="image/png"),
        FilePart(data="id", encoding="file_id", media_type="image/png"),
    ],
)
async def test_unrepresentable_chat_media_never_fetches_or_sends(part):
    req = request()
    req.messages[0].parts = [part]
    async with Wire([]) as wire:
        with pytest.raises(UnsupportedRequestError):
            await generate(wire.provider, req)
        assert not wire.requests


async def test_openrouter_opaque_reasoning_fragments_survive_replay():
    encrypted = {
        "index": 1,
        "type": "reasoning.encrypted",
        "id": "opaque-id",
        "format": "google-gemini-v1",
        "data": "aa",
    }
    text = {"index": 0, "type": "reasoning.text", "text": "think", "signature": "sig"}
    body = Bytes([
        frames(
            chunk({"reasoning_details": [encrypted, text]}),
            chunk({
                "reasoning_details": [{"index": 1, "data": "bb"}, {"index": 0, "text": "ing", "signature": "nature"}]
            }),
            chunk({"content": "seen"}, finish="stop"),
        ),
        b"data: [DONE]\n\n",
    ])
    async with Wire([
        streaming(body),
        httpx.Response(200, json=completion({"reasoning_details": [{"type": "reasoning.encrypted", "data": "full"}]})),
    ]) as wire:
        req = request()
        async with stream(wire.provider, req) as output:
            async for _ in output:
                pass
        assert output.response is not None
        req.messages.append(Response.model_validate_json(output.response.model_dump_json()).message)
        result = await generate(wire.provider, req)
        assert wire.payload(1)["messages"][1]["reasoning_details"] == [
            {**encrypted, "data": "aabb"},
            {**text, "text": "thinking", "signature": "signature"},
        ]
        assert result.message.provider_metadata == {
            "openai": {"reasoning_details": [{"type": "reasoning.encrypted", "data": "full"}]}
        }
        assert body.closed == 1 and len(wire.requests) == 2


async def test_conflicting_reasoning_identity_fails():
    body = Bytes([
        frames(
            chunk({"reasoning_details": [{"index": 0, "type": "reasoning.encrypted", "data": "a", "id": "one"}]}),
            chunk({"reasoning_details": [{"index": 0, "id": "two", "data": "b"}]}),
        )
    ])
    async with Wire([streaming(body)]) as wire:
        with pytest.raises(ProviderError):
            async with stream(wire.provider, request()) as output:
                async for _ in output:
                    pass
        assert body.closed == 1


@pytest.mark.parametrize("encoding,data", [("url", "https://assets.test/opaque-image"), ("file_id", "file-image")])
async def test_anthropic_image_reference_needs_no_guessed_mime(encoding, data):
    req = anthropic.request()
    req.messages[0].parts = [FilePart(data=data, encoding=encoding, media_type="image/*")]
    async with anthropic.Wire([httpx.Response(200, json=anthropic.message())]) as wire:
        await generate(wire.provider, req)
        assert wire.payload()["messages"][0]["content"] == [
            {
                "type": "image",
                "source": {"type": "url", "url": data} if encoding == "url" else {"type": "file", "file_id": data},
            }
        ]
