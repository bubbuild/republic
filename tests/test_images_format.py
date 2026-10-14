from __future__ import annotations

import base64
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP

import httpx2
import pytest

import republic
from republic import errors
from tests.conftest import FakeService

PNG = republic.Image("image/png", data=b"png")
MASK = republic.Image("image/png", data=b"mask")
B64_PNG = base64.b64encode(b"png").decode()


def _form_parts(request: httpx2.Request) -> list[Message]:
    """Split a multipart request into its parts, in order."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = BytesParser(policy=HTTP).parsebytes(head + request.content)
    return list(message.iter_parts())


def _azure_image_model(service: FakeService, *, retry_delay: float = 0.5) -> republic.ImageModel:
    return republic.get_image_model(
        "azure-openai:image-deployment",
        api_key="key",
        api_base="https://res.openai.azure.com/openai/v1",
        http_client=service.client(),
        retry_delay=retry_delay,
    )


class TestOpenAIImages:
    async def test_generate_sends_options_and_parses_images(self, service: FakeService) -> None:
        service.reply_json({
            "created": 1,
            "output_format": "webp",
            "data": [{"b64_json": B64_PNG, "revised_prompt": "A tabby cat"}],
            "usage": {"input_tokens": 9, "output_tokens": 272, "total_tokens": 281},
        })
        model = republic.get_image_model("openai:gpt-image-2", api_key="key", http_client=service.client())

        response = await model.generate(
            "A cat", n=1, size="1024x1024", quality="low", output_format="webp", extra_body={"user": "u1"}
        )

        assert service.requests[0].url == "https://api.openai.com/v1/images/generations"
        assert service.requests[0].headers["authorization"] == "Bearer key"
        assert service.body() == {
            "model": "gpt-image-2",
            "prompt": "A cat",
            "n": 1,
            "size": "1024x1024",
            "quality": "low",
            "output_format": "webp",
            "user": "u1",
        }
        assert response.image_parts == [republic.Image("image/webp", data=b"png")]
        assert response.revised_prompts == ["A tabby cat"]
        assert response.token_usage == republic.TokenUsage(input_tokens=9, output_tokens=272)

    async def test_url_results_default_to_png(self, service: FakeService) -> None:
        service.reply_json({"data": [{"url": "https://example.com/cat.png"}]})
        model = republic.get_image_model("openai:dall-e-3", api_key="key", http_client=service.client())

        response = await model.generate("A cat")

        assert response.image_parts == [republic.Image("image/png", url="https://example.com/cat.png")]
        assert response.revised_prompts == [None]

    async def test_edit_sends_image_references_and_mask(self, service: FakeService) -> None:
        service.reply_json({"data": [{"b64_json": B64_PNG}]})
        model = republic.get_image_model("openai:gpt-image-2", api_key="key", http_client=service.client())
        remote = republic.image("https://example.com/hat.png")

        await model.edit("Put the hat on the cat", [PNG, remote], mask=MASK, background="transparent")

        assert service.requests[0].url == "https://api.openai.com/v1/images/edits"
        assert service.body() == {
            "model": "gpt-image-2",
            "prompt": "Put the hat on the cat",
            "images": [{"image_url": PNG.data_url}, {"image_url": "https://example.com/hat.png"}],
            "mask": {"image_url": MASK.data_url},
            "background": "transparent",
        }

    async def test_edit_accepts_a_single_image(self, service: FakeService) -> None:
        service.reply_json({"data": [{"b64_json": B64_PNG}]})
        model = republic.get_image_model("openai:gpt-image-2", api_key="key", http_client=service.client())

        await model.edit("Add a hat", PNG)

        assert service.body()["images"] == [{"image_url": PNG.data_url}]

    async def test_edit_needs_an_image(self) -> None:
        model = republic.get_image_model("openai:gpt-image-2", api_key="key")

        with pytest.raises(ValueError, match="at least one image"):
            await model.edit("Add a hat", [])

    async def test_aspect_ratio_is_rejected(self) -> None:
        model = republic.get_image_model("openai:gpt-image-2", api_key="key")

        with pytest.raises(errors.UnsupportedFeatureError, match="aspect_ratio"):
            await model.generate("A cat", aspect_ratio="16:9")

    async def test_azure_uses_the_deployment_endpoint(self, service: FakeService) -> None:
        service.reply_json({"data": [{"b64_json": B64_PNG}]})
        model = republic.get_image_model(
            "azure-openai:image-deployment",
            api_key="key",
            api_base="https://res.openai.azure.com/openai/v1",
            http_client=service.client(),
        )

        await model.generate("A cat")

        assert service.requests[0].url == "https://res.openai.azure.com/openai/v1/images/generations"
        assert service.requests[0].headers["api-key"] == "key"
        assert service.body()["model"] == "image-deployment"

    async def test_azure_edits_use_multipart_form_data(self, service: FakeService) -> None:
        service.reply_json({"data": [{"b64_json": B64_PNG}]})
        jpeg = republic.Image("image/jpeg", data=b"jpeg")

        await _azure_image_model(service).edit("Add a hat", [PNG, jpeg], mask=MASK, n=2, extra_body={"stream": False})

        request = service.requests[0]
        assert request.url == "https://res.openai.azure.com/openai/v1/images/edits"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts = [
            (part.get_param("name", header="content-disposition"), part.get_filename(), part)
            for part in _form_parts(request)
        ]
        assert [(name, filename) for name, filename, _ in parts] == [
            ("model", None),
            ("prompt", None),
            ("n", None),
            ("stream", None),
            ("image[]", "image-0.png"),
            ("image[]", "image-1.jpg"),
            ("mask", "mask.png"),
        ]
        assert [part.get_payload(decode=True) for _, _, part in parts] == [
            b"image-deployment",
            b"Add a hat",
            b"2",
            b"false",
            b"png",
            b"jpeg",
            b"mask",
        ]
        assert [part.get_content_type() for _, filename, part in parts if filename] == [
            "image/png",
            "image/jpeg",
            "image/png",
        ]

    async def test_azure_edits_need_inline_images(self) -> None:
        model = republic.get_image_model(
            "azure-openai:image-deployment", api_key="key", api_base="https://res.openai.azure.com/openai/v1"
        )

        with pytest.raises(errors.UnsupportedFeatureError, match="inline image data"):
            await model.edit("Add a hat", republic.image("https://example.com/cat.png"))

    async def test_azure_edit_retries_resend_the_form(self, service: FakeService) -> None:
        service.reply_json({"error": "busy"}, status_code=503)
        service.reply_json({"data": [{"b64_json": B64_PNG}]})

        await _azure_image_model(service, retry_delay=0).edit("Add a hat", PNG)

        assert len(service.requests) == 2
        first, second = ([part.get_payload(decode=True) for part in _form_parts(r)] for r in service.requests)
        assert first == second == [b"image-deployment", b"Add a hat", b"png"]

    async def test_preferred_chat_format_keeps_images(self, service: FakeService) -> None:
        service.reply_json({"data": [{"b64_json": B64_PNG}]})
        provider = republic.get_provider("openai", api_key="key", api_format="chat", http_client=service.client())

        await provider.get_image_model("gpt-image-2").generate("A cat")

        assert service.requests[0].url.path == "/v1/images/generations"


class TestGeminiImages:
    async def test_generate_uses_generate_content(self, service: FakeService) -> None:
        service.reply_json({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Thinking about cats", "thought": True},
                            {"text": "Here is your cat."},
                            {"inlineData": {"mimeType": "image/png", "data": B64_PNG}},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1290, "thoughtsTokenCount": 10},
        })
        model = republic.get_image_model("google:gemini-image", api_key="key", http_client=service.client())

        response = await model.generate("A cat", aspect_ratio="16:9", size="2K", n=1)

        assert (
            service.requests[0].url
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-image:generateContent"
        )
        assert service.requests[0].headers["x-goog-api-key"] == "key"
        assert service.body() == {
            "contents": [{"role": "user", "parts": [{"text": "A cat"}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
                "candidateCount": 1,
            },
        }
        assert response.image_parts == [PNG]
        assert response.token_usage == republic.TokenUsage(input_tokens=5, output_tokens=1300, reasoning_tokens=10)

    async def test_edit_sends_images_after_the_prompt(self, service: FakeService) -> None:
        service.reply_json({
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": B64_PNG}}]}}]
        })
        model = republic.get_image_model("google:gemini-image", api_key="key", http_client=service.client())

        await model.edit("Add a hat", [PNG, republic.image("gs://bucket/hat.png")])

        assert service.body()["contents"][0]["parts"] == [
            {"text": "Add a hat"},
            {"inlineData": {"mimeType": "image/png", "data": B64_PNG}},
            {"fileData": {"mimeType": "image/png", "fileUri": "gs://bucket/hat.png"}},
        ]

    async def test_mask_is_rejected(self) -> None:
        model = republic.get_image_model("google:gemini-image", api_key="key")

        with pytest.raises(errors.UnsupportedFeatureError, match="masks"):
            await model.edit("Add a hat", PNG, mask=MASK)

    async def test_openai_only_options_are_rejected(self) -> None:
        model = republic.get_image_model("google:gemini-image", api_key="key")

        with pytest.raises(errors.UnsupportedFeatureError, match="quality, background"):
            await model.generate("A cat", quality="high", background="transparent")

    async def test_reply_without_images_raises(self, service: FakeService) -> None:
        service.reply_json({
            "candidates": [{"content": {"parts": [{"text": "I can't draw that."}]}, "finishReason": "IMAGE_SAFETY"}]
        })
        model = republic.get_image_model("google:gemini-image", api_key="key", http_client=service.client())

        with pytest.raises(errors.APIResponseError, match="IMAGE_SAFETY") as exc_info:
            await model.generate("A cat")
        assert "I can't draw that." in str(exc_info.value)

    async def test_blocked_prompt_raises(self, service: FakeService) -> None:
        service.reply_json({"promptFeedback": {"blockReason": "SAFETY"}})
        model = republic.get_image_model("google:gemini-image", api_key="key", http_client=service.client())

        with pytest.raises(errors.APIResponseError, match="Prompt blocked: SAFETY"):
            await model.generate("A cat")


def test_providers_without_an_image_format_reject_image_models() -> None:
    with pytest.raises(errors.UnsupportedApiFormatError, match="image"):
        republic.get_image_model("anthropic:claude-opus-5-5", api_key="key")
