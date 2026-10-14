"""Image generation formats."""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any

from republic._content import Image, Text
from republic._options import ImageOptions
from republic._response import ImageResponse, TokenUsage
from republic.errors import APIResponseError, UnsupportedFeatureError

from ._base import HttpRequest, ImageApiFormat, deep_merge
from .gemini import _part, _usage


class ImagesFormat(ImageApiFormat):
    """The OpenAI ``/images/generations`` and ``/images/edits`` endpoints.

    Edits send input images and the mask as ``image_url`` references in a JSON
    body, so both inline images and remote URLs work.
    """

    name = "images"
    OPTIONS = ("n", "size", "quality", "background", "output_format", "output_compression", "moderation")

    def generation_request(self, model: str, prompt: str, options: ImageOptions) -> HttpRequest:
        return HttpRequest("/images/generations", self._body({"model": model, "prompt": prompt}, options))

    def edit_request(
        self, model: str, prompt: str, images: Sequence[Image], *, mask: Image | None, options: ImageOptions
    ) -> HttpRequest:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "images": [{"image_url": image.data_url} for image in images],
        }
        if mask is not None:
            body["mask"] = {"image_url": mask.data_url}
        return HttpRequest("/images/edits", self._body(body, options))

    def parse_images(self, data: Mapping[str, Any]) -> ImageResponse:
        # GPT image models report the format; DALL-E returns PNG.
        media_type = f"image/{data.get('output_format') or 'png'}"
        images = [
            Image(media_type, data=base64.b64decode(item["b64_json"]))
            if item.get("b64_json")
            else Image(media_type, url=item["url"])
            for item in data["data"]
        ]
        usage = data.get("usage") or {}
        return ImageResponse(
            images,
            revised_prompts=[item.get("revised_prompt") for item in data["data"]],
            token_usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0)
            ),
        )

    def _body(self, body: dict[str, Any], options: ImageOptions) -> dict[str, Any]:
        _reject(self.name, options, "aspect_ratio")
        body.update({name: value for name, value in options.items() if name in self.OPTIONS})
        return deep_merge(body, options.get("extra_body", {}))


class MultipartImagesFormat(ImagesFormat):
    """The ``images`` format with edits sent as multipart form data, as Azure OpenAI requires.

    Input images and the mask are uploaded as files, so they need inline data.
    """

    def edit_request(
        self, model: str, prompt: str, images: Sequence[Image], *, mask: Image | None, options: ImageOptions
    ) -> HttpRequest:
        files = [("image[]", _file(f"image-{index}", image)) for index, image in enumerate(images)]
        if mask is not None:
            files.append(("mask", _file("mask", mask)))
        fields = self._body({"model": model, "prompt": prompt}, options)
        return HttpRequest("/images/edits", {name: _form_value(value) for name, value in fields.items()}, files=files)


class GeminiImageFormat(ImageApiFormat):
    """Image output from the Google Gemini ``generateContent`` endpoint.

    Edits send the input images after the prompt; masks are not supported.
    """

    name = "gemini_image"

    def generation_request(self, model: str, prompt: str, options: ImageOptions) -> HttpRequest:
        return self._request(model, [Text(prompt)], options)

    def edit_request(
        self, model: str, prompt: str, images: Sequence[Image], *, mask: Image | None, options: ImageOptions
    ) -> HttpRequest:
        if mask is not None:
            raise UnsupportedFeatureError(f"The {self.name!r} API format does not support masks")
        return self._request(model, [Text(prompt), *images], options)

    def parse_images(self, data: Mapping[str, Any]) -> ImageResponse:
        candidates = data.get("candidates") or ()
        if not candidates and (feedback := data.get("promptFeedback", {}).get("blockReason")):
            raise APIResponseError(f"Prompt blocked: {feedback}")
        images: list[Image] = []
        texts: list[str] = []
        for candidate in candidates:
            for part in candidate.get("content", {}).get("parts") or ():
                if part.get("thought"):
                    continue
                if (inline := part.get("inlineData")) and inline["mimeType"].startswith("image/"):
                    images.append(Image(inline["mimeType"], data=base64.b64decode(inline["data"])))
                elif text := part.get("text"):
                    texts.append(text)
        if not images:
            reasons = [reason for candidate in candidates if (reason := candidate.get("finishReason"))]
            raise APIResponseError(json.dumps({"finish_reasons": reasons, "text": "".join(texts)}))
        usage = _usage(data.get("usageMetadata") or {})
        return ImageResponse(
            images,
            token_usage=TokenUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                reasoning_tokens=usage.reasoning_tokens or 0,
                cached_tokens=usage.cached_tokens or 0,
            ),
        )

    def _request(self, model: str, parts: Sequence[Text | Image], options: ImageOptions) -> HttpRequest:
        _reject(self.name, options, "quality", "background", "output_format", "output_compression", "moderation")
        image_config: dict[str, Any] = {}
        if "aspect_ratio" in options:
            image_config["aspectRatio"] = options["aspect_ratio"]
        if "size" in options:
            image_config["imageSize"] = options["size"]
        config: dict[str, Any] = {"responseModalities": ["IMAGE"]}
        if image_config:
            config["imageConfig"] = image_config
        if "n" in options:
            config["candidateCount"] = options["n"]
        body = {
            "contents": [{"role": "user", "parts": [_part(part) for part in parts]}],
            "generationConfig": config,
        }
        return HttpRequest(f"/models/{model}:generateContent", deep_merge(body, options.get("extra_body", {})))


def _file(stem: str, image: Image) -> tuple[str, bytes, str]:
    if image.data is None:
        raise UnsupportedFeatureError(f"Multipart image edits need inline image data, not a URL: {image.url}")
    return f"{stem}{mimetypes.guess_extension(image.media_type) or ''}", image.data, image.media_type


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else json.dumps(value)


def _reject(api_format: str, options: ImageOptions, *unsupported: str) -> None:
    if rejected := [name for name in unsupported if name in options]:
        raise UnsupportedFeatureError(f"The {api_format!r} API format does not support {', '.join(rejected)}")
