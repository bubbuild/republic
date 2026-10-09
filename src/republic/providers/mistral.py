from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from republic._content import Message
from republic.formats import ChatRequest, EmbeddingsFormat, HttpRequest

from ._chat import MaxTokensChat
from .base import _FormatT
from .openai import OpenAICompatible


class MistralChatFormat(MaxTokensChat):
    """Mistral's chat completions, which return reasoning as ``thinking`` content chunks."""

    def chat_request(self, request: ChatRequest, *, stream: bool) -> HttpRequest:
        request.reject(self.name, "top_k")
        http_request = super().chat_request(request, stream=stream)
        body = dict(http_request.body)
        # Mistral reports usage in the last chunk without asking.
        body.pop("stream_options", None)
        if "seed" in body:
            body["random_seed"] = body.pop("seed")
        return replace(http_request, body=body)

    def reasoning_text(self, message: Mapping[str, Any]) -> str | None:
        return "".join(
            _chunks_text(chunk.get("thinking") or ())
            for chunk in _content_chunks(message)
            if chunk.get("type") == "thinking"
        )

    def content_text(self, message: Mapping[str, Any]) -> str | None:
        content = message.get("content")
        return content if content is None or isinstance(content, str) else _chunks_text(content)

    def assistant_fields(self, message: Message) -> dict[str, Any]:
        # Mistral reasons better when earlier thinking is sent back.
        if not message.reasoning:
            return {}
        content: list[dict[str, Any]] = [
            {"type": "thinking", "thinking": [{"type": "text", "text": message.reasoning}]}
        ]
        if message.text:
            content.append({"type": "text", "text": message.text})
        return {"content": content}


class MistralEmbeddingsFormat(EmbeddingsFormat):
    def embedding_request(self, model: str, texts: Sequence[str], *, dimensions: int | None) -> HttpRequest:
        http_request = super().embedding_request(model, texts, dimensions=None)
        if dimensions is None:
            return http_request
        return replace(http_request, body={**http_request.body, "output_dimension": dimensions})


_EMBEDDINGS_FORMAT = MistralEmbeddingsFormat()


class Mistral(OpenAICompatible):
    name = "mistral"
    DEFAULT_API_BASE = "https://api.mistral.ai/v1"
    SUPPORTED_API_FORMATS = ("chat", "embeddings")
    CHAT_FORMAT = MistralChatFormat()

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        api_format = super().select_api_format(format_kind, model)
        if isinstance(api_format, EmbeddingsFormat) and isinstance(_EMBEDDINGS_FORMAT, format_kind):
            return _EMBEDDINGS_FORMAT
        return api_format


def _content_chunks(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = message.get("content")
    return [] if content is None or isinstance(content, str) else content


def _chunks_text(chunks: Sequence[Mapping[str, Any]]) -> str:
    return "".join(chunk.get("text") or "" for chunk in chunks if chunk.get("type") == "text")
