"""Embedding formats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from republic._response import EmbeddingResponse, TokenUsage

from ._base import EmbeddingApiFormat, HttpRequest


class EmbeddingsFormat(EmbeddingApiFormat):
    """The OpenAI ``/embeddings`` endpoint, also served by compatible gateways."""

    name = "embeddings"

    def embedding_request(self, model: str, texts: Sequence[str], *, dimensions: int | None) -> HttpRequest:
        body: dict[str, Any] = {"model": model, "input": list(texts), "encoding_format": "float"}
        if dimensions is not None:
            body["dimensions"] = dimensions
        return HttpRequest("/embeddings", body)

    def parse_embedding(self, data: Mapping[str, Any]) -> EmbeddingResponse:
        usage = data.get("usage") or {}
        items = sorted(data["data"], key=lambda item: item["index"])
        return EmbeddingResponse(
            vectors=[item["embedding"] for item in items],
            token_usage=TokenUsage(input_tokens=usage.get("prompt_tokens", 0)),
            model=data.get("model"),
        )


class EmbedContentFormat(EmbeddingApiFormat):
    """The Google Gemini ``batchEmbedContents`` endpoint, which also serves single inputs."""

    name = "embed_content"

    def embedding_request(self, model: str, texts: Sequence[str], *, dimensions: int | None) -> HttpRequest:
        requests = []
        for text in texts:
            request: dict[str, Any] = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
            if dimensions is not None:
                request["outputDimensionality"] = dimensions
            requests.append(request)
        return HttpRequest(f"/models/{model}:batchEmbedContents", {"requests": requests})

    def parse_embedding(self, data: Mapping[str, Any]) -> EmbeddingResponse:
        return EmbeddingResponse(vectors=[embedding["values"] for embedding in data["embeddings"]])
