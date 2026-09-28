"""OpenAI /embeddings with explicit compatible endpoints and existing client policy."""

import base64
import struct
from typing import Any

import httpx
import openai

from republic.embeddings import Embedding, EmbeddingRequest, EmbeddingResponse
from republic.errors import ProviderError, UnsupportedRequestError
from republic.providers._openai_client import OpenAIClient, provider_error
from republic.types import Usage

_OPTIONS = {"user", "encoding_format", "extra_headers", "extra_query", "extra_body", "timeout"}
_MANAGED = {"model", "input", "dimensions"}


def _payload(request: EmbeddingRequest) -> dict[str, Any]:
    options = dict(request.provider_options)
    if options.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", "unsupported or managed embedding options")
    body = options.get("extra_body", {})
    if not isinstance(body, dict) or body.keys() & (_MANAGED | _OPTIONS):
        raise UnsupportedRequestError("extra_body", "cannot override managed or standard embedding fields")
    if options.get("encoding_format", "float") not in ("float", "base64"):
        raise UnsupportedRequestError("encoding_format", "expected float or base64")
    headers = options.get("extra_headers", {})
    if not isinstance(headers, dict) or any(not isinstance(value, str) for value in headers.values()):
        raise UnsupportedRequestError("extra_headers", "expected string header values")
    payload = {"model": request.model, "input": request.inputs, "encoding_format": "float", **options}
    if request.dimensions is not None:
        payload["dimensions"] = request.dimensions
    return payload


def _vector(value: Any) -> list[float]:
    if isinstance(value, str):
        data = base64.b64decode(value, validate=True)
        if not data or len(data) % 4:
            raise ValueError("invalid_vector_encoding")
        return list(struct.unpack(f"<{len(data) // 4}f", data))
    if not isinstance(value, list):
        raise TypeError("invalid_vector")
    return value


def _response(raw: dict[str, Any], request: EmbeddingRequest) -> EmbeddingResponse:
    vectors = [
        Embedding(
            index=item["index"],
            vector=_vector(item["embedding"]),
            provider_metadata={
                "openai": {key: value for key, value in item.items() if key not in {"index", "embedding"}}
            },
        )
        for item in raw["data"]
    ]
    if sorted(item.index for item in vectors) != list(range(len(request.inputs))):
        raise ValueError("invalid_embedding_indexes")
    sizes = {len(item.vector) for item in vectors}
    if len(sizes) != 1 or (request.dimensions is not None and sizes != {request.dimensions}):
        raise ValueError("invalid_embedding_dimensions")
    usage = raw.get("usage")
    return EmbeddingResponse(
        embeddings=sorted(vectors, key=lambda item: item.index),
        response_model=raw.get("model"),
        usage=Usage(input_tokens=usage.get("prompt_tokens"), output_tokens=0, raw=usage) if usage is not None else None,
        provider_metadata={
            "openai": {key: value for key, value in raw.items() if key not in {"data", "model", "usage"}}
        },
    )


class OpenAIEmbeddings(OpenAIClient):
    """One official async SDK call; no implicit batching, model routing or retrieval."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_open()
        payload = _payload(request)
        self._request_headers(payload)
        try:
            result = await self._client.embeddings.with_raw_response.create(**payload)
            return _response(result.http_response.json(), request)
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ProviderError("invalid_response", provider="openai", code="invalid_response") from exc
