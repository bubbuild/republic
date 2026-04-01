"""Embedding helpers for Republic."""

from __future__ import annotations

from typing import Any

from republic.core.errors import ErrorKind
from republic.core.execution import LLMCore
from republic.core.results import RepublicError
from republic.providers.types import EmbedRequest


class EmbeddingClient:
    """Lightweight embedding helper."""

    def __init__(self, core: LLMCore) -> None:
        self._core = core

    def _resolve_provider_model(self, model: str | None, provider: str | None) -> tuple[str, str]:
        if model is None and provider is None:
            return self._core.provider, self._core.model
        model_id = model or self._core.model
        return self._core.resolve_model_provider(model_id, provider)

    def embed(
        self,
        inputs: str | list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        provider_name, model_id = self._resolve_provider_model(model, provider)
        client = self._core.get_backend(provider_name)
        try:
            response = client.embed(EmbedRequest(model=model_id, inputs=inputs, kwargs=dict(kwargs)))
        except Exception as exc:
            if isinstance(exc, NotImplementedError):
                raise RepublicError(ErrorKind.INVALID_INPUT, f"{provider_name}:{model_id}: {exc}") from exc
            self._core.raise_wrapped(exc, provider_name, model_id)
        return response

    async def embed_async(
        self,
        inputs: str | list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        provider_name, model_id = self._resolve_provider_model(model, provider)
        client = self._core.get_backend(provider_name)
        try:
            response = await client.aembed(EmbedRequest(model=model_id, inputs=inputs, kwargs=dict(kwargs)))
        except Exception as exc:
            if isinstance(exc, NotImplementedError):
                raise RepublicError(ErrorKind.INVALID_INPUT, f"{provider_name}:{model_id}: {exc}") from exc
            self._core.raise_wrapped(exc, provider_name, model_id)
        return response
