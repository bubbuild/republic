# Copyright 2026 Vercel, Inc. Licensed under Apache-2.0.
# Data semantics adapted from ai-python c788059dd1 ops/reranking.py; see NOTICE.
"""Independent document ranking data; no retrieval or chat-model emulation."""

from typing import Protocol

from pydantic import Field, JsonValue

from republic.types import ProviderMetadata, Usage, _Data


class RerankRequest(_Data):
    model: str = Field(min_length=1)
    query: str
    documents: list[str] | list[dict[str, JsonValue]]
    top_n: int | None = Field(default=None, gt=0, strict=True)
    provider_options: ProviderMetadata = Field(default_factory=dict)


class RankedDocument(_Data):
    index: int = Field(ge=0, strict=True)
    score: float = Field(strict=True)
    provider_metadata: ProviderMetadata | None = None


class RerankResponse(_Data):
    """Server ranking order and scores, referring to the original documents."""

    ranking: list[RankedDocument]
    response_id: str | None = None
    usage: Usage | None = None
    provider_metadata: ProviderMetadata | None = None


class RerankProvider(Protocol):
    async def rerank(self, request: RerankRequest) -> RerankResponse: ...


async def rerank(provider: RerankProvider, request: RerankRequest) -> RerankResponse:
    """Delegate one logical operation. No implicit batching or retrieval."""
    return await provider.rerank(request)
