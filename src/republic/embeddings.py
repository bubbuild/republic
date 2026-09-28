# Semantics adapted from ai-python c788059dd1 (Apache-2.0); see NOTICE.
"""Independent asynchronous embedding capability, without batching or retrieval."""

from typing import Annotated, Protocol

from pydantic import Field, field_validator

from republic.types import ProviderMetadata, Usage, _Data


class EmbeddingRequest(_Data):
    model: str = Field(min_length=1)
    inputs: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0, strict=True)
    provider_options: ProviderMetadata = Field(default_factory=dict)

    @field_validator("inputs", mode="before")
    @classmethod
    def single_input(cls, value: object) -> object:
        """A single text is one input; never split text or batch implicitly."""
        return [value] if isinstance(value, str) else value


class Embedding(_Data):
    index: int = Field(ge=0, strict=True)
    vector: list[Annotated[float, Field(strict=True)]] = Field(min_length=1)
    provider_metadata: ProviderMetadata | None = None


class EmbeddingResponse(_Data):
    """Vectors in input order, retaining each provider index and metadata."""

    embeddings: list[Embedding]
    response_model: str | None = None
    usage: Usage | None = None
    provider_metadata: ProviderMetadata | None = None


class EmbeddingProvider(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


async def embed(provider: EmbeddingProvider, request: EmbeddingRequest) -> EmbeddingResponse:
    """Delegate one logical embedding operation. Client retry policy is caller-owned."""
    return await provider.embed(request)
