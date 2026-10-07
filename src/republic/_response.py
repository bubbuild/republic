from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from republic._content import Image, Message, ToolCall

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

OutputT = TypeVar("OutputT")

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "refusal", "other"]
"""Why generation ended, normalized across API formats."""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    """All prompt tokens, including those read from or written to a cache."""
    output_tokens: int = 0
    """All generated tokens, including reasoning tokens."""
    reasoning_tokens: int = 0
    """Output tokens spent on reasoning, when the provider reports them."""
    cached_tokens: int = 0
    """Prompt tokens read from the provider's prompt cache."""
    cache_write_tokens: int = 0
    """Prompt tokens written to the prompt cache, for providers that bill them separately."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Response(Generic[OutputT]):
    """A completed chat response."""

    message: Message
    """The assistant message, ready to be appended to a conversation."""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    output: OutputT | None = None
    """The structured output parsed with ``output_schema``; ``None`` when not requested or refused."""
    finish_reason: FinishReason | None = None
    refusal: str | None = None
    """Why the model declined to answer, when it refused."""
    id: str | None = None
    model: str | None = None
    """The model version that served the request, as reported by the provider."""

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def reasoning(self) -> str:
        return self.message.reasoning

    @property
    def tool_calls(self) -> list[ToolCall]:
        return list(self.message.tool_calls)

    @property
    def image_parts(self) -> list[Image]:
        return [part for part in self.message.parts if isinstance(part, Image)]

    @property
    def images(self) -> list[PILImage]:
        """Generated images decoded with Pillow (install ``republic[image]``)."""
        return [part.to_pil() for part in self.image_parts]


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    """One vector per input, in input order."""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None

    @property
    def vector(self) -> list[float]:
        """The vector of the first input."""
        return self.vectors[0]
