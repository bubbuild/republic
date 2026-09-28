"""Republic's provisional single-call provider SDK."""

from republic import events
from republic.api import Provider, Stream, generate, stream
from republic.embeddings import Embedding, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, embed
from republic.errors import (
    IncompleteStreamError,
    ProviderError,
    RepublicError,
    StreamProtocolError,
    UnsupportedRequestError,
)
from republic.types import (
    FilePart,
    FinishReason,
    Message,
    Part,
    ProviderMetadata,
    ReasoningPart,
    Request,
    RequestOptions,
    Response,
    TextPart,
    Tool,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
)

__all__ = [
    "Embedding",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FilePart",
    "FinishReason",
    "IncompleteStreamError",
    "Message",
    "Part",
    "Provider",
    "ProviderError",
    "ProviderMetadata",
    "ReasoningPart",
    "RepublicError",
    "Request",
    "RequestOptions",
    "Response",
    "Stream",
    "StreamProtocolError",
    "TextPart",
    "Tool",
    "ToolCallPart",
    "ToolChoice",
    "ToolResultPart",
    "UnsupportedRequestError",
    "Usage",
    "embed",
    "events",
    "generate",
    "stream",
]
