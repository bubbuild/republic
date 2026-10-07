"""Republic: one interface over AI providers.

Gateways, agent loops, and tool execution stay outside the package.
"""

from republic import decisions, events, history, providers
from republic._content import (
    Image,
    Message,
    ProviderData,
    Reasoning,
    Text,
    Tool,
    ToolCall,
    ToolResult,
    Video,
    assistant,
    image,
    system,
    tool_result,
    user,
    video,
)
from republic._errors import (
    APIResponseError,
    APIStatusError,
    ProviderNotFoundError,
    RepublicError,
    StreamNotFinishedError,
    UnsupportedApiFormatError,
    UnsupportedFeatureError,
)
from republic._models import ChatModel, DecisionModel, EmbeddingModel, Stream
from republic._options import ChatOptions, ReasoningEffort, ToolChoice
from republic._registry import get_decision_model, get_embedding_model, get_model, get_provider, register_provider
from republic._response import EmbeddingResponse, Response, TokenUsage

__all__ = [
    "APIResponseError",
    "APIStatusError",
    "ChatModel",
    "ChatOptions",
    "DecisionModel",
    "EmbeddingModel",
    "EmbeddingResponse",
    "Image",
    "Message",
    "ProviderData",
    "ProviderNotFoundError",
    "Reasoning",
    "ReasoningEffort",
    "RepublicError",
    "Response",
    "Stream",
    "StreamNotFinishedError",
    "Text",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "UnsupportedApiFormatError",
    "UnsupportedFeatureError",
    "Video",
    "assistant",
    "decisions",
    "events",
    "get_decision_model",
    "get_embedding_model",
    "get_model",
    "get_provider",
    "history",
    "image",
    "providers",
    "register_provider",
    "system",
    "tool_result",
    "user",
    "video",
]
