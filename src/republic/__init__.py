"""Republic: one interface over AI providers.

Gateways, agent loops, and tool execution stay outside the package.
"""

from republic import auth, decisions, errors, events, formats, history, providers, tools
from republic._content import (
    Image,
    Message,
    ProviderData,
    Reasoning,
    Text,
    Tool,
    ToolCall,
    Video,
    assistant,
    image,
    system,
    tool,
    user,
    video,
)
from republic._models import ChatModel, DecisionModel, EmbeddingModel, Stream
from republic._options import ChatOptions, ReasoningEffort, ToolChoice
from republic._registry import (
    all_providers,
    get_decision_model,
    get_embedding_model,
    get_model,
    get_provider,
    register_provider,
)
from republic._response import (
    BuiltinToolCall,
    Citation,
    EmbeddingResponse,
    FinishReason,
    ModelInfo,
    Response,
    TokenUsage,
)

__all__ = [
    "BuiltinToolCall",
    "ChatModel",
    "ChatOptions",
    "Citation",
    "DecisionModel",
    "EmbeddingModel",
    "EmbeddingResponse",
    "FinishReason",
    "Image",
    "Message",
    "ModelInfo",
    "ProviderData",
    "Reasoning",
    "ReasoningEffort",
    "Response",
    "Stream",
    "Text",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "Video",
    "all_providers",
    "assistant",
    "auth",
    "decisions",
    "errors",
    "events",
    "formats",
    "get_decision_model",
    "get_embedding_model",
    "get_model",
    "get_provider",
    "history",
    "image",
    "providers",
    "register_provider",
    "system",
    "tool",
    "tools",
    "user",
    "video",
]
