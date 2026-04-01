"""Provider hooks and runtime."""

from republic.conversation import (
    ContentPart,
    Conversation,
    ConversationMessage,
    ToolCallPart,
    ToolResultPart,
    conversation_from_messages,
)
from republic.providers.codecs import (
    conversation_to_anthropic_messages,
    conversation_to_completion_messages,
    conversation_to_openai_responses_input,
)
from republic.providers.hookspecs import providerimpl
from republic.providers.runtime import ProviderRuntime, default_provider_runtime, reset_default_provider_runtime
from republic.providers.types import (
    ANTHROPIC_COMPAT_TRANSPORTS,
    COMPLETION_TRANSPORTS,
    MESSAGES_TRANSPORTS,
    OPENAI_TRANSPORTS,
    OPENROUTER_TRANSPORTS,
    RESPONSES_TRANSPORTS,
    ChatRequest,
    EmbedRequest,
    ProviderBackend,
    ProviderCapabilities,
    ProviderContext,
    ProviderHook,
    ResponsesRequest,
    TransportName,
    transport_set,
)

__all__ = [
    "ANTHROPIC_COMPAT_TRANSPORTS",
    "COMPLETION_TRANSPORTS",
    "MESSAGES_TRANSPORTS",
    "OPENAI_TRANSPORTS",
    "OPENROUTER_TRANSPORTS",
    "RESPONSES_TRANSPORTS",
    "ChatRequest",
    "ContentPart",
    "Conversation",
    "ConversationMessage",
    "EmbedRequest",
    "ProviderBackend",
    "ProviderCapabilities",
    "ProviderContext",
    "ProviderHook",
    "ProviderRuntime",
    "ResponsesRequest",
    "ToolCallPart",
    "ToolResultPart",
    "TransportName",
    "conversation_from_messages",
    "conversation_to_anthropic_messages",
    "conversation_to_completion_messages",
    "conversation_to_openai_responses_input",
    "default_provider_runtime",
    "providerimpl",
    "reset_default_provider_runtime",
    "transport_set",
]
