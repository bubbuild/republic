"""API formats: the wire protocols providers speak.

Subclass a format to adjust a hook for one service, and return it from
:meth:`republic.providers.Provider.select_api_format`::

    class DeepSeekChat(ChatFormat):
        def reasoning_fields(self, effort, *, include_reasoning):
            return {"thinking": {"type": "disabled" if effort == "none" else "enabled"}}
"""

from __future__ import annotations

from typing import Literal

from ._base import (
    ApiFormat,
    ChatApiFormat,
    ChatRequest,
    DecisionApiFormat,
    Delta,
    EmbeddingApiFormat,
    HttpRequest,
    OutputSchema,
    ResponseInfo,
    StreamParser,
    ToolCallFragment,
    UsageReport,
    deep_merge,
    strict_schema,
)
from .chat import ChatFormat
from .embeddings import EmbedContentFormat, EmbeddingsFormat
from .gemini import GeminiFormat
from .messages import MessagesFormat
from .responses import ResponsesFormat
from .system_one import SystemOneFormat

ApiFormatName = Literal["responses", "messages", "gemini", "chat", "embeddings", "embed_content", "system_one"]

_API_FORMATS: dict[str, ApiFormat] = {
    api_format.name: api_format
    for api_format in (
        ResponsesFormat(),
        MessagesFormat(),
        GeminiFormat(),
        ChatFormat(),
        EmbeddingsFormat(),
        EmbedContentFormat(),
        SystemOneFormat(),
    )
}
"""The built-in format instances. Within each kind, earlier formats are preferred."""

__all__ = [
    "ApiFormat",
    "ApiFormatName",
    "ChatApiFormat",
    "ChatFormat",
    "ChatRequest",
    "DecisionApiFormat",
    "Delta",
    "EmbedContentFormat",
    "EmbeddingApiFormat",
    "EmbeddingsFormat",
    "GeminiFormat",
    "HttpRequest",
    "MessagesFormat",
    "OutputSchema",
    "ResponseInfo",
    "ResponsesFormat",
    "StreamParser",
    "SystemOneFormat",
    "ToolCallFragment",
    "UsageReport",
    "deep_merge",
    "strict_schema",
]
