from __future__ import annotations

from typing import Literal

from .base import ApiFormat, ChatApiFormat, DecisionApiFormat, EmbeddingApiFormat
from .chat import ChatFormat
from .embeddings import EmbedContentFormat, EmbeddingsFormat
from .gemini import GeminiFormat
from .messages import MessagesFormat
from .responses import ResponsesFormat
from .system_one import SystemOneFormat

ApiFormatName = Literal["responses", "messages", "gemini", "chat", "embeddings", "embed_content", "system_one"]

API_FORMATS: dict[str, ApiFormat] = {
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
"""All API formats. Within each kind, earlier formats are preferred."""

__all__ = ["API_FORMATS", "ApiFormat", "ApiFormatName", "ChatApiFormat", "DecisionApiFormat", "EmbeddingApiFormat"]
