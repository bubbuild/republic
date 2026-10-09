from __future__ import annotations

from .base import Provider


class OpenAI(Provider):
    name = "openai"
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    SUPPORTED_API_FORMATS = ("responses", "chat", "embeddings")


class OpenAICompatible(OpenAI):
    name = "openai-compatible"
    # Most OpenAI-compatible providers only support the chat format.
    SUPPORTED_API_FORMATS = ("chat",)
