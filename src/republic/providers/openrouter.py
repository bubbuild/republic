from __future__ import annotations

from .base import Provider


class OpenRouter(Provider):
    name = "openrouter"
    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    SUPPORTED_API_FORMATS = ("chat", "embeddings")
