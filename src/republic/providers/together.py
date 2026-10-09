from __future__ import annotations

from ._chat import MaxTokensChat
from .openai import OpenAICompatible


class Together(OpenAICompatible):
    name = "together"
    DEFAULT_API_BASE = "https://api.together.ai/v1"
    SUPPORTED_API_FORMATS = ("chat", "embeddings")
    CHAT_FORMAT = MaxTokensChat()
