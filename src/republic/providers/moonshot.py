from __future__ import annotations

from ._chat import ReasoningContentChat
from .openai import OpenAICompatible


class Moonshot(OpenAICompatible):
    """Moonshot AI's Kimi models. Use ``api_base="https://api.moonshot.cn/v1"`` for the China platform."""

    name = "moonshot"
    DEFAULT_API_BASE = "https://api.moonshot.ai/v1"
    CHAT_FORMAT = ReasoningContentChat()
