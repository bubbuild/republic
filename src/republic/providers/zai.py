from __future__ import annotations

from ._chat import MaxTokensReasoningContentChat
from .openai import OpenAICompatible


class ZAI(OpenAICompatible):
    """Z.ai's GLM models. Use ``api_base="https://open.bigmodel.cn/api/paas/v4"`` for the China platform."""

    name = "zai"
    DEFAULT_API_BASE = "https://api.z.ai/api/paas/v4"
    CHAT_FORMAT = MaxTokensReasoningContentChat()
