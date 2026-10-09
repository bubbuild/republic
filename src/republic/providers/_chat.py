"""Chat format dialects shared by OpenAI-compatible services."""

from __future__ import annotations

from typing import Any

from republic._content import Message
from republic.formats.chat import ChatFormat


class MaxTokensChat(ChatFormat):
    """Chat completions that limit output with ``max_tokens`` only."""

    def max_tokens_fields(self, max_tokens: int) -> dict[str, Any]:
        return {"max_tokens": max_tokens}


class ReasoningContentChat(ChatFormat):
    """Chat completions whose thinking models need ``reasoning_content`` sent back.

    DeepSeek, Moonshot and Z.ai reject or degrade tool-use turns whose earlier
    assistant messages lost their reasoning.
    """

    def assistant_fields(self, message: Message) -> dict[str, Any]:
        return {"reasoning_content": message.reasoning} if message.reasoning else {}


class MaxTokensReasoningContentChat(MaxTokensChat, ReasoningContentChat):
    pass
