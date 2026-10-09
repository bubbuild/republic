from __future__ import annotations

from typing import Any

from republic._options import ReasoningEffort
from republic.formats.chat import ChatFormat

from .base import Provider, _FormatT


class OpenRouterChatFormat(ChatFormat):
    """OpenRouter's chat completions, which take reasoning options as one ``reasoning`` object."""

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        reasoning: dict[str, Any] = {}
        if effort is not None:
            reasoning["effort"] = effort
        if include_reasoning:
            reasoning["exclude"] = False
        return {"reasoning": reasoning} if reasoning else {}


_CHAT_FORMAT = OpenRouterChatFormat()


class OpenRouter(Provider):
    name = "openrouter"
    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    # OpenRouter routes every API format to any model it serves, and also hosts
    # System One decision models such as TypeSafe's Jev.
    SUPPORTED_API_FORMATS = ("responses", "messages", "chat", "embeddings", "system_one")

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        api_format = super().select_api_format(format_kind, model)
        if isinstance(api_format, ChatFormat) and isinstance(_CHAT_FORMAT, format_kind):
            return _CHAT_FORMAT
        return api_format
