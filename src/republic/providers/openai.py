from __future__ import annotations

from typing import ClassVar

from republic.formats.chat import ChatFormat

from .base import Provider, _FormatT


class OpenAI(Provider):
    name = "openai"
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    SUPPORTED_API_FORMATS = ("responses", "chat", "embeddings")


class OpenAICompatible(OpenAI):
    name = "openai-compatible"
    # Most OpenAI-compatible providers only support the chat format.
    SUPPORTED_API_FORMATS = ("chat",)
    CHAT_FORMAT: ClassVar[ChatFormat] = ChatFormat()
    """The chat format used for every model; set a subclass for a service's dialect."""

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        api_format = super().select_api_format(format_kind, model)
        if isinstance(api_format, ChatFormat) and isinstance(self.CHAT_FORMAT, format_kind):
            return self.CHAT_FORMAT
        return api_format
