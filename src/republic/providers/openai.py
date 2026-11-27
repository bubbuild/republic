from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from republic.formats._base import ApiFormat, ChatRequest, HttpRequest
from republic.formats.chat import ChatFormat

from .base import Provider, _FormatT


class OpenAI(Provider):
    name = "openai"
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    SUPPORTED_API_FORMATS = ("responses", "chat", "embeddings", "images")

    def _prepare_request(
        self, api_format: ApiFormat, chat_request: ChatRequest, http_request: HttpRequest
    ) -> HttpRequest:
        if api_format.name not in {"chat", "responses"}:
            return http_request
        retention = chat_request.options.get("cache_retention")
        if retention == "none":
            return http_request
        body = dict(http_request.body)
        changed = False
        if (session_id := chat_request.options.get("session_id")) and "prompt_cache_key" not in body:
            body["prompt_cache_key"] = session_id
            changed = True
        if retention == "long" and "prompt_cache_retention" not in body:
            body["prompt_cache_retention"] = "24h"
            changed = True
        return replace(http_request, body=body) if changed else http_request


class OpenAICompatible(OpenAI):
    name = "openai-compatible"
    # Most OpenAI-compatible providers only support the chat format.
    SUPPORTED_API_FORMATS = ("chat",)
    CHAT_FORMAT: ClassVar[ChatFormat] = ChatFormat()
    """The chat format used for every model; set a subclass for a service's dialect."""

    def _prepare_request(
        self, api_format: ApiFormat, chat_request: ChatRequest, http_request: HttpRequest
    ) -> HttpRequest:
        # Generic OpenAI-compatible servers may reject OpenAI-specific prompt cache fields.
        return http_request

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        api_format = super().select_api_format(format_kind, model)
        if isinstance(api_format, ChatFormat) and isinstance(self.CHAT_FORMAT, format_kind):
            return self.CHAT_FORMAT
        return api_format
