from __future__ import annotations

from dataclasses import replace

from republic.formats._base import ApiFormat, ChatRequest, HttpRequest

from ._chat import ReasoningContentChat
from .openai import OpenAICompatible


class Moonshot(OpenAICompatible):
    """Moonshot AI's Kimi models. Use ``api_base="https://api.moonshot.cn/v1"`` for the China platform."""

    name = "moonshot"
    DEFAULT_API_BASE = "https://api.moonshot.ai/v1"
    CHAT_FORMAT = ReasoningContentChat()

    def _prepare_request(
        self, api_format: ApiFormat, chat_request: ChatRequest, http_request: HttpRequest
    ) -> HttpRequest:
        session_id = chat_request.options.get("session_id")
        if (
            not session_id
            or api_format.name != "chat"
            or chat_request.options.get("cache_retention") == "none"
            or "prompt_cache_key" in http_request.body
        ):
            return http_request
        return replace(http_request, body={**http_request.body, "prompt_cache_key": session_id})
