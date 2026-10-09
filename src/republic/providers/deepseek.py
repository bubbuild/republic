from __future__ import annotations

from dataclasses import replace

import httpx2

from republic.formats import ApiFormat, HttpRequest

from ._chat import MaxTokensReasoningContentChat
from .openai import OpenAICompatible


class DeepSeek(OpenAICompatible):
    """DeepSeek through Responses, Anthropic Messages, or chat. ``reasoning_effort="none"`` turns thinking off."""

    name = "deepseek"
    DEFAULT_API_BASE = "https://api.deepseek.com"
    SUPPORTED_API_FORMATS = ("chat", "responses", "messages")
    CHAT_FORMAT = MaxTokensReasoningContentChat()

    async def _send(
        self, client: httpx2.AsyncClient, api_format: ApiFormat, request: HttpRequest, *, stream: bool
    ) -> httpx2.Response:
        # The Anthropic-compatible API lives under its own base URL.
        if api_format.name == "messages":
            request = replace(request, path=f"/anthropic/v1{request.path}")
        return await super()._send(client, api_format, request, stream=stream)
