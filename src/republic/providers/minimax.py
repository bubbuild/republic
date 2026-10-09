from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import httpx2

from republic._options import ReasoningEffort
from republic.formats import ApiFormat, HttpRequest

from ._chat import ReasoningContentChat
from .openai import OpenAICompatible


class MiniMaxChat(ReasoningContentChat):
    """MiniMax chat, asked to return thinking in ``reasoning_content`` instead of ``<think>`` tags."""

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        return {"reasoning_split": True, **super().reasoning_fields(effort, include_reasoning=include_reasoning)}

    def reasoning_text(self, message: Mapping[str, Any]) -> str | None:
        # Older M2 models return split reasoning as ``reasoning_details``.
        details = message.get("reasoning_details") or ()
        return super().reasoning_text(message) or "".join(detail.get("text") or "" for detail in details)


class MiniMax(OpenAICompatible):
    """MiniMax through Anthropic Messages or chat. Use ``api_base="https://api.minimaxi.com"`` in China."""

    name = "minimax"
    DEFAULT_API_BASE = "https://api.minimax.io"
    SUPPORTED_API_FORMATS = ("messages", "chat")
    CHAT_FORMAT = MiniMaxChat()
    MODELS_PATH = "/v1/models"

    async def _send(
        self, client: httpx2.AsyncClient, api_format: ApiFormat, request: HttpRequest, *, stream: bool
    ) -> httpx2.Response:
        # Each API is served under its own prefix of the same host.
        prefix = "/anthropic/v1" if api_format.name == "messages" else "/v1"
        return await super()._send(client, api_format, replace(request, path=f"{prefix}{request.path}"), stream=stream)
