from __future__ import annotations

from dataclasses import replace

from republic.formats._base import ApiFormat, ChatRequest, HttpRequest

from .base import Provider


class Magpie(Provider):
    """A local Magpie gateway, which routes every API format to any model in its catalog."""

    name = "magpie"
    DEFAULT_API_BASE = "http://127.0.0.1:3425/v1"
    SUPPORTED_API_FORMATS = ("chat", "responses", "messages", "system_one")

    def _prepare_request(
        self, api_format: ApiFormat, chat_request: ChatRequest, http_request: HttpRequest
    ) -> HttpRequest:
        session_id = chat_request.options.get("session_id")
        if not session_id:
            return http_request
        headers = {**http_request.headers, "x-session-id": session_id}
        body = http_request.body
        if (
            api_format.name in {"chat", "responses"}
            and chat_request.options.get("cache_retention") != "none"
            and "prompt_cache_key" not in body
        ):
            body = {**body, "prompt_cache_key": session_id}
        return replace(http_request, headers=headers, body=body)
