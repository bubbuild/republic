from __future__ import annotations

from typing import Any

from republic.auth import Auth, HeaderAuth

from .base import Provider


class Anthropic(Provider):
    name = "anthropic"
    DEFAULT_API_BASE = "https://api.anthropic.com/v1"
    SUPPORTED_API_FORMATS = ("messages",)

    def _api_key_auth(self, api_key: str) -> Auth:
        return HeaderAuth("x-api-key", api_key)

    def _models_params(self, previous: Any) -> dict[str, str] | None:
        if previous is None:
            return {"limit": "1000"}
        return {"limit": "1000", "after_id": previous["last_id"]} if previous.get("has_more") else None
