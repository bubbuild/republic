from __future__ import annotations

import httpx2

from .base import HeaderAuth, Provider


class Anthropic(Provider):
    name = "anthropic"
    DEFAULT_API_BASE = "https://api.anthropic.com/v1"
    SUPPORTED_API_FORMATS = ("messages",)

    def _api_key_auth(self, api_key: str) -> httpx2.Auth:
        return HeaderAuth("x-api-key", api_key)
