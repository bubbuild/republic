from __future__ import annotations

import httpx2

from .base import HeaderAuth, Provider


class Google(Provider):
    name = "google"
    DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    SUPPORTED_API_FORMATS = ("gemini", "embed_content")

    def _api_key_auth(self, api_key: str) -> httpx2.Auth:
        return HeaderAuth("x-goog-api-key", api_key)
