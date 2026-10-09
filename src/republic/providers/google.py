from __future__ import annotations

from republic.auth import Auth, HeaderAuth

from .base import Provider


class Google(Provider):
    name = "google"
    DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    SUPPORTED_API_FORMATS = ("gemini", "embed_content")

    def _api_key_auth(self, api_key: str) -> Auth:
        return HeaderAuth("x-goog-api-key", api_key)
