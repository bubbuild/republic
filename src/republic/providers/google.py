from __future__ import annotations

from typing import Any

from republic._response import ModelInfo
from republic.auth import Auth, HeaderAuth

from .base import Provider


class Google(Provider):
    name = "google"
    DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    SUPPORTED_API_FORMATS = ("gemini", "embed_content")

    def _api_key_auth(self, api_key: str) -> Auth:
        return HeaderAuth("x-goog-api-key", api_key)

    def _models_params(self, previous: Any) -> dict[str, str] | None:
        if previous is None:
            return {"pageSize": "1000"}
        return {"pageSize": "1000", "pageToken": token} if (token := previous.get("nextPageToken")) else None

    def _parse_models(self, data: Any) -> list[ModelInfo]:
        # Gemini names models as "models/ID"; requests take the bare ID.
        return [
            ModelInfo(item["name"].removeprefix("models/"), item.get("displayName"), item)
            for item in data.get("models") or []
        ]
