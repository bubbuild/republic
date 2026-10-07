from __future__ import annotations

from .base import Provider


class TypeSafe(Provider):
    name = "typesafe"
    DEFAULT_API_BASE = "https://api.typesafe.ai/v1"
    SUPPORTED_API_FORMATS = ("system_one",)
