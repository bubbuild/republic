from __future__ import annotations

from .base import Provider


class Magpie(Provider):
    """A local Magpie gateway, which routes every API format to any model in its catalog."""

    name = "magpie"
    DEFAULT_API_BASE = "http://127.0.0.1:3425/v1"
    SUPPORTED_API_FORMATS = ("chat", "responses", "messages", "system_one")
