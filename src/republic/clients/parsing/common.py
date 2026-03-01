"""Common parsing utilities shared by completion and responses adapters."""

from __future__ import annotations

from typing import Any


def field(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)
