"""Built-in tools that providers run on their own servers.

Pass them in ``tools=`` next to function :class:`~republic.Tool` schemas. Each
API format maps them to its native tool and raises
:class:`~republic.UnsupportedFeatureError` for tools or settings it has no
equivalent for. Use :class:`NativeTool` for anything else a provider offers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BuiltinTool", "CodeExecution", "ImageGeneration", "NativeTool", "UserLocation", "WebFetch", "WebSearch"]


@dataclass(frozen=True)
class UserLocation:
    """An approximate location used to localize search results."""

    city: str | None = None
    region: str | None = None
    country: str | None = None
    """Two-letter ISO country code."""
    timezone: str | None = None
    """IANA time zone, such as ``"Europe/Paris"``."""


@dataclass(frozen=True)
class WebSearch:
    max_uses: int | None = None
    allowed_domains: Sequence[str] = ()
    blocked_domains: Sequence[str] = ()
    user_location: UserLocation | None = None


@dataclass(frozen=True)
class WebFetch:
    """Read the pages at URLs that appear in the conversation."""

    max_uses: int | None = None
    allowed_domains: Sequence[str] = ()
    blocked_domains: Sequence[str] = ()


@dataclass(frozen=True)
class CodeExecution:
    """Run code in a provider-hosted sandbox."""


@dataclass(frozen=True)
class ImageGeneration:
    """Let the model call the provider's image generator."""


@dataclass(frozen=True)
class NativeTool:
    """A tool definition sent verbatim, only to the API format it was written for."""

    api_format: str
    definition: Mapping[str, Any] = field(default_factory=dict)


BuiltinTool = WebSearch | WebFetch | CodeExecution | ImageGeneration | NativeTool
