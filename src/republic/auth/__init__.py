"""Authentication helpers."""

from __future__ import annotations

from collections.abc import Callable

from republic.auth.github_copilot import *  # noqa: F403
from republic.auth.openai_codex import *  # noqa: F403

APIKeyResolver = Callable[[str], str | None]


def multi_api_key_resolver(*resolvers: APIKeyResolver) -> APIKeyResolver:
    def resolver(provider: str) -> str | None:
        for r in resolvers:
            value = r(provider)
            if value is not None:
                return value
        return None

    return resolver
