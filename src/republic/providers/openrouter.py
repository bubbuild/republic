from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Self

import httpx2
from authlib.common.security import generate_token
from authlib.oauth2.rfc7636 import create_s256_code_challenge

from republic._errors import AuthenticationError
from republic._options import ReasoningEffort
from republic.auth import HeaderAuth
from republic.formats.chat import ChatFormat

from .base import Provider, _FormatT


class OpenRouterAuth(HeaderAuth):
    """An OpenRouter API key, optionally obtained through PKCE login.

    Save ``api_key`` in the caller's credential store and pass it to the
    constructor to restore a login. OpenRouter issues an API key, not a
    refreshable OAuth token.
    """

    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise AuthenticationError("OpenRouter requires an API key")
        self._api_key = api_key
        super().__init__("Authorization", f"Bearer {api_key}")

    @property
    def api_key(self) -> str:
        """The API key, for caller-managed persistence."""
        return self._api_key

    @classmethod
    async def login(cls, *, on_authorize: Callable[[str], Awaitable[str]]) -> Self:
        """Ask the caller to display an authorization URL and return its code.

        Uses OpenRouter's headless PKCE flow: the browser displays a code
        for the user to copy. The callback owns the UI and cancellation.
        """
        verifier = generate_token(48)
        url = httpx2.URL(
            "https://openrouter.ai/auth",
            params={"code_challenge": create_s256_code_challenge(verifier), "code_challenge_method": "S256"},
        )
        code = await on_authorize(str(url))
        if not isinstance(code, str) or not code.strip():
            raise AuthenticationError("OpenRouter authorization returned no code")
        try:
            async with httpx2.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/auth/keys",
                    json={"code": code.strip(), "code_verifier": verifier, "code_challenge_method": "S256"},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx2.HTTPError, ValueError):
            raise AuthenticationError("Cannot exchange the OpenRouter authorization code") from None
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str) or not key.strip():
            raise AuthenticationError("OpenRouter authorization returned no API key")
        return cls(key)


class OpenRouterChatFormat(ChatFormat):
    """OpenRouter's chat completions, which take reasoning options as one ``reasoning`` object."""

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        reasoning: dict[str, Any] = {}
        if effort is not None:
            reasoning["effort"] = effort
        if include_reasoning:
            reasoning["exclude"] = False
        return {"reasoning": reasoning} if reasoning else {}


_CHAT_FORMAT = OpenRouterChatFormat()


class OpenRouter(Provider):
    name = "openrouter"
    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    # OpenRouter routes every API format to any model it serves, and also hosts
    # System One decision models such as TypeSafe's Jev.
    SUPPORTED_API_FORMATS = ("responses", "messages", "chat", "embeddings", "system_one")

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        api_format = super().select_api_format(format_kind, model)
        if isinstance(api_format, ChatFormat) and isinstance(_CHAT_FORMAT, format_kind):
            return _CHAT_FORMAT
        return api_format
