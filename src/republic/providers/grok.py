"""Grok inference with the official CLI's xAI OAuth login."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from collections.abc import AsyncGenerator, Generator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Unpack

import httpx2
from authlib.integrations.base_client import OAuthError
from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2.rfc6749 import OAuth2Token
from filelock import FileLock

from republic._errors import AuthenticationError
from republic.auth import Auth, OAuth2Auth, _run_login

from .base import Provider

if TYPE_CHECKING:
    from republic._registry import ProviderOptions

_ISSUER = "https://auth.x.ai"
_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_SCOPE = f"{_ISSUER}::{_CLIENT_ID}"


class GrokAuth(Auth):
    """An xAI OAuth token, refreshed with Authlib before expiry.

    Caller-supplied tokens are available as ``token`` for persistence.
    ``from_file()`` reads and maintains the official Grok CLI credential store.
    Reuse one instance across requests.
    """

    requires_request_body = True

    def __init__(self, token: Mapping[str, Any]) -> None:
        self.token = _oauth_token(token)
        self._auth_file: Path | None = None
        self._lock = threading.Lock()

    @classmethod
    async def login(cls, *, executable: str = "grok", device_auth: bool = False) -> Self:
        """Run the official CLI login in the caller's terminal."""
        arguments = [executable, "login"]
        if device_auth:
            arguments.append("--device-auth")
        await _run_login(*arguments)
        return cls.from_file()

    @classmethod
    def from_file(cls, auth_file: str | Path | None = None) -> Self:
        """Reuse xAI OAuth credentials from ``$GROK_HOME/auth.json``.

        Honors ``GROK_AUTH_PATH``. Reads before each request; rotated tokens
        are saved atomically under the CLI's ``auth.json.lock`` file lock.
        """
        grok_home = Path(os.environ.get("GROK_HOME", "~/.grok")).expanduser()
        path = Path(auth_file or os.environ.get("GROK_AUTH_PATH") or grok_home / "auth.json").expanduser()
        _, token = _read_credentials(path)
        auth = cls(token)
        auth._auth_file = path
        return auth

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        yield from OAuth2Auth(self._load_token()).auth_flow(request)

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        token = await asyncio.to_thread(self._load_token)
        await request.aread()
        for signed in OAuth2Auth(token).auth_flow(request):
            yield signed

    def _load_token(self) -> OAuth2Token:
        with self._lock:
            if self._auth_file is None:
                if self.token.is_expired():
                    self.token = self._refresh(self.token)
            else:
                self.token = self._load_file_token(self._auth_file)
            return self.token

    def _load_file_token(self, path: Path) -> OAuth2Token:
        _, token = _read_credentials(path)
        if not token.is_expired():
            return token
        try:
            # Match the CLI's OS lock; never unlink it or fall back to a soft lock.
            with FileLock(
                path.with_name("auth.json.lock"),
                timeout=30,
                mode=0o600,
                preserve_lock_file=True,
                fallback_to_soft=False,
            ):
                data, token = _read_credentials(path)
                if token.is_expired():
                    token = self._refresh(token)
                    stored = data[_SCOPE]
                    stored["key"] = token["access_token"]
                    stored["refresh_token"] = token.get("refresh_token")
                    stored["expires_at"] = datetime.fromtimestamp(token["expires_at"], UTC).isoformat()
                    stored["create_time"] = datetime.now(UTC).isoformat()
                    _save_credentials(path, data)
                return token
        except (OSError, TimeoutError):
            raise AuthenticationError("Cannot update the Grok credential store") from None

    def _refresh(self, token: OAuth2Token) -> OAuth2Token:
        if not token.get("refresh_token"):
            raise AuthenticationError("Grok token has expired; run grok login")
        principal = {key: token[key] for key in ("principal_type", "principal_id") if token.get(key)}
        try:
            with OAuth2Client(
                client_id=_CLIENT_ID,
                token_endpoint_auth_method="none",  # noqa: S106 - public OAuth client
                token=token,
                timeout=30,
            ) as client:
                refreshed = client.refresh_token(f"{_ISSUER}/oauth2/token", **principal)
                return _oauth_token({**refreshed, **principal})
        except (OAuthError, httpx2.HTTPError, ValueError, TypeError):
            raise AuthenticationError("Cannot refresh the Grok token; run grok login") from None


class Grok(Provider):
    """xAI chat and Responses, using the local Grok OAuth login by default."""

    name = "grok"
    DEFAULT_API_BASE = "https://api.x.ai/v1"
    SUPPORTED_API_FORMATS = ("responses", "chat")

    def __init__(self, **options: Unpack[ProviderOptions]) -> None:
        super().__init__(**options)
        if self.auth is None and (self._http_client is None or self._http_client.auth is None):
            self.auth = GrokAuth.from_file()


def _oauth_token(value: Mapping[str, Any]) -> OAuth2Token:
    if not isinstance(value.get("access_token"), str) or not value["access_token"]:
        raise AuthenticationError("Grok credentials are missing an access token")
    try:
        token = OAuth2Token({**value, "token_type": "Bearer"})
        expiry = token["expires_at"]
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry).timestamp()
        token["expires_at"] = int(expiry)
    except (KeyError, ValueError, TypeError, OverflowError):
        raise AuthenticationError("Grok credentials are missing a valid expiry") from None
    return token


def _read_credentials(path: Path) -> tuple[dict[str, Any], OAuth2Token]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        raise AuthenticationError("Cannot read Grok credentials; run grok login") from None
    stored = data.get(_SCOPE) if isinstance(data, dict) else None
    if (
        not isinstance(stored, dict)
        or stored.get("oidc_issuer") != _ISSUER
        or stored.get("oidc_client_id") != _CLIENT_ID
    ):
        raise AuthenticationError("Grok credentials must contain an xAI OAuth login; run grok login")
    token = _oauth_token({
        "access_token": stored.get("key"),
        **{
            key: stored[key]
            for key in ("refresh_token", "expires_at", "principal_type", "principal_id")
            if key in stored
        },
    })
    return data, token


def _save_credentials(path: Path, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
