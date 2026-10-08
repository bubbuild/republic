"""ChatGPT plan access through Codex's Responses endpoint."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import tempfile
import threading
from collections.abc import AsyncGenerator, Generator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Unpack

import httpx2
from authlib.integrations.base_client import OAuthError
from authlib.integrations.httpx_client import OAuth2Client
from authlib.oauth2.rfc6749 import OAuth2Token

from republic._content import Input
from republic._errors import AuthenticationError, UnsupportedFeatureError
from republic._models import ChatModel
from republic._options import ChatOptions
from republic._response import Response
from republic.auth import Auth, OAuth2Auth, _run_login
from republic.formats import ApiFormat, ChatApiFormat, HttpRequest
from republic.history import HistoryProtocol

from .base import Provider

if TYPE_CHECKING:
    from republic._registry import ProviderOptions


class CodexAuth(Auth):
    """Authenticate with a Codex OAuth token and ChatGPT account ID.

    Expiring tokens are refreshed through Authlib. The current token is available
    as ``token``; callers own its persistence. Use ``from_file()`` to reuse and
    maintain Codex's file credential store. Reuse one instance across requests.
    """

    requires_request_body = True

    def __init__(self, token: Mapping[str, Any], *, account_id: str) -> None:
        if (
            not isinstance(token.get("access_token"), str)
            or not token["access_token"]
            or not isinstance(account_id, str)
            or not account_id
        ):
            raise AuthenticationError("Codex credentials are missing an access token or account ID")
        self.token = OAuth2Token(dict(token))
        self.token.setdefault("token_type", "Bearer")
        self.token["expires_at"] = _expires_at(self.token)
        self.account_id = account_id
        self._auth_file: Path | None = None
        self._lock = threading.Lock()

    @classmethod
    async def login(cls, *, executable: str = "codex", device_auth: bool = False) -> Self:
        """Run Codex's interactive login and return its file-backed credentials.

        Uses ``$CODEX_HOME`` and selects file storage for this invocation only.
        Set ``device_auth=True`` to use the CLI's device authorization flow.
        """
        arguments = [executable, "login", "--config", 'cli_auth_credentials_store="file"']
        if device_auth:
            arguments.append("--device-auth")
        await _run_login(*arguments)
        return cls.from_file()

    @classmethod
    def from_file(cls, auth_file: str | Path | None = None) -> Self:
        """Read a Codex login, defaulting to ``$CODEX_HOME/auth.json``.

        The file is read now and before every request. Refreshed tokens are saved
        atomically to the same file, preserving other fields.
        """
        codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
        path = Path(auth_file).expanduser() if auth_file is not None else codex_home / "auth.json"
        data, token = _read_credentials(path)
        auth = cls(token, account_id=data["tokens"]["account_id"])
        auth._auth_file = path
        return auth

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        token, account_id = self._load_token()
        request.headers["ChatGPT-Account-Id"] = account_id
        yield from OAuth2Auth(token).auth_flow(request)

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        token, account_id = await asyncio.to_thread(self._load_token)
        await request.aread()
        request.headers["ChatGPT-Account-Id"] = account_id
        for signed in OAuth2Auth(token).auth_flow(request):
            yield signed

    def _load_token(self) -> tuple[OAuth2Token, str]:
        with self._lock:
            token, account_id = self.token, self.account_id
            data = None
            if self._auth_file is not None:
                data, token = _read_credentials(self._auth_file)
                account_id = data["tokens"]["account_id"]
            if token.is_expired():
                token = self._refresh(token)
                if data is not None and self._auth_file is not None:
                    stored = data["tokens"]
                    stored.update({
                        key: token[key] for key in ("access_token", "refresh_token", "id_token") if key in token
                    })
                    expiry = token.get("expires_at")
                    if expiry is None:
                        stored.pop("expires_at", None)
                    else:
                        stored["expires_at"] = datetime.fromtimestamp(expiry, UTC).isoformat()
                    data["last_refresh"] = datetime.now(UTC).isoformat()
                    _save_credentials(self._auth_file, data)
            self.token, self.account_id = token, account_id
            return token, account_id

    def _refresh(self, token: OAuth2Token) -> OAuth2Token:
        if not token.get("refresh_token"):
            raise AuthenticationError("Codex token has expired and has no refresh token")
        try:
            with OAuth2Client(
                client_id="app_EMoamEEZ73f0CkXaXp7hrann",
                token_endpoint_auth_method="none",  # noqa: S106 - public OAuth client, not a secret
                token=token,
                token_endpoint="https://auth.openai.com/oauth/token",  # noqa: S106 - public endpoint
                timeout=30,
            ) as client:
                client.ensure_active_token()
                if not isinstance(client.token.get("access_token"), str) or not client.token["access_token"]:
                    raise AuthenticationError("Codex refresh returned no access token")
                if "id_token" in token:
                    client.token.setdefault("id_token", token["id_token"])
                return client.token
        except (OAuthError, httpx2.HTTPError, ValueError):
            raise AuthenticationError("Cannot refresh the Codex token") from None


class Codex(Provider):
    """Responses-only provider using the local Codex login by default."""

    name = "codex"
    DEFAULT_API_BASE = "https://chatgpt.com/backend-api/codex"
    SUPPORTED_API_FORMATS = ("responses",)

    def __init__(self, **options: Unpack[ProviderOptions]) -> None:
        super().__init__(**options)
        if self.auth is None and (self._http_client is None or self._http_client.auth is None):
            self.auth = CodexAuth.from_file()
        self.headers.setdefault("OpenAI-Beta", "responses=experimental")
        self.headers.setdefault("originator", "republic")

    def get_model(self, name: str, *, history: HistoryProtocol | None = None) -> ChatModel:
        return _CodexModel(self, name, self.select_api_format(ChatApiFormat, name), history=history)

    async def _send(
        self, client: httpx2.AsyncClient, api_format: ApiFormat, request: HttpRequest, *, stream: bool
    ) -> httpx2.Response:
        body = dict(request.body)
        if "max_output_tokens" in body:
            raise UnsupportedFeatureError("Codex does not support max_tokens")
        body.update(stream=True, store=False)
        body.setdefault("instructions", "You are Codex.")
        body["include"] = list(dict.fromkeys([*body.get("include", []), "reasoning.encrypted_content"]))
        return await super()._send(client, api_format, replace(request, body=body), stream=stream)


class _CodexModel(ChatModel):
    async def chat(
        self, prompt: Input, *, output_schema: type[Any] | None = None, **options: Unpack[ChatOptions]
    ) -> Response[Any]:
        # The endpoint always streams. Keep the normal parser, history and output validation.
        async with self.stream(prompt, output_schema=output_schema, **options) as stream:
            async for _ in stream:
                pass
        return stream.response


def _read_credentials(path: Path) -> tuple[dict[str, Any], OAuth2Token]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        raise AuthenticationError("Cannot read Codex credentials; run codex login with file storage") from None
    if not isinstance(data, dict) or not isinstance(data.get("tokens"), dict):
        raise AuthenticationError("Codex credentials must contain a ChatGPT login")
    stored = data["tokens"]
    if any(not isinstance(stored.get(key), str) or not stored[key] for key in ("access_token", "account_id")):
        raise AuthenticationError("Codex credentials are missing an access token or account ID")
    # A persisted expires_in cannot be interpreted relative to the current time.
    token = OAuth2Token({
        **{key: stored[key] for key in ("access_token", "refresh_token", "id_token") if key in stored},
        "expires_at": _expires_at(stored),
        "token_type": "Bearer",
    })
    return data, token


def _expires_at(stored: dict[str, Any]) -> int | None:
    """Read expiry metadata only; the server verifies the access token."""
    expiry = stored.get("expires_at")
    if isinstance(expiry, (int, float)):
        return int(expiry)
    if isinstance(expiry, str):
        try:
            return int(datetime.fromisoformat(expiry).timestamp())
        except ValueError:
            pass
    try:
        payload = stored["access_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError, binascii.Error):
        return None
    expiry = claims.get("exp") if isinstance(claims, dict) else None
    return int(expiry) if isinstance(expiry, (int, float)) else None


def _save_credentials(path: Path, data: dict[str, Any]) -> None:
    # NamedTemporaryFile creates a private (0600) file on the same filesystem.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
