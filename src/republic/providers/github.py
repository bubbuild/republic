"""Copilot inference using the GitHub CLI login or an explicit token."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Unpack

import httpx2

from republic._errors import AuthenticationError
from republic.auth import Auth
from republic.formats import ApiFormat, HttpRequest
from republic.formats._sse import ServerSentEvent

from .base import Provider

if TYPE_CHECKING:
    from republic._registry import ProviderOptions

_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_COPILOT_API_BASE = "https://api.githubcopilot.com"
_EXCHANGE_TIMEOUT = httpx2.Timeout(30)
_REFRESH_MARGIN = 60


class GitHubCLIAuth(Auth):
    """Exchange the active GitHub CLI login for Copilot inference tokens.

    ``gh auth token`` yields a GitHub login, which Copilot's inference API does
    not accept. The login is exchanged for a short-lived Copilot token that is
    renewed before it expires, and requests follow the API origin the exchange
    names. Access still depends on the account's Copilot subscription and
    policies.
    """

    requires_request_body = True

    def __init__(self, *, hostname: str = "github.com", executable: str = "gh") -> None:
        self.hostname = hostname
        self.executable = executable
        self._token: str | None = None
        self._refresh_at: float | None = None
        self._api_base = _COPILOT_API_BASE

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        if self._needs_exchange():
            self._store((yield _exchange_request(self._load_login())))
        yield self._sign(request)

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        await request.aread()
        if self._needs_exchange():
            response = yield _exchange_request(await asyncio.to_thread(self._load_login))
            await response.aread()
            self._store(response)
        yield self._sign(request)

    def _needs_exchange(self) -> bool:
        return self._token is None or (self._refresh_at is not None and time.time() >= self._refresh_at)

    def _sign(self, request: httpx2.Request) -> httpx2.Request:
        """Sign with the Copilot token, at the origin that issued it."""
        request.headers["Authorization"] = f"Bearer {self._token}"
        origin = httpx2.URL(self._api_base)
        if (request.url.host, request.url.port) != (origin.host, origin.port):
            request.url = request.url.copy_with(scheme=origin.scheme, host=origin.host, port=origin.port)
        return request

    def _store(self, response: httpx2.Response) -> None:
        """Cache the Copilot token and the API origin the exchange names."""
        if response.status_code != 200:
            raise AuthenticationError(
                f"GitHub did not exchange the GitHub CLI login for a Copilot token ({response.status_code})"
            )
        try:
            payload = response.json()
        except ValueError:
            raise AuthenticationError("The Copilot token exchange returned no JSON") from None
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise AuthenticationError("The Copilot token exchange returned no token")
        endpoints = payload.get("endpoints")
        api = endpoints.get("api") if isinstance(endpoints, dict) else None
        self._api_base = api.rstrip("/") if isinstance(api, str) and api.startswith("https://") else _COPILOT_API_BASE
        self._token = token
        self._refresh_at = _refresh_at(payload)

    def _load_login(self) -> str:
        try:
            result = subprocess.run(  # noqa: S603
                [self.executable, "auth", "token", "--hostname", self.hostname],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise AuthenticationError("Cannot read the GitHub CLI login; run gh auth login") from None
        if result.returncode or not result.stdout.strip():
            raise AuthenticationError("No active GitHub CLI token; run gh auth login")
        return result.stdout.strip()


class GitHubCopilot(Provider):
    """Copilot inference using the current GitHub CLI login by default.

    Select the format supported by your model; chat is the default.
    """

    name = "github-copilot"
    DEFAULT_API_BASE = _COPILOT_API_BASE
    SUPPORTED_API_FORMATS = ("chat", "responses", "messages")
    # The client identity Copilot's inference API expects; the first-party editor
    # sends the same set. Override any of them through headers=.
    CLIENT_HEADERS: ClassVar[Mapping[str, str]] = {
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": "vscode/1.95.0",
        "Editor-Plugin-Version": "copilot-chat/0.26.7",
        "X-GitHub-Api-Version": "2025-10-01",
    }

    def __init__(self, **options: Unpack[ProviderOptions]) -> None:
        options.setdefault("api_format", "chat")
        super().__init__(**options)
        if self.auth is None and (self._http_client is None or self._http_client.auth is None):
            self.auth = GitHubCLIAuth()
        for header, value in self.CLIENT_HEADERS.items():
            self.headers.setdefault(header, value)

    @asynccontextmanager
    async def _stream(
        self, api_format: ApiFormat, request: HttpRequest
    ) -> AsyncGenerator[AsyncIterator[ServerSentEvent]]:
        async with super()._stream(api_format, request) as events:
            yield _copilot_events(events, api_format)

    async def _send(
        self, client: httpx2.AsyncClient, api_format: ApiFormat, request: HttpRequest, *, stream: bool
    ) -> httpx2.Response:
        if api_format.name == "messages":
            request = replace(request, path="/v1/messages")
        return await super()._send(client, api_format, request, stream=stream)


def _exchange_request(login: str) -> httpx2.Request:
    """Build the exchange, which the client sending the request will also send.

    The request carries its own timeout: a client applies its timeout to requests
    it builds, not to the requests an auth flow yields.
    """
    return httpx2.Request(
        "GET",
        _EXCHANGE_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"token {login}",
            "User-Agent": "republic",
            "X-GitHub-Api-Version": "2025-04-01",
        },
        extensions={"timeout": _EXCHANGE_TIMEOUT.as_dict()},
    )


def _refresh_at(payload: dict[str, Any]) -> float | None:
    """When to exchange again: the server's advice, else the declared expiry."""
    refresh_in = _positive(payload.get("refresh_in"))
    if refresh_in is not None:
        return time.time() + refresh_in + _REFRESH_MARGIN
    expires_at = _positive(payload.get("expires_at"))
    return None if expires_at is None else expires_at - _REFRESH_MARGIN


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


async def _copilot_events(
    events: AsyncIterator[ServerSentEvent], api_format: ApiFormat
) -> AsyncIterator[ServerSentEvent]:
    item_ids: dict[int, str] = {}
    async for event in events:
        # Copilot terminates Messages streams with this marker as well.
        if event.data == "[DONE]":
            break
        yield _stable_tool_id(event, item_ids) if api_format.name == "responses" else event


def _stable_tool_id(event: ServerSentEvent, item_ids: dict[int, str]) -> ServerSentEvent:
    """Copilot re-encodes item IDs per event; output_index identifies the call."""
    payload = json.loads(event.data)
    index = payload.get("output_index")
    match payload.get("type"):
        case "response.output_item.added" if payload["item"].get("type") == "function_call":
            item_ids[index] = payload["item"].get("id") or payload["item"]["call_id"]
        case "response.function_call_arguments.delta" if index in item_ids:
            payload["item_id"] = item_ids[index]
            return replace(event, data=json.dumps(payload))
        case "response.output_item.done" if index in item_ids and payload["item"].get("type") == "function_call":
            payload["item"]["id"] = item_ids[index]
            return replace(event, data=json.dumps(payload))
    return event
