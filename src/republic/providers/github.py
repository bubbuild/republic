"""Copilot inference using the GitHub CLI login or an explicit token."""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, Unpack

import httpx2
from authlib.oauth2.rfc6749 import OAuth2Token

from republic._errors import AuthenticationError
from republic.auth import Auth, OAuth2Auth, _run_login
from republic.formats import ApiFormat, HttpRequest
from republic.formats._sse import ServerSentEvent

from .base import Provider

if TYPE_CHECKING:
    from republic._registry import ProviderOptions

_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_COPILOT_API_BASE = "https://api.githubcopilot.com"
_EXCHANGE_TIMEOUT = httpx2.Timeout(30)


class GitHubCLIAuth(Auth):
    """Send the active GitHub CLI login directly to Copilot's API.

    Reads ``gh auth token`` for each request without copying its credential
    store. Access depends on the account's Copilot subscription and policies.
    """

    requires_request_body = True

    def __init__(self, *, hostname: str = "github.com", executable: str = "gh") -> None:
        self.hostname = hostname
        self.executable = executable

    @classmethod
    async def login(cls, *, hostname: str = "github.com", executable: str = "gh") -> Self:
        """Run ``gh auth login`` in the caller's terminal; gh stores the login."""
        await _run_login(executable, "auth", "login", "--hostname", hostname, "--web")
        return cls(hostname=hostname, executable=executable)

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        token = self._load_token()
        yield from OAuth2Auth({"access_token": token, "token_type": "Bearer"}).auth_flow(request)

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        token = await asyncio.to_thread(self._load_token)
        await request.aread()
        for signed in OAuth2Auth({"access_token": token, "token_type": "Bearer"}).auth_flow(request):
            yield signed

    def _load_token(self) -> str:
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


class CopilotAuth(Auth):
    """Exchange a Copilot Plugin GitHub credential for inference tokens.

    Obtain the GitHub credential through OAuth before constructing this object;
    authorization, persistence and renewal of that credential belong to the
    caller. GitHub CLI credentials use ``GitHubCLIAuth`` instead.

    Reuse this instance to cache the Copilot token in memory. It is re-exchanged
    before expiry through the inference client's transport. Requests follow the
    returned API origin and use the Plugin's client headers unless overridden.
    """

    requires_request_body = True

    CLIENT_HEADERS: ClassVar[Mapping[str, str]] = {
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": "vscode/1.107.0",
        "Editor-Plugin-Version": "copilot-chat/0.35.0",
        "X-GitHub-Api-Version": "2025-10-01",
    }

    def __init__(self, github_token: str) -> None:
        if not github_token:
            raise AuthenticationError("Copilot exchange requires a GitHub credential")
        self._github_token = github_token
        self._token = OAuth2Token({})
        self._origin = httpx2.URL(_COPILOT_API_BASE)

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        if not self._token or self._token.is_expired():
            response = yield _exchange_request(self._github_token)
            response.read()
            self._store(response)
        yield from self._sign(request)

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        await request.aread()
        if not self._token or self._token.is_expired():
            response = yield _exchange_request(self._github_token)
            await response.aread()
            self._store(response)
        for signed in self._sign(request):
            yield signed

    def _sign(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        origin = self._origin
        request.url = request.url.copy_with(scheme=origin.scheme, host=origin.host, port=origin.port)
        request.headers["Host"] = request.url.netloc.decode("ascii")
        for name, value in self.CLIENT_HEADERS.items():
            request.headers.setdefault(name, value)
        yield from OAuth2Auth(self._token).auth_flow(request)

    def _store(self, response: httpx2.Response) -> None:
        if response.status_code != 200:
            raise AuthenticationError(f"GitHub rejected the Copilot token exchange ({response.status_code})")
        try:
            payload = response.json()
        except ValueError:
            raise AuthenticationError("The Copilot token exchange returned no JSON") from None
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise AuthenticationError("The Copilot token exchange returned no token")
        origin = _api_origin(payload)
        self._token = OAuth2Token({"access_token": token, "token_type": "Bearer", "expires_at": _expires_at(payload)})
        self._origin = origin


class GitHubCopilot(Provider):
    """Copilot inference using the current GitHub CLI login by default.

    Select the format supported by your model; chat is the default.
    """

    name = "github-copilot"
    DEFAULT_API_BASE = _COPILOT_API_BASE
    SUPPORTED_API_FORMATS = ("chat", "responses", "messages")

    def __init__(self, **options: Unpack[ProviderOptions]) -> None:
        options.setdefault("api_format", "chat")
        super().__init__(**options)
        if self.auth is None and (self._http_client is None or self._http_client.auth is None):
            self.auth = GitHubCLIAuth()

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


def _api_origin(payload: dict[str, Any]) -> httpx2.URL:
    try:
        origin = httpx2.URL(payload["endpoints"]["api"])
    except (KeyError, TypeError, httpx2.InvalidURL):
        raise AuthenticationError("The Copilot token exchange returned no valid API origin") from None
    if origin.scheme != "https" or not origin.host or origin.userinfo or origin.raw_path != b"/" or origin.fragment:
        raise AuthenticationError("The Copilot token exchange returned no valid API origin")
    return origin


def _expires_at(payload: dict[str, Any]) -> float:
    """Use the earlier server deadline; Authlib supplies the refresh leeway."""
    deadlines = []
    for field in ("refresh_in", "expires_at"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            continue
        deadlines.append(time.time() + value if field == "refresh_in" else value)
    if not deadlines:
        raise AuthenticationError("The Copilot token exchange returned no valid expiry")
    return min(deadlines)


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
