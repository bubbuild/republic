from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Unpack

import httpx2

from republic._errors import AuthenticationError
from republic.auth import Auth, OAuth2Auth
from republic.formats import ApiFormat, HttpRequest
from republic.formats._sse import ServerSentEvent

from .base import Provider

if TYPE_CHECKING:
    from republic._registry import ProviderOptions


class GitHubCLIAuth(Auth):
    """Use the active GitHub CLI login, without copying its credential store.

    This supplies a GitHub user token directly to Copilot's API. Access still
    depends on the account's Copilot subscription and policies.
    """

    requires_request_body = True

    def __init__(self, *, hostname: str = "github.com", executable: str = "gh") -> None:
        self.hostname = hostname
        self.executable = executable

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


class GitHubCopilot(Provider):
    """Copilot inference with ``auth=GitHubCLIAuth()`` and an eligible account.

    Select the format supported by your model; chat is the default.
    """

    name = "github-copilot"
    DEFAULT_API_BASE = "https://api.githubcopilot.com"
    SUPPORTED_API_FORMATS = ("chat", "responses", "messages")

    def __init__(self, **options: Unpack[ProviderOptions]) -> None:
        options.setdefault("api_format", "chat")
        super().__init__(**options)

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
