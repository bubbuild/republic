"""Synthetic Grok data and the real OpenAI SDK over an offline transport."""

import json
import time
from types import TracebackType
from typing import Any, Self

import httpx
import openai

from republic.auth.grok import GrokDeviceAuthorization, GrokTokens
from republic.providers.grok import GrokOAuth
from tests.http_fixtures import Transport


def device_payload(**extra: Any) -> dict[str, Any]:
    return {
        "device_code": "private-device",
        "user_code": "private-user-code",
        "verification_uri": "https://accounts.x.ai/device",
        "verification_uri_complete": "https://accounts.x.ai/device?user_code=private-user-code",
        "expires_in": 600,
        "interval": 5,
        **extra,
    }


def device(**extra: Any) -> GrokDeviceAuthorization:
    raw = device_payload()
    del raw["expires_in"]
    values: dict[str, Any] = {**raw, "expires_at": time.time() + 600, "client_version": "1.0.41", **extra}
    return GrokDeviceAuthorization(**values)


def token_payload(**extra: Any) -> dict[str, Any]:
    return {"access_token": "private-access", "refresh_token": "private-refresh", "token_type": "Bearer", **extra}


def tokens(**extra: Any) -> GrokTokens:
    values: dict[str, Any] = {"access_token": "private-access", "refresh_token": "private-refresh", **extra}
    return GrokTokens(**values)


class Wire:
    def __init__(self, replies: list[httpx.Response | Exception], credentials: GrokTokens | None = None) -> None:
        self.transport = Transport(replies)
        self.client = openai.AsyncOpenAI(
            api_key="unrelated-api-key",
            base_url="https://unrelated.test/v1",
            max_retries=4,
            organization="unrelated-org",
            project="unrelated-project",
            default_headers={"Authorization": "unrelated-auth", "x-unrelated": "yes"},
            default_query={"unrelated": "yes"},
            http_client=httpx.AsyncClient(transport=self.transport),
        )
        self.provider = GrokOAuth(credentials or tokens(), client_version="1.0.41", client=self.client)

    @property
    def requests(self) -> list[httpx.Request]:
        return self.transport.requests

    def payload(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.provider.aclose()
        await self.client.close()
