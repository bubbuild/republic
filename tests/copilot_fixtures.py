"""Synthetic credentials and controlled time; no account or live inference."""

import json
import time
from types import TracebackType
from typing import Any, Self

import httpx
import openai

from republic.auth.github_copilot import CopilotToken, DeviceAuthorization, GitHubToken
from republic.providers.github_copilot import GitHubCopilot
from tests.http_fixtures import Clock as Clock
from tests.http_fixtures import Transport


def device_payload(**extra: Any) -> dict[str, Any]:
    return {
        "device_code": "private-device",
        "user_code": "private-user-code",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
        **extra,
    }


def device(**extra: Any) -> DeviceAuthorization:
    raw = device_payload()
    del raw["expires_in"]
    values: dict[str, Any] = {**raw, "expires_at": time.time() + 900, **extra}
    return DeviceAuthorization(**values)


def login_payload(**extra: Any) -> dict[str, Any]:
    return {"access_token": "private-github", "scope": "user:email", "token_type": "bearer", **extra}


def login(**extra: Any) -> GitHubToken:
    values: dict[str, Any] = {"access_token": "private-github", **extra}
    return GitHubToken(**values)


def inference_payload(**extra: Any) -> dict[str, Any]:
    return {
        "token": "private-copilot",
        "expires_at": time.time() + 1800,
        "refresh_in": 1500,
        "endpoints": {"api": "https://api.individual.githubcopilot.com"},
        **extra,
    }


def token(**extra: Any) -> CopilotToken:
    values: dict[str, Any] = {
        "token": "private-copilot",
        "expires_at": time.time() + 1800,
        "api_endpoint": "https://api.individual.githubcopilot.com",
        **extra,
    }
    return CopilotToken(**values)


class Wire:
    def __init__(self, replies: list[httpx.Response | Exception], credentials: CopilotToken | None = None) -> None:
        self.transport = Transport(replies)
        self.client = openai.AsyncOpenAI(
            api_key="unrelated-api-key",
            base_url="https://unrelated.test/v1",
            max_retries=4,
            organization="unrelated-org",
            project="unrelated-project",
            default_headers={"x-unrelated": "yes"},
            default_query={"unrelated": "yes"},
            http_client=httpx.AsyncClient(transport=self.transport),
        )
        self.provider = GitHubCopilot(
            credentials or token(),
            integration_id="fixture-integration",
            client=self.client,
            base_url="https://api.individual.githubcopilot.com",
            max_retries=0,
        )

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
