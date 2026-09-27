"""Dummy credentials and HTTP transports; never use a real account."""

import json
import time
from base64 import urlsafe_b64encode
from types import TracebackType
from typing import Any, Self

import httpx
import openai

from republic.auth.codex import CodexTokens
from republic.providers.codex import OpenAICodex


def jwt(**claims: Any) -> str:
    payload = urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{payload}.fixture-signature"


def token_payload(**overrides: Any) -> dict[str, Any]:
    return {
        "access_token": jwt(**{"https://api.openai.com/auth": {"chatgpt_account_id": "acct_fixture"}}),
        "refresh_token": "fixture-refresh-private",
        "token_type": "Bearer",
        "expires_in": 1800,
        **overrides,
    }


def tokens(**overrides: Any) -> CodexTokens:
    data = token_payload()
    values: dict[str, Any] = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + 1800,
        "account_id": "acct_fixture",
        **overrides,
    }
    return CodexTokens(**values)


class Transport(httpx.MockTransport):
    def __init__(self, replies: list[httpx.Response | Exception]) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []
        self.closed = 0
        super().__init__(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def aclose(self) -> None:
        self.closed += 1
        await super().aclose()


class Wire:
    def __init__(self, replies: list[httpx.Response | Exception], credentials: CodexTokens | None = None) -> None:
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
        self.provider = OpenAICodex(credentials or tokens(), client=self.client)

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
