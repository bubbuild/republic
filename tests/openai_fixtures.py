"""HTTP/SSE fixtures exercising the real official OpenAI async client."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import httpx
from openai import AsyncOpenAI

from republic import Message, Request, TextPart
from republic.providers.openai import OpenAIChatCompletions


def request() -> Request:
    return Request(model="vendor/model", messages=[Message(role="user", parts=[TextPart(text="Hello")])])


def completion(message: dict[str, Any] | None = None, *, finish: str | None = "stop", **extra: Any) -> dict[str, Any]:
    return {
        "id": "chat-1",
        "model": "resolved-model",
        "created": 1,
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **(message or {"content": "Hello"})}, "finish_reason": finish}
        ],
        **extra,
    }


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish: str | None = None,
    logprobs: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": "chat-1",
        "model": "resolved-model",
        "created": 1,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish, "logprobs": logprobs}],
        **extra,
    }


def sse(data: dict[str, Any] | str) -> bytes:
    return f"data: {data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}\n\n".encode()


class Bytes(httpx.AsyncByteStream):
    def __init__(self, pieces: list[bytes], *, wait: bool = False, error: Exception | None = None) -> None:
        self.pieces = pieces
        self.wait = wait
        self.error = error
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = 0
        self.consumed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            self.consumed += 1
            yield piece
        if self.wait:
            self.waiting.set()
            await self.release.wait()
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        self.closed += 1


def streaming(body: Bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)


class Wire:
    def __init__(self, replies: list[httpx.Response | Exception]) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        # Deliberately enable SDK retries on the borrowed client. The adapter
        # must suppress them without mutating this caller-owned object.
        self.client = AsyncOpenAI(
            api_key="fixture-key", base_url="https://unit.test/api/v1", max_retries=3, http_client=self.http
        )
        self.provider = OpenAIChatCompletions(client=self.client)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def payload(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.provider.aclose()
        await self.client.close()
