"""Controllable HTTP response bodies shared by provider transport tests."""

import asyncio
from collections.abc import AsyncIterator

import httpx


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
