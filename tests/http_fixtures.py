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
