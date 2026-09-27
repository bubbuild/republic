"""Deterministic test transport; never exported by the SDK."""

import asyncio
from collections.abc import AsyncGenerator

from republic import Message, Request, Response, TextPart, events


class FakeProvider:
    def __init__(
        self,
        script: list[events.Event],
        *,
        error: Exception | None = None,
        wait: asyncio.Event | None = None,
        close_gate: asyncio.Event | None = None,
    ) -> None:
        self.script = script
        self.error = error
        self.wait = wait
        self.close_gate = close_gate
        self.waiting = asyncio.Event()
        self.closing = asyncio.Event()
        self.calls: list[tuple[str, Request]] = []
        self.opened = 0
        self.closed = 0
        self.emitted = 0
        self.result = Response(message=Message(role="assistant", parts=[TextPart(text="one response")]))

    async def generate(self, request: Request) -> Response:
        self.calls.append(("generate", request.model_copy(deep=True)))
        self.opened += 1
        try:
            if self.wait is not None:
                self.waiting.set()
                await self.wait.wait()
            if self.error is not None:
                raise self.error
            return self.result.model_copy(deep=True)
        finally:
            self.closed += 1

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        self.calls.append(("stream", request.model_copy(deep=True)))
        self.opened += 1
        try:
            for event in self.script:
                self.emitted += 1
                yield event
            if self.wait is not None:
                self.waiting.set()
                await self.wait.wait()
            if self.error is not None:
                raise self.error
        finally:
            self.closing.set()
            if self.close_gate is not None:
                await self.close_gate.wait()
            self.closed += 1


def request() -> Request:
    return Request(model="fake-model", messages=[Message(role="user", parts=[TextPart(text="hello")])])
