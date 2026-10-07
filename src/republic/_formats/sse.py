from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSentEvent:
    event: str
    data: str


async def iter_events(lines: AsyncIterable[str]) -> AsyncIterator[ServerSentEvent]:
    """Parse the ``text/event-stream`` format into events."""
    event = ""
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                yield ServerSentEvent(event or "message", "\n".join(data))
            event, data = "", []
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield ServerSentEvent(event or "message", "\n".join(data))
