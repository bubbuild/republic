from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSentEvent:
    event: str
    data: str


async def iter_events(lines: AsyncIterable[str], *, end_marker: str | None = None) -> AsyncIterator[ServerSentEvent]:
    """Parse SSE events, stopping at the protocol's optional data marker."""
    event = ""
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                payload = "\n".join(data)
                if payload == end_marker:
                    return
                yield ServerSentEvent(event or "message", payload)
            event, data = "", []
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        payload = "\n".join(data)
        if payload != end_marker:
            yield ServerSentEvent(event or "message", payload)
