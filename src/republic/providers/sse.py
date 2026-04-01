"""Small SSE helpers shared by HTTP-based provider backends."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any


def iter_sse_json(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    buffer: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if not buffer:
                continue
            payload = "".join(buffer)
            buffer.clear()
            if payload == "[DONE]":
                break
            yield json.loads(payload)
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].strip())


async def aiter_sse_json(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    buffer: list[str] = []
    async for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if not buffer:
                continue
            payload = "".join(buffer)
            buffer.clear()
            if payload == "[DONE]":
                break
            yield json.loads(payload)
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].strip())
