# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: no executor, replay, agent context or generate fallback.
# See NOTICE for the upstream revision and extraction details.
"""Exactly one provider operation per API call."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from republic._stream import Stream
from republic.events import Event
from republic.types import Request, Response


class Provider(Protocol):
    """Runtime adapter; clients and credentials live here, outside persisted data.

    stream must be an async generator that acquires resources inside its body
    and releases them in finally, including when aclose is called. An adapter
    implements generate explicitly; Republic does not retry or fall back.
    """

    async def generate(self, request: Request) -> Response:
        """Make one non-streaming model request."""
        ...

    def stream(self, request: Request) -> AsyncGenerator[Event, None]:
        """Make one streaming request and emit exactly one final StreamEnd."""
        ...


async def generate(provider: Provider, request: Request) -> Response:
    """Return one provider response; pass an isolated copy of caller history."""
    return await provider.generate(request.model_copy(deep=True))


@asynccontextmanager
async def stream(provider: Provider, request: Request) -> AsyncIterator[Stream]:
    """Own a provider stream until normal completion, early exit or cancellation."""
    result = Stream(provider.stream(request.model_copy(deep=True)))
    try:
        yield result
    except asyncio.CancelledError:
        result._cancel()
        raise
    finally:
        await result.aclose()
