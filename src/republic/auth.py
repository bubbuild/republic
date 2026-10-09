"""Common authentication for ``Provider(auth=...)``.

``Auth`` is HTTPX's standard auth interface. Service-specific implementations
live with their providers, such as ``republic.providers.CodexAuth``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import suppress

import httpx2
from authlib.integrations.httpx_client import OAuth2Auth
from httpx2 import Auth

from republic.errors import AuthenticationError

__all__ = ["Auth", "HeaderAuth", "OAuth2Auth"]


class HeaderAuth(Auth):
    """Send a fixed header, such as an API key, with every request."""

    def __init__(self, name: str, value: str) -> None:
        self._name = name
        self._value = value

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        request.headers[self._name] = self._value
        yield request


async def _run_login(*command: str) -> None:
    """Run a CLI login in the caller's terminal and reap it on cancellation."""
    try:
        process = await asyncio.create_subprocess_exec(*command)
    except OSError:
        raise AuthenticationError(f"Cannot start {command[0]}; install the CLI or set executable=") from None
    try:
        returncode = await process.wait()
    except asyncio.CancelledError:
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        raise
    if returncode:
        raise AuthenticationError(f"{command[0]} login failed (exit {returncode})")
