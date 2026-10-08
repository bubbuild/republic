"""Common authentication for ``Provider(auth=...)``.

``Auth`` is HTTPX's standard auth interface. Service-specific implementations
live with their providers, such as ``republic.providers.CodexAuth``.
"""

from __future__ import annotations

from collections.abc import Generator

import httpx2
from authlib.integrations.httpx_client import OAuth2Auth
from httpx2 import Auth

__all__ = ["Auth", "HeaderAuth", "OAuth2Auth"]


class HeaderAuth(Auth):
    """Send a fixed header, such as an API key, with every request."""

    def __init__(self, name: str, value: str) -> None:
        self._name = name
        self._value = value

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        request.headers[self._name] = self._value
        yield request
