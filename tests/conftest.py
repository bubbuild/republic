"""Fixtures cannot accidentally send synthetic credentials to a real service."""

import httpx
import pytest


@pytest.fixture(autouse=True)
def block_unmocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    async def deny_async(*args, **kwargs):
        pytest.fail("Tests require an explicit HTTP MockTransport")

    def deny_sync(*args, **kwargs):
        pytest.fail("Tests require an explicit HTTP MockTransport")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_sync)
