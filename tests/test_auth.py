from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import republic
from republic.auth import OAuth2Auth
from tests.conftest import FakeService


async def test_authlib_auth_works_with_existing_provider(service: FakeService) -> None:
    service.reply_json({"output": []})
    model = republic.get_model(
        "openai:test", api_key="ignored", auth=OAuth2Auth({"access_token": "oauth-token"}), http_client=service.client()
    )

    await model.chat("hello")

    assert service.requests[0].headers["authorization"] == "Bearer oauth-token"


@pytest.mark.parametrize("provider", ["codex", "github-copilot"])
@pytest.mark.parametrize("source", ["auth", "api_key", "environment", "http_client"])
async def test_explicit_credentials_override_local_login(
    provider: str, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: FakeService
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not read the GitHub login"))
    client = service.client()
    options: dict[str, Any] = {"api_format": "responses", "http_client": client, "env_prefix": "TEST_PROVIDER"}
    auth = OAuth2Auth({"access_token": "provided-token"})
    if source == "auth":
        options["auth"] = auth
    elif source == "api_key":
        options["api_key"] = "provided-token"
    elif source == "environment":
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", "provided-token")
    else:
        client.auth = auth
    service.reply_events([{"type": "response.output_text.delta", "delta": "hello"}])
    model = republic.get_model(f"{provider}:test", **options)

    async with model.stream("Hi") as stream:
        async for _ in stream:
            pass

    assert stream.text == "hello"
    assert service.requests[0].headers["authorization"] == "Bearer provided-token"
