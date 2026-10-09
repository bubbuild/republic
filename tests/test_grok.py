from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx2
import pytest
from authlib.integrations.httpx_client import OAuth2Client

import republic
from republic.providers import GrokAuth, grok
from tests.conftest import FakeService

CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = f"https://auth.x.ai::{CLIENT_ID}"


def credentials(path: Path, *, key: str = "file-token", expires_at: float | None = None) -> dict:
    data = {
        SCOPE: {
            "key": key,
            "refresh_token": "refresh-one",
            "expires_at": datetime.fromtimestamp(
                expires_at if expires_at is not None else time.time() + 3600, UTC
            ).isoformat(),
            "oidc_issuer": "https://auth.x.ai",
            "oidc_client_id": CLIENT_ID,
            "principal_type": "Team",
            "principal_id": "team-1",
            "user_id": "user-1",
            "coding_data_retention_opt_out": True,
        },
        "other-provider": {"key": "untouched"},
    }
    path.write_text(json.dumps(data))
    return data


@pytest.fixture
def auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.delenv("GROK_AUTH_PATH", raising=False)
    monkeypatch.delenv("REPUBLIC_GROK_API_KEY", raising=False)
    path = tmp_path / "auth.json"
    credentials(path)
    return path


@pytest.fixture
def refresh_service(monkeypatch: pytest.MonkeyPatch) -> FakeService:
    service = FakeService()
    monkeypatch.setattr(grok, "OAuth2Client", partial(OAuth2Client, transport=httpx2.MockTransport(service._handle)))
    return service


@pytest.mark.parametrize("api_format", ["chat", "responses"])
async def test_local_login_is_reread_for_chat_and_stream(
    auth_file: Path, service: FakeService, api_format: str
) -> None:
    original = auth_file.read_bytes()
    if api_format == "chat":
        service.reply_json({"choices": [{"message": {"content": "hello"}}]})
        event = {"choices": [{"delta": {"content": "streamed"}}]}
        path = "/v1/chat/completions"
    else:
        service.reply_json({"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]})
        event = {"type": "response.output_text.delta", "delta": "streamed"}
        path = "/v1/responses"
    model = republic.get_model("grok:test", api_format=api_format, http_client=service.client())
    response = await model.chat("Hi")
    assert response.text == "hello"
    assert auth_file.read_bytes() == original
    credentials(auth_file, key="new-login")
    service.reply_events([event, "[DONE]"])
    async with model.stream("Again") as stream:
        async for _ in stream:
            pass
    assert stream.text == "streamed"
    assert [str(r.url) for r in service.requests] == [f"https://api.x.ai{path}"] * 2
    assert [r.headers["authorization"] for r in service.requests] == ["Bearer file-token", "Bearer new-login"]


async def test_file_refresh_preserves_cli_store_and_rotates_tokens(
    auth_file: Path, refresh_service: FakeService, service: FakeService
) -> None:
    original = credentials(auth_file, expires_at=0)
    refresh_service.reply_json({"access_token": "renewed", "refresh_token": "refresh-two", "expires_in": 3600})
    service.reply_json({"output": []})
    model = republic.get_model("grok:test", http_client=service.client())
    await model.chat("Hi")
    request = refresh_service.requests[0]
    assert request.method == "POST"
    assert request.url == "https://auth.x.ai/oauth2/token"
    assert "authorization" not in request.headers
    assert parse_qs(request.content.decode()) == {
        "grant_type": ["refresh_token"],
        "refresh_token": ["refresh-one"],
        "client_id": [CLIENT_ID],
        "principal_type": ["Team"],
        "principal_id": ["team-1"],
    }
    assert service.requests[0].headers["authorization"] == "Bearer renewed"
    saved = json.loads(auth_file.read_text())
    assert saved["other-provider"] == original["other-provider"]
    for key in (
        "user_id",
        "oidc_issuer",
        "oidc_client_id",
        "principal_type",
        "principal_id",
        "coding_data_retention_opt_out",
    ):
        assert saved[SCOPE][key] == original[SCOPE][key]
    if os.name == "posix":
        assert auth_file.stat().st_mode & 0o777 == 0o600
    saved[SCOPE]["expires_at"] = "1970-01-01T00:00:00+00:00"
    auth_file.write_text(json.dumps(saved))
    refresh_service.reply_json({"access_token": "renewed-again", "expires_in": 3600})
    service.reply_json({"output": []})
    restored = republic.get_model("grok:test", http_client=service.client())
    await restored.chat("Again")
    assert parse_qs(refresh_service.requests[1].content.decode())["refresh_token"] == ["refresh-two"]
    assert service.requests[1].headers["authorization"] == "Bearer renewed-again"
    # A non-rotating refresh still leaves a restorable credential for the CLI.
    assert json.loads(auth_file.read_text())[SCOPE]["refresh_token"] == "refresh-two"  # noqa: S105 - test credential


async def test_separate_auth_instances_share_one_file_refresh(auth_file: Path, refresh_service: FakeService) -> None:
    credentials(auth_file, expires_at=0)
    first, second = GrokAuth.from_file(), GrokAuth.from_file()
    refresh_service.reply_json({"access_token": "renewed", "refresh_token": "rotated", "expires_in": 3600})
    services = [FakeService(), FakeService()]
    for service in services:
        service.reply_json({"output": []})
    await asyncio.gather(
        *(
            republic.get_model("grok:test", auth=auth, http_client=service.client()).chat("Hi")
            for auth, service in zip([first, second], services, strict=True)
        )
    )
    assert len(refresh_service.requests) == 1
    assert all(service.requests[0].headers["authorization"] == "Bearer renewed" for service in services)


async def test_explicit_tokens_refresh_without_touching_cli_store(
    auth_file: Path, refresh_service: FakeService, service: FakeService
) -> None:
    original = auth_file.read_bytes()
    auth = GrokAuth({"access_token": "old", "refresh_token": "refresh-one", "expires_at": 0})
    refresh_service.reply_json({"access_token": "renewed", "expires_in": 3600})
    for selected in [auth, auth]:
        service.reply_json({"output": []})
        await republic.get_model("grok:test", auth=selected, http_client=service.client()).chat("Hi")
    service.reply_json({"output": []})
    await republic.get_model("grok:test", auth=GrokAuth(auth.token), http_client=service.client()).chat("Restored")
    assert len(refresh_service.requests) == 1
    assert all(r.headers["authorization"] == "Bearer renewed" for r in service.requests)
    assert auth_file.read_bytes() == original


@pytest.mark.parametrize("failure", ["oauth", "json", "missing", "expiry", "network"])
async def test_failed_refresh_does_not_replace_credentials_or_send_inference(
    auth_file: Path, refresh_service: FakeService, service: FakeService, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    credentials(auth_file, expires_at=0)
    original = auth_file.read_bytes()
    if failure == "oauth":
        refresh_service.reply_json(
            {"error": "invalid_grant", "error_description": "private-refresh-token"}, status_code=400
        )
    elif failure == "json":
        refresh_service.reply_bytes(b"private-refresh-token", content_type="text/plain")
    elif failure == "network":

        def handle(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("private-refresh-token", request=request)

        monkeypatch.setattr(grok, "OAuth2Client", partial(OAuth2Client, transport=httpx2.MockTransport(handle)))
    else:
        refresh_service.reply_json({} if failure == "missing" else {"access_token": "new", "expires_in": "invalid"})
    model = republic.get_model("grok:test", http_client=service.client())
    with pytest.raises(republic.errors.AuthenticationError) as caught:
        await model.chat("Hi")
    assert "private-refresh-token" not in "".join(traceback.format_exception(caught.value))
    assert auth_file.read_bytes() == original
    assert service.requests == []


@pytest.mark.parametrize("source", ["argument", "environment"])
async def test_alternative_credential_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: FakeService, source: str
) -> None:
    path = tmp_path / "custom.json"
    credentials(path)
    if source == "environment":
        monkeypatch.setenv("GROK_AUTH_PATH", str(path))
    auth = GrokAuth.from_file(path if source == "argument" else None)
    service.reply_json({"output": []})
    await republic.get_model("grok:test", auth=auth, http_client=service.client()).chat("Hi")
    assert service.requests[0].headers["authorization"] == "Bearer file-token"


@pytest.mark.parametrize("device_auth", [False, True])
async def test_login_returns_usable_cli_credentials(
    auth_file: Path, monkeypatch: pytest.MonkeyPatch, service: FakeService, device_auth: bool
) -> None:
    async def run(*command: str) -> asyncio.subprocess.Process:
        assert command == ("/tools/grok", "login", *(("--device-auth",) if device_auth else ()))
        credentials(auth_file, key="login-token")
        process = AsyncMock(spec=asyncio.subprocess.Process)
        process.wait.return_value = 0
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", run)
    auth = await GrokAuth.login(executable="/tools/grok", device_auth=device_auth)
    service.reply_json({"output": []})
    await republic.get_model("grok:test", auth=auth, http_client=service.client()).chat("Hi")
    assert service.requests[0].headers["authorization"] == "Bearer login-token"


@pytest.mark.parametrize("body", ["bad-json", "null", "{}", '{"https://auth.x.ai::other-client": {"key": "secret"}}'])
def test_invalid_cli_store_reports_login_error(tmp_path: Path, body: str) -> None:
    path = tmp_path / "auth.json"
    path.write_text(body)
    with pytest.raises(republic.errors.AuthenticationError, match="grok login"):
        GrokAuth.from_file(path)


@pytest.mark.parametrize("token", [{}, {"access_token": "valid"}, {"access_token": "valid", "expires_at": "bad"}])
def test_invalid_explicit_credentials_are_rejected(token: dict) -> None:
    with pytest.raises(republic.errors.AuthenticationError):
        GrokAuth(token)
