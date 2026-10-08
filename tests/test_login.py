# ruff: noqa: S105 - all credentials in this module are test data
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import traceback
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx2
import pytest
from authlib.integrations.httpx_client import AsyncOAuth2Client

import republic
from republic.providers import CodexAuth, CopilotAuth, GitHubCLIAuth, github
from tests.conftest import FakeService


@pytest.mark.parametrize("device_auth", [False, True])
async def test_codex_login_returns_file_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, service: FakeService, device_auth: bool
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    config.write_text('cli_auth_credentials_store = "keyring"\n')

    async def run(*command: str) -> asyncio.subprocess.Process:
        assert command == (
            "/tools/codex",
            "login",
            "--config",
            'cli_auth_credentials_store="file"',
            *(("--device-auth",) if device_auth else ()),
        )
        (tmp_path / "auth.json").write_text(
            json.dumps({
                "tokens": {"access_token": "codex-login-token", "account_id": "account-1"},
            })
        )
        return process

    process = AsyncMock(spec=asyncio.subprocess.Process)
    process.wait.return_value = 0
    monkeypatch.setattr(asyncio, "create_subprocess_exec", run)
    auth = await CodexAuth.login(executable="/tools/codex", device_auth=device_auth)
    service.reply_events([{"type": "response.output_text.delta", "delta": "hello"}])
    response = await republic.get_model("codex:test", auth=auth, http_client=service.client()).chat("Hi")

    assert response.text == "hello"
    assert service.requests[0].headers["authorization"] == "Bearer codex-login-token"
    assert service.requests[0].headers["chatgpt-account-id"] == "account-1"
    assert config.read_text() == 'cli_auth_credentials_store = "keyring"\n'


async def test_github_login_keeps_cli_credential_lookup(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
    process = AsyncMock(spec=asyncio.subprocess.Process)
    process.wait.return_value = 0
    start = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    auth = await GitHubCLIAuth.login(hostname="github.example.com", executable="/tools/gh")
    start.assert_awaited_once_with("/tools/gh", "auth", "login", "--hostname", "github.example.com", "--web")

    def read_token(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ["/tools/gh", "auth", "token", "--hostname", "github.example.com"]
        return subprocess.CompletedProcess(command, 0, stdout="github-login-token\n")

    monkeypatch.setattr(subprocess, "run", read_token)
    service.reply_json({"choices": [{"message": {"content": "hello"}}]})
    response = await republic.get_model("github-copilot:test", auth=auth, http_client=service.client()).chat("Hi")

    assert response.text == "hello"
    assert len(service.requests) == 1
    assert service.requests[0].headers["authorization"] == "Bearer github-login-token"


@pytest.mark.parametrize("auth_type", [CodexAuth, GitHubCLIAuth])
async def test_cli_login_reports_missing_executable(auth_type: type[CodexAuth | GitHubCLIAuth], tmp_path: Path) -> None:
    with pytest.raises(republic.AuthenticationError, match="Cannot start"):
        await auth_type.login(executable=str(tmp_path / "missing-cli"))


@pytest.mark.parametrize("auth_type", [CodexAuth, GitHubCLIAuth])
async def test_cli_login_reports_failure(
    auth_type: type[CodexAuth | GitHubCLIAuth], monkeypatch: pytest.MonkeyPatch
) -> None:
    process = AsyncMock(spec=asyncio.subprocess.Process)
    process.wait.return_value = 2
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    with pytest.raises(republic.AuthenticationError, match="exit 2"):
        await auth_type.login()


@pytest.mark.parametrize("auth_type", [CodexAuth, GitHubCLIAuth])
async def test_cancelling_cli_login_reaps_child(
    auth_type: type[CodexAuth | GitHubCLIAuth], monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes = []

    async def run(*command: str) -> asyncio.subprocess.Process:
        process = await spawn(sys.executable, "-c", "import time; time.sleep(60)")
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", run)
    task = asyncio.create_task(auth_type.login())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert processes[0].returncode is not None
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()


@pytest.fixture
def device_service(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> FakeService:
    monkeypatch.setattr(
        github, "AsyncOAuth2Client", partial(AsyncOAuth2Client, transport=httpx2.MockTransport(service._handle))
    )
    return service


def device_response(**overrides: object) -> dict[str, object]:
    return {
        "device_code": "private-device-code",
        "user_code": "USER-CODE",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        **overrides,
    }


@pytest.mark.parametrize("custom_display", [False, True])
async def test_copilot_login_can_be_used_and_restored(
    device_service: FakeService,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    custom_display: bool,
) -> None:
    service = device_service
    service.reply_json(device_response())
    service.reply_json({"error": "authorization_pending"})
    service.reply_json({"error": "slow_down", "interval": 10})
    service.reply_json({"error": "slow_down", "interval": 15})
    service.reply_json({"access_token": "plugin-login-token", "token_type": "bearer"})
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    display = AsyncMock()

    auth = await CopilotAuth.login(on_authorize=display if custom_display else None)

    assert [call.args[0] for call in sleep.await_args_list] == [5, 5, 10, 15]
    output = capsys.readouterr()
    if custom_display:
        display.assert_awaited_once_with("https://github.com/login/device", "USER-CODE")
        assert output.err == ""
    else:
        assert "https://github.com/login/device" in output.err
        assert "USER-CODE" in output.err
    assert "private-device-code" not in output.err + output.out
    assert "plugin-login-token" not in output.err + output.out
    device_request, *polls = service.requests
    assert device_request.method == "POST"
    assert str(device_request.url) == "https://github.com/login/device/code"
    assert parse_qs(device_request.content.decode()) == {"client_id": ["Iv1.b507a08c87ecfe98"], "scope": ["read:user"]}
    for request in polls:
        assert request.method == "POST"
        assert str(request.url) == "https://github.com/login/oauth/access_token"
        assert "authorization" not in request.headers
        assert request.headers["accept"] == "application/json"
        assert parse_qs(request.content.decode()) == {
            "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
            "device_code": ["private-device-code"],
            "client_id": ["Iv1.b507a08c87ecfe98"],
        }
    assert auth.github_token == "plugin-login-token"
    for selected in [auth, CopilotAuth(auth.github_token)]:
        service.reply_json({
            "token": "inference-token",
            "refresh_in": 300,
            "endpoints": {"api": "https://api.githubcopilot.com"},
        })
        service.reply_json({"choices": [{"message": {"content": "hello"}}]})
        response = await republic.get_model("github-copilot:test", auth=selected, http_client=service.client()).chat(
            "Hi"
        )
        assert response.text == "hello"
        assert service.requests[-2].headers["authorization"] == "token plugin-login-token"
        assert service.requests[-1].headers["authorization"] == "Bearer inference-token"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
        ("token_expired", "expired"),
        ("invalid_client", "rejected"),
    ],
)
async def test_copilot_login_reports_oauth_failure(
    device_service: FakeService, monkeypatch: pytest.MonkeyPatch, error: str, message: str
) -> None:
    device_service.reply_json(device_response(interval=1))
    device_service.reply_json({"error": error, "error_description": "private-device-code"}, status_code=400)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(republic.AuthenticationError, match=message) as caught:
        await CopilotAuth.login(on_authorize=AsyncMock())

    assert len(device_service.requests) == 2
    assert "private-device-code" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("body", [{}, {"access_token": ""}, {"access_token": 42}])
async def test_copilot_login_requires_an_access_token(
    device_service: FakeService, monkeypatch: pytest.MonkeyPatch, body: dict[str, object]
) -> None:
    device_service.reply_json(device_response())
    device_service.reply_json(body)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with pytest.raises(republic.AuthenticationError, match="no token"):
        await CopilotAuth.login(on_authorize=AsyncMock())


@pytest.mark.parametrize(
    "overrides", [{"expires_in": 0}, {"interval": 0}, {"verification_uri": "not-a-url"}, {"device_code": ""}]
)
async def test_copilot_login_rejects_invalid_device_response(
    device_service: FakeService, overrides: dict[str, object]
) -> None:
    device_service.reply_json(device_response(**overrides))
    display = AsyncMock()
    with pytest.raises(republic.AuthenticationError, match="Cannot start") as caught:
        await CopilotAuth.login(on_authorize=display)
    display.assert_not_awaited()
    assert "private-device-code" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("during_display", [False, True])
async def test_copilot_login_stops_at_device_expiry(device_service: FakeService, during_display: bool) -> None:
    device_service.reply_json(device_response(expires_in=1))

    async def display(url: str, code: str) -> None:
        if during_display:
            await asyncio.Event().wait()

    with pytest.raises(republic.AuthenticationError, match="expired"):
        await asyncio.wait_for(CopilotAuth.login(on_authorize=display), timeout=5)
    assert len(device_service.requests) == 1


@pytest.mark.parametrize("stage", ["device", "token"])
@pytest.mark.parametrize("failure", ["http", "json", "null"])
async def test_copilot_login_reports_invalid_server_responses(
    device_service: FakeService, monkeypatch: pytest.MonkeyPatch, stage: str, failure: str
) -> None:
    if stage == "token":
        device_service.reply_json(device_response())
    if failure == "http":
        device_service.reply_json({"detail": "private-device-code"}, status_code=502)
    elif failure == "json":
        device_service.reply_bytes(b"private-device-code", content_type="text/plain")
    else:
        device_service.reply_json(None)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(republic.AuthenticationError, match="Cannot") as caught:
        await CopilotAuth.login(on_authorize=AsyncMock())

    assert len(device_service.requests) == (1 if stage == "device" else 2)
    assert "private-device-code" not in "".join(traceback.format_exception(caught.value))


async def test_copilot_login_can_be_cancelled(device_service: FakeService) -> None:
    device_service.reply_json(device_response())
    displayed = asyncio.Event()

    async def display(url: str, code: str) -> None:
        displayed.set()

    task = asyncio.create_task(CopilotAuth.login(on_authorize=display))
    try:
        await asyncio.wait_for(displayed.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(device_service.requests) == 1


async def test_copilot_login_preserves_callback_errors(device_service: FakeService) -> None:
    device_service.reply_json(device_response())
    error = TimeoutError("UI timeout")
    with pytest.raises(TimeoutError) as caught:
        await CopilotAuth.login(on_authorize=AsyncMock(side_effect=error))
    assert caught.value is error
    assert len(device_service.requests) == 1
