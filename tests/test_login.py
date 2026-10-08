from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import republic
from republic.providers import CodexAuth, GitHubCLIAuth
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
