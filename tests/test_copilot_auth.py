"""Real Authlib device grant, explicit exchange and persistence on offline HTTP."""

import asyncio
import logging
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from republic import generate
from republic.auth import github_copilot as auth
from republic.providers.github_copilot import GitHubCopilot
from tests.copilot_fixtures import Clock, Wire, device, device_payload, inference_payload, login, login_payload, token
from tests.http_fixtures import Transport
from tests.openai_fixtures import completion, request


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(auth, "time", value)
    monkeypatch.setattr(auth, "asyncio", SimpleNamespace(sleep=value.sleep, timeout=asyncio.timeout))
    return value


@pytest.mark.asyncio
async def test_device_login_store_inference_and_explicit_renewal(
    clock: Clock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    begin = Transport([httpx.Response(200, json=device_payload())])
    pending = await auth.start_device_authorization(transport=begin)
    assert begin.closed == 1 and len(begin.requests) == 1
    sent = begin.requests[0]
    assert str(sent.url) == "https://github.com/login/device/code"
    assert parse_qs(sent.content.decode()) == {"client_id": ["01ab8ac9400c4e429b23"], "scope": ["user:email"]}
    assert pending.expires_at == clock.now + 900
    assert "authorization" not in sent.headers
    poll = Transport([
        httpx.Response(200, json=raw)
        for raw in [
            {"error": "authorization_pending"},
            {"error": "slow_down", "interval": 12},
            {"error": "authorization_pending", "interval": 3},
            login_payload(),
        ]
    ])
    github = await auth.wait_for_token(pending, timeout=100, transport=poll)
    assert clock.sleeps == [5, 5, 12, 12]
    assert github.expires_at is None and not github.is_expired()
    assert github.refresh_token is None  # No fabricated expiry or refresh grant.
    assert len(poll.requests) == 4 and poll.closed == 1
    for sent in poll.requests:
        assert str(sent.url) == "https://github.com/login/oauth/access_token"
        assert "authorization" not in sent.headers
        assert parse_qs(sent.content.decode()) == {
            "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
            "device_code": [pending.device_code],
            "client_id": [pending.client_id],
        }
    path = tmp_path / "explicit.json"
    auth.write_token(path, github)
    assert path.stat().st_mode & 0o777 == 0o600 and auth.read_token(path) == github
    exchange = Transport([httpx.Response(200, json=inference_payload(expires_at=clock.now + 1800))])
    copilot = await auth.exchange_copilot_token(github, transport=exchange)
    assert copilot.refresh_at == clock.now + 1500
    assert len(exchange.requests) == 1 and exchange.closed == 1
    sent = exchange.requests[0]
    assert sent.method == "GET" and str(sent.url) == "https://api.github.com/copilot_internal/v2/token"
    assert sent.headers["authorization"] == f"token {github.access_token}"
    assert sent.headers["x-github-api-version"] == "2025-04-01"
    auth.write_token(path, copilot)
    assert auth.read_token(path) == copilot
    async with Wire([httpx.Response(200, json=completion()), httpx.Response(200, json=completion())], copilot) as wire:
        await generate(wire.provider, request())
        renew = Transport([
            httpx.Response(200, json=inference_payload(**{"token": "private-renewed"}, expires_at=clock.now + 2000))
        ])
        updated = await auth.exchange_copilot_token(github, transport=renew)
        async with GitHubCopilot(updated, integration_id="fixture-integration", client=wire.client) as refreshed:
            await generate(refreshed, request())
        assert len(wire.requests) == 2 and len(renew.requests) == 1
        assert wire.requests[0].headers["authorization"] == "Bearer private-copilot"
        assert wire.requests[1].headers["authorization"] == "Bearer private-renewed"
        assert not wire.client.is_closed()
    for secret in (pending.device_code, pending.user_code, github.access_token, copilot.token, updated.token):
        assert secret not in caplog.text + repr(pending) + repr(github) + repr(copilot)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["authorization_pending", "slow_down"])
async def test_default_interval_and_slow_down_increment(clock: Clock, error: str) -> None:
    raw = device_payload()
    del raw["interval"]
    pending = await auth.start_device_authorization(transport=Transport([httpx.Response(200, json=raw)]))
    transport = Transport([httpx.Response(400, json={"error": error}), httpx.Response(200, json=login_payload())])
    await auth.wait_for_token(pending, timeout=60, transport=transport)
    assert clock.sleeps == [5, 10 if error == "slow_down" else 5]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code",
    [
        ("access_denied", "access_denied"),
        ("expired_token", "expired_token"),
        ("incorrect_device_code", "incorrect_device_code"),
        ("private-unknown", "oauth_error"),
    ],
)
async def test_poll_stops_on_denial_expiry_or_unknown(clock: Clock, error: str, code: str) -> None:
    transport = Transport([httpx.Response(400, json={"error": error, "error_description": "private-server"})])
    with pytest.raises(auth.CopilotAuthError) as caught:
        await auth.wait_for_token(device(expires_at=clock.now + 600), timeout=60, transport=transport)
    assert caught.value.code == code
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "private-" not in "".join(traceback.format_exception(caught.value))
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,expires,code,sleeps,requests",
    [
        (12, 600, "deadline_exceeded", [5, 5, 2], 2),
        (60, 7, "expired_token", [5, 2], 1),
        (60, -1, "expired_token", [], 0),
    ],
)
async def test_poll_obeys_both_deadlines(
    clock: Clock, budget: int, expires: int, code: str, sleeps: list[int], requests: int
) -> None:
    transport = Transport([httpx.Response(200, json={"error": "authorization_pending"}) for _ in range(requests)])
    with pytest.raises(auth.CopilotAuthError, match=code):
        await auth.wait_for_token(device(expires_at=clock.now + expires), timeout=budget, transport=transport)
    assert clock.sleeps == sleeps and len(transport.requests) == requests and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["sleep", "request"])
async def test_poll_cancel_closes_client(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    waiting = asyncio.Event()

    async def block(*args: Any, **kwargs: Any) -> None:
        waiting.set()
        await asyncio.Event().wait()

    transport = Transport([])
    if phase == "sleep":
        monkeypatch.setattr(auth, "asyncio", SimpleNamespace(sleep=block, timeout=asyncio.timeout))
    else:
        monkeypatch.setattr(auth, "asyncio", SimpleNamespace(sleep=lambda _: asyncio.sleep(0), timeout=asyncio.timeout))
        monkeypatch.setattr(transport, "handle_async_request", block)
    task = asyncio.create_task(auth.wait_for_token(device(), timeout=60, transport=transport))
    await asyncio.wait_for(waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed == 1


@pytest.mark.asyncio
async def test_poll_timeout_inflight_is_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "asyncio", SimpleNamespace(sleep=lambda _: asyncio.sleep(0), timeout=asyncio.timeout))

    async def blocked(_: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError

    transport = Transport([])
    monkeypatch.setattr(transport, "handle_async_request", blocked)
    with pytest.raises(auth.CopilotAuthError, match="deadline_exceeded"):
        await auth.wait_for_token(device(), timeout=0.02, transport=transport)
    assert transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,code",
    [
        ({}, "invalid_token_type"),
        ({"token_type": "bearer"}, "invalid_response"),
        (login_payload(access_token=""), "invalid_token"),
        (login_payload(**{"token_type": "basic"}), "invalid_token_type"),
        (login_payload(expires_in="3600"), "invalid_expiry"),
        (login_payload(expires_in=None), "invalid_expiry"),
        (login_payload(expires_at=True), "invalid_expiry"),
        (login_payload(expires_in=0), "invalid_expiry"),
        (login_payload(refresh_token_expires_in=-1), "invalid_expiry"),
        ({"error": "slow_down", "interval": 0}, "invalid_interval"),
    ],
)
async def test_invalid_poll_data_never_keeps_waiting(clock: Clock, raw: Any, code: str) -> None:
    transport = Transport([httpx.Response(200, json=raw)])
    with pytest.raises(auth.CopilotAuthError, match=code):
        await auth.wait_for_token(device(expires_at=clock.now + 600), timeout=60, transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"expires_in": 0},
        {"expires_in": "900"},
        {"interval": False},
        {"device_code": None},
        {"user_code": "a\nb"},
        {"verification_uri": "https://attacker.test"},
    ],
)
async def test_device_response_validated(extra: dict[str, Any]) -> None:
    transport = Transport([httpx.Response(200, json=device_payload(**extra))])
    with pytest.raises(auth.CopilotAuthError):
        await auth.start_device_authorization(transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
async def test_explicit_client_id_is_carried_into_token_request(clock: Clock) -> None:
    start = Transport([httpx.Response(200, json=device_payload())])
    pending = await auth.start_device_authorization(client_id="caller-app", transport=start)
    poll = Transport([httpx.Response(200, json=login_payload())])
    result = await auth.wait_for_token(pending, timeout=60, transport=poll)
    assert result.client_id == "caller-app"
    assert parse_qs(poll.requests[0].content.decode())["client_id"] == ["caller-app"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rotate", [False, True])
async def test_oauth_refresh_only_for_a_real_refresh_token(clock: Clock, rotate: bool) -> None:
    original = login(**{"refresh_token": "private-refresh"}, refresh_expires_at=clock.now + 1000)
    raw = login_payload(**{"access_token": "private-new"}, expires_in=500)
    if rotate:
        raw.update(**{"refresh_token": "private-rotated"}, refresh_token_expires_in=2000)
    transport = Transport([httpx.Response(200, json=raw)])
    result = await auth.refresh_github_token(original, transport=transport)
    assert result.expires_at == clock.now + 500
    assert result.refresh_token == ("private-rotated" if rotate else "private-refresh")
    assert result.refresh_expires_at == clock.now + (2000 if rotate else 1000)
    assert parse_qs(transport.requests[0].content.decode()) == {
        "grant_type": ["refresh_token"],
        "refresh_token": [original.refresh_token],
        "client_id": [original.client_id],
    }
    assert "authorization" not in transport.requests[0].headers
    assert transport.closed == 1 and len(transport.requests) == 1


@pytest.mark.asyncio
async def test_absent_expired_and_rejected_refresh() -> None:
    transport = Transport([])
    with pytest.raises(auth.CopilotAuthError, match="no_refresh_token"):
        await auth.refresh_github_token(login(), transport=transport)
    assert not transport.requests
    transport = Transport([httpx.Response(400, json={"error": "invalid_grant", "error_description": "private-secret"})])
    with pytest.raises(auth.CopilotAuthError, match="refresh_rejected") as caught:
        await auth.refresh_github_token(
            login(**{"refresh_token": "private"}, refresh_expires_at=1), transport=transport
        )
    assert caught.value.__context__ is None and len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://models.github.ai/inference",
        "https://api.githubcopilot.com.attacker.test",
        "http://api.githubcopilot.com",
        "https://user@api.githubcopilot.com",
        "https://api.githubcopilot.com:443",
        "https://api.githubcopilot.com/path",
        "https://api.githubcopilot.com?x=y",
        [],
    ],
)
async def test_untrusted_exchange_endpoint_rejected(endpoint: Any) -> None:
    transport = Transport([httpx.Response(200, json=inference_payload(endpoints={"api": endpoint}))])
    with pytest.raises(auth.CopilotAuthError, match="untrusted_endpoint"):
        await auth.exchange_copilot_token(login(), transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
async def test_exchange_default_origin_expiry_and_no_profile_collection() -> None:
    raw = inference_payload()
    del raw["endpoints"]
    del raw["refresh_in"]
    transport = Transport([httpx.Response(200, json=raw)])
    result = await auth.exchange_copilot_token(login(), transport=transport)
    assert result.api_endpoint == "https://api.githubcopilot.com" and result.refresh_at is None
    assert result.expires_at == raw["expires_at"] and len(transport.requests) == 1
    renewed = await auth.exchange_copilot_token(
        login(expires_at=1), transport=Transport([httpx.Response(200, json=raw)])
    )
    assert renewed.token == result.token


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["device", "poll", "exchange", "refresh"])
@pytest.mark.parametrize(
    "reply,code",
    [
        (httpx.Response(401, json={"message": "private-secret"}), "unauthorized"),
        (httpx.Response(403, json={"error_details": {"message": "private-secret"}}), "forbidden"),
        (httpx.Response(429, json={"error": "private-secret"}), "rate_limit"),
        (httpx.Response(307, headers={"location": "https://attacker.test"}), "http_error"),
        (httpx.ConnectError("private-secret"), "transport_error"),
    ],
)
async def test_auth_http_failures_no_retry_redirect_secret_or_entitlement_fallback(
    clock: Clock, phase: str, reply: httpx.Response | Exception, code: str
) -> None:
    transport = Transport([reply])
    credentials = login(**{"refresh_token": "private-secret"})
    with pytest.raises(auth.CopilotAuthError) as caught:
        if phase == "device":
            await auth.start_device_authorization(transport=transport)
        elif phase == "poll":
            await auth.wait_for_token(device(expires_at=clock.now + 600), timeout=60, transport=transport)
        elif phase == "refresh":
            await auth.refresh_github_token(credentials, transport=transport)
        else:
            await auth.exchange_copilot_token(login(), transport=transport)
    assert caught.value.code == code
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert "private-secret" not in "".join(traceback.format_exception(caught.value))
    assert len(transport.requests) == 1 and transport.closed == 1


def test_token_files_failed_writes_are_atomic_and_errors_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "chosen.json"
    original = login()
    auth.write_token(path, original)

    def fail(*args: Any) -> None:
        raise OSError("private-secret")

    monkeypatch.setattr("republic.auth._files.os.replace", fail)
    with pytest.raises(auth.CopilotAuthError, match="credential_write_failed") as caught:
        auth.write_token(path, token())
    assert auth.read_token(path) == original
    assert list(tmp_path.iterdir()) == [path]
    assert caught.value.__context__ is None and "private-secret" not in str(caught.value)
    with pytest.raises(auth.CopilotAuthError, match="credential_read_failed"):
        auth.read_token(tmp_path / "missing")
    path.write_text('{"kind":"unknown", "data": "private-secret"}')
    with pytest.raises(auth.CopilotAuthError, match="invalid_token_file"):
        auth.read_token(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"token": ""},
        {"token": None},
        {"expires_at": True},
        {"expires_at": "2000000000"},
        {"expires_at": None},
        {"refresh_in": 0},
        {"endpoints": []},
    ],
)
async def test_exchange_response_rejects_malformed_tokens(extra: dict[str, Any]) -> None:
    transport = Transport([httpx.Response(200, json=inference_payload(**extra))])
    with pytest.raises(auth.CopilotAuthError):
        await auth.exchange_copilot_token(login(), transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), 10**400, -1])
def test_expiry_checks_are_finite_and_never_guess(expiry: Any) -> None:
    with pytest.raises(auth.CopilotAuthError, match="invalid_expiry"):
        token(expires_at=expiry)
    with pytest.raises(auth.CopilotAuthError, match="invalid_expiry"):
        login(expires_at=expiry)


@pytest.mark.asyncio
async def test_real_optional_login_expiry_and_refresh_fields(clock: Clock) -> None:
    raw = login_payload(**{"expires_in": 30, "refresh_token": "private-refresh", "refresh_token_expires_in": 300})
    result = await auth.wait_for_token(
        device(expires_at=clock.now + 600), timeout=60, transport=Transport([httpx.Response(200, json=raw)])
    )
    assert result.expires_at == clock.now + 30 and result.refresh_expires_at == clock.now + 300
    clock.now += 31
    assert result.is_expired()
