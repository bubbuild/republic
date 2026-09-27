"""Real Authlib device/refresh requests on synthetic HTTP; no live credentials."""

import asyncio
import json
import logging
import traceback
from base64 import urlsafe_b64encode
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from republic import generate
from republic.auth import grok as auth
from republic.providers.grok import GrokOAuth
from tests.grok_fixtures import Wire, device, device_payload, token_payload, tokens
from tests.http_fixtures import Bytes, Clock, Transport, streaming
from tests.openai_fixtures import request, sse
from tests.responses_fixtures import terminal


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(auth, "time", value)
    monkeypatch.setattr(auth, "asyncio", SimpleNamespace(sleep=value.sleep, timeout=asyncio.timeout))
    return value


@pytest.mark.asyncio
async def test_device_save_inference_refresh_and_inference(
    clock: Clock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    begin = Transport([httpx.Response(200, json=device_payload())])
    pending = await auth.start_device_authorization(client_version="1.0.41", transport=begin)
    assert begin.closed == 1 and len(begin.requests) == 1
    sent = begin.requests[0]
    assert str(sent.url) == "https://auth.x.ai/oauth2/device/code"
    assert parse_qs(sent.content.decode()) == {
        "client_id": ["b1a00492-073a-47ea-816f-4c329264a828"],
        "scope": ["offline_access grok-cli:access api:access"],
        "referrer": ["republic"],
    }
    assert sent.headers["x-grok-client-version"] == "1.0.41"
    assert sent.headers["x-grok-client-surface"] == "headless" and "authorization" not in sent.headers
    assert pending.expires_at == clock.now + 600
    poll = Transport([
        httpx.Response(400, json={"error": "authorization_pending"}),
        httpx.Response(400, json={"error": "slow_down", "interval": 12}),
        httpx.Response(400, json={"error": "authorization_pending", "interval": 3}),
        httpx.Response(200, json=token_payload(expires_in=60)),
    ])
    credentials = await auth.wait_for_tokens(pending, timeout=100, transport=poll)
    assert clock.sleeps == [5, 5, 12, 12]
    assert credentials.expires_at == clock.now + 60 and not credentials.is_expired()
    assert len(poll.requests) == 4 and poll.closed == 1
    for sent in poll.requests:
        assert str(sent.url) == "https://auth.x.ai/oauth2/token"
        assert sent.headers["x-grok-client-version"] == "1.0.41"
        assert "authorization" not in sent.headers
        assert parse_qs(sent.content.decode()) == {
            "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
            "device_code": [pending.device_code],
            "client_id": ["b1a00492-073a-47ea-816f-4c329264a828"],
        }
    path = tmp_path / "chosen.json"
    auth.write_tokens(path, credentials)
    assert path.stat().st_mode & 0o777 == 0o600 and auth.read_tokens(path) == credentials
    replies = [streaming(Bytes([sse(terminal())])) for _ in range(2)]
    async with Wire(replies, auth.read_tokens(path)) as wire:
        await generate(wire.provider, request())
        clock.now += 61
        assert credentials.is_expired()
        renewal = Transport([
            httpx.Response(200, json=token_payload(**{"access_token": "private-new"}, expires_in=300))
        ])
        updated = await auth.refresh_tokens(credentials, transport=renewal)
        assert updated.expires_at == clock.now + 300
        assert parse_qs(renewal.requests[0].content.decode()) == {
            "grant_type": ["refresh_token"],
            "refresh_token": [credentials.refresh_token],
            "client_id": ["b1a00492-073a-47ea-816f-4c329264a828"],
        }
        assert len(renewal.requests) == 1 and renewal.closed == 1
        async with GrokOAuth(updated, client_version="1.0.41", client=wire.client) as refreshed:
            await generate(refreshed, request())
        assert len(wire.requests) == 2 and not wire.client.is_closed()
        assert wire.requests[0].headers["authorization"] == f"Bearer {credentials.access_token}"
        assert wire.requests[1].headers["authorization"] == f"Bearer {updated.access_token}"
    for secret in (pending.device_code, pending.user_code, credentials.access_token, credentials.refresh_token):
        assert secret is not None
        assert secret not in caplog.text + repr(pending) + repr(credentials)


@pytest.mark.asyncio
@pytest.mark.parametrize("rotate", [False, True])
async def test_refresh_retains_or_rotates_tokens_and_unverified_principal(clock: Clock, rotate: bool) -> None:
    previous = tokens(principal_type="Team", principal_id="old-team", expires_at=clock.now + 10)
    claims = {"principalType": "Team", "principalId": "new-team", "exp": 1}
    encoded = urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    raw = token_payload(access_token=f"unverified.{encoded}.unsigned", expires_in=300)
    if rotate:
        raw.update({"refresh_token": "private-rotated"})
    else:
        del raw["refresh_token"]
    transport = Transport([httpx.Response(200, json=raw)])
    updated = await auth.refresh_tokens(previous, transport=transport)
    assert updated.refresh_token == ("private-rotated" if rotate else previous.refresh_token)
    assert updated.principal_id == "new-team" and updated.principal_type == "Team"
    assert updated.expires_at == clock.now + 300  # The unverified JWT exp is not used.
    form = parse_qs(transport.requests[0].content.decode())
    assert form["principal_type"] == ["Team"] and form["principal_id"] == ["old-team"]
    assert "authorization" not in transport.requests[0].headers
    second = token_payload()
    del second["refresh_token"]
    next_tokens = await auth.refresh_tokens(updated, transport=Transport([httpx.Response(200, json=second)]))
    assert next_tokens.expires_at is None and not next_tokens.is_expired()
    assert next_tokens.principal_id == updated.principal_id


@pytest.mark.asyncio
async def test_missing_expiry_refresh_and_default_interval_are_not_invented(clock: Clock) -> None:
    raw = device_payload()
    del raw["interval"]
    pending = await auth.start_device_authorization(
        client_version="1.0.41", transport=Transport([httpx.Response(200, json=raw)])
    )
    result = await auth.wait_for_tokens(
        pending,
        timeout=60,
        transport=Transport([
            httpx.Response(400, json={"error": "slow_down"}),
            httpx.Response(200, json={"access_token": "opaque", "token_type": "bearer"}),
        ]),
    )
    assert clock.sleeps == [5, 10]
    assert result.expires_at is None and not result.is_expired() and result.refresh_token is None
    unused = Transport([])
    with pytest.raises(auth.GrokAuthError, match="no_refresh_token"):
        await auth.refresh_tokens(result, transport=unused)
    assert not unused.requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code",
    [
        ("access_denied", "access_denied"),
        ("expired_token", "expired_token"),
        ("invalid_client", "invalid_client"),
        ("private-unknown", "oauth_error"),
    ],
)
async def test_poll_denial_expiry_and_unknown_stop(clock: Clock, error: str, code: str) -> None:
    transport = Transport([httpx.Response(400, json={"error": error, "error_description": "private-secret"})])
    with pytest.raises(auth.GrokAuthError) as caught:
        await auth.wait_for_tokens(device(expires_at=clock.now + 600), timeout=60, transport=transport)
    assert caught.value.code == code
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert "private-" not in "".join(traceback.format_exception(caught.value))
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,expires,code,sleeps,count",
    [
        (12, 600, "deadline_exceeded", [5, 5, 2], 2),
        (60, 7, "expired_token", [5, 2], 1),
        (60, -1, "expired_token", [], 0),
    ],
)
async def test_deadline_and_actual_device_expiry(
    clock: Clock, budget: int, expires: int, code: str, sleeps: list[int], count: int
) -> None:
    transport = Transport([httpx.Response(400, json={"error": "authorization_pending"}) for _ in range(count)])
    with pytest.raises(auth.GrokAuthError, match=code):
        await auth.wait_for_tokens(device(expires_at=clock.now + expires), timeout=budget, transport=transport)
    assert clock.sleeps == sleeps and len(transport.requests) == count and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["sleep", "request", "refresh", "start"])
async def test_cancellation_releases_auth_client(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
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
    if phase == "refresh":
        coroutine = auth.refresh_tokens(tokens(), transport=transport)
    elif phase == "start":
        coroutine = auth.start_device_authorization(client_version="1.0.41", transport=transport)
    else:
        coroutine = auth.wait_for_tokens(device(), timeout=60, transport=transport)
    task = asyncio.create_task(coroutine)
    await asyncio.wait_for(waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed == 1


@pytest.mark.asyncio
async def test_deadline_cancels_inflight_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "asyncio", SimpleNamespace(sleep=lambda _: asyncio.sleep(0), timeout=asyncio.timeout))

    async def block(_: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError

    transport = Transport([])
    monkeypatch.setattr(transport, "handle_async_request", block)
    with pytest.raises(auth.GrokAuthError, match="deadline_exceeded"):
        await auth.wait_for_tokens(device(), timeout=0.02, transport=transport)
    assert transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"expires_in": 0},
        {"expires_in": "600"},
        {"expires_in": None},
        {"interval": False},
        {"device_code": None},
        {"user_code": "a\nb"},
        {"verification_uri": []},
        {"verification_uri": "https://auth.x.ai.attacker.test/device"},
        {"verification_uri": "https://user@auth.x.ai/device"},
        {"verification_uri": "https://auth.x.ai:443/device"},
        {"verification_uri_complete": "http://accounts.x.ai/device"},
        {"verification_uri": "https://auth.x.ai/\x00device"},
    ],
)
async def test_invalid_device_data(extra: dict[str, Any]) -> None:
    transport = Transport([httpx.Response(200, json=device_payload(**extra))])
    with pytest.raises(auth.GrokAuthError):
        await auth.start_device_authorization(client_version="1.0.41", transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,code",
    [
        ({}, "invalid_token_type"),
        ({"token_type": "bearer"}, "invalid_response"),
        (token_payload(access_token=""), "invalid_token"),
        (token_payload(**{"token_type": "basic"}), "invalid_token_type"),
        (token_payload(expires_in="3600"), "invalid_expiry"),
        (token_payload(expires_in=None), "invalid_expiry"),
        (token_payload(expires_at=True), "invalid_expiry"),
        (token_payload(expires_in=0), "invalid_expiry"),
        (token_payload(refresh_token=""), "invalid_token"),
        ({"error": "slow_down", "interval": 0}, "invalid_interval"),
        ({"error": []}, "invalid_response"),
    ],
)
async def test_invalid_poll_data_stops(clock: Clock, raw: Any, code: str) -> None:
    transport = Transport([httpx.Response(200, json=raw)])
    with pytest.raises(auth.GrokAuthError, match=code):
        await auth.wait_for_tokens(device(expires_at=clock.now + 600), timeout=60, transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["start", "poll", "refresh"])
@pytest.mark.parametrize(
    "reply,code",
    [
        (httpx.Response(401, json={"error": "private-secret"}), "unauthorized"),
        (httpx.Response(403, json={"error": "private-secret"}), "forbidden"),
        (httpx.Response(429, json={"error": "private-secret"}), "rate_limit"),
        (httpx.Response(307, headers={"location": "https://attacker.test"}), "http_error"),
        (httpx.Response(200, content=b"private-secret"), "invalid_response"),
        (httpx.Response(200, json=[]), "invalid_response"),
        (httpx.ConnectError("private-secret"), "transport_error"),
    ],
)
async def test_auth_failures_do_not_retry_redirect_or_leak(
    clock: Clock, phase: str, reply: httpx.Response | Exception, code: str
) -> None:
    transport = Transport([reply])
    with pytest.raises(auth.GrokAuthError) as caught:
        if phase == "start":
            await auth.start_device_authorization(client_version="1.0.41", transport=transport)
        elif phase == "poll":
            await auth.wait_for_tokens(device(expires_at=clock.now + 600), timeout=60, transport=transport)
        else:
            await auth.refresh_tokens(tokens(), transport=transport)
    assert caught.value.code == code
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert "private-secret" not in "".join(traceback.format_exception(caught.value))
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,code",
    [
        ({"error": "invalid_grant", "error_description": "private-secret"}, "refresh_rejected"),
        (token_payload(expires_in="bad"), "invalid_expiry"),
        (token_payload(access_token=None), "invalid_token"),
    ],
)
async def test_refresh_rejection_and_malformed_reply(raw: dict[str, Any], code: str) -> None:
    transport = Transport([httpx.Response(400 if "error" in raw else 200, json=raw)])
    with pytest.raises(auth.GrokAuthError, match=code) as caught:
        await auth.refresh_tokens(tokens(), transport=transport)
    assert caught.value.__context__ is None and len(transport.requests) == 1 and transport.closed == 1


def test_atomic_explicit_files_and_sanitized_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "chosen.json"
    original = tokens(principal_type="User", principal_id="private-principal")
    auth.write_tokens(path, original)

    def fail(*args: Any) -> None:
        raise OSError("private-secret")

    monkeypatch.setattr("republic.auth._files.os.replace", fail)
    with pytest.raises(auth.GrokAuthError, match="credential_write_failed") as caught:
        auth.write_tokens(path, tokens(**{"access_token": "private-new"}))
    assert auth.read_tokens(path) == original and list(tmp_path.iterdir()) == [path]
    assert caught.value.__context__ is None and "private-secret" not in str(caught.value)
    with pytest.raises(auth.GrokAuthError, match="credential_read_failed"):
        auth.read_tokens(tmp_path / "missing")
    for raw in ("[]", '{"access_token": "private-secret", "extra": 1}', "private-secret"):
        path.write_text(raw)
        with pytest.raises(auth.GrokAuthError, match="invalid_token_file") as caught:
            auth.read_tokens(path)
        assert caught.value.__context__ is None


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), 10**400, -1, True, "123"])
def test_invalid_expiry_is_never_guessed(expiry: Any) -> None:
    with pytest.raises(auth.GrokAuthError, match="invalid_expiry"):
        tokens(expires_at=expiry)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["", "latest", "1.0.41\nprivate", None])
async def test_explicit_version_validated_without_network(version: Any) -> None:
    transport = Transport([])
    with pytest.raises(auth.GrokAuthError, match="invalid_client_version"):
        await auth.start_device_authorization(client_version=version, transport=transport)
    with pytest.raises(auth.GrokAuthError, match="invalid_client_version"):
        GrokOAuth(tokens(), client_version=version)
    assert not transport.requests
