# Selected scenarios adapted from Bub 357901db (Apache-2.0); see NOTICE.
"""Real Authlib exchanges against offline HTTP fixtures, with explicit storage."""

import asyncio
import hashlib
import json
import logging
import time
import traceback
from base64 import urlsafe_b64encode
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from republic import generate
from republic.auth.codex import (
    CodexAuthError,
    create_authorization,
    exchange_code,
    read_tokens,
    refresh_tokens,
    write_tokens,
)
from republic.providers.codex import OpenAICodex
from tests.codex_fixtures import Transport, Wire, jwt, token_payload, tokens
from tests.http_fixtures import Bytes, streaming
from tests.openai_fixtures import request, sse
from tests.responses_fixtures import terminal


@pytest.mark.asyncio
async def test_pkce_s256_and_full_offline_login_inference_refresh_flow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    authorization = await create_authorization()
    query = parse_qs(urlsplit(authorization.url).query)
    assert query["code_challenge_method"] == ["S256"]
    challenge = urlsafe_b64encode(hashlib.sha256(authorization.code_verifier.encode()).digest()).decode().rstrip("=")
    assert query["code_challenge"] == [challenge]
    assert query["state"] == [authorization.state]
    assert query["scope"] == ["openid profile email offline_access"]
    assert query["originator"] == ["codex_cli_rs"]
    assert query["response_type"] == ["code"]
    callback = authorization.redirect_uri + "?" + urlencode({"code": "private-code", "state": authorization.state})
    exchange = Transport([httpx.Response(200, json=token_payload())])
    credentials = await exchange_code(authorization, callback, transport=exchange)
    assert exchange.closed == 1 and len(exchange.requests) == 1
    req = exchange.requests[0]
    assert str(req.url) == "https://auth.openai.com/oauth/token"
    assert "authorization" not in req.headers
    assert parse_qs(req.content.decode()) == {
        "grant_type": ["authorization_code"],
        "code": ["private-code"],
        "client_id": query["client_id"],
        "redirect_uri": [authorization.redirect_uri],
        "code_verifier": [authorization.code_verifier],
    }
    assert time.time() + 1700 < credentials.expires_at <= time.time() + 1800
    path = tmp_path / "chosen.json"
    write_tokens(path, credentials)
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_tokens(path) == credentials
    bodies = [Bytes([sse(terminal())]), Bytes([sse(terminal())])]
    async with Wire([streaming(b) for b in bodies], read_tokens(path)) as wire:
        assert (await generate(wire.provider, request())).message.text == "Hello"
        rotated_access = jwt(**{"https://api.openai.com/auth": {"chatgpt_account_id": "acct_new"}})
        refresh = Transport([
            httpx.Response(
                200, json=token_payload(**{"access_token": rotated_access, "refresh_token": "rotated-private"})
            )
        ])
        updated = await refresh_tokens(credentials, transport=refresh)
        assert len(refresh.requests) == 1 and refresh.closed == 1
        assert parse_qs(refresh.requests[0].content.decode()) == {
            "grant_type": ["refresh_token"],
            "client_id": query["client_id"],
            "refresh_token": [credentials.refresh_token],
        }
        assert "authorization" not in refresh.requests[0].headers
        assert updated.account_id == "acct_new" and updated.refresh_token != credentials.refresh_token
        async with OpenAICodex(updated, client=wire.client) as refreshed:
            await generate(refreshed, request())
        assert len(wire.requests) == 2
        assert wire.requests[0].headers["authorization"] == f"Bearer {credentials.access_token}"
        assert wire.requests[1].headers["authorization"] == f"Bearer {updated.access_token}"
        assert wire.requests[1].headers["chatgpt-account-id"] == "acct_new"
        assert not wire.client.is_closed()
    assert all(body.closed == 1 for body in bodies)
    for secret in (
        credentials.access_token,
        credentials.refresh_token,
        updated.access_token,
        "rotated-private",
        "private-code",
        authorization.code_verifier,
    ):
        assert secret not in caplog.text + repr(credentials) + repr(authorization)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "code"),
    [
        ({"code": "x"}, "state_mismatch"),
        ({"code": "x", "state": "wrong"}, "state_mismatch"),
        ({"code": ""}, "missing_code"),
        ({}, "missing_code"),
        ({"error": "access_denied", "error_description": "private-error"}, "denied"),
        ({"error": "server_error"}, "callback_error"),
    ],
)
async def test_callback_failures_before_network(query: dict[str, str], code: str) -> None:
    authorization = await create_authorization()
    if code != "state_mismatch":
        query = {"state": authorization.state, **query}
    transport = Transport([])
    with pytest.raises(CodexAuthError) as caught:
        await exchange_code(authorization, authorization.redirect_uri + "?" + urlencode(query), transport=transport)
    assert caught.value.code == code
    assert not transport.requests
    assert "private-error" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["#fragment", "&state=duplicate", "&code=duplicate", "&error=x&error=y"])
async def test_ambiguous_callback_rejected(suffix: str) -> None:
    auth = await create_authorization()
    with pytest.raises(CodexAuthError, match="invalid_callback"):
        await exchange_code(auth, auth.redirect_uri + "?code=x&state=" + auth.state + suffix)


@pytest.mark.asyncio
async def test_redirect_and_verifier_validation() -> None:
    with pytest.raises(CodexAuthError, match="invalid_redirect"):
        await create_authorization(redirect_uri="https://remote.test/callback")
    auth = await create_authorization()
    with pytest.raises(CodexAuthError, match="invalid_pkce"):
        replace(auth, code_verifier="short")
    with pytest.raises(CodexAuthError, match="invalid_state"):
        replace(auth, state="")
    with pytest.raises(CodexAuthError, match="invalid_callback"):
        await exchange_code(auth, "http://other.test/callback?code=x&state=" + auth.state)
    with pytest.raises(CodexAuthError, match="invalid_authorization"):
        replace(auth, url=auth.url.replace("code_challenge_method=S256", "code_challenge_method=plain"))
    with pytest.raises(CodexAuthError, match="invalid_authorization"):
        replace(auth, code_verifier="x" * 64)


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", [None, True, False, "3600", "private-invalid", -1, 0, {}, []])
@pytest.mark.parametrize("key", ["expires_in", "expires_at"])
async def test_malformed_expiry_not_coerced_or_replaced(key: str, expiry: Any) -> None:
    transport = Transport([httpx.Response(200, json=token_payload(**{key: expiry}))])
    with pytest.raises(CodexAuthError, match="invalid_expiry"):
        await refresh_tokens(tokens(), transport=transport)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.parametrize("expiry", [float("inf"), float("nan"), 10**400])
def test_nonfinite_expiry_rejected(expiry: float) -> None:
    with pytest.raises(CodexAuthError, match="invalid_expiry"):
        tokens(expires_at=expiry)


@pytest.mark.asyncio
async def test_refresh_keeps_unrotated_token_and_updates_claim_hints() -> None:
    raw = token_payload()
    del raw["refresh_token"]
    del raw["expires_in"]
    raw["access_token"] = jwt(exp=2000000000)
    raw["id_token"] = jwt(**{"https://api.openai.com/auth": {"chatgpt_account_id": "acct_hint"}})
    updated = await refresh_tokens(tokens(), transport=Transport([httpx.Response(200, json=raw)]))
    assert updated.refresh_token == tokens().refresh_token
    assert updated.expires_at == 2000000000
    assert updated.account_id == "acct_hint"
    raw.update({"access_token": "opaque-access"})
    raw["expires_in"] = 100
    del raw["id_token"]
    updated = await refresh_tokens(tokens(), transport=Transport([httpx.Response(200, json=raw)]))
    assert updated.account_id == "acct_fixture"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        [],
        {},
        {"access_token": "opaque"},
        token_payload(access_token=""),
        token_payload(refresh_token=""),
        token_payload(**{"token_type": "mac"}),
    ],
)
async def test_invalid_token_payload(raw: Any) -> None:
    with pytest.raises(CodexAuthError):
        await refresh_tokens(tokens(), transport=Transport([httpx.Response(200, json=raw)]))


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("kind", ["oauth", "http", "network", "json"])
async def test_errors_never_include_response_or_native_cause(refresh: bool, kind: str) -> None:
    secret = tokens().refresh_token
    replies = {
        "oauth": httpx.Response(400, json={"error": "invalid_grant", "error_description": secret}),
        "http": httpx.Response(503, text=secret),
        "network": httpx.ConnectError(secret),
        "json": httpx.Response(200, text=secret),
    }
    transport = Transport([replies[kind]])
    auth = await create_authorization()
    with pytest.raises(CodexAuthError) as caught:
        if refresh:
            await refresh_tokens(tokens(), transport=transport)
        else:
            await exchange_code(auth, auth.redirect_uri + "?code=x&state=" + auth.state, transport=transport)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert len(transport.requests) == 1 and transport.closed == 1


def test_expiry_and_explicit_file_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert tokens(expires_at=1).is_expired()
    assert not tokens().is_expired()
    assert tokens().is_expired(leeway=3600)
    with pytest.raises(CodexAuthError, match="invalid_leeway"):
        tokens().is_expired(leeway=-1)
    with pytest.raises(CodexAuthError, match="credential_read_failed"):
        read_tokens(tmp_path / "absent")
    path = tmp_path / "chosen.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(CodexAuthError, match="invalid_token_file"):
        read_tokens(path)

    def fail_replace(*args: Any) -> None:
        raise OSError("fixture")

    monkeypatch.setattr("republic.auth.codex.os.replace", fail_replace)
    with pytest.raises(CodexAuthError, match="credential_write_failed"):
        write_tokens(path, tokens())
    assert path.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(CodexAuthError, match="credential_write_failed"):
        write_tokens(tmp_path / "absent" / "tokens", tokens())


@pytest.mark.parametrize("raw", [[], {"unexpected": "private"}, {"access_token": "private"}, {"expires_at": "private"}])
def test_bad_file_does_not_echo_values(tmp_path: Path, raw: Any) -> None:
    path = tmp_path / "chosen.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(CodexAuthError) as caught:
        read_tokens(path)
    assert "private" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", [False, True])
async def test_cancel_auth_exchange_closes_response_and_transport(refresh: bool) -> None:
    body = Bytes([], wait=True)
    transport = Transport([httpx.Response(200, stream=body)])
    auth = await create_authorization()
    operation = (
        refresh_tokens(tokens(), transport=transport)
        if refresh
        else exchange_code(
            auth,
            auth.redirect_uri + "?code=x&state=" + auth.state,
            transport=transport,
        )
    )
    task = asyncio.create_task(operation)
    await asyncio.wait_for(body.waiting.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed == 1 and transport.closed == 1 and len(transport.requests) == 1
