# Copyright 2023-2026 SpaceXAI. Licensed under Apache-2.0.
# Protocol subset adapted for Republic from grok-build f0e3be11; see NOTICE.
"""Explicit Grok device authorization and refresh, without a login UI/runtime."""

import asyncio
import json
import math
import re
import time
from base64 import urlsafe_b64decode
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from authlib.common.errors import AuthlibBaseError
from authlib.integrations.httpx_client import AsyncOAuth2Client

from republic.auth._files import read_json, write_json
from republic.errors import RepublicError

_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_DEVICE_URL = "https://auth.x.ai/oauth2/device/code"
_OAUTH_URL = "https://auth.x.ai/oauth2/token"
# The CLI also asks for identity, conversation and workspace scopes. This SDK
# requests only renewal/API access; acceptance of the reduced set is unverified.
_SCOPE = "offline_access grok-cli:access api:access"
_CLIENT_AUTH = "none"
_ERRORS = {"access_denied", "expired_token", "invalid_grant", "invalid_client", "invalid_scope", "unauthorized_client"}


class GrokAuthError(RepublicError):
    """Fixed diagnostics without server text, credentials or native causes."""

    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(f"Grok authentication: {code}")


def _positive(value: Any, code: str = "invalid_expiry") -> float:
    if type(value) not in (int, float) or not 0 < value <= 1e308 or not math.isfinite(value):
        raise GrokAuthError(code)
    return value


def _secret(value: Any) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or any(not 33 <= ord(c) <= 126 for c in value):
        raise GrokAuthError("invalid_token")
    return value


def _version(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", value):
        raise GrokAuthError("invalid_client_version")


def _interval(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise GrokAuthError("invalid_interval")
    return value


def _verification_uri(value: str) -> None:
    # Only the first-party issuer/accounts app, no custom enterprise redirect.
    if not isinstance(value, str):
        raise GrokAuthError("invalid_verification_uri")
    try:
        url = urlsplit(value)
        valid = (
            value.isascii()
            and all(33 <= ord(c) <= 126 for c in value)
            and url.scheme == "https"
            and url.netloc in {"auth.x.ai", "accounts.x.ai"}
            and not url.fragment
            and "\\" not in value
        )
    except ValueError:
        valid = False
    if not valid:
        raise GrokAuthError("invalid_verification_uri")


@dataclass(frozen=True)
class GrokDeviceAuthorization:
    """Pending device data; the caller displays the URL/code and owns cancellation."""

    device_code: str = field(repr=False)
    user_code: str = field(repr=False)
    verification_uri: str = field(repr=False)
    expires_at: float
    interval: int
    client_version: str
    verification_uri_complete: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _secret(self.device_code)
        _secret(self.user_code)
        _positive(self.expires_at)
        _interval(self.interval)
        _version(self.client_version)
        _verification_uri(self.verification_uri)
        if self.verification_uri_complete is not None:
            _verification_uri(self.verification_uri_complete)


@dataclass(frozen=True)
class GrokTokens:
    """Opaque access/refresh data. Principal fields are unverified routing hints."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    principal_type: str | None = field(default=None, repr=False)
    principal_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _secret(self.access_token)
        if self.refresh_token is not None:
            _secret(self.refresh_token)
        if self.expires_at is not None:
            _positive(self.expires_at)
        if (self.principal_type is None) != (self.principal_id is None):
            raise GrokAuthError("invalid_principal")
        if self.principal_type is not None:
            _secret(self.principal_type)
            _secret(self.principal_id)

    def is_expired(self) -> bool:
        """A local expiry check, not identity/entitlement validation; None is unknown."""
        return self.expires_at is not None and time.time() >= self.expires_at


def _client(transport: httpx.AsyncBaseTransport | None, timeout: float) -> AsyncOAuth2Client:
    _positive(timeout, "invalid_timeout")
    return AsyncOAuth2Client(
        client_id=_CLIENT_ID,
        token_endpoint_auth_method=_CLIENT_AUTH,
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
    )


def _body(response: httpx.Response) -> dict[str, Any]:
    if response.status_code not in (200, 400):
        code = {401: "unauthorized", 403: "forbidden", 429: "rate_limit"}.get(response.status_code, "http_error")
        raise GrokAuthError(code, status_code=response.status_code)
    raw = response.json()
    if not isinstance(raw, dict):
        raise GrokAuthError("invalid_response")
    if "error" not in raw and response.status_code != 200:
        raise GrokAuthError("http_error", status_code=response.status_code)
    return raw


def _device_headers(client_version: str) -> dict[str, str]:
    return {"Accept": "application/json", "x-grok-client-version": client_version, "x-grok-client-surface": "headless"}


async def start_device_authorization(
    *,
    client_version: str,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 30,
) -> GrokDeviceAuthorization:
    """One request to auth.x.ai. No browser, discovery request, file or fallback."""
    _version(client_version)
    try:
        async with _client(transport, timeout) as client:
            started = time.time()
            response = await client.request(
                "POST",
                _DEVICE_URL,
                withhold_token=True,
                data={"client_id": _CLIENT_ID, "scope": _SCOPE, "referrer": "republic"},
                headers=_device_headers(client_version),
            )
            raw = _body(response)
            if "error" in raw:
                raise GrokAuthError(raw["error"] if raw["error"] in _ERRORS else "device_authorization_failed")
            return GrokDeviceAuthorization(
                device_code=raw["device_code"],
                user_code=raw["user_code"],
                verification_uri=raw["verification_uri"],
                verification_uri_complete=raw.get("verification_uri_complete"),
                expires_at=started + _positive(raw["expires_in"]),
                interval=raw.get("interval", 5),
                client_version=client_version,
            )
    except httpx.HTTPError:
        error = GrokAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = GrokAuthError("invalid_response")
    raise error


def _principal_hint(access_token: str) -> tuple[str | None, str | None]:
    """Decode only routing hints; no JWT authentication or permission decisions."""
    try:
        pieces = access_token.split(".")
        if len(pieces) != 3:
            return None, None
        raw = json.loads(urlsafe_b64decode(pieces[1] + "=" * (-len(pieces[1]) % 4)))
        if isinstance(raw, dict):
            kind = raw.get("principal_type", raw.get("principalType"))
            identity = raw.get("principal_id", raw.get("principalId"))
            if isinstance(kind, str) and kind and isinstance(identity, str) and identity:
                return kind, identity
    except (ValueError, UnicodeDecodeError):
        pass
    return None, None


def _tokens(raw: dict[str, Any], previous: GrokTokens | None = None) -> GrokTokens:
    # Validate before Authlib's token object can coerce strings/booleans to numbers.
    if str(raw.get("token_type", "")).lower() != "bearer":
        raise GrokAuthError("invalid_token_type")
    for key in ("expires_at", "expires_in"):
        if key in raw:
            _positive(raw[key])
    expiry = raw.get("expires_at")
    if expiry is None and "expires_in" in raw:
        expiry = time.time() + raw["expires_in"]
    access = _secret(raw["access_token"])
    kind, identity = _principal_hint(access)
    if kind is None and previous is not None:
        kind, identity = previous.principal_type, previous.principal_id
    return GrokTokens(
        access_token=access,
        refresh_token=raw.get("refresh_token", previous.refresh_token if previous else None),
        expires_at=expiry,
        principal_type=kind,
        principal_id=identity,
    )


async def _poll(
    client: AsyncOAuth2Client,
    device: GrokDeviceAuthorization,
    interval: int,
) -> tuple[GrokTokens | None, int]:
    result = None
    next_interval = interval

    def validate(response: httpx.Response) -> httpx.Response:
        nonlocal result, next_interval
        raw = _body(response)
        if raw.get("error") in ("authorization_pending", "slow_down"):
            minimum = interval + (5 if raw["error"] == "slow_down" else 0)
            next_interval = max(minimum, _interval(raw.get("interval", minimum)))
        elif "error" not in raw:
            result = _tokens(raw)
        return response

    client.register_compliance_hook("access_token_response", validate)
    try:
        await client.fetch_token(
            _OAUTH_URL,
            grant_type="urn:ietf:params:oauth:grant-type:device_code",
            device_code=device.device_code,
            headers=_device_headers(device.client_version),
        )
    except AuthlibBaseError as exc:
        code = getattr(exc, "error", None)
        if code in ("authorization_pending", "slow_down"):
            return None, next_interval
        error = GrokAuthError(code if code in _ERRORS else "oauth_error")
    else:
        if result is None:
            raise GrokAuthError("invalid_token")
        return result, next_interval
    finally:
        client.compliance_hook["access_token_response"].discard(validate)
    raise error


async def wait_for_tokens(
    device: GrokDeviceAuthorization,
    *,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
    request_timeout: float = 30,
) -> GrokTokens:
    """Poll with interval/slow-down and the earlier of caller budget/device expiry."""
    _positive(timeout, "invalid_timeout")
    remaining = device.expires_at - time.time()
    deadline = time.monotonic() + min(timeout, remaining)
    code = "expired_token" if remaining <= timeout else "deadline_exceeded"
    interval = device.interval
    try:
        async with _client(transport, request_timeout) as client:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GrokAuthError(code)
                await asyncio.sleep(min(interval, remaining))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GrokAuthError(code)
                async with asyncio.timeout(remaining):
                    result, interval = await _poll(client, device, interval)
                if time.monotonic() >= deadline:
                    raise GrokAuthError(code)
                if result is not None:
                    return result
    except TimeoutError:
        error = GrokAuthError(code)
    except httpx.HTTPError:
        error = GrokAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = GrokAuthError("invalid_response")
    raise error


async def refresh_tokens(
    tokens: GrokTokens,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 30,
) -> GrokTokens:
    """One explicit refresh grant, retaining an unrotated token and principal hint."""
    if tokens.refresh_token is None:
        raise GrokAuthError("no_refresh_token")
    result = None

    def validate(response: httpx.Response) -> httpx.Response:
        nonlocal result
        raw = _body(response)
        if "error" not in raw:
            result = _tokens(raw, tokens)
        return response

    principal = {}
    if tokens.principal_type is not None:
        principal = {"principal_type": tokens.principal_type, "principal_id": tokens.principal_id}
    try:
        async with _client(transport, timeout) as client:
            client.register_compliance_hook("refresh_token_response", validate)
            await client.refresh_token(_OAUTH_URL, refresh_token=tokens.refresh_token, **principal)
    except AuthlibBaseError:
        error = GrokAuthError("refresh_rejected")
    except httpx.HTTPError:
        error = GrokAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = GrokAuthError("invalid_response")
    else:
        if result is None:
            raise GrokAuthError("invalid_token")
        return result
    raise error


def read_tokens(path: str | Path) -> GrokTokens:
    """Read only a caller-selected Republic token file; no CLI file discovery."""
    try:
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise GrokAuthError("invalid_token_file")
        return GrokTokens(**raw)
    except OSError:
        error = GrokAuthError("credential_read_failed")
    except (ValueError, TypeError):
        error = GrokAuthError("invalid_token_file")
    raise error


def write_tokens(path: str | Path, tokens: GrokTokens) -> None:
    """Atomically write mode 0600 at the explicit path; parent must already exist."""
    try:
        write_json(path, asdict(tokens))
    except OSError:
        error = GrokAuthError("credential_write_failed")
    else:
        return
    raise error
