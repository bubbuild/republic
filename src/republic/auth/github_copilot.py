# Protocol references: Microsoft VS Code/Copilot Chat (MIT); see NOTICE.
"""Explicit GitHub device login and Copilot inference-token exchange."""

import asyncio
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from authlib.common.errors import AuthlibBaseError
from authlib.integrations.httpx_client import AsyncOAuth2Client

from republic.auth._files import read_json, write_json
from republic.errors import RepublicError

_CLIENT_ID = "01ab8ac9400c4e429b23"  # Official VS Code GitHub authentication app.
_CLIENT_AUTH = "none"
_DEVICE_URL = "https://github.com/login/device/code"
_OAUTH_URL = "https://github.com/login/oauth/access_token"
_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_VERIFICATION_URI = "https://github.com/login/device"
_API_ENDPOINTS = {
    "https://api.githubcopilot.com",
    "https://api.individual.githubcopilot.com",
    "https://api.business.githubcopilot.com",
    "https://api.enterprise.githubcopilot.com",
}
_ERRORS = {
    "access_denied",
    "expired_token",
    "incorrect_client_credentials",
    "incorrect_device_code",
    "device_flow_disabled",
    "unsupported_grant_type",
    "invalid_grant",
}


class CopilotAuthError(RepublicError):
    """Fixed diagnostics, with no secret-bearing server text or native causes."""

    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(f"Copilot authentication: {code}")


def _positive(value: Any, code: str = "invalid_expiry") -> float:
    if type(value) not in (int, float) or not 0 < value <= 1e308 or not math.isfinite(value):
        raise CopilotAuthError(code)
    return value


def _secret(value: Any) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or any(not 33 <= ord(c) <= 126 for c in value):
        raise CopilotAuthError("invalid_token")
    return value


def _interval(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise CopilotAuthError("invalid_interval")
    return value


def _client_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise CopilotAuthError("invalid_client_id")


@dataclass(frozen=True)
class DeviceAuthorization:
    """A pending GitHub.com device flow. The caller displays the URL/user_code."""

    device_code: str = field(repr=False)
    user_code: str = field(repr=False)
    verification_uri: str
    expires_at: float
    interval: int
    client_id: str = _CLIENT_ID

    def __post_init__(self) -> None:
        _secret(self.device_code)
        _secret(self.user_code)
        _positive(self.expires_at)
        _interval(self.interval)
        _client_id(self.client_id)
        if self.verification_uri != _VERIFICATION_URI:
            raise CopilotAuthError("invalid_verification_uri")


@dataclass(frozen=True)
class GitHubToken:
    """GitHub login data, not proof of Copilot entitlement. Expiry may be unknown."""

    access_token: str = field(repr=False)
    expires_at: float | None = None
    scope: str | None = None
    client_id: str = _CLIENT_ID
    refresh_token: str | None = field(default=None, repr=False)
    refresh_expires_at: float | None = None

    def __post_init__(self) -> None:
        _secret(self.access_token)
        _client_id(self.client_id)
        for expiry in (self.expires_at, self.refresh_expires_at):
            if expiry is not None:
                _positive(expiry)
        if self.refresh_token is not None:
            _secret(self.refresh_token)
        if self.scope is not None and not isinstance(self.scope, str):
            raise CopilotAuthError("invalid_scope")

    def is_expired(self) -> bool:
        """False when no expiry was declared; this does not validate access."""
        return self.expires_at is not None and time.time() >= self.expires_at


@dataclass(frozen=True)
class CopilotToken:
    """Short-lived inference token. Renew by explicitly exchanging GitHubToken again."""

    token: str = field(repr=False)
    expires_at: float
    api_endpoint: str
    refresh_at: float | None = None

    def __post_init__(self) -> None:
        _secret(self.token)
        _positive(self.expires_at)
        if self.refresh_at is not None:
            _positive(self.refresh_at)
        # Exact, public service origins only: no path/query/userinfo/port/redirect.
        if not isinstance(self.api_endpoint, str) or self.api_endpoint not in _API_ENDPOINTS:
            raise CopilotAuthError("untrusted_endpoint")

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


def _client(client_id: str, transport: httpx.AsyncBaseTransport | None, timeout: float) -> AsyncOAuth2Client:
    _client_id(client_id)
    _positive(timeout, "invalid_timeout")
    return AsyncOAuth2Client(
        client_id=client_id,
        token_endpoint_auth_method=_CLIENT_AUTH,
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
    )


def _body(response: httpx.Response) -> dict[str, Any]:
    if response.status_code not in (200, 400):
        codes = {401: "unauthorized", 403: "forbidden", 429: "rate_limit"}
        raise CopilotAuthError(codes.get(response.status_code, "http_error"), status_code=response.status_code)
    raw = response.json()
    if not isinstance(raw, dict):
        raise CopilotAuthError("invalid_response")
    return raw


def _oauth_body(response: httpx.Response) -> dict[str, Any]:
    raw = _body(response)
    if "error" not in raw and response.status_code != 200:
        raise CopilotAuthError("http_error", status_code=response.status_code)
    return raw


async def start_device_authorization(
    *, client_id: str = _CLIENT_ID, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30
) -> DeviceAuthorization:
    """One device-code request. No browser, profile lookup, persistence or polling."""
    try:
        async with _client(client_id, transport, timeout) as client:
            started = time.time()
            response = await client.request(
                "POST",
                _DEVICE_URL,
                data={"client_id": client_id, "scope": "user:email"},
                headers={"Accept": "application/json"},
                withhold_token=True,
            )
            raw = _body(response)
            if "error" in raw:
                raise CopilotAuthError(raw["error"] if raw["error"] in _ERRORS else "device_authorization_failed")
            if response.status_code != 200:
                raise CopilotAuthError("http_error", status_code=response.status_code)
            return DeviceAuthorization(
                device_code=raw["device_code"],
                user_code=raw["user_code"],
                verification_uri=raw["verification_uri"],
                interval=raw.get("interval", 5),
                expires_at=started + _positive(raw["expires_in"]),
                client_id=client_id,
            )
    except httpx.HTTPError:
        error = CopilotAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = CopilotAuthError("invalid_response")
    raise error


def _github_token(raw: dict[str, Any], client_id: str, previous: GitHubToken | None = None) -> GitHubToken:
    # Called on raw response JSON *before* Authlib can coerce invalid expiry fields.
    if str(raw.get("token_type", "")).lower() != "bearer":
        raise CopilotAuthError("invalid_token_type")
    expires_at = None
    for key in ("expires_at", "expires_in", "refresh_token_expires_in"):
        if key in raw:
            _positive(raw[key])
    if "expires_at" in raw:
        expires_at = raw["expires_at"]
    elif "expires_in" in raw:
        expires_at = time.time() + raw["expires_in"]
    refresh_at = (
        time.time() + raw["refresh_token_expires_in"]
        if "refresh_token_expires_in" in raw
        else previous.refresh_expires_at
        if previous and "refresh_token" not in raw
        else None
    )
    return GitHubToken(
        access_token=raw["access_token"],
        expires_at=expires_at,
        scope=raw.get("scope", previous.scope if previous else None),
        client_id=client_id,
        refresh_token=raw.get("refresh_token", previous.refresh_token if previous else None),
        refresh_expires_at=refresh_at,
    )


async def _poll(
    client: AsyncOAuth2Client, device: DeviceAuthorization, interval: int
) -> tuple[GitHubToken | None, int]:
    next_interval = interval
    result = None

    def validate(response: httpx.Response) -> httpx.Response:
        nonlocal next_interval, result
        raw = _oauth_body(response)
        if raw.get("error") in ("authorization_pending", "slow_down"):
            minimum = interval + (5 if raw["error"] == "slow_down" else 0)
            next_interval = max(minimum, _interval(raw.get("interval", minimum)))
        elif "error" not in raw:
            result = _github_token(raw, device.client_id)
        return response

    client.register_compliance_hook("access_token_response", validate)
    try:
        await client.fetch_token(
            _OAUTH_URL, grant_type="urn:ietf:params:oauth:grant-type:device_code", device_code=device.device_code
        )
    except AuthlibBaseError as exc:
        code = getattr(exc, "error", None)
        if code in ("authorization_pending", "slow_down"):
            return None, next_interval
        error = CopilotAuthError(code if code in _ERRORS else "oauth_error")
    else:
        if result is None:
            raise CopilotAuthError("invalid_token")
        return result, next_interval
    finally:
        client.compliance_hook["access_token_response"].discard(validate)
    raise error


async def wait_for_token(
    device: DeviceAuthorization,
    *,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
    request_timeout: float = 30,
) -> GitHubToken:
    """Wait within a caller time budget AND device expiry; pending/slow_down only."""
    _positive(timeout, "invalid_timeout")
    remaining = device.expires_at - time.time()
    deadline = time.monotonic() + min(timeout, remaining)
    end_code = "expired_token" if remaining <= timeout else "deadline_exceeded"
    interval = device.interval
    try:
        async with _client(device.client_id, transport, request_timeout) as client:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CopilotAuthError(end_code)
                await asyncio.sleep(min(interval, remaining))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CopilotAuthError(end_code)
                async with asyncio.timeout(remaining):
                    result, interval = await _poll(client, device, interval)
                if time.monotonic() >= deadline:
                    raise CopilotAuthError(end_code)
                if result is not None:
                    return result
    except TimeoutError:
        error = CopilotAuthError(end_code)
    except httpx.HTTPError:
        error = CopilotAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = CopilotAuthError("invalid_response")
    raise error


async def refresh_github_token(
    token: GitHubToken,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 30,
) -> GitHubToken:
    """Explicit OAuth refresh ONLY when GitHub actually issued a refresh token."""
    if token.refresh_token is None:
        raise CopilotAuthError("no_refresh_token")
    if token.refresh_expires_at is not None and time.time() >= token.refresh_expires_at:
        raise CopilotAuthError("refresh_expired")
    result = None

    def validate(response: httpx.Response) -> httpx.Response:
        nonlocal result
        raw = _oauth_body(response)
        if "error" not in raw:
            result = _github_token(raw, token.client_id, token)
        return response

    try:
        async with _client(token.client_id, transport, timeout) as client:
            client.register_compliance_hook("refresh_token_response", validate)
            await client.refresh_token(_OAUTH_URL, refresh_token=token.refresh_token)
    except AuthlibBaseError:
        error = CopilotAuthError("refresh_rejected")
    except httpx.HTTPError:
        error = CopilotAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = CopilotAuthError("invalid_response")
    else:
        if result is None:
            raise CopilotAuthError("invalid_token")
        return result
    raise error


async def exchange_copilot_token(
    github_token: GitHubToken,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 30,
) -> CopilotToken:
    """Exchange once (also how to renew). GitHub login alone does not grant Copilot."""
    if github_token.is_expired():
        raise CopilotAuthError("github_token_expired")
    _positive(timeout, "invalid_timeout")
    try:
        async with httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False) as client:
            response = await client.get(
                _EXCHANGE_URL,
                headers={
                    "Authorization": f"token {github_token.access_token}",
                    "Accept": "application/json",
                    "User-Agent": "republic",
                    "X-GitHub-Api-Version": "2025-04-01",
                },
            )
            raw = _body(response)
            if response.status_code != 200:
                raise CopilotAuthError("exchange_rejected", status_code=response.status_code)
            endpoints = raw.get("endpoints", {})
            if not isinstance(endpoints, dict):
                raise CopilotAuthError("invalid_endpoint")
            refresh_at = time.time() + _positive(raw["refresh_in"]) if "refresh_in" in raw else None
            return CopilotToken(
                token=raw["token"],
                expires_at=raw["expires_at"],
                refresh_at=refresh_at,
                api_endpoint=endpoints.get("api", "https://api.githubcopilot.com"),
            )
    except httpx.HTTPError:
        error = CopilotAuthError("transport_error")
    except (ValueError, TypeError, KeyError):
        error = CopilotAuthError("invalid_response")
    raise error


def read_token(path: str | Path) -> GitHubToken | CopilotToken:
    """Read a tagged Republic token file at exactly the supplied path."""
    try:
        raw = read_json(path)
        if not isinstance(raw, dict) or set(raw) != {"kind", "data"}:
            raise CopilotAuthError("invalid_token_file")
        if raw["kind"] == "github":
            return GitHubToken(**raw["data"])
        if raw["kind"] == "copilot":
            return CopilotToken(**raw["data"])
        raise CopilotAuthError("invalid_token_file")
    except OSError:
        error = CopilotAuthError("credential_read_failed")
    except (ValueError, TypeError):
        error = CopilotAuthError("invalid_token_file")
    raise error


def write_token(path: str | Path, token: GitHubToken | CopilotToken) -> None:
    """Atomically write mode 0600; parent must exist. No discovery or implicit save."""
    if not isinstance(token, GitHubToken | CopilotToken):
        raise CopilotAuthError("invalid_token")
    try:
        write_json(path, {"kind": "github" if isinstance(token, GitHubToken) else "copilot", "data": asdict(token)})
    except OSError:
        error = CopilotAuthError("credential_write_failed")
    else:
        return
    raise error
