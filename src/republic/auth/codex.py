# Adapted from Bub 357901db and OpenAI Codex 21eb3551 (Apache-2.0); see NOTICE.
"""ChatGPT PKCE and explicit token persistence. Login UX belongs to the caller."""

import json
import math
import re
import time
from base64 import urlsafe_b64decode
from dataclasses import asdict, dataclass, field
from pathlib import Path
from secrets import compare_digest
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from authlib.common.errors import AuthlibBaseError
from authlib.common.security import generate_token
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.oauth2.rfc6749.parameters import parse_authorization_code_response
from authlib.oauth2.rfc7636 import create_s256_code_challenge

from republic.auth._files import read_json, write_json
from republic.errors import RepublicError

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_EXCHANGE_URL = "https://auth.openai.com/oauth/token"
_CLIENT_AUTH = "none"
_REDIRECT_URI = "http://localhost:1455/auth/callback"


class CodexAuthError(RepublicError):
    """A stable local code, without server text, credentials or native causes."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Codex authentication: {code}")


def _number(value: Any, code: str = "invalid_expiry") -> float:
    if type(value) not in (int, float) or not 0 < value <= 1e308 or not math.isfinite(value):
        raise CodexAuthError(code)
    return value


def _secret(value: Any) -> str:
    if not isinstance(value, str) or not value or not value.isprintable() or any(c.isspace() for c in value):
        raise CodexAuthError("invalid_token")
    return value


@dataclass(frozen=True)
class CodexTokens:
    """Plain token data; repr hides credentials. expires_at is Unix seconds.

    account_id and JWT expiry are routing/freshness hints, never verified identity.
    Explicit access to the fields (or serialization) reveals secrets by design.
    """

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    account_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _secret(self.access_token)
        if self.refresh_token is not None:
            _secret(self.refresh_token)
        if not self.access_token.isascii():
            raise CodexAuthError("invalid_token")
        if self.expires_at is not None:
            _number(self.expires_at)
        if self.account_id is not None and (
            not isinstance(self.account_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.account_id)
        ):
            raise CodexAuthError("invalid_account")

    def is_expired(self, *, leeway: float = 0) -> bool:
        """Check locally; never refresh or validate the token with a server."""
        if type(leeway) not in (int, float) or not math.isfinite(leeway) or leeway < 0:
            raise CodexAuthError("invalid_leeway")
        return self.expires_at is not None and time.time() + leeway >= self.expires_at


@dataclass(frozen=True)
class CodexAuthorization:
    """One pending login. Keep private; the caller owns cancellation and lifetime."""

    url: str = field(repr=False)
    state: str = field(repr=False)
    code_verifier: str = field(repr=False)
    redirect_uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or not self.state or not self.state.isascii():
            raise CodexAuthError("invalid_state")
        if not re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", self.code_verifier):
            raise CodexAuthError("invalid_pkce")
        _redirect(self.redirect_uri)
        query = parse_qs(urlsplit(self.url).query)
        if (
            query.get("code_challenge_method") != ["S256"]
            or query.get("code_challenge") != [create_s256_code_challenge(self.code_verifier)]
            or query.get("state") != [self.state]
            or query.get("redirect_uri") != [self.redirect_uri]
        ):
            raise CodexAuthError("invalid_authorization")


def _redirect(uri: str) -> None:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CodexAuthError("invalid_redirect")


def _client(
    *, redirect_uri: str | None = None, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30
) -> AsyncOAuth2Client:
    _number(timeout, "invalid_timeout")
    client = AsyncOAuth2Client(
        client_id=_CLIENT_ID,
        token_endpoint_auth_method=_CLIENT_AUTH,
        code_challenge_method="S256",
        redirect_uri=redirect_uri,
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
    )
    # Validate the original JSON before Authlib's permissive expiry coercion.
    client.register_compliance_hook("access_token_response", _token_response)
    client.register_compliance_hook("refresh_token_response", _token_response)
    return client


async def create_authorization(*, redirect_uri: str = _REDIRECT_URI) -> CodexAuthorization:
    """Build an S256 authorization URL, without opening a browser or a socket."""
    _redirect(redirect_uri)
    verifier = generate_token(64)
    async with _client(redirect_uri=redirect_uri) as client:
        url, state = client.create_authorization_url(
            _AUTHORIZE_URL,
            code_verifier=verifier,
            scope="openid profile email offline_access",
            **{"id_token_add_organizations": "true", "codex_cli_simplified_flow": "true", "originator": "codex_cli_rs"},
        )
    return CodexAuthorization(url=url, state=state, code_verifier=verifier, redirect_uri=redirect_uri)


def _callback(authorization: CodexAuthorization, callback_url: str) -> str:
    parsed = urlsplit(callback_url)
    expected = urlsplit(authorization.redirect_uri)
    if (parsed.scheme, parsed.netloc, parsed.path) != (
        expected.scheme,
        expected.netloc,
        expected.path,
    ) or parsed.fragment:
        raise CodexAuthError("invalid_callback")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if any(len(query.get(key, [])) > 1 for key in ("state", "code", "error")):
        raise CodexAuthError("invalid_callback")
    # Authlib's code parser checks state only on success. Check denial callbacks
    # as well, before looking at their error; never echo error_description.
    state = query.get("state", [""])[0]
    if not state or not compare_digest(state.encode(), authorization.state.encode()):
        raise CodexAuthError("state_mismatch")
    if "error" in query:
        raise CodexAuthError("denied" if query["error"] == ["access_denied"] else "callback_error")
    result = parse_authorization_code_response(callback_url, state=authorization.state)
    if not result["code"].strip():
        raise CodexAuthError("missing_code")
    return result["code"]


def _claims(token: str) -> dict[str, Any]:
    """Decode hints only. No signature/issuer/audience verification is implied."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        raw = json.loads(urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, UnicodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _expiry(raw: dict[str, Any]) -> float | None:
    # Validate *all* supplied expiry fields, even when another one takes priority.
    for key in ("expires_in", "expires_at"):
        if key in raw:
            _number(raw[key])
    if "expires_at" in raw:
        return raw["expires_at"]
    if "expires_in" in raw:
        return time.time() + raw["expires_in"]
    expiry = _claims(raw["access_token"]).get("exp")
    return _number(expiry) if expiry is not None else None


def _token_response(response: httpx.Response) -> httpx.Response:
    raw = response.json()
    if not isinstance(raw, dict):
        raise CodexAuthError("invalid_token")
    if "error" in raw:
        # Let Authlib handle OAuth errors, then expose only our fixed error code.
        return response
    if response.status_code != 200:
        raise CodexAuthError("token_http_error")
    _secret(raw.get("access_token"))
    for key in ("refresh_token", "id_token"):
        if key in raw:
            _secret(raw[key])
    if str(raw.get("token_type", "bearer")).lower() != "bearer":
        raise CodexAuthError("invalid_token")
    _expiry(raw)
    return response


def _tokens(raw: dict[str, Any], previous: CodexTokens | None = None) -> CodexTokens:
    account = None
    for key in ("access_token", "id_token"):
        hints = _claims(raw.get(key, "")).get("https://api.openai.com/auth", {})
        if isinstance(hints, dict) and hints.get("chatgpt_account_id") is not None:
            account = hints["chatgpt_account_id"]
            break
    return CodexTokens(
        access_token=raw["access_token"],
        refresh_token=raw.get("refresh_token", previous.refresh_token if previous else None),
        expires_at=_expiry(raw),
        account_id=account if account is not None else previous.account_id if previous else None,
    )


async def exchange_code(
    authorization: CodexAuthorization,
    callback_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 30,
) -> CodexTokens:
    """Validate the full callback and exchange once. Own/close the supplied transport."""
    try:
        code = _callback(authorization, callback_url)
    except AuthlibBaseError:
        error = CodexAuthError("missing_code")
    else:
        return await exchange_authorization_code(authorization, code, transport=transport, timeout=timeout)
    raise error


async def exchange_authorization_code(
    authorization: CodexAuthorization,
    code: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 30,
) -> CodexTokens:
    """Exchange a caller-received code with PKCE. The caller validates its callback state.

    Use exchange_code for complete callback URLs with built-in state validation.
    """
    if not isinstance(code, str) or not code.strip():
        raise CodexAuthError("missing_code")
    try:
        async with _client(redirect_uri=authorization.redirect_uri, transport=transport, timeout=timeout) as client:
            raw = await client.fetch_token(
                _EXCHANGE_URL, grant_type="authorization_code", code=code, code_verifier=authorization.code_verifier
            )
        return _tokens(raw)
    except AuthlibBaseError as exc:
        error = CodexAuthError("missing_code" if getattr(exc, "error", None) == "missing_code" else "exchange_rejected")
    except httpx.HTTPError:
        error = CodexAuthError("exchange_transport_error")
    except (ValueError, TypeError, KeyError):
        error = CodexAuthError("invalid_token")
    # Deliberately outside except: no secret-bearing cause/context is retained.
    raise error


async def refresh_tokens(
    tokens: CodexTokens, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30
) -> CodexTokens:
    """Refresh once, explicitly; never mutate tokens, save or replay inference."""
    if tokens.refresh_token is None:
        raise CodexAuthError("no_refresh_token")
    try:
        async with _client(transport=transport, timeout=timeout) as client:
            raw = await client.refresh_token(_EXCHANGE_URL, refresh_token=tokens.refresh_token)
        return _tokens(raw, tokens)
    except AuthlibBaseError:
        error = CodexAuthError("refresh_rejected")
    except httpx.HTTPError:
        error = CodexAuthError("refresh_transport_error")
    except (ValueError, TypeError, KeyError):
        error = CodexAuthError("invalid_token")
    raise error


def read_tokens(path: str | Path) -> CodexTokens:
    """Read only the explicit file, in Republic's format. No default-path discovery."""
    try:
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise CodexAuthError("invalid_token_file")
        return CodexTokens(**raw)
    except OSError:
        error = CodexAuthError("credential_read_failed")
    except (ValueError, TypeError):
        error = CodexAuthError("invalid_token_file")
    raise error


def write_tokens(path: str | Path, tokens: CodexTokens) -> None:
    """Atomically replace the explicit file with mode 0600. Parent must exist."""
    try:
        write_json(path, asdict(tokens))
    except OSError:
        error = CodexAuthError("credential_write_failed")
    else:
        return
    raise error
