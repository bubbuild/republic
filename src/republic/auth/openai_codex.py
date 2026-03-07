"""Authentication helper utilities."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from authlib.integrations.httpx_client import OAuth2Client

_CODEX_PROVIDERS = {"openai"}
# Keep aligned with the official Codex client default (codex-rs core/src/auth.rs::CLIENT_ID).
_DEFAULT_CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_DEFAULT_CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105
_DEFAULT_CODEX_OAUTH_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_DEFAULT_CODEX_OAUTH_SCOPE = "openid profile email offline_access"
_DEFAULT_CODEX_OAUTH_ORIGINATOR = "codex_cli_rs"


class CodexOAuthResponseError(TypeError):
    """Raised when Codex OAuth token response is malformed."""


class CodexOAuthLoginError(RuntimeError):
    """Raised when Codex OAuth login flow cannot complete."""


class CodexOAuthStateMismatchError(CodexOAuthLoginError):
    """Raised when OAuth state validation fails."""


class CodexOAuthMissingCodeError(CodexOAuthLoginError):
    """Raised when OAuth redirect does not include authorization code."""


def _build_oauth_callback_error_message(*, redirect_uri: str, timeout_seconds: float) -> str:
    return (
        "Did not receive OAuth callback. "
        f"redirect_uri={redirect_uri!r}, timeout_seconds={timeout_seconds}. "
        "Possible causes: callback wait timed out, local callback port is unavailable, "
        "or redirect_uri is not a loopback HTTP address. "
        "Try increasing timeout_seconds or use prompt_for_redirect for manual paste."
    )


def codex_cli_api_key_resolver(codex_home: str | Path | None = None) -> Callable[[str], str | None]:
    """Build a provider-scoped resolver that reads Codex CLI OAuth token.

    The resolver only returns a token for provider `openai`.
    It reads from `$CODEX_HOME/auth.json` (default `~/.codex/auth.json`).
    """

    auth_path = _resolve_codex_auth_path(codex_home)

    def _resolver(provider: str) -> str | None:
        if provider not in _CODEX_PROVIDERS:
            return None
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None

        access_token = tokens.get("access_token")
        if not isinstance(access_token, str):
            return None
        token = access_token.strip()
        return token or None

    return _resolver


@dataclass(frozen=True)
class OpenAICodexOAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: int
    account_id: str | None = None


def _resolve_codex_auth_path(codex_home: str | Path | None = None) -> Path:
    if codex_home is None:
        codex_home = os.getenv("CODEX_HOME", "~/.codex")
    return Path(codex_home).expanduser() / "auth.json"


def _parse_tokens(payload: dict[str, Any]) -> OpenAICodexOAuthTokens | None:
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return None

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(access_token, str) or not isinstance(refresh_token, str):
        return None
    access = access_token.strip()
    refresh = refresh_token.strip()
    if not access or not refresh:
        return None

    expires_raw = tokens.get("expires_at")
    if isinstance(expires_raw, (int, float)):
        expires_at = int(expires_raw)
    else:
        # Codex CLI file may not persist explicit expiry.
        # Use last_refresh + 1h or "now + 1h" as best-effort fallback.
        last_refresh_raw = payload.get("last_refresh")
        last_refresh = int(last_refresh_raw) if isinstance(last_refresh_raw, (int, float)) else int(time.time())
        expires_at = last_refresh + 3600

    account_id = tokens.get("account_id")
    if not isinstance(account_id, str):
        account_id = None
    return OpenAICodexOAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        account_id=account_id,
    )


def load_openai_codex_oauth_tokens(codex_home: str | Path | None = None) -> OpenAICodexOAuthTokens | None:
    auth_path = _resolve_codex_auth_path(codex_home)
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_tokens(payload)


def save_openai_codex_oauth_tokens(
    tokens: OpenAICodexOAuthTokens,
    codex_home: str | Path | None = None,
) -> Path:
    auth_path = _resolve_codex_auth_path(codex_home)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    try:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    payload = raw if isinstance(raw, dict) else {}

    tokens_node = payload.get("tokens")
    if not isinstance(tokens_node, dict):
        tokens_node = {}
    tokens_node.update({
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at": tokens.expires_at,
    })
    if tokens.account_id:
        tokens_node["account_id"] = tokens.account_id
    payload["tokens"] = tokens_node
    payload["last_refresh"] = int(time.time())

    auth_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    with suppress(OSError):
        os.chmod(auth_path, 0o600)
    return auth_path


def refresh_openai_codex_oauth_tokens(
    refresh_token: str,
    *,
    timeout_seconds: float = 15.0,
    client_id: str = _DEFAULT_CODEX_OAUTH_CLIENT_ID,
    token_url: str = _DEFAULT_CODEX_OAUTH_TOKEN_URL,
) -> OpenAICodexOAuthTokens:
    with OAuth2Client(client_id=client_id, timeout=timeout_seconds, trust_env=False) as oauth:
        payload = oauth.refresh_token(
            url=token_url,
            refresh_token=refresh_token,
        )
    return _tokens_from_token_payload(payload, account_id=None)


def _build_pkce_pair() -> str:
    verifier = urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    return verifier


def _build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    authorize_url: str,
    scope: str,
    originator: str,
) -> str:
    with OAuth2Client(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge_method="S256",
        trust_env=False,
    ) as oauth:
        url, _ = oauth.create_authorization_url(
            authorize_url,
            state=state,
            code_verifier=code_challenge,
            id_token_add_organizations="true",  # noqa: S106
            codex_cli_simplified_flow="true",
            originator=originator,
        )
    return str(url)


def _extract_code_and_state(input_value: str) -> tuple[str | None, str | None]:
    raw = input_value.strip()
    if not raw:
        return None, None

    parsed = urllib.parse.urlsplit(raw)
    query = urllib.parse.parse_qs(parsed.query)
    code = query.get("code", [None])[0]
    state = query.get("state", [None])[0]
    if isinstance(code, str) or isinstance(state, str):
        return code if isinstance(code, str) else None, state if isinstance(state, str) else None

    if "code=" in raw:
        parsed_qs = urllib.parse.parse_qs(raw)
        code = parsed_qs.get("code", [None])[0]
        state = parsed_qs.get("state", [None])[0]
        return code if isinstance(code, str) else None, state if isinstance(state, str) else None

    return raw, None


def _is_loopback_redirect_uri(redirect_uri: str) -> bool:
    parsed = urllib.parse.urlsplit(redirect_uri)
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


def _wait_for_local_oauth_callback(
    *,
    redirect_uri: str,
    timeout_seconds: float,
) -> tuple[str | None, str | None] | None:
    if not _is_loopback_redirect_uri(redirect_uri):
        return None

    parsed_redirect = urllib.parse.urlsplit(redirect_uri)
    host = parsed_redirect.hostname or "localhost"
    port = parsed_redirect.port
    path = parsed_redirect.path or "/"
    if port is None:
        return None

    lock = threading.Lock()
    state: dict[str, str | None] = {"code": None, "state": None}
    done = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != path:
                self.send_response(404)
                self.end_headers()
                return

            query = urllib.parse.parse_qs(parsed.query)
            code = query.get("code", [None])[0]
            returned_state = query.get("state", [None])[0]
            with lock:
                state["code"] = code if isinstance(code, str) else None
                state["state"] = returned_state if isinstance(returned_state, str) else None
            done.set()

            body = (
                b"<!doctype html><html><body><p>Authentication successful. Return to your terminal.</p></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError:
        return None

    server.timeout = 0.2
    deadline = time.monotonic() + timeout_seconds
    try:
        while not done.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    if not done.is_set():
        return None

    with lock:
        return state["code"], state["state"]


def extract_openai_codex_account_id(access_token: str) -> str | None:
    parts = access_token.split(".")
    if len(parts) != 3:
        return None
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        payload = json.loads(urlsafe_b64decode((payload_segment + padding).encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    auth = payload.get("https://api.openai.com/auth")
    if not isinstance(auth, dict):
        return None
    account_id = auth.get("chatgpt_account_id")
    if not isinstance(account_id, str):
        return None
    normalized = account_id.strip()
    return normalized or None


def _exchange_openai_codex_authorization_code(
    code: str,
    *,
    verifier: str,
    redirect_uri: str,
    timeout_seconds: float,
    client_id: str,
    token_url: str,
) -> OpenAICodexOAuthTokens:
    with OAuth2Client(
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge_method="S256",
        timeout=timeout_seconds,
        trust_env=False,
    ) as oauth:
        payload = oauth.fetch_token(
            url=token_url,
            grant_type="authorization_code",
            code=code,
            code_verifier=verifier,
        )
    account_id = extract_openai_codex_account_id(str(payload.get("access_token", "")))
    return _tokens_from_token_payload(payload, account_id=account_id)


def login_openai_codex_oauth(
    *,
    codex_home: str | Path | None = None,
    prompt_for_redirect: Callable[[str], str] | None = None,
    open_browser: bool = True,
    browser_opener: Callable[[str], Any] | None = None,
    redirect_uri: str = "http://localhost:1455/auth/callback",
    timeout_seconds: float = 300.0,
    client_id: str = _DEFAULT_CODEX_OAUTH_CLIENT_ID,
    authorize_url: str = _DEFAULT_CODEX_OAUTH_AUTHORIZE_URL,
    token_url: str = _DEFAULT_CODEX_OAUTH_TOKEN_URL,
    scope: str = _DEFAULT_CODEX_OAUTH_SCOPE,
    originator: str = _DEFAULT_CODEX_OAUTH_ORIGINATOR,
) -> OpenAICodexOAuthTokens:
    """Run minimal OpenAI Codex OAuth login flow and persist tokens."""

    verifier = _build_pkce_pair()
    state = secrets.token_hex(16)
    oauth_url = _build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=verifier,
        state=state,
        authorize_url=authorize_url,
        scope=scope,
        originator=originator,
    )

    if open_browser:
        opener = browser_opener or webbrowser.open
        opener(oauth_url)

    if prompt_for_redirect is not None:
        callback_input = prompt_for_redirect(oauth_url)
        code, returned_state = _extract_code_and_state(callback_input)
    else:
        callback_values = _wait_for_local_oauth_callback(
            redirect_uri=redirect_uri,
            timeout_seconds=timeout_seconds,
        )
        if callback_values is None:
            message = _build_oauth_callback_error_message(
                redirect_uri=redirect_uri,
                timeout_seconds=timeout_seconds,
            )
            raise CodexOAuthLoginError(message)
        code, returned_state = callback_values

    if returned_state and returned_state != state:
        raise CodexOAuthStateMismatchError
    if not isinstance(code, str) or not code.strip():
        raise CodexOAuthMissingCodeError

    tokens = _exchange_openai_codex_authorization_code(
        code=code.strip(),
        verifier=verifier,
        redirect_uri=redirect_uri,
        timeout_seconds=timeout_seconds,
        client_id=client_id,
        token_url=token_url,
    )
    save_openai_codex_oauth_tokens(tokens, codex_home)
    return tokens


def openai_codex_oauth_resolver(
    codex_home: str | Path | None = None,
    *,
    refresh_skew_seconds: int = 120,
    refresh_timeout_seconds: float = 15.0,
    client_id: str = _DEFAULT_CODEX_OAUTH_CLIENT_ID,
    token_url: str = _DEFAULT_CODEX_OAUTH_TOKEN_URL,
    refresher: Callable[[str], OpenAICodexOAuthTokens] | None = None,
) -> Callable[[str], str | None]:
    """Build a resolver for OpenAI Codex OAuth tokens with auto-refresh."""

    lock = threading.Lock()
    if refresher is None:
        refresher = lambda refresh_token: refresh_openai_codex_oauth_tokens(
            refresh_token,
            timeout_seconds=refresh_timeout_seconds,
            client_id=client_id,
            token_url=token_url,
        )

    def _resolver(provider: str) -> str | None:
        if provider not in _CODEX_PROVIDERS:
            return None
        with lock:
            tokens = load_openai_codex_oauth_tokens(codex_home)
            if tokens is None:
                return None
            now = int(time.time())
            if tokens.expires_at > now + refresh_skew_seconds:
                return tokens.access_token

            try:
                refreshed = refresher(tokens.refresh_token)
            except Exception:
                # Keep serving current token if it has not expired yet.
                if tokens.expires_at > now:
                    return tokens.access_token
                return None

            persisted = OpenAICodexOAuthTokens(
                access_token=refreshed.access_token,
                refresh_token=refreshed.refresh_token,
                expires_at=refreshed.expires_at,
                account_id=refreshed.account_id or tokens.account_id,
            )
            save_openai_codex_oauth_tokens(persisted, codex_home)
            return persisted.access_token

    return _resolver


def _tokens_from_token_payload(
    payload: dict[str, Any],
    *,
    account_id: str | None,
) -> OpenAICodexOAuthTokens:
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access_token, str) or not isinstance(refresh_token, str):
        raise CodexOAuthResponseError
    if not isinstance(expires_in, (int, float)):
        raise CodexOAuthResponseError
    normalized_access = access_token.strip()
    return OpenAICodexOAuthTokens(
        access_token=normalized_access,
        refresh_token=refresh_token.strip(),
        expires_at=int(time.time() + float(expires_in)),
        account_id=account_id or extract_openai_codex_account_id(normalized_access),
    )


__all__ = [
    "CodexOAuthLoginError",
    "CodexOAuthMissingCodeError",
    "CodexOAuthStateMismatchError",
    "OpenAICodexOAuthTokens",
    "codex_cli_api_key_resolver",
    "extract_openai_codex_account_id",
    "load_openai_codex_oauth_tokens",
    "login_openai_codex_oauth",
    "openai_codex_oauth_resolver",
    "refresh_openai_codex_oauth_tokens",
    "save_openai_codex_oauth_tokens",
]
