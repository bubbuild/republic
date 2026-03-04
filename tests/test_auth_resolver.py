from __future__ import annotations

import pytest

import republic.core.execution as execution
from republic import (
    LLM,
    login_openai_codex_oauth,
    openai_codex_oauth_resolver,
)
from republic.auth.openai_codex import (
    CodexOAuthLoginError,
    CodexOAuthMissingCodeError,
    CodexOAuthStateMismatchError,
    OpenAICodexOAuthTokens,
    codex_cli_api_key_resolver,
    save_openai_codex_oauth_tokens,
)

from .fakes import FakeAnyLLMFactory, make_response


def test_llm_uses_api_key_resolver_when_api_key_is_missing(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []
    factory = FakeAnyLLMFactory()

    def _create(provider: str, **kwargs: object):
        created.append((provider, dict(kwargs)))
        return factory.create(provider, **kwargs)

    monkeypatch.setattr(execution.AnyLLM, "create", _create)

    client = factory.ensure("openai")
    client.queue_completion(make_response(text="ok"))

    llm = LLM(
        model="openai:gpt-5.3-codex",
        api_key_resolver=lambda provider: "oauth-token" if provider == "openai" else None,
    )
    assert llm.chat("hello") == "ok"
    assert created[0][0] == "openai"
    assert created[0][1]["api_key"] == "oauth-token"


def test_explicit_api_key_has_priority_over_resolver(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []
    factory = FakeAnyLLMFactory()

    def _create(provider: str, **kwargs: object):
        created.append((provider, dict(kwargs)))
        return factory.create(provider, **kwargs)

    monkeypatch.setattr(execution.AnyLLM, "create", _create)

    client = factory.ensure("openai")
    client.queue_completion(make_response(text="ok"))

    llm = LLM(
        model="openai:gpt-5.3-codex",
        api_key={"openai": "explicit-key"},
        api_key_resolver=lambda _: "oauth-token",
    )
    assert llm.chat("hello") == "ok"
    assert created[0][1]["api_key"] == "explicit-key"


def test_provider_map_falls_back_to_resolver_for_missing_provider(monkeypatch) -> None:
    created: list[tuple[str, dict[str, object]]] = []
    factory = FakeAnyLLMFactory()

    def _create(provider: str, **kwargs: object):
        created.append((provider, dict(kwargs)))
        return factory.create(provider, **kwargs)

    monkeypatch.setattr(execution.AnyLLM, "create", _create)

    client = factory.ensure("openai")
    client.queue_completion(make_response(text="ok"))

    llm = LLM(
        model="openai:gpt-5.3-codex",
        api_key={"anthropic": "anthropic-key"},
        api_key_resolver=lambda provider: "oauth-token" if provider == "openai" else None,
    )
    assert llm.chat("hello") == "ok"
    assert created[0][1]["api_key"] == "oauth-token"


def test_codex_cli_api_key_resolver_reads_access_token(tmp_path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens": {"access_token": "  token-123  "}}', encoding="utf-8")

    resolver = codex_cli_api_key_resolver(tmp_path)
    assert resolver("openai") == "token-123"
    assert resolver("openai-codex") is None
    assert resolver("anthropic") is None


def test_openai_codex_oauth_resolver_refreshes_expiring_token(tmp_path) -> None:
    save_openai_codex_oauth_tokens(
        OpenAICodexOAuthTokens(
            access_token="old-token",  # noqa: S106
            refresh_token="refresh-1",  # noqa: S106
            expires_at=1,  # expired
            account_id="acct-1",
        ),
        tmp_path,
    )

    calls: list[str] = []

    def _refresher(refresh_token: str) -> OpenAICodexOAuthTokens:
        calls.append(refresh_token)
        return OpenAICodexOAuthTokens(
            access_token="new-token",  # noqa: S106
            refresh_token="refresh-2",  # noqa: S106
            expires_at=4_102_444_800,  # 2100-01-01
        )

    resolver = openai_codex_oauth_resolver(tmp_path, refresher=_refresher)
    assert resolver("openai") == "new-token"
    assert calls == ["refresh-1"]

    # Should persist refreshed token and avoid another refresh.
    assert resolver("openai") == "new-token"
    assert calls == ["refresh-1"]


def test_openai_codex_oauth_resolver_returns_none_when_expired_and_refresh_fails(tmp_path) -> None:
    save_openai_codex_oauth_tokens(
        OpenAICodexOAuthTokens(
            access_token="old-token",  # noqa: S106
            refresh_token="refresh-1",  # noqa: S106
            expires_at=1,  # expired
        ),
        tmp_path,
    )

    resolver = openai_codex_oauth_resolver(
        tmp_path,
        refresher=lambda _: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    assert resolver("openai") is None


def test_openai_codex_oauth_resolver_uses_current_token_if_refresh_fails_but_not_expired(tmp_path) -> None:
    save_openai_codex_oauth_tokens(
        OpenAICodexOAuthTokens(
            access_token="still-valid",  # noqa: S106
            refresh_token="refresh-1",  # noqa: S106
            expires_at=4_102_444_800,  # far future
        ),
        tmp_path,
    )

    resolver = openai_codex_oauth_resolver(
        tmp_path,
        refresh_skew_seconds=4_102_444_799,
        refresher=lambda _: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    assert resolver("openai") == "still-valid"


def test_login_openai_codex_oauth_success_persists_tokens(monkeypatch, tmp_path) -> None:
    exchange_calls: list[tuple[str, str, str, float, str, str]] = []

    def _exchange(
        code: str,
        *,
        verifier: str,
        redirect_uri: str,
        timeout_seconds: float,
        client_id: str,
        token_url: str,
    ) -> OpenAICodexOAuthTokens:
        exchange_calls.append((code, verifier, redirect_uri, timeout_seconds, client_id, token_url))
        return OpenAICodexOAuthTokens(
            access_token="access-token",  # noqa: S106
            refresh_token="refresh-token",  # noqa: S106
            expires_at=4_102_444_800,
            account_id="acct-1",
        )

    monkeypatch.setattr("republic.auth.openai_codex._exchange_openai_codex_authorization_code", _exchange)
    monkeypatch.setattr("republic.auth.openai_codex.secrets.token_hex", lambda _: "state-fixed")

    opened: list[str] = []

    def _open(url: str):
        opened.append(url)
        return True

    def _prompt(url: str) -> str:
        assert "state=state-fixed" in url
        return "http://127.0.0.1:1455/auth/callback?code=auth-code&state=state-fixed"

    tokens = login_openai_codex_oauth(
        codex_home=tmp_path,
        prompt_for_redirect=_prompt,
        browser_opener=_open,
    )

    expected_access_token = "access-token"  # noqa: S105
    assert tokens.access_token == expected_access_token
    assert len(exchange_calls) == 1
    assert exchange_calls[0][0] == "auth-code"
    assert opened

    resolver = codex_cli_api_key_resolver(tmp_path)
    assert resolver("openai") == expected_access_token


def test_login_openai_codex_oauth_raises_on_state_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "republic.auth.openai_codex._exchange_openai_codex_authorization_code",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr("republic.auth.openai_codex.secrets.token_hex", lambda _: "state-fixed")

    def _prompt(_: str) -> str:
        return "http://127.0.0.1:1455/auth/callback?code=auth-code&state=wrong"

    with pytest.raises(CodexOAuthStateMismatchError):
        login_openai_codex_oauth(
            codex_home=tmp_path,
            prompt_for_redirect=_prompt,
            open_browser=False,
        )


def test_login_openai_codex_oauth_raises_on_missing_code(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "republic.auth.openai_codex._exchange_openai_codex_authorization_code",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr("republic.auth.openai_codex.secrets.token_hex", lambda _: "state-fixed")

    def _prompt(_: str) -> str:
        return "http://127.0.0.1:1455/auth/callback?state=state-fixed"

    with pytest.raises(CodexOAuthMissingCodeError):
        login_openai_codex_oauth(
            codex_home=tmp_path,
            prompt_for_redirect=_prompt,
            open_browser=False,
        )


def test_login_openai_codex_oauth_uses_local_callback_without_prompt(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("republic.auth.openai_codex.secrets.token_hex", lambda _: "state-fixed")
    monkeypatch.setattr(
        "republic.auth.openai_codex._wait_for_local_oauth_callback",
        lambda **_: ("auth-code", "state-fixed"),
    )

    def _exchange(
        code: str,
        *,
        verifier: str,
        redirect_uri: str,
        timeout_seconds: float,
        client_id: str,
        token_url: str,
    ) -> OpenAICodexOAuthTokens:
        assert code == "auth-code"
        return OpenAICodexOAuthTokens(
            access_token="access-token",  # noqa: S106
            refresh_token="refresh-token",  # noqa: S106
            expires_at=4_102_444_800,
        )

    monkeypatch.setattr("republic.auth.openai_codex._exchange_openai_codex_authorization_code", _exchange)

    tokens = login_openai_codex_oauth(
        codex_home=tmp_path,
        prompt_for_redirect=None,
        open_browser=False,
    )
    assert tokens.access_token == "access-token"  # noqa: S105


def test_login_openai_codex_oauth_raises_without_prompt_and_without_callback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("republic.auth.openai_codex.secrets.token_hex", lambda _: "state-fixed")
    monkeypatch.setattr("republic.auth.openai_codex._wait_for_local_oauth_callback", lambda **_: None)

    with pytest.raises(CodexOAuthLoginError, match="Did not receive OAuth callback"):
        login_openai_codex_oauth(
            codex_home=tmp_path,
            prompt_for_redirect=None,
            open_browser=False,
        )
