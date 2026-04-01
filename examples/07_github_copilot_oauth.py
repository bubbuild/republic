from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from republic import LLM, github_copilot_oauth_resolver, login_github_copilot_oauth
from republic.auth.github_copilot import (
    load_github_cli_oauth_token,
    load_github_cli_oauth_token_via_command,
)


class SmokeTestError(RuntimeError):
    @classmethod
    def unexpected_post_url(cls, url: str) -> SmokeTestError:
        return cls(f"Unexpected POST url: {url}")

    @classmethod
    def unexpected_get_url(cls, url: str) -> SmokeTestError:
        return cls(f"Unexpected GET url: {url}")

    @classmethod
    def missing_live_token(cls) -> SmokeTestError:
        return cls("No GitHub OAuth token found in env, gh config, or `gh auth token`.")


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return dict(self._payload)

    def raise_for_status(self) -> None:
        return None


class FakeHTTPClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __enter__(self) -> FakeHTTPClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any] | None = None, headers=None):
        del json, headers
        if url.endswith("/login/device/code"):
            return FakeHTTPResponse({
                "device_code": "device-1",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "interval": 1,
                "expires_in": 900,
            })
        if url.endswith("/login/oauth/access_token"):
            return FakeHTTPResponse({
                "access_token": "gho_mock_access",
                "token_type": "bearer",
                "scope": "read:user user:email",
            })
        raise SmokeTestError.unexpected_post_url(url)

    def get(self, url: str, *, headers=None):
        del headers
        if url.endswith("/user"):
            return FakeHTTPResponse({
                "id": 7,
                "login": "psiace",
                "email": "psiace@example.com",
            })
        raise SmokeTestError.unexpected_get_url(url)


class FakeGitHubModelsClient:
    def completion(self, **_: Any) -> Any:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="mock-ok",
                        tool_calls=[],
                    )
                )
            ],
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    async def acompletion(self, **kwargs: Any) -> Any:
        return self.completion(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test GitHub Copilot OAuth support in Republic.")
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "live"),
        default="auto",
        help="`mock` runs a fully mocked end-to-end flow; `live` uses your current GitHub token source.",
    )
    parser.add_argument(
        "--model",
        default="github-copilot:gpt-4.1",
        help="Model to use for the chat request.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly OK.",
        help="Prompt to send in live mode.",
    )
    return parser.parse_args()


def _run_mock(model: str) -> None:
    import republic.auth.github_copilot as auth_module
    import republic.providers.github_copilot as github_provider_module

    original_http_client = auth_module.httpx.Client
    original_post_chat = github_provider_module.GitHubCopilotBackend._post_chat

    def _post_chat(self, request):
        del self, request
        return {
            "choices": [
                {
                    "message": {
                        "content": "mock-ok",
                        "tool_calls": [],
                    }
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

    auth_module.httpx.Client = FakeHTTPClient
    github_provider_module.GitHubCopilotBackend._post_chat = _post_chat

    try:
        with tempfile.TemporaryDirectory(prefix="republic-copilot-smoke-") as temp_dir:
            config_home = Path(temp_dir)
            tokens = login_github_copilot_oauth(
                config_home=config_home,
                open_browser=False,
                device_code_notifier=lambda verification_uri, user_code: print(
                    f"mock device flow: open {verification_uri} code={user_code}"
                ),
            )
            print("mock login: ok")
            print("mock github token stored:", bool(tokens.github_token))

            llm = LLM(
                model=model,
                api_key_resolver=github_copilot_oauth_resolver(config_home),
            )
            text = llm.chat("Reply with exactly OK.")
            print("mock chat:", text)
    finally:
        auth_module.httpx.Client = original_http_client
        github_provider_module.GitHubCopilotBackend._post_chat = original_post_chat


def _run_live(model: str, prompt: str) -> None:
    if not _has_live_token():
        raise SmokeTestError.missing_live_token()

    llm = LLM(
        model=model,
        api_key_resolver=github_copilot_oauth_resolver(),
    )
    text = llm.chat(prompt, max_tokens=32)
    print("live chat:", text)


def _has_live_token() -> bool:
    return load_github_cli_oauth_token() is not None or load_github_cli_oauth_token_via_command() is not None


def main() -> None:
    args = parse_args()
    mode = args.mode
    if mode == "auto":
        mode = "live" if _has_live_token() else "mock"

    print("mode:", mode)
    if mode == "mock":
        _run_mock(args.model)
        return
    _run_live(args.model, args.prompt)


if __name__ == "__main__":
    main()
