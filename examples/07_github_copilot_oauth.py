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
        if url.endswith("/login/device/code"):
            return FakeHTTPResponse(
                {
                    "device_code": "device-1",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                }
            )
        if url.endswith("/login/oauth/access_token"):
            return FakeHTTPResponse(
                {
                    "access_token": "gho_mock_access",
                    "token_type": "bearer",
                    "scope": "read:user user:email",
                }
            )
        raise RuntimeError(f"Unexpected POST url: {url}")

    def get(self, url: str, *, headers=None):
        if url.endswith("/user"):
            return FakeHTTPResponse(
                {
                    "id": 7,
                    "login": "psiace",
                    "email": "psiace@example.com",
                }
            )
        raise RuntimeError(f"Unexpected GET url: {url}")


class FakeSessionEventType:
    ASSISTANT_MESSAGE_DELTA = "assistant.message.delta"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_USAGE = "assistant.usage"
    SESSION_ERROR = "session.error"
    SESSION_IDLE = "session.idle"
    EXTERNAL_TOOL_REQUESTED = "external_tool.requested"


class FakePermissionRequestResult:
    def __init__(self, kind: str = "denied-no-approval-rule-and-could-not-request-from-user") -> None:
        self.kind = kind


class FakeCopilotSession:
    def __init__(self) -> None:
        self._handler = None

    def on(self, handler):
        self._handler = handler

        def _unsubscribe():
            self._handler = None

        return _unsubscribe

    async def send_and_wait(self, options: dict[str, Any], timeout: float | None = None):
        if self._handler is not None:
            self._handler(SimpleNamespace(type=FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, data=SimpleNamespace(delta_content="mock-")))
            self._handler(SimpleNamespace(type=FakeSessionEventType.ASSISTANT_MESSAGE_DELTA, data=SimpleNamespace(delta_content="ok")))
            self._handler(SimpleNamespace(type=FakeSessionEventType.ASSISTANT_USAGE, data=SimpleNamespace(input_tokens=3, output_tokens=2)))
            final = SimpleNamespace(type=FakeSessionEventType.ASSISTANT_MESSAGE, data=SimpleNamespace(content="mock-ok"))
            self._handler(final)
            self._handler(SimpleNamespace(type=FakeSessionEventType.SESSION_IDLE, data=SimpleNamespace()))
            return final
        return None

    async def destroy(self) -> None:
        return None

    async def send(self, options: dict[str, Any]) -> str:
        await self.send_and_wait(options)
        return "msg_1"

    async def abort(self) -> None:
        return None


class FakeCopilotClient:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = dict(options)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def create_session(self, config: dict[str, Any]) -> FakeCopilotSession:
        return FakeCopilotSession()


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
    import republic.clients.github_copilot as client_module

    original_http_client = auth_module.httpx.Client
    original_sdk_loader = client_module._load_copilot_sdk
    auth_module.httpx.Client = FakeHTTPClient
    client_module._load_copilot_sdk = lambda: SimpleNamespace(
        client_type=FakeCopilotClient,
        permission_result_type=FakePermissionRequestResult,
        event_type=FakeSessionEventType,
        tool_type=SimpleNamespace,
    )

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
        client_module._load_copilot_sdk = original_sdk_loader


def _run_live(model: str, prompt: str) -> None:
    if not _has_live_token():
        raise RuntimeError("No GitHub OAuth token found in env, gh config, or `gh auth token`.")

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
