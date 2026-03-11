"""Helpers for GitHub Copilot OAuth-backed sessions."""

from __future__ import annotations

from typing import Any

from any_llm.providers.openai.base import BaseOpenAIProvider

DEFAULT_GITHUB_COPILOT_API_BASE = "https://models.github.ai/inference"
DEFAULT_GITHUB_API_VERSION = "2022-11-28"


class GitHubCopilotProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "github-copilot"
    PROVIDER_DOCUMENTATION_URL = "https://docs.github.com/en/copilot"
    ENV_API_KEY_NAME = "GITHUB_TOKEN"
    API_BASE = DEFAULT_GITHUB_COPILOT_API_BASE
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        client_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"cli_path", "cli_url", "use_stdio", "log_level", "config_dir", "session_timeout"}
        }
        default_headers = build_github_copilot_default_headers()
        default_headers.update(dict(client_kwargs.pop("default_headers", {})))
        super().__init__(
            api_key=api_key,
            api_base=resolve_github_copilot_api_base(api_base),
            default_headers=default_headers,
            **client_kwargs,
        )


def should_use_github_copilot_backend(provider: str) -> bool:
    return provider == "github-copilot"


def resolve_github_copilot_api_base(api_base: str | None) -> str:
    raw = (api_base or DEFAULT_GITHUB_COPILOT_API_BASE).rstrip("/")
    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")]
    if raw.endswith("/inference"):
        return raw
    if raw.endswith("models.github.ai"):
        return f"{raw}/inference"
    return raw


def build_github_copilot_default_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": DEFAULT_GITHUB_API_VERSION,
    }


__all__ = [
    "DEFAULT_GITHUB_API_VERSION",
    "DEFAULT_GITHUB_COPILOT_API_BASE",
    "GitHubCopilotProvider",
    "build_github_copilot_default_headers",
    "resolve_github_copilot_api_base",
    "should_use_github_copilot_backend",
]
