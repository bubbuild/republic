"""Helpers for GitHub Copilot OAuth-backed sessions."""

from __future__ import annotations

from typing import Any

from any_llm.constants import INSIDE_NOTEBOOK

from republic.clients._async_bridge import threaded_async_call_to_sync_iter

DEFAULT_GITHUB_COPILOT_API_BASE = "https://models.github.ai/inference"
DEFAULT_GITHUB_API_VERSION = "2022-11-28"


class GitHubCopilotClient:
    SUPPORTS_RESPONSES = False

    def __init__(self, base_client: Any) -> None:
        self._base_client = base_client
        self.client = getattr(base_client, "client", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_client, name)

    def completion(
        self,
        *,
        allow_running_loop: bool = INSIDE_NOTEBOOK,
        **kwargs: Any,
    ) -> Any:
        if not kwargs.get("stream"):
            return self._base_client.completion(
                allow_running_loop=allow_running_loop,
                **kwargs,
            )

        del allow_running_loop
        return threaded_async_call_to_sync_iter(self._base_client.acompletion(**kwargs))

    async def acompletion(self, **kwargs: Any) -> Any:
        return await self._base_client.acompletion(**kwargs)


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
    "GitHubCopilotClient",
    "build_github_copilot_default_headers",
    "resolve_github_copilot_api_base",
    "should_use_github_copilot_backend",
]
