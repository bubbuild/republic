"""Provider-specific client creation helpers."""

from __future__ import annotations

from typing import Any

from any_llm import AnyLLM

from republic.clients.github_copilot import (
    GitHubCopilotProvider,
    should_use_github_copilot_backend,
)
from republic.clients.openai_codex import (
    DEFAULT_CODEX_INCLUDE,
    DEFAULT_CODEX_INSTRUCTIONS,
    DEFAULT_CODEX_TEXT_CONFIG,
    OpenAICodexProvider,
    should_use_openai_codex_backend,
)


def create_anyllm_client(
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    client_args: dict[str, Any],
) -> AnyLLM:
    if should_use_openai_codex_backend(provider, api_key):
        return OpenAICodexProvider(
            api_key=api_key,
            api_base=api_base,
            default_instructions=DEFAULT_CODEX_INSTRUCTIONS,
            default_include=DEFAULT_CODEX_INCLUDE,
            default_text=DEFAULT_CODEX_TEXT_CONFIG,
            store=False,
            **client_args,
        )
    if should_use_github_copilot_backend(provider):
        return GitHubCopilotProvider(
            api_key=api_key,
            api_base=api_base,
            **client_args,
        )
    return AnyLLM.create(provider, api_key=api_key, api_base=api_base, **client_args)


__all__ = [
    "create_anyllm_client",
]
