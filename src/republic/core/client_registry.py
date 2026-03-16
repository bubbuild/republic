"""Provider-specific client creation helpers."""

from __future__ import annotations

from typing import Any

from any_llm import AnyLLM
from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.utils.aio import async_coro_to_sync_iter

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


class AnyLLMClientAdapter:
    """Normalize client behavior that Republic relies on across providers."""

    def __init__(self, client: AnyLLM) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def messages(self, **kwargs: Any) -> Any:
        if not kwargs.get("stream"):
            return self._client.messages(**kwargs)

        stream_kwargs = dict(kwargs)
        allow_running_loop = stream_kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return async_coro_to_sync_iter(
            self._client.amessages(**stream_kwargs),
            allow_running_loop=allow_running_loop,
        )


def create_anyllm_client(
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    client_args: dict[str, Any],
) -> AnyLLMClientAdapter:
    client: AnyLLM
    if should_use_openai_codex_backend(provider, api_key):
        client = OpenAICodexProvider(
            api_key=api_key,
            api_base=api_base,
            default_instructions=DEFAULT_CODEX_INSTRUCTIONS,
            default_include=DEFAULT_CODEX_INCLUDE,
            default_text=DEFAULT_CODEX_TEXT_CONFIG,
            store=False,
            **client_args,
        )
    elif should_use_github_copilot_backend(provider):
        client = GitHubCopilotProvider(
            api_key=api_key,
            api_base=api_base,
            **client_args,
        )
    else:
        client = AnyLLM.create(provider, api_key=api_key, api_base=api_base, **client_args)
    return AnyLLMClientAdapter(client)


__all__ = [
    "AnyLLMClientAdapter",
    "create_anyllm_client",
]
