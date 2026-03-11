"""Provider-specific client builders used by execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from republic.clients.github_copilot import (
    GitHubCopilotClient,
    build_github_copilot_default_headers,
    resolve_github_copilot_api_base,
    should_use_github_copilot_backend,
)
from republic.clients.openai_codex import (
    DEFAULT_CODEX_INCLUDE,
    DEFAULT_CODEX_INSTRUCTIONS,
    DEFAULT_CODEX_TEXT_CONFIG,
    OpenAICodexResponsesClient,
    build_openai_codex_default_headers,
    resolve_openai_codex_api_base,
    should_use_openai_codex_backend,
)


@dataclass(frozen=True)
class ProviderClientBuildContext:
    provider: str
    api_key: str | None
    api_base: str | None
    client_args: dict[str, Any]
    create_client: Callable[..., Any]


@dataclass(frozen=True)
class ProviderClientFactory:
    matches: Callable[[ProviderClientBuildContext], bool]
    build: Callable[[ProviderClientBuildContext], Any]


def build_special_client(
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    client_args: dict[str, Any],
    create_client: Callable[..., Any],
) -> Any | None:
    context = ProviderClientBuildContext(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        client_args=client_args,
        create_client=create_client,
    )
    for factory in _SPECIAL_CLIENT_FACTORIES:
        if factory.matches(context):
            return factory.build(context)
    return None


def _build_openai_codex_client(context: ProviderClientBuildContext) -> Any:
    default_headers = dict(context.client_args.get("default_headers", {}))
    default_headers.update(build_openai_codex_default_headers(context.api_key or ""))
    base_client = context.create_client(
        context.provider,
        api_key=context.api_key,
        api_base=resolve_openai_codex_api_base(context.api_base),
        **{
            **context.client_args,
            "default_headers": default_headers,
        },
    )
    return OpenAICodexResponsesClient(
        base_client,
        default_instructions=DEFAULT_CODEX_INSTRUCTIONS,
        default_include=DEFAULT_CODEX_INCLUDE,
        default_text=DEFAULT_CODEX_TEXT_CONFIG,
        store=False,
    )


def _build_github_copilot_client(context: ProviderClientBuildContext) -> Any:
    client_args = {
        key: value
        for key, value in context.client_args.items()
        if key not in {"cli_path", "cli_url", "use_stdio", "log_level", "config_dir", "session_timeout"}
    }
    default_headers = build_github_copilot_default_headers()
    default_headers.update(dict(client_args.get("default_headers", {})))
    base_client = context.create_client(
        "openai",
        api_key=context.api_key,
        api_base=resolve_github_copilot_api_base(context.api_base),
        **{
            **client_args,
            "default_headers": default_headers,
        },
    )
    return GitHubCopilotClient(base_client)


def _matches_openai_codex(context: ProviderClientBuildContext) -> bool:
    return should_use_openai_codex_backend(context.provider, context.api_key)


def _matches_github_copilot(context: ProviderClientBuildContext) -> bool:
    return should_use_github_copilot_backend(context.provider)


_SPECIAL_CLIENT_FACTORIES: tuple[ProviderClientFactory, ...] = (
    ProviderClientFactory(
        matches=_matches_openai_codex,
        build=_build_openai_codex_client,
    ),
    ProviderClientFactory(
        matches=_matches_github_copilot,
        build=_build_github_copilot_client,
    ),
)


__all__ = [
    "ProviderClientBuildContext",
    "ProviderClientFactory",
    "build_special_client",
]
