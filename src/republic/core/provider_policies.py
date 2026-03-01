"""Provider policy decisions shared across request paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPolicy:
    enable_responses_without_capability: bool = False
    include_usage_in_completion_stream: bool = False
    completion_max_tokens_arg: str = "max_tokens"


_DEFAULT_POLICY = ProviderPolicy()
_POLICIES: dict[str, ProviderPolicy] = {
    "openai": ProviderPolicy(
        include_usage_in_completion_stream=True,
        completion_max_tokens_arg="max_completion_tokens",
    ),
    # any-llm supports OpenRouter responses in practice but still reports SUPPORTS_RESPONSES=False.
    "openrouter": ProviderPolicy(
        enable_responses_without_capability=True,
        include_usage_in_completion_stream=True,
    ),
}


def provider_policy(provider_name: str) -> ProviderPolicy:
    lowered = provider_name.lower()
    for key, policy in _POLICIES.items():
        if key in lowered:
            return policy
    return _DEFAULT_POLICY


def should_use_responses(*, provider_name: str, use_responses: bool, supports_responses: bool) -> bool:
    if not use_responses:
        return False
    if supports_responses:
        return True
    return provider_policy(provider_name).enable_responses_without_capability


def should_include_completion_stream_usage(provider_name: str) -> bool:
    return provider_policy(provider_name).include_usage_in_completion_stream


def completion_max_tokens_arg(provider_name: str) -> str:
    return provider_policy(provider_name).completion_max_tokens_arg
