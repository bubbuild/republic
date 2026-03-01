"""Provider policy decisions shared across request paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPolicy:
    enable_responses_without_capability: bool = False
    include_usage_in_completion_stream: bool = False
    completion_max_tokens_arg: str = "max_tokens"
    responses_tools_blocked_model_prefixes: tuple[str, ...] = ()


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
        responses_tools_blocked_model_prefixes=("anthropic/",),
    ),
}


def _normalize_provider_name(provider_name: str) -> str:
    return provider_name.strip().lower()


def provider_policy(provider_name: str) -> ProviderPolicy:
    return _POLICIES.get(_normalize_provider_name(provider_name), _DEFAULT_POLICY)


def _responses_tools_blocked_for_model(provider_name: str, model_id: str) -> bool:
    policy = provider_policy(provider_name)
    lowered_model = model_id.strip().lower()
    return any(lowered_model.startswith(prefix) for prefix in policy.responses_tools_blocked_model_prefixes)


def should_attempt_responses(
    *,
    provider_name: str,
    model_id: str,
    has_tools: bool,
    supports_responses: bool,
) -> bool:
    if has_tools and _responses_tools_blocked_for_model(provider_name, model_id):
        return False
    if supports_responses:
        return True
    return provider_policy(provider_name).enable_responses_without_capability


def responses_rejection_reason(
    *,
    provider_name: str,
    model_id: str,
    has_tools: bool,
    supports_responses: bool,
) -> str | None:
    if has_tools and _responses_tools_blocked_for_model(provider_name, model_id):
        return "responses format is not supported for this model when tools are enabled"
    if supports_responses:
        return None
    if provider_policy(provider_name).enable_responses_without_capability:
        return None
    return "responses format is not supported by this provider"


def supports_anthropic_messages_format(*, provider_name: str, model_id: str) -> bool:
    normalized_provider = _normalize_provider_name(provider_name)
    normalized_model = model_id.strip().lower()
    return normalized_provider == "anthropic" or normalized_model.startswith("anthropic/")


def should_include_completion_stream_usage(provider_name: str) -> bool:
    return provider_policy(provider_name).include_usage_in_completion_stream


def completion_max_tokens_arg(provider_name: str) -> str:
    return provider_policy(provider_name).completion_max_tokens_arg
