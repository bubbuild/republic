"""Provider capability decisions shared across request paths."""

from __future__ import annotations

from dataclasses import dataclass

from any_llm import AnyLLM
from any_llm.exceptions import UnsupportedProviderError
from any_llm.types.provider import ProviderMetadata

from republic.clients.github_copilot import GitHubCopilotProvider


@dataclass(frozen=True)
class ProviderPolicy:
    include_usage_in_completion_stream: bool = False
    metadata: ProviderMetadata | None = None


_DEFAULT_POLICY = ProviderPolicy()
_POLICIES: dict[str, ProviderPolicy] = {
    "github-copilot": ProviderPolicy(
        include_usage_in_completion_stream=True,
        metadata=GitHubCopilotProvider.get_provider_metadata(),
    ),
    # Stream usage is not represented in any-llm provider metadata. Keep this as
    # a narrow default for providers whose SDK path accepts OpenAI stream_options.
    "openai": ProviderPolicy(
        include_usage_in_completion_stream=True,
    ),
    "openrouter": ProviderPolicy(include_usage_in_completion_stream=True),
}


def _normalize_provider_name(provider_name: str) -> str:
    return provider_name.strip().lower()


def provider_policy(provider_name: str) -> ProviderPolicy:
    return _POLICIES.get(_normalize_provider_name(provider_name), _DEFAULT_POLICY)


def provider_metadata(provider_name: str) -> ProviderMetadata | None:
    normalized_provider = _normalize_provider_name(provider_name)
    local_metadata = provider_policy(normalized_provider).metadata
    if local_metadata is not None:
        return local_metadata
    try:
        return AnyLLM.get_provider_class(normalized_provider).get_provider_metadata()
    except (AttributeError, ImportError, UnsupportedProviderError):
        return None


def responses_rejection_reason(
    *,
    provider_name: str,
    model_id: str,
    supports_responses: bool,
) -> str | None:
    if supports_responses:
        return None
    metadata = provider_metadata(provider_name)
    if metadata is not None and metadata.responses:
        return None
    return "responses format is not supported by this provider"


def supports_messages_format(*, provider_name: str, model_id: str) -> bool:
    metadata = provider_metadata(provider_name)
    if metadata is not None:
        return metadata.messages
    return model_id.strip().lower().startswith("anthropic/")


def should_include_completion_stream_usage(provider_name: str) -> bool:
    return provider_policy(provider_name).include_usage_in_completion_stream
