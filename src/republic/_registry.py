from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, TypeVar, Unpack

import httpx2

from republic._errors import ProviderNotFoundError
from republic._models import ChatModel, DecisionModel, EmbeddingModel
from republic.auth import Auth
from republic.formats import ApiFormatName
from republic.history import HistoryProtocol
from republic.providers import (
    ZAI,
    Anthropic,
    AzureOpenAI,
    Codex,
    DeepSeek,
    GitHubCopilot,
    Google,
    Grok,
    Magpie,
    MiniMax,
    Mistral,
    Moonshot,
    Ollama,
    OpenAI,
    OpenRouter,
    Provider,
    Together,
    TypeSafe,
)

P = TypeVar("P", bound=Provider)

_PROVIDERS: dict[str, type[Provider]] = {
    "codex": Codex,
    "github-copilot": GitHubCopilot,
    "openai": OpenAI,
    "anthropic": Anthropic,
    "google": Google,
    "grok": Grok,
    "openrouter": OpenRouter,
    "typesafe": TypeSafe,
    "ollama": Ollama,
    "deepseek": DeepSeek,
    "moonshot": Moonshot,
    "azure-openai": AzureOpenAI,
    "together": Together,
    "zai": ZAI,
    "mistral": Mistral,
    "magpie": Magpie,
    "minimax": MiniMax,
}


class ProviderOptions(TypedDict, total=False):
    api_key: str
    api_base: str
    auth: Auth
    api_format: ApiFormatName
    headers: Mapping[str, str]
    extra_body: Mapping[str, Any]
    env_prefix: str
    http_client: httpx2.AsyncClient
    timeout: httpx2.Timeout | float
    max_retries: int
    retry_delay: float
    max_retry_delay: float


def register_provider(provider_class: type[P], name: str | None = None) -> type[P]:
    """Make ``provider_class`` available as ``name`` in model specs such as ``"name:model"``."""
    _PROVIDERS[name or provider_class.name] = provider_class
    return provider_class


def all_providers() -> Sequence[str]:
    """The names usable in ``"provider:model"`` specs, including registered custom providers, sorted."""
    return sorted(_PROVIDERS)


def get_provider(name: str, **options: Unpack[ProviderOptions]) -> Provider:
    """Create the provider registered as ``name``.

    Environment variables are read with the ``REPUBLIC_{NAME}`` prefix, with hyphens as underscores, unless
    ``env_prefix`` is given.
    """
    try:
        provider_class = _PROVIDERS[name]
    except KeyError:
        raise ProviderNotFoundError(f"No provider registered as {name!r}; known: {all_providers()}") from None
    options.setdefault("env_prefix", _env_prefix(name))
    return provider_class(**options)


def get_model(spec: str, *, history: HistoryProtocol | None = None, **options: Unpack[ProviderOptions]) -> ChatModel:
    """Create a chat model from a ``"provider:model"`` spec."""
    provider_name, model_name = _split_spec(spec)
    return get_provider(provider_name, **options).get_model(model_name, history=history)


def get_embedding_model(spec: str, **options: Unpack[ProviderOptions]) -> EmbeddingModel:
    """Create an embedding model from a ``"provider:model"`` spec."""
    provider_name, model_name = _split_spec(spec)
    return get_provider(provider_name, **options).get_embedding_model(model_name)


def get_decision_model(spec: str, **options: Unpack[ProviderOptions]) -> DecisionModel:
    """Create a decision model from a ``"provider:model"`` spec."""
    provider_name, model_name = _split_spec(spec)
    return get_provider(provider_name, **options).get_decision_model(model_name)


def _env_prefix(name: str) -> str:
    return f"REPUBLIC_{name.upper().replace('-', '_')}"


def _split_spec(spec: str) -> tuple[str, str]:
    provider_name, separator, model_name = spec.partition(":")
    if not separator or not model_name:
        raise ValueError(f"Expected a 'provider:model' spec, got {spec!r}")
    return provider_name, model_name
