from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, Unpack

import httpx

from republic._errors import ProviderNotFoundError
from republic._formats import ApiFormatName
from republic._models import ChatModel, DecisionModel, EmbeddingModel
from republic.history import HistoryProtocol
from republic.providers import Anthropic, Google, OpenAI, OpenRouter, Provider, TypeSafe

_PROVIDERS: dict[str, type[Provider]] = {
    "openai": OpenAI,
    "anthropic": Anthropic,
    "google": Google,
    "openrouter": OpenRouter,
    "typesafe": TypeSafe,
}


class ProviderOptions(TypedDict, total=False):
    api_key: str
    api_base: str
    auth: httpx.Auth
    api_format: ApiFormatName
    headers: Mapping[str, str]
    env_prefix: str
    http_client: httpx.AsyncClient
    timeout: httpx.Timeout | float


def register_provider(provider_class: type[Provider], name: str) -> None:
    """Make ``provider_class`` available as ``name`` in model specs such as ``"name:model"``."""
    _PROVIDERS[name] = provider_class


def get_provider(name: str, **options: Unpack[ProviderOptions]) -> Provider:
    """Create the provider registered as ``name``.

    Environment variables are read with the ``REPUBLIC_{NAME}`` prefix unless
    ``env_prefix`` is given.
    """
    try:
        provider_class = _PROVIDERS[name]
    except KeyError:
        raise ProviderNotFoundError(f"No provider registered as {name!r}; known: {sorted(_PROVIDERS)}") from None
    options.setdefault("env_prefix", f"REPUBLIC_{name.upper()}")
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


def _split_spec(spec: str) -> tuple[str, str]:
    provider_name, separator, model_name = spec.partition(":")
    if not separator or not model_name:
        raise ValueError(f"Expected a 'provider:model' spec, got {spec!r}")
    return provider_name, model_name
