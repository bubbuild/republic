from __future__ import annotations

import importlib.metadata
from typing import Any

import pytest

import republic.core.execution as execution
from republic import LLM
from republic.core.errors import ErrorKind
from republic.core.execution import TransportResponse
from republic.core.results import RepublicError
from republic.providers.openai import OpenAIBackend
from republic.providers.runtime import ProviderRuntime
from republic.providers.types import (
    MESSAGES_TRANSPORTS,
    ChatRequest,
    ProviderBackend,
    ProviderCapabilities,
    ProviderContext,
    ProviderHook,
)


class _MessagesOnlyBackend(ProviderBackend):
    capabilities = ProviderCapabilities(transports=MESSAGES_TRANSPORTS)

    def chat(self, request) -> TransportResponse:
        return TransportResponse(
            transport="messages",
            payload={"choices": [{"message": {"content": "ok"}}]},
        )

    async def achat(self, request) -> TransportResponse:
        return self.chat(request)


class _MessagesOnlyHook(ProviderHook):
    name = "custom"

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        return _MessagesOnlyBackend()


class _RuntimeWithCustomMessagesProvider:
    def provider_for(self, provider_name: str) -> ProviderHook | None:
        if provider_name == "custom":
            return _MessagesOnlyHook()
        return None


def test_runtime_ignores_broken_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenEntryPoint:
        name = "broken"

        def load(self) -> object:
            raise ImportError("boom")

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group=None: [_BrokenEntryPoint()] if group == "republic.providers" else [],
    )

    runtime = ProviderRuntime()
    provider = runtime.provider_for("openai")

    assert provider is not None
    assert provider.name == "openai"


def test_runtime_maps_openrouter_to_openai_backend() -> None:
    runtime = ProviderRuntime()

    provider = runtime.provider_for("openrouter")

    assert provider is not None
    assert provider.name == "openai"


@pytest.mark.parametrize("provider_name", ["gemini", "mistral"])
def test_runtime_loads_builtin_native_http_providers(provider_name: str) -> None:
    runtime = ProviderRuntime()

    provider = runtime.provider_for(provider_name)

    assert provider is not None
    assert provider.name == provider_name


def test_openrouter_backend_supports_messages_transport() -> None:
    backend = OpenAIBackend(
        ProviderContext(
            provider="openrouter",
            api_key="dummy",
            api_base="https://openrouter.ai/api/v1",
        )
    )

    assert backend.capabilities.transports == frozenset({"completion", "responses", "messages"})
    assert backend.capabilities.completion_max_tokens_arg == "max_tokens"


def test_openrouter_backend_defaults_to_openrouter_api_base() -> None:
    backend = OpenAIBackend(
        ProviderContext(
            provider="openrouter",
            api_key="dummy",
            api_base=None,
        )
    )

    assert backend._resolved_api_base() == "https://openrouter.ai/api/v1"


def test_openai_backend_omits_null_responses_tools() -> None:
    captured: dict[str, Any] = {}

    class _ResponsesClient:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}

    class _Client:
        def __init__(self) -> None:
            self.responses = _ResponsesClient()

    backend = OpenAIBackend(
        ProviderContext(
            provider="openrouter",
            api_key="dummy",
            api_base="https://openrouter.ai/api/v1",
        )
    )
    backend._sync_client = _Client()

    backend.chat(
        ChatRequest(
            transport="responses",
            model="openai/gpt-5.4-nano",
            conversation=execution.conversation_from_messages([{"role": "user", "content": "hi"}]),
            stream=False,
            reasoning_effort=None,
            kwargs={},
            tools=None,
        )
    )

    assert "tools" not in captured


def test_messages_transport_is_driven_by_backend_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "default_provider_runtime", lambda: _RuntimeWithCustomMessagesProvider())

    llm = LLM(model="custom:model-x", api_key="dummy", api_format="messages")

    assert llm.chat("hi") == "ok"


def test_embeddings_use_invalid_input_for_unsupported_provider() -> None:
    llm = LLM(model="anthropic:claude-3-5-haiku-latest", api_key="dummy")

    with pytest.raises(RepublicError) as exc_info:
        llm.embeddings.embed("hi")

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT
