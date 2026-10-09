from __future__ import annotations

from collections.abc import Generator

import httpx2
import pytest

import republic
from republic import _registry
from tests.conftest import FakeService

CHAT_REPLY = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
RESPONSES_REPLY = {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}


class TestGetModel:
    async def test_splits_provider_and_model_on_first_colon(self, service: FakeService) -> None:
        service.reply_json({"output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]})
        model = republic.get_model("openrouter:meta-llama/llama-4:free", http_client=service.client())

        response = await model.chat("Hello")

        assert service.requests[0].url == "https://openrouter.ai/api/v1/responses"
        assert service.body()["model"] == "meta-llama/llama-4:free"
        assert response.text == "hi"

    def test_rejects_spec_without_provider(self) -> None:
        with pytest.raises(ValueError, match="provider:model"):
            republic.get_model("gpt-6-sol")

    def test_rejects_unknown_provider(self) -> None:
        with pytest.raises(republic.errors.ProviderNotFoundError):
            republic.get_model("nowhere:model")


class TestCredentials:
    async def test_reads_api_key_and_base_from_default_env_prefix(
        self, monkeypatch: pytest.MonkeyPatch, service: FakeService
    ) -> None:
        monkeypatch.setenv("REPUBLIC_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("REPUBLIC_OPENAI_API_BASE", "https://gateway.test/v1/")
        service.reply_json(RESPONSES_REPLY)
        model = republic.get_model("openai:gpt-6-sol", http_client=service.client())

        await model.chat("Hello")

        assert service.requests[0].url == "https://gateway.test/v1/responses"
        assert service.requests[0].headers["authorization"] == "Bearer env-key"

    async def test_reads_custom_env_prefix(self, monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
        monkeypatch.setenv("MY_CUSTOM_PREFIX_API_KEY", "custom-key")
        service.reply_json(RESPONSES_REPLY)
        provider = republic.get_provider("openai", env_prefix="MY_CUSTOM_PREFIX", http_client=service.client())

        await provider.get_model("gpt-6-sol").chat("Hello")

        assert service.requests[0].headers["authorization"] == "Bearer custom-key"

    async def test_explicit_api_key_wins_over_env(self, monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
        monkeypatch.setenv("REPUBLIC_ANTHROPIC_API_KEY", "env-key")
        service.reply_json({"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}})
        model = republic.get_model("anthropic:claude-opus-5-5", api_key="explicit", http_client=service.client())

        await model.chat("Hello")

        assert service.requests[0].headers["x-api-key"] == "explicit"
        assert service.requests[0].headers["anthropic-version"] == "2023-06-01"

    async def test_custom_auth_is_used_as_is(self, service: FakeService) -> None:
        class TokenAuth(republic.auth.Auth):
            def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
                request.headers["x-token"] = "secret"
                yield request

        service.reply_json({"candidates": [], "usageMetadata": {}})
        model = republic.get_model("google:gemini-3-pro", auth=TokenAuth(), http_client=service.client())

        await model.chat("Hello")

        assert service.requests[0].headers["x-token"] == "secret"
        assert "x-goog-api-key" not in service.requests[0].headers


class TestApiFormat:
    def test_provider_order_sets_the_preference(self) -> None:
        class ChatFirst(republic.providers.OpenAI):
            SUPPORTED_API_FORMATS = ("chat", "responses", "embeddings")

        assert ChatFirst().get_model("gpt-6-sol").api_format.name == "chat"
        assert ChatFirst(api_format="responses").get_model("gpt-6-sol").api_format.name == "responses"

    def test_missing_kind_is_rejected_when_getting_the_model(self) -> None:
        provider = republic.get_provider("anthropic")

        with pytest.raises(republic.errors.UnsupportedApiFormatError, match="embedding"):
            provider.get_embedding_model("claude-opus-5-5")

    async def test_requested_chat_format_preserves_embeddings(self, service: FakeService) -> None:
        service.reply_json(CHAT_REPLY)
        service.reply_json({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
        provider = republic.get_provider("openai", api_format="chat", http_client=service.client())

        response = await provider.get_model("gpt-6-sol").chat("Hello")
        embedding = await provider.get_embedding_model("text-embedding-4").embed("Hello")

        assert service.requests[0].url.path == "/v1/chat/completions"
        assert response.text == "hi"
        assert service.requests[1].url.path == "/v1/embeddings"
        assert embedding.vector == [0.1, 0.2]

    def test_rejects_unsupported_format(self) -> None:
        with pytest.raises(republic.errors.UnsupportedApiFormatError):
            republic.get_model("openai:gpt-6-sol", api_format="messages")


class TestRegisterProvider:
    async def test_custom_provider_uses_its_key_for_env_and_formats(
        self, monkeypatch: pytest.MonkeyPatch, service: FakeService
    ) -> None:
        class MyCustomProvider(republic.providers.OpenAI):
            SUPPORTED_API_FORMATS = ("messages", "chat")

        republic.register_provider(MyCustomProvider, "custom")
        monkeypatch.setenv("REPUBLIC_CUSTOM_API_KEY", "custom-key")
        service.reply_json({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1}})
        model = republic.get_model("custom:gpt-6-sol", http_client=service.client())

        response = await model.chat("Hello")

        assert service.requests[0].url.path == "/v1/messages"
        assert service.requests[0].headers["authorization"] == "Bearer custom-key"
        assert response.text == "ok"

    def test_all_providers_lists_built_in_and_registered_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_registry, "_PROVIDERS", dict(_registry._PROVIDERS))

        republic.register_provider(republic.providers.OpenAICompatible, "listed")

        names = republic.all_providers()
        assert {"openai", "azure-openai", "minimax", "listed"} <= set(names)
        assert list(names) == sorted(names)


async def test_http_errors_carry_status_and_body(service: FakeService) -> None:
    service.reply_json({"error": "bad key"}, status_code=401)
    model = republic.get_model("openai:gpt-6-sol", http_client=service.client())

    with pytest.raises(republic.errors.APIStatusError) as exc_info:
        await model.chat("Hello")

    assert exc_info.value.status_code == 401
    assert "bad key" in exc_info.value.body


class TestOpenRouter:
    async def test_serves_the_messages_format(self, service: FakeService) -> None:
        service.reply_json({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1}})
        model = republic.get_model(
            "openrouter:anthropic/claude-opus-5-5", api_key="key", api_format="messages", http_client=service.client()
        )

        response = await model.chat("Hi")

        assert service.requests[0].url == "https://openrouter.ai/api/v1/messages"
        assert service.requests[0].headers["authorization"] == "Bearer key"
        assert response.text == "ok"

    async def test_serves_system_one_decisions(self, service: FakeService) -> None:
        service.reply_json({"answers": {"spam": {"type": "noul", "noul": 0.02}}})
        model = republic.get_decision_model("openrouter:typesafe/jev-latest", http_client=service.client())

        response = await model.decide("Hello", questions={"spam": republic.decisions.Noul("Is it spam?")})

        assert service.requests[0].url == "https://openrouter.ai/api/v1/systemone"
        assert response.spam == republic.decisions.NoulAnswer(0.02)
