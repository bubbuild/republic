from __future__ import annotations

import pytest

import republic
from republic.events import ReasoningDelta, TextDelta
from tests.conftest import FakeService

WEATHER = republic.Tool("get_weather", "Look up the weather", {"type": "object", "properties": {}})
CHAT_REPLY = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
EMBEDDING_REPLY = {"data": [{"index": 0, "embedding": [0.5]}], "usage": {"prompt_tokens": 1}}


@pytest.mark.parametrize(
    ("spec", "url", "max_tokens_field"),
    [
        ("ollama:gpt-oss", "http://localhost:11434/v1/chat/completions", "max_tokens"),
        ("moonshot:kimi-k3", "https://api.moonshot.ai/v1/chat/completions", "max_completion_tokens"),
        ("together:openai/gpt-oss-120b", "https://api.together.ai/v1/chat/completions", "max_tokens"),
        ("zai:glm-5.2", "https://api.z.ai/api/paas/v4/chat/completions", "max_tokens"),
        ("mistral:mistral-medium-3-5", "https://api.mistral.ai/v1/chat/completions", "max_tokens"),
    ],
)
async def test_chat_reaches_the_default_endpoint(
    service: FakeService, spec: str, url: str, max_tokens_field: str
) -> None:
    service.reply_json(CHAT_REPLY)

    response = await republic.get_model(spec, api_key="key", http_client=service.client()).chat("Hello", max_tokens=9)

    assert service.requests[0].url == url
    assert service.requests[0].headers["authorization"] == "Bearer key"
    assert service.body()[max_tokens_field] == 9
    assert response.text == "hi"


async def test_magpie_defaults_to_local_chat(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)

    response = await republic.get_model("magpie:openrouter/moonshotai/kimi-k3", http_client=service.client()).chat("Hi")

    assert service.requests[0].url == "http://127.0.0.1:3425/v1/chat/completions"
    assert service.body()["model"] == "openrouter/moonshotai/kimi-k3"
    assert response.text == "hi"


class TestDeepSeek:
    async def test_responses_use_the_root_endpoint(self, service: FakeService) -> None:
        service.reply_json({"output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]})
        model = republic.get_model(
            "deepseek:deepseek-flash", api_key="key", api_format="responses", http_client=service.client()
        )

        response = await model.chat("Hi")

        assert service.requests[0].url == "https://api.deepseek.com/responses"
        assert response.text == "hi"

    async def test_messages_use_the_anthropic_endpoint(self, service: FakeService) -> None:
        service.reply_json({"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}})
        model = republic.get_model(
            "deepseek:deepseek-v4-pro", api_key="key", api_format="messages", http_client=service.client()
        )

        response = await model.chat("Hi")

        assert service.requests[0].url == "https://api.deepseek.com/anthropic/v1/messages"
        assert service.requests[0].headers["authorization"] == "Bearer key"
        assert response.text == "hi"

    async def test_defaults_to_chat_with_max_tokens(self, service: FakeService) -> None:
        service.reply_json(CHAT_REPLY)
        model = republic.get_model("deepseek:deepseek-flash", api_key="key", http_client=service.client())

        await model.chat("Hi", max_tokens=9)

        assert service.requests[0].url == "https://api.deepseek.com/chat/completions"
        assert service.body()["max_tokens"] == 9


class TestMiniMax:
    async def test_defaults_to_the_anthropic_endpoint(self, service: FakeService) -> None:
        service.reply_json({"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}})

        response = await republic.get_model("minimax:MiniMax-M3", api_key="key", http_client=service.client()).chat("Hi")

        assert service.requests[0].url == "https://api.minimax.io/anthropic/v1/messages"
        assert service.requests[0].headers["authorization"] == "Bearer key"
        assert response.text == "hi"

    async def test_chat_splits_reasoning(self, service: FakeService) -> None:
        service.reply_json({
            "choices": [{"message": {"content": "hi", "reasoning_details": [{"type": "reasoning.text", "text": "Hm."}]}}]
        })
        model = republic.get_model("minimax:MiniMax-M2.7", api_key="key", api_format="chat", http_client=service.client())

        response = await model.chat("Hi", max_tokens=9)

        assert service.requests[0].url == "https://api.minimax.io/v1/chat/completions"
        assert service.body()["reasoning_split"] is True
        assert service.body()["max_completion_tokens"] == 9
        assert (response.reasoning, response.text) == ("Hm.", "hi")


async def test_ollama_needs_no_api_key(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
    monkeypatch.delenv("REPUBLIC_OLLAMA_API_KEY", raising=False)
    service.reply_json(CHAT_REPLY)

    await republic.get_model("ollama:qwen3", http_client=service.client()).chat("Hello")

    assert "authorization" not in service.requests[0].headers


@pytest.mark.parametrize(("spec", "path"), [("ollama:embeddinggemma", "/v1/embeddings"), ("together:m", "/v1/embeddings")])
async def test_embeddings_use_the_openai_format(service: FakeService, spec: str, path: str) -> None:
    service.reply_json(EMBEDDING_REPLY)

    response = await republic.get_embedding_model(spec, api_key="key", http_client=service.client()).embed("Hi")

    assert service.requests[0].url.path == path
    assert response.vectors == [[0.5]]


@pytest.mark.parametrize("provider", ["deepseek", "moonshot", "zai", "minimax"])
async def test_reasoning_content_is_sent_back(service: FakeService, provider: str) -> None:
    service.reply_json({
        "choices": [
            {
                "message": {
                    "reasoning_content": "Need the weather.",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                }
            }
        ]
    })
    service.reply_json(CHAT_REPLY)
    model = republic.get_model(f"{provider}:model", api_key="key", api_format="chat", http_client=service.client())

    first = await model.chat("Weather?", tools=[WEATHER], reasoning_effort="high")
    results = [republic.tool_result(call, "sunny") for call in first.tool_calls]
    await model.chat(["Weather?", first.message, republic.assistant(tool_results=results)], tools=[WEATHER])

    assert service.body(0)["reasoning_effort"] == "high"
    assert first.reasoning == "Need the weather."
    assert service.body(1)["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        "reasoning_content": "Need the weather.",
    }


async def test_openai_compatible_does_not_send_reasoning_back(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    provider = republic.providers.OpenAICompatible(
        api_key="key", api_base="https://compat.example/v1", http_client=service.client()
    )

    await provider.get_model("m").chat(["Hi", republic.Message("assistant", (republic.Reasoning("Greet back."), republic.Text("Hello"))), "Bye"])

    assert service.body()["messages"][1] == {"role": "assistant", "content": "Hello"}


class TestAzureOpenAI:
    async def test_uses_v1_api_with_api_key_header(self, monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
        monkeypatch.setenv("REPUBLIC_AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("REPUBLIC_AZURE_OPENAI_API_BASE", "https://res.openai.azure.com/openai/v1/")
        service.reply_json({"output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]})

        response = await republic.get_model("azure-openai:my-deployment", http_client=service.client()).chat("Hello")

        request = service.requests[0]
        assert request.url == "https://res.openai.azure.com/openai/v1/responses"
        assert request.headers["api-key"] == "azure-key"
        assert "authorization" not in request.headers
        assert service.body()["model"] == "my-deployment"
        assert response.text == "hi"

    async def test_builds_api_base_from_resource(self, service: FakeService) -> None:
        service.reply_json(CHAT_REPLY)
        provider = republic.providers.AzureOpenAI(
            resource="my-res", api_key="key", api_format="chat", http_client=service.client()
        )

        await provider.get_model("my-deployment").chat("Hello")

        assert service.requests[0].url == "https://my-res.openai.azure.com/openai/v1/chat/completions"
        assert service.body()["model"] == "my-deployment"

    async def test_reads_resource_from_env(self, monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
        monkeypatch.delenv("REPUBLIC_AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.setenv("REPUBLIC_AZURE_OPENAI_RESOURCE", "env-res")
        service.reply_json({"output": []})

        await republic.get_model("azure-openai:my-deployment", api_key="key", http_client=service.client()).chat("Hi")

        assert service.requests[0].url == "https://env-res.openai.azure.com/openai/v1/responses"

    def test_api_base_wins_over_resource(self) -> None:
        provider = republic.providers.AzureOpenAI(resource="my-res", api_base="https://gateway.test/openai/v1")

        assert provider.api_base == "https://gateway.test/openai/v1"

    def test_requires_resource_or_api_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REPUBLIC_AZURE_OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("REPUBLIC_AZURE_OPENAI_RESOURCE", raising=False)

        with pytest.raises(ValueError, match="REPUBLIC_AZURE_OPENAI_RESOURCE"):
            republic.get_provider("azure-openai", api_key="key")


class TestMistral:
    async def test_maps_options_to_mistral_fields(self, service: FakeService) -> None:
        service.reply_events([{"choices": [{"delta": {"content": "hi"}}]}, "[DONE]"])
        model = republic.get_model("mistral:mistral-small-latest", api_key="key", http_client=service.client())

        async with model.stream("Hello", seed=7) as stream:
            async for _ in stream:
                pass

        body = service.body()
        assert body["random_seed"] == 7
        assert "seed" not in body
        assert "stream_options" not in body

    async def test_rejects_top_k(self, service: FakeService) -> None:
        model = republic.get_model("mistral:mistral-small-latest", api_key="key", http_client=service.client())

        with pytest.raises(republic.UnsupportedFeatureError, match="top_k"):
            await model.chat("Hello", top_k=5)

    async def test_reads_thinking_chunks_and_sends_them_back(self, service: FakeService) -> None:
        service.reply_json({
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": [{"type": "text", "text": "Hmm."}]},
                            {"type": "text", "text": "Hello!"},
                        ]
                    }
                }
            ]
        })
        service.reply_json(CHAT_REPLY)
        model = republic.get_model("mistral:mistral-medium-3-5", api_key="key", http_client=service.client())

        first = await model.chat("Hi", reasoning_effort="high")
        await model.chat(["Hi", first.message, "Bye"])

        assert (first.reasoning, first.text) == ("Hmm.", "Hello!")
        assert service.body(1)["messages"][1] == {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "Hmm."}]},
                {"type": "text", "text": "Hello!"},
            ],
        }

    async def test_streams_thinking_chunks(self, service: FakeService) -> None:
        service.reply_events([
            {"choices": [{"delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "Hm"}]}]}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "thinking", "thinking": [{"type": "text", "text": "m."}]},
                                {"type": "text", "text": "Hel"},
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "lo!"}, "finish_reason": "stop"}]},
            "[DONE]",
        ])
        model = republic.get_model("mistral:mistral-medium-3-5", api_key="key", http_client=service.client())

        async with model.stream("Hi") as stream:
            events = [event async for event in stream if isinstance(event, TextDelta | ReasoningDelta)]

        assert events == [ReasoningDelta("Hm"), ReasoningDelta("m."), TextDelta("Hel"), TextDelta("lo!")]

    async def test_embeddings_send_output_dimension(self, service: FakeService) -> None:
        service.reply_json(EMBEDDING_REPLY)
        model = republic.get_embedding_model("mistral:mistral-embed", api_key="key", http_client=service.client())

        await model.embed("Hi", dimensions=256)

        assert service.requests[0].url == "https://api.mistral.ai/v1/embeddings"
        assert service.body()["output_dimension"] == 256
        assert "dimensions" not in service.body()
