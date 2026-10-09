from __future__ import annotations

import pytest

import republic
from republic import ModelInfo, errors
from tests.conftest import FakeService


@pytest.mark.parametrize(
    ("name", "url"),
    [
        ("openai", "https://api.openai.com/v1/models"),
        ("deepseek", "https://api.deepseek.com/models"),
        ("minimax", "https://api.minimax.io/v1/models"),
        ("ollama", "http://localhost:11434/v1/models"),
        ("openrouter", "https://openrouter.ai/api/v1/models"),
    ],
)
async def test_lists_openai_style_models(service: FakeService, name: str, url: str) -> None:
    service.reply_json({"object": "list", "data": [{"id": "model-a"}, {"id": "model-b", "name": "Model B"}]})
    provider = republic.get_provider(name, api_key="key", http_client=service.client())

    models = await provider.list_models()

    assert service.requests[0].method == "GET"
    assert service.requests[0].url == url
    assert service.requests[0].headers["authorization"] == "Bearer key"
    assert models == [
        ModelInfo("model-a", raw={"id": "model-a"}),
        ModelInfo("model-b", "Model B", raw={"id": "model-b", "name": "Model B"}),
    ]


async def test_lists_a_bare_array(service: FakeService) -> None:
    service.reply_json([{"id": "meta/llama", "display_name": "Llama"}])
    provider = republic.get_provider("together", api_key="key", http_client=service.client())

    models = await provider.list_models()

    assert [(model.id, model.display_name) for model in models] == [("meta/llama", "Llama")]


async def test_lists_models_keyed_by_name(service: FakeService) -> None:
    service.reply_json({"models": [{"name": "jev-latest", "description": "Decisions"}]})
    provider = republic.get_provider("typesafe", api_key="key", http_client=service.client())

    models = await provider.list_models()

    assert [(model.id, model.display_name) for model in models] == [("jev-latest", None)]


async def test_anthropic_follows_pages(service: FakeService) -> None:
    service.reply_json({"data": [{"id": "claude-a", "display_name": "Claude A"}], "has_more": True, "last_id": "a"})
    service.reply_json({"data": [{"id": "claude-b"}], "has_more": False, "last_id": "claude-b"})
    provider = republic.get_provider("anthropic", api_key="key", http_client=service.client())

    models = await provider.list_models()

    assert [model.id for model in models] == ["claude-a", "claude-b"]
    assert models[0].display_name == "Claude A"
    first, second = service.requests
    assert first.url == "https://api.anthropic.com/v1/models?limit=1000"
    assert second.url == "https://api.anthropic.com/v1/models?limit=1000&after_id=a"
    assert first.headers["x-api-key"] == "key"
    assert first.headers["anthropic-version"] == "2023-06-01"


async def test_google_strips_the_resource_prefix_and_follows_pages(service: FakeService) -> None:
    service.reply_json({"models": [{"name": "models/gemini-a", "displayName": "Gemini A"}], "nextPageToken": "t"})
    service.reply_json({"models": [{"name": "models/gemini-b"}]})
    provider = republic.get_provider("google", api_key="key", http_client=service.client())

    models = await provider.list_models()

    assert [(model.id, model.display_name) for model in models] == [("gemini-a", "Gemini A"), ("gemini-b", None)]
    assert (
        service.requests[1].url == "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&pageToken=t"
    )
    assert service.requests[0].headers["x-goog-api-key"] == "key"


async def test_error_status_raises(service: FakeService) -> None:
    service.reply_json({"error": "nope"}, status_code=401)
    provider = republic.get_provider("openai", api_key="key", http_client=service.client())

    with pytest.raises(republic.errors.APIStatusError) as error:
        await provider.list_models()

    assert error.value.status_code == 401


async def test_codex_cannot_list_models(service: FakeService) -> None:
    provider = republic.get_provider("codex", auth=republic.auth.HeaderAuth("Authorization", "Bearer t"))

    with pytest.raises(errors.UnsupportedFeatureError):
        await provider.list_models()
