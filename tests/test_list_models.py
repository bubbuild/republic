from __future__ import annotations

import pytest

import republic
from republic import ModelInfo
from republic.providers import CodexAuth
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


@pytest.mark.parametrize("name", ["openai", "codex"])
async def test_error_status_raises(service: FakeService, name: str) -> None:
    service.reply_json({"error": "nope"}, status_code=401)
    provider = republic.get_provider(name, api_key="key", http_client=service.client())

    with pytest.raises(republic.errors.APIStatusError) as error:
        await provider.list_models()

    assert error.value.status_code == 401


async def test_codex_lists_catalog_with_auth_and_preserves_metadata_and_order(service: FakeService) -> None:
    catalog = [
        {
            "slug": "model-b",
            "display_name": "Model B",
            "visibility": "list",
            "priority": 20,
            "supported_in_api": False,
            "context_window": 272000,
            "supported_reasoning_levels": [{"effort": "high", "description": "High reasoning"}],
        },
        {"slug": "internal-review", "display_name": "Internal Review", "visibility": "hide", "priority": 0},
        {"slug": "model-a", "display_name": "Model A", "visibility": "list", "priority": 10},
    ]
    service.reply_json({"models": catalog})
    provider = republic.get_provider(
        "codex",
        auth=CodexAuth({"access_token": "provided-token"}, account_id="provided-account"),
        http_client=service.client(),
    )

    models = await provider.list_models()

    assert models == [
        ModelInfo("model-b", "Model B", raw=catalog[0]),
        ModelInfo("internal-review", "Internal Review", raw=catalog[1]),
        ModelInfo("model-a", "Model A", raw=catalog[2]),
    ]
    request = service.requests[0]
    assert request.method == "GET"
    assert request.url == "https://chatgpt.com/backend-api/codex/models?client_version=0.167.0"
    assert request.headers["authorization"] == "Bearer provided-token"
    assert request.headers["chatgpt-account-id"] == "provided-account"


async def test_codex_lists_an_empty_catalog(service: FakeService) -> None:
    service.reply_json({"models": []})
    provider = republic.get_provider("codex", api_key="key", http_client=service.client())

    assert await provider.list_models() == []
