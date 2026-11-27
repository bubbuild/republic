from __future__ import annotations

import asyncio
from typing import Any, Literal, TypeVar

import pytest

import republic
from republic.formats import ApiFormat, ChatFormat, ChatRequest
from republic.providers import (
    Codex,
    CodexAuth,
    DeepSeek,
    Grok,
    Magpie,
    Mistral,
    Moonshot,
    OpenAI,
    OpenAICompatible,
    OpenRouter,
)
from tests.conftest import FakeService

FormatT = TypeVar("FormatT", bound=ApiFormat)

CHAT_REPLY = {"choices": [{"message": {"content": "ok"}}]}
RESPONSES_REPLY = {"id": "resp_1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}
RESPONSES_EVENTS = [
    {"type": "response.output_text.delta", "delta": "ok"},
    {"type": "response.completed", "response": RESPONSES_REPLY},
    "[DONE]",
]
MESSAGES_REPLY = {
    "id": "msg_1",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


@pytest.mark.parametrize("api_format", ["chat", "responses"])
async def test_openai_maps_session_id_and_long_retention(
    service: FakeService, api_format: Literal["chat", "responses"]
) -> None:
    service.reply_json(CHAT_REPLY if api_format == "chat" else RESPONSES_REPLY)
    model = OpenAI(api_key="key", api_format=api_format, http_client=service.client()).get_model("gpt-5")

    await model.chat("Hi", session_id="sess-1", cache_retention="long")

    body = service.body()
    assert body["prompt_cache_key"] == "sess-1"
    assert body["prompt_cache_retention"] == "24h"


async def test_openai_respects_none_retention_and_extra_body_override(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    service.reply_json(CHAT_REPLY)
    model = OpenAI(api_key="key", api_format="chat", http_client=service.client()).get_model("gpt-5")

    await model.chat("Hi", session_id="sess-1", cache_retention="none")
    await model.chat("Hi", session_id="sess-1", extra_body={"prompt_cache_key": "custom-key"})

    assert "prompt_cache_key" not in service.body(0)
    assert "prompt_cache_retention" not in service.body(0)
    assert service.body(1)["prompt_cache_key"] == "custom-key"


async def test_openai_compatible_does_not_inject_openai_cache_fields(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    model = OpenAICompatible(
        api_key="key", api_base="https://compat.example/v1", http_client=service.client()
    ).get_model("local-model")

    await model.chat("Hi", session_id="sess-1", cache_retention="long")

    body = service.body()
    assert "prompt_cache_key" not in body
    assert "prompt_cache_retention" not in body


async def test_codex_sets_session_header_and_prompt_cache_key(service: FakeService) -> None:
    service.reply_events(RESPONSES_EVENTS)
    service.reply_events(RESPONSES_EVENTS)
    auth = CodexAuth({"access_token": "tok"}, account_id="acct-1")
    model = Codex(auth=auth, http_client=service.client()).get_model("gpt-5-codex")

    await model.chat("Hi", session_id="sess-codex")
    await model.chat("Hi", session_id="sess-codex", cache_retention="none")

    assert service.requests[0].headers["session-id"] == "sess-codex"
    assert service.requests[0].headers["thread-id"] == "sess-codex"
    assert service.body(0)["prompt_cache_key"] == "sess-codex"
    assert service.requests[1].headers["session-id"] == "sess-codex"
    assert service.requests[1].headers["thread-id"] == "sess-codex"
    assert "prompt_cache_key" not in service.body(1)


async def test_grok_sets_conv_id_header_and_responses_prompt_cache_key(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    service.reply_json(RESPONSES_REPLY)
    service.reply_json(RESPONSES_REPLY)
    chat_model = Grok(api_key="key", api_format="chat", http_client=service.client()).get_model("grok-4")
    resp_model = Grok(api_key="key", api_format="responses", http_client=service.client()).get_model("grok-4")

    await chat_model.chat("Hi", session_id="sess-grok")
    await resp_model.chat("Hi", session_id="sess-grok")
    await resp_model.chat("Hi", session_id="sess-grok", cache_retention="none")

    assert service.requests[0].headers["x-grok-conv-id"] == "sess-grok"
    assert "prompt_cache_key" not in service.body(0)
    assert service.requests[1].headers["x-grok-conv-id"] == "sess-grok"
    assert service.body(1)["prompt_cache_key"] == "sess-grok"
    assert service.requests[2].headers["x-grok-conv-id"] == "sess-grok"
    assert "prompt_cache_key" not in service.body(2)


async def test_mistral_sets_affinity_header_and_prompt_cache_key(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    service.reply_json(CHAT_REPLY)
    model = Mistral(api_key="key", http_client=service.client()).get_model("mistral-large-latest")

    await model.chat("Hi", session_id="sess-mistral")
    await model.chat("Hi", session_id="sess-mistral", cache_retention="none")

    assert service.requests[0].headers["x-affinity"] == "sess-mistral"
    assert service.body(0)["prompt_cache_key"] == "sess-mistral"
    assert service.requests[1].headers["x-affinity"] == "sess-mistral"
    assert "prompt_cache_key" not in service.body(1)


async def test_moonshot_sets_prompt_cache_key_while_deepseek_stays_untouched(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    service.reply_json(CHAT_REPLY)
    service.reply_json(CHAT_REPLY)
    moonshot = Moonshot(api_key="key", http_client=service.client()).get_model("kimi-k2")
    deepseek = DeepSeek(api_key="key", http_client=service.client()).get_model("deepseek-chat")

    await moonshot.chat("Hi", session_id="sess-kimi")
    await moonshot.chat("Hi", session_id="sess-kimi", cache_retention="none")
    await deepseek.chat("Hi", session_id="sess-ds")

    assert service.body(0)["prompt_cache_key"] == "sess-kimi"
    assert "prompt_cache_key" not in service.body(1)
    assert "prompt_cache_key" not in service.body(2)
    assert "x-session-id" not in service.requests[2].headers


async def test_magpie_sets_session_header_across_formats_and_cache_key_for_openai_formats(
    service: FakeService,
) -> None:
    service.reply_json(CHAT_REPLY)
    service.reply_json(RESPONSES_REPLY)
    service.reply_json(MESSAGES_REPLY)
    provider = Magpie(api_key="magpie", http_client=service.client())

    await provider.get_model("m-chat").chat("Hi", session_id="sess-m", cache_retention="short")
    resp_provider = Magpie(api_key="magpie", api_format="responses", http_client=service.client())
    await resp_provider.get_model("m-resp").chat("Hi", session_id="sess-m", cache_retention="none")
    msg_provider = Magpie(api_key="magpie", api_format="messages", http_client=service.client())
    await msg_provider.get_model("m-msg").chat("Hi", session_id="sess-m", cache_retention="short")

    assert service.requests[0].headers["x-session-id"] == "sess-m"
    assert service.body(0)["prompt_cache_key"] == "sess-m"
    assert service.requests[1].headers["x-session-id"] == "sess-m"
    assert "prompt_cache_key" not in service.body(1)
    assert service.requests[2].headers["x-session-id"] == "sess-m"
    assert service.body(2)["cache_control"] == {"type": "ephemeral"}
    assert "prompt_cache_key" not in service.body(2)


async def test_openrouter_sets_session_header_without_mutating_provider(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    service.reply_json(CHAT_REPLY)
    provider = OpenRouter(api_key="key", api_format="chat", http_client=service.client())
    model = provider.get_model("vendor/model")

    await asyncio.gather(
        model.chat("A", session_id="sess-a"),
        model.chat("B", session_id="sess-b"),
    )

    seen = {req.headers["x-session-id"] for req in service.requests}
    assert seen == {"sess-a", "sess-b"}
    assert "x-session-id" not in provider.headers


@pytest.mark.parametrize(
    ("retention", "expected"),
    [
        ("short", {"type": "ephemeral"}),
        ("long", {"type": "ephemeral", "ttl": "1h"}),
        ("none", None),
        (None, None),
    ],
)
async def test_messages_format_maps_cache_retention(
    service: FakeService, retention: republic.CacheRetention | None, expected: dict[str, str] | None
) -> None:
    service.reply_json(MESSAGES_REPLY)
    model = republic.get_model("anthropic:claude-sonnet-4-6", api_key="key", http_client=service.client())

    options: dict[str, Any] = {"session_id": "sess-a"}
    if retention is not None:
        options["cache_retention"] = retention
    await model.chat("Hi", **options)

    assert service.body().get("cache_control") == expected


async def test_gemini_ignores_session_id_and_cache_retention(service: FakeService) -> None:
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    model = republic.get_model("google:gemini-2.5-pro", api_key="key", http_client=service.client())

    await model.chat("Hi", session_id="sess-g", cache_retention="long")

    body = service.body()
    assert "prompt_cache_key" not in body
    assert "cache_control" not in body
    assert "x-session-id" not in service.requests[0].headers


async def test_chat_and_responses_parse_cache_write_tokens(service: FakeService) -> None:
    service.reply_json({
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 25},
        },
    })
    service.reply_json({
        **RESPONSES_REPLY,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30},
        },
    })
    chat_model = OpenAI(api_key="key", api_format="chat", http_client=service.client()).get_model("gpt-5")
    resp_model = OpenAI(api_key="key", api_format="responses", http_client=service.client()).get_model("gpt-5")

    chat_res = await chat_model.chat("Hi")
    resp_res = await resp_model.chat("Hi")

    assert chat_res.token_usage.cached_tokens == 40
    assert chat_res.token_usage.cache_write_tokens == 25
    assert resp_res.token_usage.cached_tokens == 50
    assert resp_res.token_usage.cache_write_tokens == 30


async def test_custom_format_cache_fields_hook(service: FakeService) -> None:
    class CustomCacheChat(ChatFormat):
        def cache_fields(self, request: ChatRequest) -> dict[str, Any]:
            session_id = request.options.get("session_id")
            return {"custom_cache_key": session_id} if session_id else {}

    class CustomProvider(OpenAICompatible):
        name = "custom-cache"
        CHAT_FORMAT = CustomCacheChat()

    service.reply_json(CHAT_REPLY)
    model = CustomProvider(api_key="key", api_base="https://custom.example/v1", http_client=service.client()).get_model(
        "m"
    )

    await model.chat("Hi", session_id="sess-custom")

    assert service.body()["custom_cache_key"] == "sess-custom"
