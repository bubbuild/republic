from __future__ import annotations

from typing import Any, TypeVar

import republic
from republic.formats import ChatFormat, GeminiFormat
from tests.conftest import FakeService

FormatT = TypeVar("FormatT", bound=republic.formats.ApiFormat)

CHAT_REPLY = {"choices": [{"message": {"content": "ok"}}]}


class DeepSeekChat(ChatFormat):
    def reasoning_fields(self, effort: republic.ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        return {"thinking": {"type": "disabled" if effort == "none" else "enabled"}} if effort else {}

    def max_tokens_fields(self, max_tokens: int) -> dict[str, Any]:
        return {"max_tokens": max_tokens}

    def reasoning_text(self, message: Any) -> str | None:
        return message.get("thoughts")


class DeepSeek(republic.providers.OpenAICompatible):
    name = "deepseek"

    def select_api_format(self, format_kind: type[FormatT], model: str) -> FormatT:
        api_format = super().select_api_format(format_kind, model)
        chat = DeepSeekChat()
        return chat if isinstance(chat, format_kind) else api_format


class Gemini25(GeminiFormat):
    def reasoning_fields(self, effort: republic.ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        return {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}} if effort else {}


class VersionedGoogle(republic.providers.Google):
    def select_api_format(self, format_kind: type[FormatT], model: str) -> FormatT:
        api_format = super().select_api_format(format_kind, model)
        legacy = Gemini25()
        return legacy if model.startswith("gemini-2.5") and isinstance(legacy, format_kind) else api_format


async def test_provider_can_swap_in_a_format_subclass(service: FakeService) -> None:
    service.reply_json({"choices": [{"message": {"content": "ok", "thoughts": "Thinking it over."}}]})
    model = DeepSeek(api_key="key", http_client=service.client()).get_model("deepseek-chat")

    response = await model.chat("Hi", reasoning_effort="high", max_tokens=100)

    body = service.body()
    assert body["thinking"] == {"type": "enabled"}
    assert body["max_tokens"] == 100
    assert "reasoning_effort" not in body
    assert "max_completion_tokens" not in body
    assert response.reasoning == "Thinking it over."


async def test_hooks_apply_to_streams(service: FakeService) -> None:
    service.reply_events([{"choices": [{"delta": {"thoughts": "Hmm."}}]}, "[DONE]"])
    model = DeepSeek(api_key="key", http_client=service.client()).get_model("deepseek-chat")

    async with model.stream("Hi") as stream:
        [event async for event in stream]

    assert stream.reasoning == "Hmm."


async def test_format_can_depend_on_the_model(service: FakeService) -> None:
    service.reply_json({"candidates": []})
    service.reply_json({"candidates": []})
    provider = VersionedGoogle(api_key="key", http_client=service.client())

    await provider.get_model("gemini-2.5-pro").chat("Hi", reasoning_effort="high")
    await provider.get_model("gemini-3-pro").chat("Hi", reasoning_effort="high")

    assert service.body(0)["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    assert service.body(1)["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


async def test_openrouter_chat_sends_one_reasoning_object(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    model = republic.get_model("openrouter:vendor/model", api_format="chat", http_client=service.client())

    await model.chat("Hi", reasoning_effort="low", include_reasoning=True)

    body = service.body()
    assert body["reasoning"] == {"effort": "low", "exclude": False}
    assert "reasoning_effort" not in body


async def test_extra_body_deep_merges_with_provider_defaults_and_format_fields(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    model = republic.get_model(
        "openrouter:vendor/model",
        api_format="chat",
        extra_body={"provider": {"order": ["anthropic"]}, "reasoning": {"max_tokens": 2000}},
        http_client=service.client(),
    )

    await model.chat("Hi", reasoning_effort="high", extra_body={"provider": {"allow_fallbacks": False}})

    body = service.body()
    assert body["reasoning"] == {"effort": "high", "max_tokens": 2000}
    assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
