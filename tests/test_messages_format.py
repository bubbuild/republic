from __future__ import annotations

import pydantic
import pytest

import republic
from republic.events import Completed, ReasoningDelta, TextDelta, ToolCallDelta, ToolCallReady, UsageDelta
from tests.conftest import FakeService

THINKING = {"type": "thinking", "thinking": "Need the weather tool.", "signature": "sig"}


class Answer(pydantic.BaseModel):
    value: int


def make_model(service: FakeService) -> republic.ChatModel:
    return republic.get_model("anthropic:claude-opus-5-5", api_key="key", http_client=service.client())


async def test_chat_moves_system_to_top_level_and_defaults_max_tokens(service: FakeService) -> None:
    service.reply_json({
        "content": [{"type": "text", "text": "Hi!"}],
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 3,
            "output_tokens": 2,
        },
    })

    response = await make_model(service).chat([
        republic.system("Be brief."),
        republic.user("Hello", republic.image("https://example.com/cat.png")),
    ])

    assert service.body() == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": "Be brief.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
                ],
            }
        ],
    }
    assert response.text == "Hi!"
    assert response.token_usage == republic.TokenUsage(
        input_tokens=18, output_tokens=2, cached_tokens=5, cache_write_tokens=3
    )


async def test_tool_results_follow_calls_with_thinking_preserved(service: FakeService) -> None:
    service.reply_json({
        "content": [THINKING, {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    service.reply_json({
        "content": [{"type": "text", "text": "Sunny."}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    model = make_model(service)

    first = await model.chat("Weather?")
    await model.chat([
        "Weather?",
        first.message,
        republic.tool(first.tool_calls[0], "sunny"),
        "Thanks!",
    ])

    assert service.body()["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
        {
            "role": "assistant",
            "content": [
                THINKING,
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny", "is_error": False},
                {"type": "text", "text": "Thanks!"},
            ],
        },
    ]


async def test_tool_results_can_hold_multiple_parts(service: FakeService) -> None:
    service.reply_json({"content": [{"type": "text", "text": "A cat."}], "usage": {"input_tokens": 1}})
    call = republic.ToolCall("toolu_1", "screenshot", "{}")
    result = republic.tool(call, "Captured", republic.Image("image/png", data=b"png"), is_error=True)

    await make_model(service).chat(["Look", result])

    assert service.body()["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {"type": "text", "text": "Captured"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"}},
                ],
                "is_error": True,
            }
        ],
    }


async def test_structured_output_closes_object_schemas(service: FakeService) -> None:
    service.reply_json({"content": [{"type": "text", "text": '{"value": 42}'}], "usage": {"input_tokens": 1}})

    response = await make_model(service).chat("Answer?", output_schema=Answer)

    schema = service.body()["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert response.output == Answer(value=42)


async def test_video_input_is_rejected(service: FakeService) -> None:
    with pytest.raises(republic.errors.UnsupportedFeatureError):
        await make_model(service).chat(republic.user(republic.video("https://example.com/clip.mp4")))


async def test_stream_events(service: FakeService) -> None:
    service.reply_events([
        {
            "type": "message_start",
            "message": {"id": "msg_1", "model": "claude-opus-5-5", "usage": {"input_tokens": 12, "output_tokens": 1}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Hmm."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Checking"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"city"'}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": ': "Paris"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
        {"type": "message_stop"},
    ])

    async with make_model(service).stream("Weather?") as stream:
        events = [event async for event in stream]

    assert events[:-1] == [
        UsageDelta(republic.TokenUsage(input_tokens=12, output_tokens=1)),
        ReasoningDelta("Hmm."),
        TextDelta("Checking"),
        ToolCallDelta("toolu_1", "get_weather", '{"city"'),
        ToolCallDelta("toolu_1", "get_weather", ': "Paris"}'),
        ToolCallReady(republic.ToolCall("toolu_1", "get_weather", '{"city": "Paris"}')),
        UsageDelta(republic.TokenUsage(output_tokens=29)),
    ]
    assert stream.response.finish_reason == "tool_calls"
    assert (stream.response.id, stream.response.model) == ("msg_1", "claude-opus-5-5")
    assert isinstance(events[-1], Completed)
    assert stream.response.reasoning == "Hmm."
    assert stream.response.message.parts[0] == republic.ProviderData(
        "messages", {"type": "thinking", "thinking": "Hmm.", "signature": "sig"}
    )
    assert stream.token_usage == republic.TokenUsage(12, 30)


async def test_generation_options_map_to_messages_fields(service: FakeService) -> None:
    service.reply_json({"content": [{"type": "text", "text": '{"value": 1}'}], "usage": {"input_tokens": 1}})

    await make_model(service).chat(
        "Hi",
        output_schema=Answer,
        tools=[republic.Tool("lookup")],
        tool_choice="required",
        parallel_tool_calls=False,
        reasoning_effort="max",
        max_tokens=64000,
        stop=["END"],
    )

    body = service.body()
    assert body["max_tokens"] == 64000
    assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert body["output_config"]["effort"] == "max"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["stop_sequences"] == ["END"]


async def test_seed_is_rejected(service: FakeService) -> None:
    with pytest.raises(republic.errors.UnsupportedFeatureError, match="seed"):
        await make_model(service).chat("Hi", seed=1)


async def test_reasoning_options_enable_adaptive_thinking(service: FakeService) -> None:
    service.reply_json({"content": [], "usage": {"input_tokens": 1}})

    await make_model(service).chat("Hi", reasoning_effort="high", include_reasoning=True, top_k=5)

    body = service.body()
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["top_k"] == 5


async def test_no_reasoning_disables_thinking(service: FakeService) -> None:
    service.reply_json({"content": [], "usage": {"input_tokens": 1}})

    await make_model(service).chat("Hi", reasoning_effort="none")

    body = service.body()
    assert body["thinking"] == {"type": "disabled"}
    assert "output_config" not in body


async def test_refusal_stop_reason(service: FakeService) -> None:
    service.reply_json({
        "id": "msg_1",
        "model": "claude-opus-5-5",
        "content": [],
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": "Declined."},
        "usage": {"input_tokens": 1},
    })

    response = await make_model(service).chat("Hi", output_schema=Answer)

    assert response.finish_reason == "refusal"
    assert response.refusal == "Declined."
    assert response.output is None


async def test_empty_refusal_turn_is_not_sent_back(service: FakeService) -> None:
    refusal = {
        "content": [],
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "explanation": "Declined."},
        "usage": {"input_tokens": 1},
    }
    service.reply_json(refusal)
    service.reply_json(refusal)
    model = republic.get_model(
        "anthropic:claude-opus-5-5",
        api_key="key",
        http_client=service.client(),
        history=republic.history.InMemoryHistory(),
    )

    await model.chat("hello")
    await model.chat("and now?")

    assert service.body()["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "and now?"}]}
    ]


async def test_strict_tool_is_flagged(service: FakeService) -> None:
    service.reply_json({"content": [], "usage": {"input_tokens": 1}})

    await make_model(service).chat("Hi", tools=[republic.Tool("lookup", strict=True)])

    assert service.body()["tools"][0]["strict"] is True
