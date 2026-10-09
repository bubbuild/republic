from __future__ import annotations

import base64

import pydantic
import pytest

import republic
from republic.events import (
    BuiltinToolCallReady,
    Completed,
    ImageReady,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallReady,
    UsageDelta,
)
from tests.conftest import FakeService

REASONING = {"type": "reasoning", "id": "rs_1", "summary": []}
FUNCTION_CALL = {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city":"Paris"}'}


def make_model(service: FakeService, history: republic.history.HistoryProtocol | None = None) -> republic.ChatModel:
    return republic.get_model("openai:gpt-6-sol", api_key="key", http_client=service.client(), history=history)


async def test_chat_reads_message_items_and_usage(service: FakeService) -> None:
    service.reply_json({
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 1}},
    })

    response = await make_model(service).chat([republic.system("Be kind."), "Hi"], max_tokens=50)

    assert service.requests[0].url.path == "/v1/responses"
    assert service.body() == {
        "model": "gpt-6-sol",
        "input": [
            {"role": "system", "content": "Be kind."},
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        ],
        "max_output_tokens": 50,
    }
    assert response.text == "Hello!"
    assert response.token_usage == republic.TokenUsage(input_tokens=3, output_tokens=2, cached_tokens=1)


async def test_reasoning_items_are_sent_back_with_tool_results(service: FakeService) -> None:
    service.reply_json({"output": [REASONING, FUNCTION_CALL]})
    service.reply_json({"output": []})
    model = make_model(service, history=republic.history.InMemoryHistory())

    first = await model.chat("Weather in Paris?")
    await model.chat(republic.assistant(tool_results=[republic.tool_result(first.tool_calls[0], "sunny")]))

    assert service.body()["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]},
        REASONING,
        FUNCTION_CALL,
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]


async def test_reasoning_summaries_are_exposed(service: FakeService) -> None:
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Checked the map."}]}
    service.reply_json({"output": [reasoning]})

    response = await make_model(service).chat("Where?")

    assert response.reasoning == "Checked the map."
    assert republic.ProviderData("responses", reasoning) in response.message.parts


async def test_generated_images_are_returned(service: FakeService) -> None:
    service.reply_json({
        "output": [
            {"type": "image_generation_call", "result": base64.b64encode(b"png").decode(), "output_format": "png"}
        ]
    })

    response = await make_model(service).chat("Draw", extra_body={"tools": [{"type": "image_generation"}]})

    assert service.body()["tools"] == [{"type": "image_generation"}]
    assert response.image_parts == [republic.Image("image/png", data=b"png")]


async def test_video_input_is_rejected(service: FakeService) -> None:
    clip = republic.video(b"mp4", media_type="video/mp4")

    with pytest.raises(republic.UnsupportedFeatureError):
        await make_model(service).chat(republic.user("Describe", clip))


async def test_stream_events(service: FakeService) -> None:
    image_item = {"type": "image_generation_call", "result": base64.b64encode(b"png").decode()}
    service.reply_events([
        {"type": "response.created", "response": {}},
        {"type": "response.reasoning_summary_part.added", "summary_index": 0},
        {"type": "response.reasoning_summary_text.delta", "delta": "Plan"},
        {"type": "response.reasoning_summary_part.added", "summary_index": 1},
        {"type": "response.reasoning_summary_text.delta", "delta": "Answer"},
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {
            "type": "response.output_item.done",
            "item": {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]},
        },
        {"type": "response.output_item.added", "item": {**FUNCTION_CALL, "id": "fc_1", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"city":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '"Paris"}'},
        {"type": "response.output_item.done", "item": {**FUNCTION_CALL, "id": "fc_1"}},
        {"type": "response.output_item.done", "item": image_item},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-6-sol-2026",
                "status": "completed",
                "usage": {"input_tokens": 4, "output_tokens": 6, "output_tokens_details": {"reasoning_tokens": 2}},
            },
        },
    ])

    async with make_model(service).stream("Hi") as stream:
        events = [event async for event in stream]

    assert events[:-1] == [
        ReasoningDelta("Plan"),
        ReasoningDelta("\n\n"),
        ReasoningDelta("Answer"),
        TextDelta("Hel"),
        TextDelta("lo"),
        ToolCallDelta("call_1", "get_weather", '{"city":'),
        ToolCallDelta("call_1", "get_weather", '"Paris"}'),
        ToolCallReady(republic.ToolCall("call_1", "get_weather", '{"city":"Paris"}')),
        ImageReady(republic.Image("image/png", data=b"png")),
        BuiltinToolCallReady(republic.BuiltinToolCall("image_generation")),
        UsageDelta(republic.TokenUsage(input_tokens=4, output_tokens=6, reasoning_tokens=2)),
    ]
    assert isinstance(events[-1], Completed)
    assert stream.response.finish_reason == "tool_calls"
    assert (stream.response.id, stream.response.model) == ("resp_1", "gpt-6-sol-2026")
    assert stream.response.reasoning == "Plan\n\nAnswer"
    assert stream.text == "Hello"
    assert stream.token_usage == republic.TokenUsage(input_tokens=4, output_tokens=6, reasoning_tokens=2)


async def test_stream_failure_raises(service: FakeService) -> None:
    service.reply_events([{"type": "response.failed", "response": {"error": {"message": "overloaded"}}}])

    with pytest.raises(republic.APIResponseError, match="overloaded"):
        async with make_model(service).stream("Hi") as stream:
            async for _ in stream:
                pass


async def test_reading_unfinished_stream_raises(service: FakeService) -> None:
    service.reply_events([{"type": "response.output_text.delta", "delta": "Hel"}])

    async with make_model(service).stream("Hi") as stream:
        with pytest.raises(republic.StreamNotFinishedError):
            _ = stream.text


async def test_generation_options_map_to_responses_fields(service: FakeService) -> None:
    service.reply_json({"output": []})
    tool = republic.Tool("lookup")

    await make_model(service).chat("Hi", tools=[tool], tool_choice=tool, reasoning_effort="xhigh", top_p=0.5)

    body = service.body()
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["top_p"] == 0.5


async def test_options_without_a_responses_field_are_rejected(service: FakeService) -> None:
    with pytest.raises(republic.UnsupportedFeatureError, match="stop, seed"):
        await make_model(service).chat("Hi", stop=["END"], seed=1)


async def test_tools_send_explicit_strict_and_reasoning_summary(service: FakeService) -> None:
    service.reply_json({"output": []})

    await make_model(service).chat(
        "Hi", tools=[republic.Tool("lookup")], reasoning_effort="high", include_reasoning=True
    )

    body = service.body()
    assert body["tools"][0]["strict"] is False
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}


async def test_refusal_and_incomplete_status(service: FakeService) -> None:
    service.reply_json({
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}],
    })

    response = await make_model(service).chat("Hi")

    assert response.refusal == "No."
    assert response.finish_reason == "length"
    assert response.id == "resp_1"


async def test_stream_accepts_function_calls_delivered_whole(service: FakeService) -> None:
    service.reply_events([
        {"type": "response.output_item.done", "item": FUNCTION_CALL},
        {"type": "response.completed", "response": {"status": "completed"}},
    ])

    async with make_model(service).stream("Hi") as stream:
        events = [event async for event in stream]

    assert events[:2] == [
        ToolCallDelta("call_1", "get_weather", '{"city":"Paris"}'),
        ToolCallReady(republic.ToolCall("call_1", "get_weather", '{"city":"Paris"}')),
    ]
    assert stream.response.finish_reason == "tool_calls"


class Location(pydantic.BaseModel):
    city: str


class Trip(pydantic.BaseModel):
    origin: Location = pydantic.Field(description="Where the trip starts")
    note: str | None = None


async def test_structured_output_schema_meets_strict_mode(service: FakeService) -> None:
    text = '{"origin": {"city": "Paris"}, "note": null}'
    service.reply_json({"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]})

    response = await make_model(service).chat("Plan", output_schema=Trip)

    text_format = service.body()["text"]["format"]
    schema = text_format["schema"]
    assert (text_format["name"], text_format["strict"]) == ("Trip", True)
    assert (schema["additionalProperties"], schema["required"]) == (False, ["origin", "note"])
    assert "default" not in schema["properties"]["note"]
    assert schema["properties"]["origin"]["additionalProperties"] is False
    assert schema["properties"]["origin"]["description"] == "Where the trip starts"
    assert schema["$defs"]["Location"]["additionalProperties"] is False
    assert response.output == Trip(origin=Location(city="Paris"))
