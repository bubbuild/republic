from __future__ import annotations

import base64

import pydantic
import pytest

import republic
from republic.events import (
    Completed,
    ImageReady,
    ReasoningDelta,
    RefusalDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallReady,
    UsageDelta,
)
from tests.conftest import FakeService

WEATHER = republic.Tool(
    "get_weather",
    "Look up the weather",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
PIXEL = base64.b64encode(b"\x89PNG fake").decode()


class Forecast(pydantic.BaseModel):
    city: str
    sunny: bool


def make_model(service: FakeService) -> republic.ChatModel:
    return republic.get_model("openrouter:vendor/model", api_key="key", api_format="chat", http_client=service.client())


async def test_chat_sends_messages_and_reads_text_and_usage(service: FakeService) -> None:
    service.reply_json({
        "choices": [{"message": {"content": "Fine, thanks."}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}},
    })

    response = await make_model(service).chat([
        republic.system("Be brief."),
        republic.user("Hello", republic.image(b"img", media_type="image/png")),
    ])

    assert service.body()["messages"] == [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
            ],
        },
    ]
    assert response.text == "Fine, thanks."
    assert response.token_usage == republic.TokenUsage(input_tokens=7, output_tokens=3, cached_tokens=4)
    assert response.token_usage.total_tokens == 10


async def test_tool_calls_round_trip_through_tool_results(service: FakeService) -> None:
    service.reply_json({
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                }
            }
        ]
    })
    service.reply_json({"choices": [{"message": {"content": "Sunny in Paris."}}]})
    model = make_model(service)

    first = await model.chat("Weather in Paris?", tools=[WEATHER])
    call = first.tool_calls[0]
    await model.chat(["Weather in Paris?", republic.assistant(tool_results=[republic.tool_result(call, "sunny")])])

    assert service.body(0)["tools"] == [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Look up the weather", "parameters": WEATHER.parameters},
        }
    ]
    assert call.args == {"city": "Paris"}
    assert service.body()["messages"][1:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]


async def test_structured_output_is_requested_and_parsed(service: FakeService) -> None:
    service.reply_json({"choices": [{"message": {"content": '{"city": "Paris", "sunny": true}'}}]})

    response = await make_model(service).chat("Forecast?", output_schema=Forecast)

    assert service.body()["response_format"]["json_schema"]["name"] == "Forecast"
    assert response.output == Forecast(city="Paris", sunny=True)


async def test_generated_images_are_decoded(service: FakeService) -> None:
    service.reply_json({
        "choices": [
            {
                "message": {
                    "content": "Here it is.",
                    "images": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PIXEL}"}}],
                }
            }
        ]
    })

    response = await make_model(service).chat("Draw a cat")

    assert response.image_parts == [republic.Image("image/png", data=b"\x89PNG fake")]


async def test_stream_yields_text_tool_calls_and_completion(service: FakeService) -> None:
    service.reply_events([
        {"choices": [{"delta": {"reasoning_content": "Weather needs a tool."}}]},
        {"choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"content": "check."}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}}]},
        {"id": "chatcmpl-1", "model": "vendor/model-2", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 4}},
        },
        "[DONE]",
    ])

    async with make_model(service).stream("Weather?", tools=[WEATHER]) as stream:
        events = [event async for event in stream]

    assert service.body()["stream"] is True
    assert events[:-1] == [
        ReasoningDelta("Weather needs a tool."),
        TextDelta("Let me "),
        TextDelta("check."),
        ToolCallDelta("call_1", "get_weather", '{"city":'),
        ToolCallDelta("call_1", "get_weather", '"Paris"}'),
        UsageDelta(republic.TokenUsage(input_tokens=5, output_tokens=9, reasoning_tokens=4)),
        ToolCallReady(republic.ToolCall("call_1", "get_weather", '{"city":"Paris"}')),
    ]
    assert isinstance(events[-1], Completed)
    assert stream.response.reasoning == "Weather needs a tool."
    assert stream.response.finish_reason == "tool_calls"
    assert (stream.response.id, stream.response.model) == ("chatcmpl-1", "vendor/model-2")
    assert stream.text == "Let me check."
    assert stream.token_usage == republic.TokenUsage(input_tokens=5, output_tokens=9, reasoning_tokens=4)
    assert not any(isinstance(event, ImageReady) for event in events)


async def test_stream_parses_structured_output_at_the_end(service: FakeService) -> None:
    service.reply_events([
        {"choices": [{"delta": {"content": '{"city": "Paris",'}}]},
        {"choices": [{"delta": {"content": ' "sunny": false}'}}]},
        "[DONE]",
    ])

    async with make_model(service).stream("Forecast?", output_schema=Forecast) as stream:
        chunks = [event.chunk async for event in stream if isinstance(event, TextDelta)]

    assert "".join(chunks) == '{"city": "Paris", "sunny": false}'
    assert stream.output == Forecast(city="Paris", sunny=False)


async def test_embedding(service: FakeService) -> None:
    service.reply_json({
        "model": "embedder-v2",
        "data": [{"index": 1, "embedding": [0.3]}, {"index": 0, "embedding": [0.1]}],
        "usage": {"prompt_tokens": 4},
    })
    model = republic.get_embedding_model("openrouter:embedder", http_client=service.client())

    response = await model.embed_many(["Hello", "World"], dimensions=1)

    assert service.requests[0].url.path == "/api/v1/embeddings"
    assert service.body() == {
        "model": "embedder",
        "input": ["Hello", "World"],
        "encoding_format": "float",
        "dimensions": 1,
    }
    assert response.vectors == [[0.1], [0.3]]
    assert response.vector == [0.1]
    assert response.model == "embedder-v2"
    assert response.token_usage.input_tokens == 4


async def test_images_decode_with_pillow(service: FakeService) -> None:
    from io import BytesIO

    from PIL import Image as PILImage

    buffer = BytesIO()
    PILImage.new("RGB", (2, 3)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    service.reply_json({
        "choices": [{"message": {"images": [{"image_url": {"url": f"data:image/png;base64,{encoded}"}}]}}]
    })

    response = await make_model(service).chat("Draw")

    assert response.images[0].size == (2, 3)


async def test_generation_options_map_to_chat_fields(service: FakeService) -> None:
    service.reply_json({"choices": [{"message": {"content": "ok"}}]})

    await make_model(service).chat(
        "Hi",
        tools=[WEATHER],
        tool_choice=WEATHER,
        parallel_tool_calls=False,
        max_tokens=100,
        temperature=0.3,
        top_p=0.9,
        stop=["END"],
        seed=7,
        reasoning_effort="high",
        extra_body={"top_k": 20},
    )

    body = service.body()
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert {key: body[key] for key in body if key not in {"model", "messages", "tools", "tool_choice"}} == {
        "parallel_tool_calls": False,
        "max_completion_tokens": 100,
        "temperature": 0.3,
        "top_p": 0.9,
        "stop": ["END"],
        "seed": 7,
        "reasoning_effort": "high",
        "top_k": 20,
    }


async def test_refusal_skips_structured_output(service: FakeService) -> None:
    service.reply_json({
        "choices": [{"message": {"content": None, "refusal": "I can't help with that."}, "finish_reason": "stop"}]
    })

    response = await make_model(service).chat("Forecast?", output_schema=Forecast)

    assert response.refusal == "I can't help with that."
    assert response.finish_reason == "refusal"
    assert response.output is None


async def test_stream_refusal_and_length_finish(service: FakeService) -> None:
    service.reply_events([
        {"choices": [{"delta": {"refusal": "No."}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        "[DONE]",
    ])

    async with make_model(service).stream("Hi") as stream:
        events = [event async for event in stream]

    assert events[0] == RefusalDelta("No.")
    assert stream.response.finish_reason == "length"


async def test_strict_tools_and_sampling_options(service: FakeService) -> None:
    service.reply_json({"choices": [{"message": {"content": "ok"}}]})
    strict_tool = republic.Tool("lookup", parameters=WEATHER.parameters, strict=True)

    await make_model(service).chat(
        "Hi", tools=[strict_tool], top_k=40, presence_penalty=0.5, frequency_penalty=0.2, include_reasoning=True
    )

    body = service.body()
    assert body["tools"][0]["function"]["strict"] is True
    assert (body["top_k"], body["presence_penalty"], body["frequency_penalty"]) == (40, 0.5, 0.2)
    assert "include_reasoning" not in body


async def test_provider_headers_are_sent(service: FakeService) -> None:
    service.reply_json({"choices": [{"message": {"content": "ok"}}]})
    model = republic.get_model(
        "openrouter:vendor/model",
        api_format="chat",
        headers={"HTTP-Referer": "https://example.com"},
        http_client=service.client(),
    )

    await model.chat("Hi")

    assert service.requests[0].headers["http-referer"] == "https://example.com"


async def test_stream_can_only_be_iterated_once(service: FakeService) -> None:
    service.reply_events([{"choices": [{"delta": {"content": "Hi"}}]}, "[DONE]"])

    async with make_model(service).stream("Hi") as stream:
        [event async for event in stream]
        with pytest.raises(RuntimeError, match="once"):
            [event async for event in stream]

    assert stream.text == "Hi"
