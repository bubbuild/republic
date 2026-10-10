from __future__ import annotations

import base64

import pytest

import republic
from republic.events import ImageReady, ReasoningDelta, TextDelta, ToolCallDelta, ToolCallReady, UsageDelta
from tests.conftest import FakeService

VIDEO = base64.b64encode(b"mp4").decode()
PNG = base64.b64encode(b"png").decode()


def make_model(service: FakeService) -> republic.ChatModel:
    return republic.get_model("google:gemini-3-pro", api_key="key", http_client=service.client())


async def test_chat_builds_contents_and_reads_usage(service: FakeService) -> None:
    service.reply_json({
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "Thinking", "thought": True}, {"text": "A cat."}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 8,
            "cachedContentTokenCount": 6,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 2,
        },
    })

    response = await make_model(service).chat(
        [republic.system("Be brief."), republic.user("What is this?", republic.video(b"mp4", media_type="video/mp4"))],
        temperature=0.2,
    )

    assert service.requests[0].url.path == "/v1beta/models/gemini-3-pro:generateContent"
    assert service.requests[0].headers["x-goog-api-key"] == "key"
    assert service.body() == {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "What is this?"}, {"inlineData": {"mimeType": "video/mp4", "data": VIDEO}}],
            }
        ],
        "systemInstruction": {"parts": [{"text": "Be brief."}]},
        "generationConfig": {"temperature": 0.2},
    }
    assert response.text == "A cat."
    assert response.token_usage == republic.TokenUsage(
        input_tokens=8, output_tokens=5, reasoning_tokens=2, cached_tokens=6
    )
    assert response.reasoning == "Thinking"


async def test_function_calls_keep_signatures_and_omit_generated_ids(service: FakeService) -> None:
    service.reply_json({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "sig"}
                    ]
                }
            }
        ]
    })
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "Sunny."}]}}]})
    model = make_model(service)

    first = await model.chat("Weather?", tools=[republic.Tool("get_weather")])
    call = first.tool_calls[0]
    await model.chat(["Weather?", republic.tool(call, "sunny")])

    assert service.body(0)["tools"] == [
        {
            "functionDeclarations": [
                {"name": "get_weather", "description": "", "parametersJsonSchema": {"type": "object", "properties": {}}}
            ]
        }
    ]
    assert call.args == {"city": "Paris"}
    assert service.body()["contents"][1:] == [
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "sig"}],
        },
        {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"output": "sunny"}}}]},
    ]


async def test_tool_results_can_hold_multiple_parts(service: FakeService) -> None:
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "A cat."}]}}]})
    call = republic.ToolCall("call_1", "screenshot", "{}")
    result = republic.tool(call, "Captured", republic.Image("image/png", data=b"png"), is_error=True)

    await make_model(service).chat(["Look", result])

    assert service.body()["contents"][-1] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "screenshot",
                    "response": {"error": "Captured"},
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}],
                }
            }
        ],
    }


async def test_tool_results_reject_remote_media(service: FakeService) -> None:
    call = republic.ToolCall("call_1", "screenshot", "{}")
    result = republic.tool(call, "Captured", republic.Image("image/png", url="https://example.com/shot.png"))

    with pytest.raises(republic.errors.UnsupportedFeatureError, match="inline media"):
        await make_model(service).chat(["Look", result])

    assert service.requests == []


async def test_blocked_prompt_raises(service: FakeService) -> None:
    service.reply_json({"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(republic.errors.APIResponseError, match="SAFETY"):
        await make_model(service).chat("Hi")


async def test_stream_text_images_and_calls(service: FakeService) -> None:
    service.reply_events([
        {"candidates": [{"content": {"parts": [{"text": "Sketching first.", "thought": True}, {"text": "Here"}]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}}]},
        {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"id": "fc_1", "name": "save", "args": {}}}]},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3-pro-001",
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 4},
        },
    ])

    async with make_model(service).stream("Draw and save") as stream:
        events = [event async for event in stream]

    assert service.requests[0].url.path == "/v1beta/models/gemini-3-pro:streamGenerateContent"
    assert service.requests[0].url.params["alt"] == "sse"
    assert events[:-1] == [
        ReasoningDelta("Sketching first."),
        TextDelta("Here"),
        ImageReady(republic.Image("image/png", data=b"png")),
        ToolCallDelta("fc_1", "save", "{}"),
        ToolCallReady(republic.ToolCall("fc_1", "save", "{}")),
        UsageDelta(republic.TokenUsage(input_tokens=2, output_tokens=4)),
    ]
    assert stream.response.finish_reason == "tool_calls"
    assert stream.response.model == "gemini-3-pro-001"
    assert stream.token_usage == republic.TokenUsage(2, 4)


async def test_embedding(service: FakeService) -> None:
    service.reply_json({"embeddings": [{"values": [0.5, 0.25]}]})
    model = republic.get_embedding_model("google:gemini-embedding-001", http_client=service.client())

    response = await model.embed("Hello", dimensions=2)

    assert service.requests[0].url.path == "/v1beta/models/gemini-embedding-001:batchEmbedContents"
    assert service.body() == {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "Hello"}]},
                "outputDimensionality": 2,
            }
        ]
    }
    assert response.vector == [0.5, 0.25]


async def test_generation_options_map_to_generation_config(service: FakeService) -> None:
    service.reply_json({"candidates": []})
    tool = republic.Tool("lookup")

    await make_model(service).chat(
        "Hi", tools=[tool], tool_choice=tool, top_p=0.8, stop=["END"], seed=3, reasoning_effort="low", max_tokens=10
    )

    body = service.body()
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["lookup"]}}
    assert body["generationConfig"] == {
        "maxOutputTokens": 10,
        "topP": 0.8,
        "seed": 3,
        "stopSequences": ["END"],
        "thinkingConfig": {"thinkingLevel": "low"},
    }


async def test_parallel_tool_calls_is_rejected(service: FakeService) -> None:
    with pytest.raises(republic.errors.UnsupportedFeatureError, match="parallel_tool_calls"):
        await make_model(service).chat("Hi", parallel_tool_calls=False)


async def test_reasoning_and_sampling_options(service: FakeService) -> None:
    service.reply_json({"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]})

    response = await make_model(service).chat(
        "Hi", include_reasoning=True, top_k=10, presence_penalty=0.1, frequency_penalty=0.2
    )

    assert service.body()["generationConfig"] == {
        "topK": 10,
        "presencePenalty": 0.1,
        "frequencyPenalty": 0.2,
        "thinkingConfig": {"includeThoughts": True},
    }
    assert response.finish_reason == "content_filter"


async def test_strict_tools_are_rejected(service: FakeService) -> None:
    with pytest.raises(republic.errors.UnsupportedFeatureError, match="strict"):
        await make_model(service).chat("Hi", tools=[republic.Tool("lookup", strict=True)])
