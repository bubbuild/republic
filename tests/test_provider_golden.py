"""Provider correctness cases migrated from Goose and Fantasy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Unpack

import pytest

import republic
from republic.events import Completed, Event, ReasoningDelta, TextDelta, ToolCallReady, UsageDelta
from tests.conftest import FakeService

CASES = Path(__file__).with_suffix("") / "cases"
MODELS = {
    "chat": "openai:gpt-4o-mini",
    "responses": "openai:gpt-4o-mini",
    "messages": "anthropic:claude-sonnet-4-20250514",
    "gemini": "google:gemini-2.5-flash",
}
WEATHER = republic.Tool(
    "weather",
    "Get weather information for a location",
    {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "the city"}},
        "required": ["location"],
    },
)
NUMBERS = {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "description": "first number"},
        "b": {"type": "integer", "description": "second number"},
    },
    "required": ["a", "b"],
}
ARITHMETIC = [
    republic.Tool("add", "Add two numbers", NUMBERS),
    republic.Tool("multiply", "Multiply two numbers", NUMBERS),
]


@pytest.fixture(params=MODELS)
def api_format(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=[False, True], ids=["chat", "stream"])
def streaming(request: pytest.FixtureRequest) -> bool:
    return request.param


def load_case(service: FakeService, name: str) -> None:
    paths = sorted((CASES / name).iterdir())
    assert paths, "A provider case must have at least one response"
    for path in paths:
        content_type = "text/event-stream" if path.suffix == ".sse" else "application/json"
        service.reply_bytes(path.read_bytes(), content_type=content_type)


def make_model(service: FakeService, api_format: str) -> republic.ChatModel:
    return republic.get_model(
        MODELS[api_format], api_format=api_format, api_key="test-key", http_client=service.client()
    )


async def respond(
    model: republic.ChatModel,
    prompt: list[republic.Message],
    *,
    streaming: bool = False,
    **options: Unpack[republic.ChatOptions],
) -> tuple[republic.Response[None], list[Event]]:
    if not streaming:
        return await model.chat(prompt, **options), []

    async with model.stream(prompt, **options) as stream:
        events = [event async for event in stream]
        response = stream.response

    assert isinstance(events[-1], Completed)
    assert events[-1].response is response
    assert "".join(event.chunk for event in events if isinstance(event, TextDelta)) == response.text
    assert "".join(event.chunk for event in events if isinstance(event, ReasoningDelta)) == response.reasoning
    ready_calls = [event.call for event in events if isinstance(event, ToolCallReady)]
    assert len(ready_calls) == len(response.tool_calls)
    assert {call.id: call for call in ready_calls} == {call.id: call for call in response.tool_calls}
    assert {call.id: call.metadata for call in ready_calls} == {call.id: call.metadata for call in response.tool_calls}
    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens"):
        assert sum(getattr(event.usage, field) for event in events if isinstance(event, UsageDelta)) == getattr(
            response.token_usage, field
        )
    return response, events


def assert_fantasy_requests(
    service: FakeService,
    api_format: str,
    streaming: bool,
    prompt: list[republic.Message],
    tools: Sequence[republic.Tool] = (),
) -> None:
    action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    urls = {
        "chat": "https://api.openai.com/v1/chat/completions",
        "responses": "https://api.openai.com/v1/responses",
        "messages": "https://api.anthropic.com/v1/messages",
        "gemini": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:{action}",
    }
    for index, request in enumerate(service.requests):
        assert request.method == "POST"
        assert str(request.url) == urls[api_format]
        body = service.body(index)
        if api_format == "chat":
            assert body["model"] == "gpt-4o-mini"
            assert body["max_completion_tokens"] == 4000
            assert body.get("stream", False) is streaming
            assert body.get("stream_options") == ({"include_usage": True} if streaming else None)
            assert body["messages"][:2] == [
                {"role": "system", "content": prompt[0].text},
                {"role": "user", "content": prompt[1].text},
            ]
            assert body.get("tools", []) == [
                {
                    "type": "function",
                    "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
                }
                for tool in tools
            ]
        elif api_format == "responses":
            assert body["model"] == "gpt-4o-mini"
            assert body["max_output_tokens"] == 4000
            assert body.get("stream", False) is streaming
            assert body["input"][:2] == [
                {"role": "system", "content": prompt[0].text},
                {"role": "user", "content": [{"type": "input_text", "text": prompt[1].text}]},
            ]
            assert body.get("tools", []) == [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": False,
                }
                for tool in tools
            ]
        elif api_format == "messages":
            assert body["model"] == "claude-sonnet-4-20250514"
            assert body["max_tokens"] == 4000
            assert body.get("stream", False) is streaming
            assert body["system"] == prompt[0].text
            assert body["messages"][0] == {"role": "user", "content": [{"type": "text", "text": prompt[1].text}]}
            assert body.get("tools", []) == [
                {"name": tool.name, "description": tool.description, "input_schema": tool.parameters} for tool in tools
            ]
        else:
            assert body["generationConfig"] == {"maxOutputTokens": 4000}
            assert body["systemInstruction"] == {"parts": [{"text": prompt[0].text}]}
            assert body["contents"][0] == {"role": "user", "parts": [{"text": prompt[1].text}]}
            declarations = [
                {"name": tool.name, "description": tool.description, "parametersJsonSchema": tool.parameters}
                for tool in tools
            ]
            assert body.get("tools", []) == ([{"functionDeclarations": declarations}] if tools else [])
        if tools:
            assert (
                body.get("toolConfig" if api_format == "gemini" else "tool_choice")
                == {
                    "chat": "auto",
                    "responses": "auto",
                    "messages": {"type": "auto"},
                    "gemini": {"functionCallingConfig": {"mode": "AUTO"}},
                }[api_format]
            )


def assert_gemini_recorded_signatures(case_name: str, calls: list[republic.ToolCall], *, streaming: bool) -> None:
    path = CASES / case_name / ("0.sse" if streaming else "0.json")
    # These recorded first tool responses each contain one JSON payload.
    payload = json.loads(path.read_text().removeprefix("data: "))
    expected = {
        part["functionCall"]["name"]: part.get("thoughtSignature")
        for part in payload["candidates"][0]["content"]["parts"]
        if "functionCall" in part
    }
    assert {call.name: call.metadata.get("gemini_thought_signature") for call in calls} == expected


def assert_tool_exchange(
    service: FakeService, api_format: str, calls: list[republic.ToolCall], outputs: dict[str, str]
) -> None:
    body = service.body()
    if api_format == "chat":
        assistant = body["messages"][2]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"] == [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls
        ]
        assert body["messages"][-len(calls) :] == [
            {"role": "tool", "tool_call_id": call.id, "content": outputs[call.name]} for call in calls
        ]
    elif api_format == "responses":
        assert [item for item in body["input"] if item.get("type") == "function_call"] == [
            {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments}
            for call in calls
        ]
        assert body["input"][-len(calls) :] == [
            {"type": "function_call_output", "call_id": call.id, "output": outputs[call.name]} for call in calls
        ]
    elif api_format == "messages":
        assert body["messages"][1]["role"] == "assistant"
        assert [block for block in body["messages"][1]["content"] if block["type"] == "tool_use"] == [
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args} for call in calls
        ]
        assert body["messages"][-1]["content"] == [
            {"type": "tool_result", "tool_use_id": call.id, "content": outputs[call.name], "is_error": False}
            for call in calls
        ]
    else:
        assert body["contents"][-2]["role"] == "model"
        assert [part for part in body["contents"][-2]["parts"] if "functionCall" in part] == [
            {
                "functionCall": {"name": call.name, "args": call.args},
                **(
                    {"thoughtSignature": call.metadata["gemini_thought_signature"]}
                    if "gemini_thought_signature" in call.metadata
                    else {}
                ),
            }
            for call in calls
        ]
        assert body["contents"][-1]["parts"] == [
            {"functionResponse": {"name": call.name, "response": {"output": outputs[call.name]}}} for call in calls
        ]


async def test_fantasy_text(service: FakeService, api_format: str, streaming: bool) -> None:
    suffix = "_streaming" if streaming else ""
    load_case(service, f"fantasy_{api_format}_simple{suffix}")
    model = make_model(service, api_format)
    prompt = [republic.system("You are a helpful assistant"), republic.user("Say hi in Portuguese")]

    response, _ = await respond(
        model,
        prompt,
        streaming=streaming,
        max_tokens=4000,
    )

    assert any(greeting in response.text for greeting in ("Oi", "oi", "Olá", "olá"))
    assert response.tool_calls == []
    assert response.finish_reason == "stop"
    assert response.token_usage.input_tokens > 0
    assert response.token_usage.output_tokens > 0
    assert len(service.requests) == 1
    assert_fantasy_requests(service, api_format, streaming, prompt)


async def test_fantasy_weather_tool(service: FakeService, api_format: str, streaming: bool) -> None:
    suffix = "_streaming" if streaming else ""
    case_name = f"fantasy_{api_format}_tool{suffix}"
    load_case(service, case_name)
    model = make_model(service, api_format)
    prompt = [republic.system("You are a helpful assistant"), republic.user("What's the weather in Florence,Italy?")]

    first, _ = await respond(model, prompt, streaming=streaming, max_tokens=4000, tools=[WEATHER], tool_choice="auto")

    assert first.finish_reason == "tool_calls"
    assert len(first.tool_calls) == 1
    call = first.tool_calls[0]
    assert call.id
    assert call.name == "weather"
    assert isinstance(call.args, dict)
    assert "Florence" in call.args["location"]
    if api_format == "gemini":
        assert_gemini_recorded_signatures(case_name, first.tool_calls, streaming=streaming)

    second, _ = await respond(
        model,
        [*prompt, first.message, republic.assistant(tool_results=[republic.tool_result(call, "40 C")])],
        streaming=streaming,
        max_tokens=4000,
        tools=[WEATHER],
        tool_choice="auto",
    )

    assert "Florence" in second.text
    assert "40" in second.text
    assert second.tool_calls == []
    assert second.finish_reason == "stop"
    assert len(service.requests) == 2
    assert_fantasy_requests(service, api_format, streaming, prompt, [WEATHER])
    assert_tool_exchange(service, api_format, [call], {"weather": "40 C"})


async def test_fantasy_parallel_tools(service: FakeService, api_format: str, streaming: bool) -> None:
    suffix = "_streaming" if streaming else ""
    case_name = f"fantasy_{api_format}_multi_tool{suffix}"
    load_case(service, case_name)
    model = make_model(service, api_format)
    system = (
        "You are a helpful assistant. Always use both add and multiply at the same time."
        if streaming
        else "You are a helpful assistant. CRITICAL: Always use both add and multiply at the same time ALWAYS."
    )
    prompt = [republic.system(system), republic.user("Add and multiply the number 2 and 3")]

    first, _ = await respond(model, prompt, streaming=streaming, max_tokens=4000, tools=ARITHMETIC, tool_choice="auto")

    assert first.finish_reason == "tool_calls"
    assert len(first.tool_calls) == 2
    assert {call.name for call in first.tool_calls} == {"add", "multiply"}
    assert all(call.id for call in first.tool_calls)
    assert len({call.id for call in first.tool_calls}) == 2
    assert all(call.args == {"a": 2, "b": 3} for call in first.tool_calls)
    if api_format == "gemini":
        assert_gemini_recorded_signatures(case_name, first.tool_calls, streaming=streaming)
    outputs = {"add": "5", "multiply": "6"}

    second, _ = await respond(
        model,
        [
            *prompt,
            first.message,
            republic.assistant(
                tool_results=[republic.tool_result(call, outputs[call.name]) for call in first.tool_calls]
            ),
        ],
        streaming=streaming,
        max_tokens=4000,
        tools=ARITHMETIC,
        tool_choice="auto",
    )

    assert "5" in second.text
    assert "6" in second.text
    assert second.tool_calls == []
    assert second.finish_reason == "stop"
    assert len(service.requests) == 2
    assert_fantasy_requests(service, api_format, streaming, prompt, ARITHMETIC)
    assert_tool_exchange(service, api_format, first.tool_calls, outputs)


async def test_goose_length_finish_reason(service: FakeService) -> None:
    load_case(service, "goose_chat_length")

    response, _ = await respond(make_model(service, "chat"), [republic.user("Hello")])

    assert response.text == "Partial answer"
    assert response.finish_reason == "length"


async def test_goose_empty_tool_arguments(service: FakeService) -> None:
    load_case(service, "goose_chat_empty_arguments")

    response, _ = await respond(make_model(service, "chat"), [republic.user("Hello")])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "1"
    assert response.tool_calls[0].name == "example_fn"
    assert response.tool_calls[0].args == {}


async def test_goose_anthropic_cache_write_usage(service: FakeService) -> None:
    load_case(service, "goose_messages_cache_write")

    response, _ = await respond(make_model(service, "messages"), [republic.user("Hello")])

    assert response.text == "Hello! How can I assist you today?"
    assert response.token_usage == republic.TokenUsage(input_tokens=24, output_tokens=15, cache_write_tokens=12)
    assert response.token_usage.total_tokens == 39


async def test_goose_unsigned_thinking(service: FakeService) -> None:
    load_case(service, "goose_messages_unsigned_thinking")

    response, _ = await respond(make_model(service, "messages"), [republic.user("Hello")])

    assert response.reasoning == "internal reasoning"
    assert response.text == ""
    assert response.message.parts[0] == republic.ProviderData(
        "messages", {"type": "thinking", "thinking": "internal reasoning"}
    )


async def test_goose_thinking_signature_fragments_round_trip(service: FakeService) -> None:
    load_case(service, "goose_messages_thinking_streaming")
    model = make_model(service, "messages")
    prompt = [republic.user("Hello")]

    response, _ = await respond(model, prompt, streaming=True)

    thinking = {"type": "thinking", "thinking": "Let me analyze this problem.", "signature": "sig_abc123"}
    assert response.reasoning == thinking["thinking"]
    assert response.text == "Here is the answer."
    assert response.token_usage == republic.TokenUsage(input_tokens=10, output_tokens=25)
    block = response.message.parts[0]
    assert isinstance(block, republic.ProviderData)

    await respond(model, [*prompt, response.message, republic.user("Continue.")], streaming=True)

    outgoing = service.body()["messages"][1]["content"][0]
    assert outgoing == block.payload
    assert block == republic.ProviderData("messages", thinking)
    assert outgoing == thinking


async def test_goose_redacted_thinking_round_trip(service: FakeService) -> None:
    load_case(service, "goose_messages_redacted_streaming")
    model = make_model(service, "messages")
    prompt = [republic.user("Hello")]

    response, _ = await respond(model, prompt, streaming=True)

    block = {"type": "redacted_thinking", "data": "opaque_base64_data"}
    assert response.reasoning == ""
    assert response.text == "Done."
    assert response.message.parts[0] == republic.ProviderData("messages", block)

    await respond(model, [*prompt, response.message, republic.user("Continue.")], streaming=True)

    assert service.body()["messages"][1]["content"][0] == block


async def test_goose_interleaved_tool_arguments(service: FakeService) -> None:
    load_case(service, "goose_messages_parallel_streaming")

    response, events = await respond(make_model(service, "messages"), [republic.user("Hello")], streaming=True)

    assert len(response.tool_calls) == 2
    assert {call.name: (call.id, call.args) for call in response.tool_calls} == {
        "search": ("tool_a", {"query": "rust"}),
        "write": ("tool_b", {"path": "/tmp/a.md"}),  # noqa: S108 -- Recorded argument; no filesystem access.
    }
    assert {event.call.id for event in events if isinstance(event, ToolCallReady)} == {"tool_a", "tool_b"}


async def test_goose_cache_usage_survives_stream_deltas(service: FakeService) -> None:
    load_case(service, "goose_messages_cache_streaming")

    response, _ = await respond(make_model(service, "messages"), [republic.user("Hello")], streaming=True)

    assert response.text == "Hi"
    assert response.id == "msg_1"
    assert response.finish_reason == "stop"
    assert response.token_usage == republic.TokenUsage(
        input_tokens=15007, output_tokens=25, cached_tokens=5000, cache_write_tokens=10000
    )
    assert response.token_usage.total_tokens == 15032


@pytest.mark.parametrize(
    "case_name,expected_usage,expected_total",
    [
        ("goose_gemini_cached", republic.TokenUsage(input_tokens=100, output_tokens=20, cached_tokens=80), 120),
        (
            "goose_gemini_thinking_usage",
            republic.TokenUsage(input_tokens=100, output_tokens=250, reasoning_tokens=200),
            350,
        ),
    ],
    ids=["cached", "thinking"],
)
async def test_goose_gemini_usage(
    service: FakeService, case_name: str, expected_usage: republic.TokenUsage, expected_total: int
) -> None:
    load_case(service, case_name)

    response, _ = await respond(make_model(service, "gemini"), [republic.user("Hello")])

    assert response.token_usage == expected_usage
    assert response.token_usage.total_tokens == expected_total


async def test_goose_distinct_gemini_signatures_round_trip(service: FakeService) -> None:
    load_case(service, "goose_gemini_signature")
    model = make_model(service, "gemini")
    prompt = [republic.user("List files")]

    first, _ = await respond(model, prompt)

    assert first.reasoning == "Let me think..."
    assert first.text == ""
    assert len(first.tool_calls) == 2
    assert all(call.id for call in first.tool_calls)
    assert len({call.id for call in first.tool_calls}) == 2
    assert {call.name: call.args for call in first.tool_calls} == {"shell": {"cmd": "ls"}, "read": {}}
    shell = next(call for call in first.tool_calls if call.name == "shell")
    assert shell.metadata["gemini_thought_signature"] == "thought_sig_abc"
    prompt.extend([
        first.message,
        republic.assistant(tool_results=[republic.tool_result(call, "output") for call in first.tool_calls]),
    ])

    second, _ = await respond(model, prompt)

    assert len(second.tool_calls) == 1
    echo = second.tool_calls[0]
    assert echo.name == "echo"
    assert echo.args == {}
    assert echo.id not in {call.id for call in first.tool_calls}
    assert echo.metadata["gemini_thought_signature"] == "sig_456"
    prompt.extend([second.message, republic.assistant(tool_results=[republic.tool_result(echo, "output")])])

    final, _ = await respond(model, prompt)

    assert final.text == "Done!"
    assert final.reasoning == ""
    assert final.tool_calls == []
    contents = service.body()["contents"]
    assert contents[1]["parts"][0]["thoughtSignature"] == "thought_sig_abc"
    assert contents[3]["parts"][0]["thoughtSignature"] == "sig_456"
    assert_tool_exchange(service, "gemini", [echo], {"echo": "output"})


async def test_goose_responses_keepalive_and_done(service: FakeService) -> None:
    load_case(service, "goose_responses_keepalive_streaming")

    response, _ = await respond(make_model(service, "responses"), [republic.user("Hello")], streaming=True)

    assert response.text == "Hello world"
    assert response.id == "resp_1"
    assert response.model == "gpt-5.2-pro"
    assert response.token_usage == republic.TokenUsage(input_tokens=10, output_tokens=4, cached_tokens=6)
    assert response.token_usage.total_tokens == 14


async def test_goose_refusal(service: FakeService) -> None:
    load_case(service, "goose_responses_refusal")

    response, _ = await respond(make_model(service, "responses"), [republic.user("Hello")])

    assert response.refusal == "I cannot help with that request."
    assert response.finish_reason == "refusal"


async def test_goose_function_call_uses_call_id(service: FakeService) -> None:
    load_case(service, "goose_responses_call_id")

    response, _ = await respond(make_model(service, "responses"), [republic.user("Hello")])

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "test__get_person_zip_code"
    assert call.args == {"name": "Alice Burns"}
