from __future__ import annotations

import pytest

import republic
from republic._formats import ApiFormatName
from republic.events import BuiltinToolCallReady, CitationAdded
from republic.tools import BuiltinTool, CodeExecution, ImageGeneration, NativeTool, UserLocation, WebFetch, WebSearch
from tests.conftest import FakeService

PARIS = UserLocation(city="Paris", country="FR")
LOOKUP = republic.Tool("lookup")


def make_model(service: FakeService, spec: str, api_format: ApiFormatName | None = None) -> republic.ChatModel:
    return republic.get_model(spec, api_key="key", api_format=api_format, http_client=service.client())


class TestResponses:
    async def test_builtin_tools_map_to_native_tools(self, service: FakeService) -> None:
        service.reply_json({"output": []})
        file_search = NativeTool("responses", {"type": "file_search", "vector_store_ids": ["vs_1"]})

        await make_model(service, "openai:gpt-6-sol").chat(
            "Hi",
            tools=[
                LOOKUP,
                WebSearch(allowed_domains=["example.com"], user_location=PARIS),
                CodeExecution(),
                ImageGeneration(),
                file_search,
            ],
        )

        body = service.body()
        assert body["tools"][1:] == [
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["example.com"]},
                "user_location": {"type": "approximate", "city": "Paris", "country": "FR"},
            },
            {"type": "code_interpreter", "container": {"type": "auto"}},
            {"type": "image_generation"},
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
        ]
        assert body["include"] == ["web_search_call.action.sources"]

    @pytest.mark.parametrize(
        "tool",
        [WebSearch(max_uses=3), WebFetch(), NativeTool("messages", {"type": "bash_20250124", "name": "bash"})],
    )
    async def test_unsupported_tools_are_rejected(self, service: FakeService, tool: BuiltinTool) -> None:
        with pytest.raises(republic.UnsupportedFeatureError):
            await make_model(service, "openai:gpt-6-sol").chat("Hi", tools=[tool])

    async def test_search_results_round_trip(self, service: FakeService) -> None:
        search_call = {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "paris weather"},
        }
        message = {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": "Sunny.",
                    "annotations": [{"type": "url_citation", "url": "https://weather.example", "title": "Weather"}],
                }
            ],
        }
        service.reply_json({"output": [search_call, message]})
        service.reply_json({"output": []})
        model = make_model(service, "openai:gpt-6-sol")

        response = await model.chat("Weather in Paris?", tools=[WebSearch()])
        await model.chat(["Weather in Paris?", response.message, "Thanks"])

        assert response.citations == (republic.Citation("https://weather.example", title="Weather"),)
        assert response.builtin_tool_calls == (
            republic.BuiltinToolCall("web_search", input={"action": search_call["action"]}, id="ws_1"),
        )
        assert service.body()["input"][1:3] == [search_call, {"role": "assistant", "content": "Sunny."}]

    async def test_stream_reports_citations_and_builtin_calls(self, service: FakeService) -> None:
        service.reply_events([
            {"type": "response.output_text.delta", "delta": "Sunny."},
            {
                "type": "response.output_text.annotation.added",
                "annotation": {"type": "url_citation", "url": "https://weather.example"},
            },
            {"type": "response.output_item.done", "item": {"type": "web_search_call", "id": "ws_1", "action": {}}},
        ])

        async with make_model(service, "openai:gpt-6-sol").stream("Hi", tools=[WebSearch()]) as stream:
            events = [event async for event in stream]

        assert CitationAdded(republic.Citation("https://weather.example")) in events
        assert BuiltinToolCallReady(republic.BuiltinToolCall("web_search", input={"action": {}}, id="ws_1")) in events


class TestMessages:
    async def test_builtin_tools_map_to_server_tools(self, service: FakeService) -> None:
        service.reply_json({"content": [], "usage": {"input_tokens": 1}})

        await make_model(service, "anthropic:claude-opus-5-5").chat(
            "Hi",
            tools=[
                WebSearch(max_uses=2, allowed_domains=["example.com"], user_location=PARIS),
                WebFetch(blocked_domains=["private.example"]),
                CodeExecution(),
            ],
        )

        assert service.body()["tools"] == [
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 2,
                "allowed_domains": ["example.com"],
                "user_location": {"type": "approximate", "city": "Paris", "country": "FR"},
            },
            {"type": "web_fetch_20260209", "name": "web_fetch", "blocked_domains": ["private.example"]},
            {"type": "code_execution_20260521", "name": "code_execution"},
        ]

    async def test_image_generation_is_rejected(self, service: FakeService) -> None:
        with pytest.raises(republic.UnsupportedFeatureError, match="ImageGeneration"):
            await make_model(service, "anthropic:claude-opus-5-5").chat("Hi", tools=[ImageGeneration()])

    async def test_server_tool_blocks_round_trip(self, service: FakeService) -> None:
        tool_use = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "paris"}}
        tool_result = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "url": "https://weather.example", "title": "Weather"}],
        }
        text = {
            "type": "text",
            "text": "Sunny.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://weather.example",
                    "title": "Weather",
                    "cited_text": "Sunny all week",
                }
            ],
        }
        service.reply_json({"content": [tool_use, tool_result, text], "stop_reason": "end_turn", "usage": {}})
        service.reply_json({"content": [], "usage": {}})
        model = make_model(service, "anthropic:claude-opus-5-5")

        response = await model.chat("Weather?", tools=[WebSearch()])
        await model.chat(["Weather?", response.message, "Thanks"])

        assert response.builtin_tool_calls == (
            republic.BuiltinToolCall(
                "web_search", input={"query": "paris"}, output=tool_result["content"], id="srvtoolu_1"
            ),
        )
        assert response.citations == (
            republic.Citation("https://weather.example", title="Weather", cited_text="Sunny all week"),
        )
        assert service.body()["messages"][1] == {
            "role": "assistant",
            "content": [tool_use, tool_result, {"type": "text", "text": "Sunny."}],
        }

    async def test_stream_assembles_server_tool_input_and_reports_pause(self, service: FakeService) -> None:
        service.reply_events([
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query"'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ': "paris"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "citations_delta", "citation": {"url": "https://weather.example"}},
            },
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "pause_turn"}, "usage": {"output_tokens": 5}},
        ])

        async with make_model(service, "anthropic:claude-opus-5-5").stream("Hi", tools=[WebSearch()]) as stream:
            events = [event async for event in stream]

        call = republic.BuiltinToolCall("web_search", input={"query": "paris"}, output=[], id="srvtoolu_1")
        assert BuiltinToolCallReady(call) in events
        assert stream.response.citations == (republic.Citation("https://weather.example"),)
        assert stream.response.finish_reason == "pause"
        assert stream.response.message.parts[0] == republic.ProviderData(
            "messages",
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "paris"}},
        )


class TestGemini:
    async def test_builtin_tools_map_to_native_tools(self, service: FakeService) -> None:
        service.reply_json({"candidates": []})

        await make_model(service, "google:gemini-3-pro").chat(
            "Hi", tools=[LOOKUP, WebSearch(), WebFetch(), CodeExecution()]
        )

        assert service.body()["tools"][1:] == [{"googleSearch": {}}, {"urlContext": {}}, {"codeExecution": {}}]

    @pytest.mark.parametrize("tool", [WebSearch(allowed_domains=["example.com"]), ImageGeneration()])
    async def test_unsupported_tools_are_rejected(self, service: FakeService, tool: BuiltinTool) -> None:
        with pytest.raises(republic.UnsupportedFeatureError):
            await make_model(service, "google:gemini-3-pro").chat("Hi", tools=[tool])

    async def test_code_execution_and_grounding_round_trip(self, service: FakeService) -> None:
        code = {"executableCode": {"language": "PYTHON", "code": "print(2 + 2)"}}
        result = {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "4\n"}}
        service.reply_json({
            "candidates": [
                {
                    "content": {"role": "model", "parts": [code, result, {"text": "It is 4."}]},
                    "groundingMetadata": {
                        "webSearchQueries": ["two plus two"],
                        "groundingChunks": [{"web": {"uri": "https://math.example", "title": "Math"}}],
                    },
                    "finishReason": "STOP",
                }
            ]
        })
        service.reply_json({"candidates": []})
        model = make_model(service, "google:gemini-3-pro")

        response = await model.chat("2 + 2?", tools=[CodeExecution(), WebSearch()])
        await model.chat(["2 + 2?", response.message, "Thanks"])

        assert response.builtin_tool_calls == (
            republic.BuiltinToolCall(
                "code_execution", input=code["executableCode"], output=result["codeExecutionResult"]
            ),
            republic.BuiltinToolCall("web_search", input={"queries": ["two plus two"]}),
        )
        assert response.citations == (republic.Citation("https://math.example", title="Math"),)
        assert service.body()["contents"][1] == {"role": "model", "parts": [code, result, {"text": "It is 4."}]}


class TestChat:
    async def test_web_search_maps_to_search_options(self, service: FakeService) -> None:
        service.reply_json({
            "choices": [
                {
                    "message": {
                        "content": "Sunny.",
                        "annotations": [
                            {"type": "url_citation", "url_citation": {"url": "https://weather.example", "title": "W"}}
                        ],
                    }
                }
            ]
        })

        response = await make_model(service, "openrouter:vendor/model", "chat").chat(
            "Hi", tools=[WebSearch(user_location=UserLocation(country="FR"))]
        )

        assert service.body()["web_search_options"] == {
            "user_location": {"type": "approximate", "approximate": {"country": "FR"}}
        }
        assert "tools" not in service.body()
        assert response.citations == (republic.Citation("https://weather.example", title="W"),)

    async def test_code_execution_is_rejected(self, service: FakeService) -> None:
        with pytest.raises(republic.UnsupportedFeatureError, match="CodeExecution"):
            await make_model(service, "openrouter:vendor/model", "chat").chat("Hi", tools=[CodeExecution()])
