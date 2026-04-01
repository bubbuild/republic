from __future__ import annotations

from types import SimpleNamespace

from republic.clients.chat import ChatClient
from republic.clients.parsing import parser_for_transport
from republic.clients.parsing.types import BaseTransportParser, ParsedChunk, ParsedResponse

from .fakes import make_responses_function_call, make_responses_response


def test_parser_for_transport_returns_parser_objects() -> None:
    parser = parser_for_transport("completion")
    assert isinstance(parser, BaseTransportParser)
    assert callable(parser.parse_response)
    assert callable(parser.parse_chunk)

    parser = parser_for_transport("responses")
    assert isinstance(parser, BaseTransportParser)
    assert callable(parser.parse_response)
    assert callable(parser.parse_chunk)

    parser = parser_for_transport("messages")
    assert isinstance(parser, BaseTransportParser)
    assert callable(parser.parse_response)
    assert callable(parser.parse_chunk)


def test_responses_extract_tool_calls_accepts_full_response() -> None:
    response = make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    responses_parser = parser_for_transport("responses")
    parsed = responses_parser.parse_response(response)
    assert isinstance(parsed, ParsedResponse)
    assert parsed.tool_calls[0]["function"]["name"] == "echo"


def test_chat_client_resolve_transport_treats_output_text_as_responses() -> None:
    payload = SimpleNamespace(output_text="hello")
    assert ChatClient._is_non_stream_response(payload) is True
    parsed = ChatClient._parse_response(payload)
    assert isinstance(parsed, ParsedResponse)
    assert parsed.text == "hello"


def test_responses_parse_chunk_captures_text_tool_and_completion_metadata() -> None:
    parser = parser_for_transport("responses")
    parsed = parser.parse_chunk({
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"tokyo"}',
        },
        "usage": {"total_tokens": 5},
    })

    assert isinstance(parsed, ParsedChunk)
    assert parsed.output_item_type == "function_call"
    assert parsed.tool_call_deltas
    assert parsed.usage == {"total_tokens": 5}


def test_completion_parser_preserves_tool_call_metadata() -> None:
    parser = parser_for_transport("completion")
    parsed = parser.parse_response({
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text":"tokyo"}'},
                            "thought_signature": "sig_123",
                        }
                    ],
                }
            }
        ]
    })

    assert parsed.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text":"tokyo"}'},
            "thought_signature": "sig_123",
        }
    ]


def test_completion_parser_supports_metadata_only_marker() -> None:
    parser = parser_for_transport("completion")
    parsed = parser.parse_response({
        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}}],
        "usage": {"total_tokens": 7},
        "republic_metadata_only": True,
    })

    assert parsed.metadata_only is True
    assert parsed.usage == {"total_tokens": 7}
