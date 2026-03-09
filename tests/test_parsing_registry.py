from __future__ import annotations

from types import SimpleNamespace

from republic.clients.chat import ChatClient
from republic.clients.parsing import parser_for_transport
from republic.clients.parsing.types import BaseTransportParser

from .fakes import make_responses_function_call, make_responses_response


def test_parser_for_transport_returns_parser_objects() -> None:
    parser = parser_for_transport("completion")
    assert isinstance(parser, BaseTransportParser)
    assert callable(parser.extract_text)
    assert callable(parser.extract_tool_calls)
    assert callable(parser.extract_usage)

    parser = parser_for_transport("responses")
    assert isinstance(parser, BaseTransportParser)
    assert callable(parser.extract_text)
    assert callable(parser.extract_tool_calls)
    assert callable(parser.extract_usage)

    parser = parser_for_transport("messages")
    assert isinstance(parser, BaseTransportParser)
    assert callable(parser.extract_text)
    assert callable(parser.extract_tool_calls)
    assert callable(parser.extract_usage)


def test_responses_extract_tool_calls_accepts_full_response() -> None:
    response = make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    responses_parser = parser_for_transport("responses")
    calls = responses_parser.extract_tool_calls(response)
    assert calls[0]["function"]["name"] == "echo"


def test_chat_client_resolve_transport_treats_output_text_as_responses() -> None:
    payload = SimpleNamespace(output_text="hello")
    assert ChatClient._is_non_stream_response(payload) is True
    assert ChatClient._extract_text(payload) == "hello"
