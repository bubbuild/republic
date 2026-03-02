from __future__ import annotations

from republic.clients.parsing import parser_for_transport
from republic.clients.parsing import responses as responses_parser

from .fakes import make_responses_function_call, make_responses_response


def test_parser_for_transport_returns_expected_modules() -> None:
    assert parser_for_transport("completion").__name__.endswith(".completion")
    assert parser_for_transport("responses").__name__.endswith(".responses")
    assert parser_for_transport("messages").__name__.endswith(".messages")


def test_responses_extract_tool_calls_accepts_full_response() -> None:
    response = make_responses_response(tool_calls=[make_responses_function_call("echo", '{"text":"tokyo"}')])
    calls = responses_parser.extract_tool_calls(response)
    assert calls[0]["function"]["name"] == "echo"
