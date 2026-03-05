"""Parsing helpers for provider response payloads."""

from __future__ import annotations

from republic.clients.parsing.completion import PARSER as completion_parser
from republic.clients.parsing.messages import PARSER as messages_parser
from republic.clients.parsing.responses import PARSER as responses_parser
from republic.clients.parsing.types import BaseTransportParser, TransportKind

_PARSERS: dict[TransportKind, BaseTransportParser] = {
    "completion": completion_parser,
    "responses": responses_parser,
    "messages": messages_parser,
}


def parser_for_transport(transport: TransportKind) -> BaseTransportParser:
    return _PARSERS[transport]


__all__ = ["BaseTransportParser", "TransportKind", "parser_for_transport"]
