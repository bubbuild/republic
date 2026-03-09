"""Parsing helpers for provider response payloads."""

from __future__ import annotations

from republic.clients.parsing.completion import CompletionTransportParser
from republic.clients.parsing.responses import ResponseTransportParser
from republic.clients.parsing.types import BaseTransportParser, TransportKind

_PARSERS: dict[TransportKind, BaseTransportParser] = {
    "completion": CompletionTransportParser(),
    "responses": ResponseTransportParser(),
    "messages": CompletionTransportParser(),
}


def parser_for_transport(transport: TransportKind) -> BaseTransportParser:
    return _PARSERS[transport]


__all__ = ["BaseTransportParser", "TransportKind", "parser_for_transport"]
