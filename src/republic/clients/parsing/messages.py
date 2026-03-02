"""Anthropic messages parsing.

Currently any-llm exposes Anthropic messages in completion-compatible payload
shapes, so this parser intentionally reuses completion parsing behavior.
"""

from __future__ import annotations

from republic.clients.parsing.completion import PARSER

__all__ = ["PARSER"]
