"""Context building for tape entries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery


class _LastAnchor:
    def __repr__(self) -> str:
        return "LAST_ANCHOR"


LAST_ANCHOR = _LastAnchor()
AnchorSelector: TypeAlias = str | None | _LastAnchor
SelectedMessages: TypeAlias = list[dict[str, Any]] | Coroutine[Any, Any, list[dict[str, Any]]]
ContextSelector: TypeAlias = Callable[[Iterable[TapeEntry], "TapeContext"], SelectedMessages]
MESSAGE_PART_TEXT = "text"


@dataclass(frozen=True)
class TapeContext:
    """Rules for selecting tape entries into a prompt context.

    anchor: LAST_ANCHOR for the most recent anchor, None for the full tape, or an anchor name.
    select: Optional selector called after anchor slicing that returns messages.
    state: Optional state dictionary to be passed along with the context.
    """

    anchor: AnchorSelector = LAST_ANCHOR
    select: ContextSelector | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def build_query(self, query: TapeQuery) -> TapeQuery:
        if self.anchor is None:
            return query
        if isinstance(self.anchor, _LastAnchor):
            return query.last_anchor()
        return query.after_anchor(self.anchor)


def build_messages(entries: Iterable[TapeEntry], context: TapeContext) -> SelectedMessages:
    if context.select is not None:
        return context.select(entries, context)
    return _default_messages(entries)


def _default_messages(entries: Iterable[TapeEntry]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if entry.kind != "message":
            continue
        payload = entry.payload
        if not isinstance(payload, dict):
            continue
        message = _model_message_from_tape_payload(payload)
        if message is not None:
            messages.append(message)
    return messages


def _model_message_from_tape_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    message = dict(payload)
    if "parts" not in message:
        return message

    parts = message.pop("parts", None)
    content = _text_content_from_parts(parts)
    if content:
        message["content"] = content
        return message
    if message.get("tool_calls"):
        return message
    return None


def _text_content_from_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") != MESSAGE_PART_TEXT:
            continue
        content = part.get("content")
        if isinstance(content, str):
            text_parts.append(content)
    return "".join(text_parts)
