"""Tape format primitives for Republic."""

from __future__ import annotations

import json
from collections.abc import Coroutine, Hashable, Iterable
from typing import Any, Protocol, TypeAlias

from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry

SelectedMessages: TypeAlias = list[dict[str, Any]] | Coroutine[Any, Any, list[dict[str, Any]]]


class TapeFormat(Protocol):
    """Owns tape entry shape, search semantics, and context injection for a tape."""

    name: str
    version: str

    def message(self, message: dict[str, Any], **meta: Any) -> TapeEntry: ...

    def system(self, content: str, **meta: Any) -> TapeEntry: ...

    def anchor(self, name: str, state: dict[str, Any] | None = None, **meta: Any) -> TapeEntry: ...

    def tool_call(self, calls: list[dict[str, Any]], **meta: Any) -> TapeEntry: ...

    def tool_result(self, results: list[Any], **meta: Any) -> TapeEntry: ...

    def error(self, error: RepublicError, **meta: Any) -> TapeEntry: ...

    def event(self, name: str, data: dict[str, Any] | None = None, **meta: Any) -> TapeEntry: ...

    def anchor_name(self, entry: TapeEntry) -> str | None: ...

    def entry_kind(self, entry: TapeEntry) -> str: ...

    def matches(self, entry: TapeEntry, query: str) -> bool:
        """Decide whether ``entry`` matches ``query``. Format owns the matching semantics."""
        ...

    def dedup_key(self, entry: TapeEntry) -> Hashable:
        """Return a hashable key for deduplicating semantically equivalent entries."""
        ...

    def select_messages(self, entries: Iterable[TapeEntry], context: object) -> SelectedMessages: ...


class RepublicTapeFormat:
    """Default tape format used by Republic."""

    name = "republic"
    version = "1"

    def message(self, message: dict[str, Any], **meta: Any) -> TapeEntry:
        return TapeEntry.message(message, **meta)

    def system(self, content: str, **meta: Any) -> TapeEntry:
        return TapeEntry.system(content, **meta)

    def anchor(self, name: str, state: dict[str, Any] | None = None, **meta: Any) -> TapeEntry:
        return TapeEntry.anchor(name, state=state, **meta)

    def tool_call(self, calls: list[dict[str, Any]], **meta: Any) -> TapeEntry:
        return TapeEntry.tool_call(calls, **meta)

    def tool_result(self, results: list[Any], **meta: Any) -> TapeEntry:
        return TapeEntry.tool_result(results, **meta)

    def error(self, error: RepublicError, **meta: Any) -> TapeEntry:
        return TapeEntry.error(error, **meta)

    def event(self, name: str, data: dict[str, Any] | None = None, **meta: Any) -> TapeEntry:
        return TapeEntry.event(name, data, **meta)

    def anchor_name(self, entry: TapeEntry) -> str | None:
        if entry.kind != "anchor":
            return None
        name = entry.payload.get("name")
        return name if isinstance(name, str) else None

    def entry_kind(self, entry: TapeEntry) -> str:
        return entry.kind

    def matches(self, entry: TapeEntry, query: str) -> bool:
        return query.casefold() in self._serialize(entry).casefold()

    def dedup_key(self, entry: TapeEntry) -> Hashable:
        return self._serialize(entry)

    def _serialize(self, entry: TapeEntry) -> str:
        return json.dumps(
            {
                "kind": self.entry_kind(entry),
                "date": entry.date,
                "payload": entry.payload,
                "meta": entry.meta,
            },
            sort_keys=True,
            default=str,
        )

    def select_messages(self, entries: Iterable[TapeEntry], context: object) -> SelectedMessages:
        del context
        messages: list[dict[str, Any]] = []
        for entry in entries:
            if entry.kind != "message":
                continue
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            messages.append(dict(payload))
        return messages


DEFAULT_TAPE_FORMAT = RepublicTapeFormat()
