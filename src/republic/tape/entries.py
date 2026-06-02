"""Tape entries for Republic."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import republic.tape.anchor as tape_anchor
from republic.core.results import RepublicError


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TapeEntry:
    """A single append-only entry in a tape."""

    id: int
    kind: str
    payload: Any
    meta: dict[str, Any] = field(default_factory=dict)
    date: str = field(default_factory=utc_now)

    def copy(self) -> TapeEntry:
        return TapeEntry(self.id, self.kind, copy.deepcopy(self.payload), copy.deepcopy(self.meta), self.date)

    @classmethod
    def record(
        cls,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> TapeEntry:
        entry_meta = dict(meta)
        if content_type is not None:
            entry_meta["content_type"] = content_type
        return cls(id=0, kind=kind, payload=copy.deepcopy(payload), meta=entry_meta)

    @classmethod
    def message(cls, message: dict[str, Any], **meta: Any) -> TapeEntry:
        return cls(id=0, kind="message", payload=copy.deepcopy(message), meta=dict(meta))

    @classmethod
    def system(cls, content: str, **meta: Any) -> TapeEntry:
        return cls(id=0, kind="system", payload={"content": content}, meta=dict(meta))

    @classmethod
    def anchor(cls, name: str, payload: Any | None = None, **meta: Any) -> TapeEntry:
        tape_anchor.validate(name)
        entry_meta = dict(meta)
        entry_meta[tape_anchor.TAPE_ANCHOR_NAME_KEY] = name
        return cls(
            id=0,
            kind=tape_anchor.TAPE_ANCHOR_KIND,
            payload=copy.deepcopy(payload),
            meta=entry_meta,
        )

    @classmethod
    def tool_call(cls, calls: list[dict[str, Any]], **meta: Any) -> TapeEntry:
        return cls(id=0, kind="tool_call", payload={"calls": copy.deepcopy(calls)}, meta=dict(meta))

    @classmethod
    def tool_result(cls, results: list[Any], **meta: Any) -> TapeEntry:
        return cls(id=0, kind="tool_result", payload={"results": copy.deepcopy(results)}, meta=dict(meta))

    @classmethod
    def error(cls, error: RepublicError, **meta: Any) -> TapeEntry:
        return cls(id=0, kind="error", payload=error.as_dict(), meta=dict(meta))

    @classmethod
    def event(cls, name: str, data: dict[str, Any] | None = None, **meta: Any) -> TapeEntry:
        payload: dict[str, Any] = {"name": name}
        if data is not None:
            payload["data"] = copy.deepcopy(data)
        return cls(id=0, kind="event", payload=payload, meta=dict(meta))
