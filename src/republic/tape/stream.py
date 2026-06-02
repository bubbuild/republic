"""Append-only tape stream handles."""

from __future__ import annotations

from typing import Any

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery
from republic.tape.store import AsyncTapeStore, TapeStore
from republic.tape.view import (
    TAPE_NOW,
    TapeInfo,
    TapeView,
    closed_entry,
    entry_offset,
    is_close_offset,
    offset_id,
    read_tape_view,
    tape_info,
)


def _validate_record_kind(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise RepublicError(ErrorKind.INVALID_INPUT, "Tape entry kind must be a non-empty string.")
    if kind in tape_anchor.STRUCTURAL_ENTRY_KINDS:
        raise RepublicError(ErrorKind.INVALID_INPUT, f"'{kind}' is reserved for structural tape entries.")


def _read_view(entries: list[TapeEntry], query: TapeQuery[Any]) -> TapeView:
    return read_tape_view(
        entries,
        offset=query.offset,
        limit=query.limit_value,
        include_anchors=query.includes_anchors,
        stop_at_close=query.stops_at_close,
    )


class TapeStream:
    """Schemaless, offset-addressed stream for one named tape."""

    def __init__(self, store: TapeStore, tape: str) -> None:
        self._store = store
        self._tape = tape

    def append(
        self,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> str:
        _validate_record_kind(kind)
        if self.info().closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append to a closed tape stream.")
        stored = self._store.append(
            self._tape,
            TapeEntry.record(payload, kind=kind, content_type=content_type, **meta),
        )
        return entry_offset(stored)

    def anchor(self, name: str, payload: Any | None = None, **meta: Any) -> str:
        tape_anchor.validate(name, custom=True)
        if self.info().closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append anchor entries to a closed tape stream.")
        stored = self._store.append(self._tape, TapeEntry.anchor(name, payload, **meta))
        return entry_offset(stored)

    def read(self, query: TapeQuery[Any] | None = None) -> TapeView:
        active_query = query or TapeQuery()
        if active_query.offset == TAPE_NOW:
            entries = self._store.read(self._tape)
        else:
            start_id = offset_id(active_query.offset)
            entries = self._store.read(self._tape, after=start_id)
            if (
                not entries
                and start_id > 0
                and active_query.stops_at_close
                and is_close_offset(self._store.read(self._tape, after=start_id - 1, limit=1), start_id)
            ):
                return TapeView((), active_query.offset, up_to_date=True, closed=True)
        return _read_view(entries, active_query)

    def close(self, payload: Any | None = None) -> str:
        entries = self._store.read(self._tape)
        closed = closed_entry(entries)
        if closed is not None:
            return entry_offset(closed)
        stored = self._store.append(self._tape, TapeEntry.anchor(tape_anchor.TAPE_CLOSE_ANCHOR, payload))
        return entry_offset(stored)

    def info(self) -> TapeInfo:
        return tape_info(self._store.read(self._tape))


class AsyncTapeStream:
    """Async schemaless, offset-addressed stream for one named tape."""

    def __init__(self, store: AsyncTapeStore, tape: str) -> None:
        self._store = store
        self._tape = tape

    async def append(
        self,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> str:
        _validate_record_kind(kind)
        if (await self.info()).closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append to a closed tape stream.")
        stored = await self._store.append(
            self._tape,
            TapeEntry.record(payload, kind=kind, content_type=content_type, **meta),
        )
        return entry_offset(stored)

    async def anchor(self, name: str, payload: Any | None = None, **meta: Any) -> str:
        tape_anchor.validate(name, custom=True)
        if (await self.info()).closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append anchor entries to a closed tape stream.")
        stored = await self._store.append(self._tape, TapeEntry.anchor(name, payload, **meta))
        return entry_offset(stored)

    async def read(self, query: TapeQuery[Any] | None = None) -> TapeView:
        active_query = query or TapeQuery()
        if active_query.offset == TAPE_NOW:
            entries = await self._store.read(self._tape)
        else:
            start_id = offset_id(active_query.offset)
            entries = await self._store.read(self._tape, after=start_id)
            if not entries and start_id > 0:
                current = await self._store.read(self._tape, after=start_id - 1, limit=1)
                if active_query.stops_at_close and is_close_offset(current, start_id):
                    return TapeView((), active_query.offset, up_to_date=True, closed=True)
        return _read_view(entries, active_query)

    async def close(self, payload: Any | None = None) -> str:
        entries = await self._store.read(self._tape)
        closed = closed_entry(entries)
        if closed is not None:
            return entry_offset(closed)
        stored = await self._store.append(self._tape, TapeEntry.anchor(tape_anchor.TAPE_CLOSE_ANCHOR, payload))
        return entry_offset(stored)

    async def info(self) -> TapeInfo:
        return tape_info(await self._store.read(self._tape))
