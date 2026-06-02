"""Offset stream views over append-only tapes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.store import AsyncTapeStore, TapeStore

TAPE_START = "-1"
TAPE_NOW = "now"
OFFSET_WIDTH = 20


@dataclass(frozen=True)
class TapeStreamView:
    """A stream view read from an opaque tape offset."""

    entries: tuple[TapeEntry, ...]
    next_offset: str
    up_to_date: bool
    closed: bool = False


@dataclass(frozen=True)
class TapeStreamInfo:
    """Current stream position and lifecycle state."""

    tail_offset: str
    entry_count: int
    closed: bool = False
    closed_offset: str | None = None


@dataclass(frozen=True)
class TapeStreamQuery:
    """Rules for reading one view from a tape stream."""

    _offset: str = TAPE_START
    _limit: int | None = None
    _include_anchors: bool = False
    _stop_at_close: bool = True

    def after_offset(self, offset: str) -> TapeStreamQuery:
        offset_id(offset)
        return replace(self, _offset=offset)

    def now(self) -> TapeStreamQuery:
        return replace(self, _offset=TAPE_NOW)

    def start(self) -> TapeStreamQuery:
        return replace(self, _offset=TAPE_START)

    def limit(self, value: int) -> TapeStreamQuery:
        if value < 1:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Tape stream read limit must be positive.")
        return replace(self, _limit=value)

    def include_anchors(self, value: bool = True) -> TapeStreamQuery:
        return replace(self, _include_anchors=value)

    def stop_at_close(self, value: bool = True) -> TapeStreamQuery:
        return replace(self, _stop_at_close=value)


def entry_offset(entry: TapeEntry) -> str:
    """Return the opaque offset for a stored tape entry."""

    if entry.id < 1:
        raise RepublicError(ErrorKind.INVALID_INPUT, "Only stored tape entries have offsets.")
    return f"{entry.id:0{OFFSET_WIDTH}d}"


def offset_id(offset: str) -> int:
    """Return the stored entry id represented by an opaque tape offset."""

    if offset == TAPE_START:
        return 0
    if offset == TAPE_NOW:
        raise RepublicError(ErrorKind.INVALID_INPUT, "'now' must be resolved against a concrete tape.")
    if len(offset) != OFFSET_WIDTH or not offset.isdigit():
        raise RepublicError(ErrorKind.INVALID_INPUT, f"Invalid tape offset: '{offset}'.")
    offset_id = int(offset)
    if offset_id == 0:
        raise RepublicError(ErrorKind.INVALID_INPUT, f"Invalid tape offset: '{offset}'. Use TAPE_START.")
    return offset_id


def _is_anchor_entry(entry: TapeEntry) -> bool:
    return entry.kind == tape_anchor.TAPE_ANCHOR_KIND


def _is_close_entry(entry: TapeEntry) -> bool:
    return tape_anchor.name(entry) == tape_anchor.TAPE_CLOSE_ANCHOR


def _closed_entry(entries: list[TapeEntry]) -> TapeEntry | None:
    for entry in entries:
        if _is_close_entry(entry):
            return entry
    return None


def _visible_entries(entries: list[TapeEntry]) -> list[TapeEntry]:
    closed = _closed_entry(entries)
    boundary = closed.id if closed is not None else None
    return [entry for entry in entries if not _is_anchor_entry(entry) and (boundary is None or entry.id < boundary)]


def _tail_offset(entries: list[TapeEntry]) -> str:
    visible = _visible_entries(entries)
    if visible:
        return entry_offset(visible[-1])
    closed = _closed_entry(entries)
    if closed is not None:
        return entry_offset(closed)
    return TAPE_START


def _is_close_offset(entries: list[TapeEntry], offset_entry_id: int) -> bool:
    return len(entries) == 1 and entries[0].id == offset_entry_id and _is_close_entry(entries[0])


def _read_view(
    stored_entries: list[TapeEntry],
    *,
    query: TapeStreamQuery,
) -> TapeStreamView:
    if query._limit is not None and query._limit < 1:
        raise RepublicError(ErrorKind.INVALID_INPUT, "Tape stream read limit must be positive.")
    offset = query._offset
    if offset == TAPE_NOW:
        offset = _tail_offset(stored_entries)
    start_id = offset_id(offset)
    closed = _closed_entry(stored_entries)
    closed_id = closed.id if closed is not None and query._stop_at_close else None
    view_entries: list[TapeEntry] = []
    next_offset = offset

    for entry in stored_entries:
        if entry.id <= start_id:
            continue
        if closed_id is not None and entry.id > closed_id:
            break
        if _is_anchor_entry(entry):
            if query._include_anchors:
                view_entries.append(entry.copy())
            next_offset = entry_offset(entry)
            if query._stop_at_close and _is_close_entry(entry):
                return TapeStreamView(tuple(view_entries), next_offset, up_to_date=True, closed=True)
            continue
        view_entries.append(entry.copy())
        next_offset = entry_offset(entry)
        if query._limit is not None and len(view_entries) >= query._limit:
            return TapeStreamView(tuple(view_entries), next_offset, up_to_date=False, closed=False)

    return TapeStreamView(
        entries=tuple(view_entries),
        next_offset=next_offset,
        up_to_date=True,
        closed=closed is not None,
    )


def _stream_info(entries: list[TapeEntry]) -> TapeStreamInfo:
    closed = _closed_entry(entries)
    return TapeStreamInfo(
        tail_offset=_tail_offset(entries),
        entry_count=len(_visible_entries(entries)),
        closed=closed is not None,
        closed_offset=entry_offset(closed) if closed is not None else None,
    )


def _validate_record_kind(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise RepublicError(ErrorKind.INVALID_INPUT, "Tape entry kind must be a non-empty string.")
    if kind in tape_anchor.STRUCTURAL_ENTRY_KINDS:
        raise RepublicError(ErrorKind.INVALID_INPUT, f"'{kind}' is reserved for structural tape entries.")


class TapeStream(Protocol):
    """Schemaless, offset-addressed tape stream contract."""

    def append(
        self,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> str: ...

    def anchor(self, name: str, payload: Any | None = None, **meta: Any) -> str: ...

    def read(self, query: TapeStreamQuery | None = None) -> TapeStreamView: ...

    def close(self, payload: Any | None = None) -> str: ...

    def info(self) -> TapeStreamInfo: ...


class AsyncTapeStream(Protocol):
    """Async schemaless, offset-addressed tape stream contract."""

    async def append(
        self,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> str: ...

    async def anchor(self, name: str, payload: Any | None = None, **meta: Any) -> str: ...

    async def read(self, query: TapeStreamQuery | None = None) -> TapeStreamView: ...

    async def close(self, payload: Any | None = None) -> str: ...

    async def info(self) -> TapeStreamInfo: ...


class _TapeStreamHandle:
    """Concrete stream handle backed by a TapeManager."""

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

    def read(self, query: TapeStreamQuery | None = None) -> TapeStreamView:
        active_query = query or TapeStreamQuery()
        if active_query._offset == TAPE_NOW:
            entries = self._store.read(self._tape)
        else:
            start_id = offset_id(active_query._offset)
            entries = self._store.read(self._tape, after=start_id)
            if (
                not entries
                and start_id > 0
                and active_query._stop_at_close
                and _is_close_offset(self._store.read(self._tape, after=start_id - 1, limit=1), start_id)
            ):
                return TapeStreamView((), active_query._offset, up_to_date=True, closed=True)
        return _read_view(entries, query=active_query)

    def close(self, payload: Any | None = None) -> str:
        entries = self._store.read(self._tape)
        closed = _closed_entry(entries)
        if closed is not None:
            return entry_offset(closed)
        stored = self._store.append(self._tape, TapeEntry.anchor(tape_anchor.TAPE_CLOSE_ANCHOR, payload))
        return entry_offset(stored)

    def info(self) -> TapeStreamInfo:
        return _stream_info(self._store.read(self._tape))


class _AsyncTapeStreamHandle:
    """Concrete async stream handle backed by an AsyncTapeManager."""

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

    async def read(self, query: TapeStreamQuery | None = None) -> TapeStreamView:
        active_query = query or TapeStreamQuery()
        if active_query._offset == TAPE_NOW:
            entries = await self._store.read(self._tape)
        else:
            start_id = offset_id(active_query._offset)
            entries = await self._store.read(self._tape, after=start_id)
            if not entries and start_id > 0:
                current = await self._store.read(self._tape, after=start_id - 1, limit=1)
                if active_query._stop_at_close and _is_close_offset(current, start_id):
                    return TapeStreamView((), active_query._offset, up_to_date=True, closed=True)
        return _read_view(entries, query=active_query)

    async def close(self, payload: Any | None = None) -> str:
        entries = await self._store.read(self._tape)
        closed = _closed_entry(entries)
        if closed is not None:
            return entry_offset(closed)
        stored = await self._store.append(self._tape, TapeEntry.anchor(tape_anchor.TAPE_CLOSE_ANCHOR, payload))
        return entry_offset(stored)

    async def info(self) -> TapeStreamInfo:
        return _stream_info(await self._store.read(self._tape))
