"""Query helpers for tape entries."""

from __future__ import annotations

import inspect
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, overload

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.view import TAPE_NOW, TAPE_START, TapeInfo, TapeView, entry_offset, offset_id

if TYPE_CHECKING:
    from republic.tape.store import AsyncTapeStore, TapeStore

T = TypeVar("T", bound="TapeStore | AsyncTapeStore", covariant=True)


def _is_anchor_entry(entry: TapeEntry) -> bool:
    return entry.kind == tape_anchor.TAPE_ANCHOR_KIND


def _is_close_entry(entry: TapeEntry) -> bool:
    return tape_anchor.name(entry) == tape_anchor.TAPE_CLOSE_ANCHOR


def _closed_entry(entries: Sequence[TapeEntry]) -> TapeEntry | None:
    for entry in entries:
        if _is_close_entry(entry):
            return entry
    return None


def _visible_entries(entries: Sequence[TapeEntry]) -> list[TapeEntry]:
    closed = _closed_entry(entries)
    boundary = closed.id if closed is not None else None
    return [entry for entry in entries if not _is_anchor_entry(entry) and (boundary is None or entry.id < boundary)]


def _tail_offset(entries: Sequence[TapeEntry]) -> str:
    visible = _visible_entries(entries)
    if visible:
        return entry_offset(visible[-1])
    closed = _closed_entry(entries)
    if closed is not None:
        return entry_offset(closed)
    return TAPE_START


def _is_close_offset(entries: Sequence[TapeEntry], entry_id: int) -> bool:
    return len(entries) == 1 and entries[0].id == entry_id and _is_close_entry(entries[0])


def _tape_info(entries: Sequence[TapeEntry]) -> TapeInfo:
    closed = _closed_entry(entries)
    return TapeInfo(
        tail_offset=_tail_offset(entries),
        entry_count=len(_visible_entries(entries)),
        closed=closed is not None,
        closed_offset=entry_offset(closed) if closed is not None else None,
    )


def _tape_view(entries: list[TapeEntry], query: TapeQuery[Any]) -> TapeView:
    offset = _tail_offset(entries) if query.offset == TAPE_NOW else query.offset
    start_id = offset_id(offset)
    closed = _closed_entry(entries)
    closed_id = closed.id if closed is not None and query.stops_at_close else None
    view_entries: list[TapeEntry] = []
    next_offset = offset

    for entry in entries:
        if entry.id <= start_id:
            continue
        if closed_id is not None and entry.id > closed_id:
            break
        if _is_anchor_entry(entry):
            if query.includes_anchors:
                view_entries.append(entry.copy())
            next_offset = entry_offset(entry)
            if query.stops_at_close and _is_close_entry(entry):
                return TapeView(tuple(view_entries), next_offset, up_to_date=True, closed=True)
            continue
        view_entries.append(entry.copy())
        next_offset = entry_offset(entry)
        if query.limit_value is not None and len(view_entries) >= query.limit_value:
            return TapeView(tuple(view_entries), next_offset, up_to_date=False, closed=False)

    return TapeView(
        entries=tuple(view_entries),
        next_offset=next_offset,
        up_to_date=True,
        closed=closed is not None,
    )


async def _async_view(query: TapeQuery[AsyncTapeStore]) -> TapeView:
    tape, store = query._bound()
    if query.offset == TAPE_NOW:
        entries = await store.read(tape)
    else:
        start_id = offset_id(query.offset)
        entries = await store.read(tape, after=start_id)
        if not entries and start_id > 0:
            current = await store.read(tape, after=start_id - 1, limit=1)
            if query.stops_at_close and _is_close_offset(current, start_id):
                return TapeView((), query.offset, up_to_date=True, closed=True)
    return _tape_view(entries, query)


async def _async_info(query: TapeQuery[AsyncTapeStore]) -> TapeInfo:
    tape, store = query._bound()
    return _tape_info(await store.read(tape))


@dataclass(frozen=True)
class TapeQuery(Generic[T]):
    tape: str | None = None
    store: T | None = None
    _query: str | None = None
    _after_anchor: str | None = None
    _after_last: bool = False
    _offset: str = TAPE_START
    _between_anchors: tuple[str, str] | None = None
    _between_dates: tuple[str, str] | None = None
    _kinds: tuple[str, ...] = field(default_factory=tuple)
    _limit: int | None = None
    _include_anchors: bool = False
    _stop_at_close: bool = True

    @property
    def offset(self) -> str:
        return self._offset

    @property
    def limit_value(self) -> int | None:
        return self._limit

    @property
    def includes_anchors(self) -> bool:
        return self._include_anchors

    @property
    def stops_at_close(self) -> bool:
        return self._stop_at_close

    def bind(self, tape: str, store: T) -> TapeQuery[T]:
        return replace(self, tape=tape, store=store)

    def query(self, value: str) -> Self:
        return replace(self, _query=value)

    def after_anchor(self, name: str) -> Self:
        if not name:
            return replace(self, _after_anchor=None, _after_last=False)
        return replace(self, _after_anchor=name, _after_last=False)

    def last_anchor(self) -> Self:
        return replace(self, _after_anchor=None, _after_last=True)

    def after_offset(self, offset: str) -> Self:
        offset_id(offset)
        return replace(self, _offset=offset)

    def now(self) -> Self:
        return replace(self, _offset=TAPE_NOW)

    def start(self) -> Self:
        return replace(self, _offset=TAPE_START)

    def between_anchors(self, start: str, end: str) -> Self:
        return replace(self, _between_anchors=(start, end))

    def between_dates(self, start: str | date_type, end: str | date_type) -> Self:
        start_value = start.isoformat() if isinstance(start, date_type) else start
        end_value = end.isoformat() if isinstance(end, date_type) else end
        return replace(self, _between_dates=(start_value, end_value))

    def kinds(self, *kinds: str) -> Self:
        return replace(self, _kinds=kinds)

    def limit(self, value: int) -> Self:
        if value < 1:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Tape query limit must be positive.")
        return replace(self, _limit=value)

    def include_anchors(self, value: bool = True) -> Self:
        return replace(self, _include_anchors=value)

    def stop_at_close(self, value: bool = True) -> Self:
        return replace(self, _stop_at_close=value)

    @overload
    def all(self: TapeQuery[TapeStore]) -> list[TapeEntry]: ...

    @overload
    async def all(self: TapeQuery[AsyncTapeStore]) -> list[TapeEntry]: ...

    def all(self) -> list[TapeEntry] | Coroutine[Any, Any, list[TapeEntry]]:
        _, store = self._bound()
        entries = store.fetch_all(self)
        if inspect.isawaitable(entries):
            return entries
        return list(entries)

    @overload
    def view(self: TapeQuery[TapeStore]) -> TapeView: ...

    @overload
    async def view(self: TapeQuery[AsyncTapeStore]) -> TapeView: ...

    def view(self) -> TapeView | Coroutine[Any, Any, TapeView]:
        tape, store = self._bound()
        if inspect.iscoroutinefunction(store.read):
            return _async_view(self)  # type: ignore[arg-type]

        if self.offset == TAPE_NOW:
            entries = store.read(tape)
        else:
            start_id = offset_id(self.offset)
            entries = store.read(tape, after=start_id)
            if (
                not entries
                and start_id > 0
                and self.stops_at_close
                and _is_close_offset(store.read(tape, after=start_id - 1, limit=1), start_id)
            ):
                return TapeView((), self.offset, up_to_date=True, closed=True)
        return _tape_view(entries, self)

    @overload
    def info(self: TapeQuery[TapeStore]) -> TapeInfo: ...

    @overload
    async def info(self: TapeQuery[AsyncTapeStore]) -> TapeInfo: ...

    def info(self) -> TapeInfo | Coroutine[Any, Any, TapeInfo]:
        tape, store = self._bound()
        if inspect.iscoroutinefunction(store.read):
            return _async_info(self)  # type: ignore[arg-type]
        return _tape_info(store.read(tape))

    def _bound(self) -> tuple[str, T]:
        if self.tape is None or self.store is None:
            raise RepublicError(ErrorKind.INVALID_INPUT, "TapeQuery must be bound to a tape and store before all().")
        return self.tape, self.store
