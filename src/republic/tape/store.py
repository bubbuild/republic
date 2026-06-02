"""Tape stores for Republic."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from collections.abc import Sequence
from datetime import UTC, datetime, time
from datetime import date as date_type
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeGuard

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.view import TAPE_NOW, offset_id

if TYPE_CHECKING:
    from republic.tape.query import TapeQuery


class TapeStore(Protocol):
    """Append-only tape persistence interface."""

    def list_tapes(self) -> list[str]: ...

    def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]: ...

    def fetch_all(self, query: TapeQuery) -> list[TapeEntry]: ...

    def append(self, tape: str, entry: TapeEntry) -> TapeEntry: ...


class AsyncTapeStore(Protocol):
    """Async append-only tape persistence interface."""

    async def list_tapes(self) -> list[str]: ...

    async def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]: ...

    async def fetch_all(self, query: TapeQuery) -> list[TapeEntry]: ...

    async def append(self, tape: str, entry: TapeEntry) -> TapeEntry: ...


def is_async_tape_store(store: TapeStore | AsyncTapeStore) -> TypeGuard[AsyncTapeStore]:
    return hasattr(store, "append") and inspect.iscoroutinefunction(store.append)


def _anchor_index(
    entries: Sequence[TapeEntry],
    name: str | None,
    *,
    default: int,
    forward: bool,
    start: int = 0,
) -> int:
    rng = range(start, len(entries)) if forward else range(len(entries) - 1, start - 1, -1)
    for idx in rng:
        anchor_name = tape_anchor.name(entries[idx])
        if anchor_name is None:
            continue
        if name is not None and anchor_name != name:
            continue
        return idx
    return default


def _parse_datetime_boundary(value: str, *, is_end: bool) -> datetime:
    if "T" not in value and " " not in value:
        try:
            parsed_date = date_type.fromisoformat(value)
        except ValueError:
            pass
        else:
            boundary_time = time.max if is_end else time.min
            return datetime.combine(parsed_date, boundary_time, tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed_date = date_type.fromisoformat(value)
        except ValueError as exc:
            raise RepublicError(ErrorKind.INVALID_INPUT, f"Invalid ISO date or datetime: '{value}'.") from exc
        boundary_time = time.max if is_end else time.min
        parsed = datetime.combine(parsed_date, boundary_time, tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _entry_in_datetime_range(entry: TapeEntry, start_dt: datetime, end_dt: datetime) -> bool:
    entry_dt = _parse_datetime_boundary(entry.date, is_end=False)
    return start_dt <= entry_dt <= end_dt


def _entry_matches_query(entry: TapeEntry, query: str) -> bool:
    needle = query.casefold()
    haystack = json.dumps(
        {
            "kind": entry.kind,
            "date": entry.date,
            "payload": entry.payload,
            "meta": entry.meta,
        },
        sort_keys=True,
        default=str,
    ).casefold()
    return needle in haystack


def _anchor_query_slice(entries: Sequence[TapeEntry], query: TapeQuery) -> list[TapeEntry]:
    start_index = 0
    end_index: int | None = None

    if query._between_anchors is not None:
        start_name, end_name = query._between_anchors
        start_idx = _anchor_index(entries, start_name, default=-1, forward=False)
        if start_idx < 0:
            raise RepublicError(ErrorKind.NOT_FOUND, f"Anchor '{start_name}' was not found.")
        end_idx = _anchor_index(entries, end_name, default=-1, forward=True, start=start_idx + 1)
        if end_idx < 0:
            raise RepublicError(ErrorKind.NOT_FOUND, f"Anchor '{end_name}' was not found.")
        start_index = min(start_idx + 1, len(entries))
        end_index = min(max(start_index, end_idx), len(entries))
    elif query._after_last:
        anchor_index = _anchor_index(entries, None, default=-1, forward=False)
        if anchor_index < 0:
            raise RepublicError(ErrorKind.NOT_FOUND, "No anchors found in tape.")
        start_index = min(anchor_index + 1, len(entries))
    elif query._after_anchor is not None:
        anchor_index = _anchor_index(entries, query._after_anchor, default=-1, forward=False)
        if anchor_index < 0:
            raise RepublicError(ErrorKind.NOT_FOUND, f"Anchor '{query._after_anchor}' was not found.")
        start_index = min(anchor_index + 1, len(entries))

    return list(entries[start_index:end_index])


def _date_query_filter(entries: list[TapeEntry], query: TapeQuery) -> list[TapeEntry]:
    if query._between_dates is None:
        return entries

    start_date, end_date = query._between_dates
    start_dt = _parse_datetime_boundary(start_date, is_end=False)
    end_dt = _parse_datetime_boundary(end_date, is_end=True)
    if start_dt > end_dt:
        raise RepublicError(ErrorKind.INVALID_INPUT, "Start date must be earlier than or equal to end date.")
    return [entry for entry in entries if _entry_in_datetime_range(entry, start_dt, end_dt)]


def _text_query_filter(entries: list[TapeEntry], query: TapeQuery) -> list[TapeEntry]:
    if not query._query:
        return entries
    return [entry for entry in entries if _entry_matches_query(entry, query._query)]


def _kind_query_filter(entries: list[TapeEntry], query: TapeQuery) -> list[TapeEntry]:
    if not query._kinds:
        return entries
    return [entry for entry in entries if entry.kind in query._kinds]


def _limit_query_filter(entries: list[TapeEntry], query: TapeQuery) -> list[TapeEntry]:
    if query.limit_value is None:
        return entries
    return entries[: query.limit_value]


class InMemoryQueryMixin:
    """Reusable in-memory query implementation for simple TapeStore backends."""

    def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]:
        raise NotImplementedError("InMemoryQueryMixin requires read() to be implemented.")

    def fetch_all(self, query: TapeQuery) -> list[TapeEntry]:
        if query.tape is None:
            raise RepublicError(ErrorKind.INVALID_INPUT, "TapeQuery must be bound to a tape before fetch_all().")
        if query.offset == TAPE_NOW:
            return []

        entries = self.read(query.tape, after=offset_id(query.offset))
        selected = _anchor_query_slice(entries, query)
        selected = _date_query_filter(selected, query)
        selected = _text_query_filter(selected, query)
        selected = _kind_query_filter(selected, query)
        return _limit_query_filter(selected, query)


class InMemoryTapeStore(InMemoryQueryMixin):
    """In-memory tape storage (not thread-safe)."""

    def __init__(self) -> None:
        self._tapes: dict[str, list[TapeEntry]] = {}
        self._next_id: dict[str, int] = {}

    def list_tapes(self) -> list[str]:
        return sorted(self._tapes.keys())

    def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]:
        if after < 0:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Tape read offset must be non-negative.")
        if limit is not None and limit < 1:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Tape read limit must be positive.")
        entries = [entry.copy() for entry in self._tapes.get(tape, []) if entry.id > after]
        if limit is None:
            return entries
        return entries[:limit]

    def append(self, tape: str, entry: TapeEntry) -> TapeEntry:
        next_id = self._next_id.get(tape, 1)
        self._next_id[tape] = next_id + 1
        stored = TapeEntry(next_id, entry.kind, copy.deepcopy(entry.payload), copy.deepcopy(entry.meta), entry.date)
        self._tapes.setdefault(tape, []).append(stored)
        return stored.copy()


class AsyncTapeStoreAdapter:
    """Adapt a sync TapeStore to AsyncTapeStore."""

    def __init__(self, store: TapeStore) -> None:
        self._store = store

    async def list_tapes(self) -> list[str]:
        return await asyncio.to_thread(self._store.list_tapes)

    async def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]:
        return await asyncio.to_thread(self._store.read, tape, after=after, limit=limit)

    async def fetch_all(self, query: TapeQuery) -> list[TapeEntry]:
        return await asyncio.to_thread(self._store.fetch_all, query)

    async def append(self, tape: str, entry: TapeEntry) -> TapeEntry:
        return await asyncio.to_thread(self._store.append, tape, entry)


class UnavailableTapeStore:
    """Sync TapeStore sentinel that always fails with a clear message."""

    def __init__(self, message: str) -> None:
        self._message = message

    def _raise(self) -> NoReturn:
        raise RepublicError(ErrorKind.INVALID_INPUT, self._message)

    def list_tapes(self) -> list[str]:
        self._raise()

    def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]:
        self._raise()

    def fetch_all(self, query: TapeQuery) -> list[TapeEntry]:
        self._raise()

    def append(self, tape: str, entry: TapeEntry) -> TapeEntry:
        self._raise()
