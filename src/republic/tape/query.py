"""Query helpers for tape entries."""

from __future__ import annotations

import json
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, time
from datetime import date as date_type
from typing import Generic, Self, TypeVar, overload

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.store import AsyncTapeStore, TapeStore
from republic.tape.stream import offset_id

T = TypeVar("T", bound="TapeStore | AsyncTapeStore", covariant=True)


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


def _apply_query(entries: Sequence[TapeEntry], query: TapeQuery) -> list[TapeEntry]:  # noqa: C901
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

    sliced = list(entries[start_index:end_index])
    if query._between_dates is not None:
        start_date, end_date = query._between_dates
        start_dt = _parse_datetime_boundary(start_date, is_end=False)
        end_dt = _parse_datetime_boundary(end_date, is_end=True)
        if start_dt > end_dt:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Start date must be earlier than or equal to end date.")
        sliced = [entry for entry in sliced if _entry_in_datetime_range(entry, start_dt, end_dt)]
    if query._query:
        sliced = [entry for entry in sliced if _entry_matches_query(entry, query._query)]
    if query._kinds:
        sliced = [entry for entry in sliced if entry.kind in query._kinds]
    if query._limit is not None:
        sliced = sliced[: query._limit]
    return sliced


@dataclass(frozen=True)
class TapeQuery(Generic[T]):
    tape: str
    store: T
    _query: str | None = None
    _after_anchor: str | None = None
    _after_last: bool = False
    _after_offset: str | None = None
    _between_anchors: tuple[str, str] | None = None
    _between_dates: tuple[str, str] | None = None
    _kinds: tuple[str, ...] = field(default_factory=tuple)
    _limit: int | None = None

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
        return replace(self, _after_offset=offset)

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

    @overload
    def all(self: TapeQuery[TapeStore]) -> list[TapeEntry]: ...

    @overload
    async def all(self: TapeQuery[AsyncTapeStore]) -> list[TapeEntry]: ...

    def all(self) -> list[TapeEntry] | Coroutine[None, None, list[TapeEntry]]:
        after = offset_id(self._after_offset) if self._after_offset is not None else 0
        entries = self.store.read(self.tape, after=after)
        if isinstance(entries, Coroutine):
            return self._all_async(entries)
        return _apply_query(entries, self)

    async def _all_async(self, entries: Coroutine[None, None, list[TapeEntry]]) -> list[TapeEntry]:
        return _apply_query(await entries, self)
