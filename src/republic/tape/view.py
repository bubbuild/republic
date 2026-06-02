"""Tape offsets and stream views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry

TAPE_START = "-1"
TAPE_NOW = "now"
OFFSET_WIDTH = 20


@dataclass(frozen=True)
class TapeView:
    """A bounded view read from a tape offset."""

    entries: tuple[TapeEntry, ...]
    next_offset: str
    up_to_date: bool
    closed: bool = False


@dataclass(frozen=True)
class TapeInfo:
    """Current tape stream position and lifecycle state."""

    tail_offset: str
    entry_count: int
    closed: bool = False
    closed_offset: str | None = None


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
    entry_id = int(offset)
    if entry_id == 0:
        raise RepublicError(ErrorKind.INVALID_INPUT, f"Invalid tape offset: '{offset}'. Use TAPE_START.")
    return entry_id


def is_anchor_entry(entry: TapeEntry) -> bool:
    return entry.kind == tape_anchor.TAPE_ANCHOR_KIND


def is_close_entry(entry: TapeEntry) -> bool:
    return tape_anchor.name(entry) == tape_anchor.TAPE_CLOSE_ANCHOR


def closed_entry(entries: Sequence[TapeEntry]) -> TapeEntry | None:
    for entry in entries:
        if is_close_entry(entry):
            return entry
    return None


def visible_entries(entries: Sequence[TapeEntry]) -> list[TapeEntry]:
    closed = closed_entry(entries)
    boundary = closed.id if closed is not None else None
    return [entry for entry in entries if not is_anchor_entry(entry) and (boundary is None or entry.id < boundary)]


def tail_offset(entries: Sequence[TapeEntry]) -> str:
    visible = visible_entries(entries)
    if visible:
        return entry_offset(visible[-1])
    closed = closed_entry(entries)
    if closed is not None:
        return entry_offset(closed)
    return TAPE_START


def is_close_offset(entries: Sequence[TapeEntry], offset_entry_id: int) -> bool:
    return len(entries) == 1 and entries[0].id == offset_entry_id and is_close_entry(entries[0])


def tape_info(entries: Sequence[TapeEntry]) -> TapeInfo:
    closed = closed_entry(entries)
    return TapeInfo(
        tail_offset=tail_offset(entries),
        entry_count=len(visible_entries(entries)),
        closed=closed is not None,
        closed_offset=entry_offset(closed) if closed is not None else None,
    )


def read_tape_view(
    stored_entries: Sequence[TapeEntry],
    *,
    offset: str,
    limit: int | None,
    include_anchors: bool = False,
    stop_at_close: bool = True,
) -> TapeView:
    offset = tail_offset(stored_entries) if offset == TAPE_NOW else offset
    start_id = offset_id(offset)
    closed = closed_entry(stored_entries)
    closed_id = closed.id if closed is not None and stop_at_close else None
    view_entries: list[TapeEntry] = []
    next_offset = offset

    for entry in stored_entries:
        if entry.id <= start_id:
            continue
        if closed_id is not None and entry.id > closed_id:
            break
        if is_anchor_entry(entry):
            if include_anchors:
                view_entries.append(entry.copy())
            next_offset = entry_offset(entry)
            if stop_at_close and is_close_entry(entry):
                return TapeView(tuple(view_entries), next_offset, up_to_date=True, closed=True)
            continue
        view_entries.append(entry.copy())
        next_offset = entry_offset(entry)
        if limit is not None and len(view_entries) >= limit:
            return TapeView(tuple(view_entries), next_offset, up_to_date=False, closed=False)

    return TapeView(
        entries=tuple(view_entries),
        next_offset=next_offset,
        up_to_date=True,
        closed=closed is not None,
    )
