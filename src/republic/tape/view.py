"""Tape view return values and offsets."""

from __future__ import annotations

from dataclasses import dataclass

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
