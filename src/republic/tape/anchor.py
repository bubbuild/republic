"""Anchor entry constants and helpers for tape streams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError

if TYPE_CHECKING:
    from republic.tape.entries import TapeEntry

TAPE_ANCHOR_KIND = "anchor"
TAPE_ANCHOR_NAME_KEY = "anchor"
TAPE_CLOSE_ANCHOR = "close"
BUILTIN_TAPE_ANCHORS = frozenset({TAPE_CLOSE_ANCHOR})
STRUCTURAL_ENTRY_KINDS = frozenset({TAPE_ANCHOR_KIND})


def validate(name: str, *, custom: bool = False) -> None:
    if not isinstance(name, str) or not name or name != name.strip():
        raise RepublicError(ErrorKind.INVALID_INPUT, "Tape anchor name must be a non-empty string.")
    if custom and name in BUILTIN_TAPE_ANCHORS:
        raise RepublicError(ErrorKind.INVALID_INPUT, f"'{name}' is a built-in tape anchor.")


def name(entry: TapeEntry) -> str | None:
    if entry.kind != TAPE_ANCHOR_KIND:
        return None
    value = entry.meta.get(TAPE_ANCHOR_NAME_KEY)
    return value if isinstance(value, str) else None


def data(entry: TapeEntry) -> Any | None:
    if entry.kind != TAPE_ANCHOR_KIND:
        return None
    return entry.payload
