"""Context building for tape entries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.format import DEFAULT_TAPE_FORMAT, SelectedMessages, TapeFormat
from republic.tape.query import TapeQuery


class _LastAnchor:
    def __repr__(self) -> str:
        return "LAST_ANCHOR"


LAST_ANCHOR = _LastAnchor()
AnchorSelector: TypeAlias = str | None | _LastAnchor
ContextSelector: TypeAlias = Callable[[Iterable[TapeEntry], "TapeContext"], SelectedMessages]


@dataclass(frozen=True)
class TapeContext:
    """Rules for selecting tape entries into a prompt context.

    anchor: LAST_ANCHOR for the most recent anchor, None for the full tape, or an anchor name.
    select: Optional selector called after anchor slicing that returns messages.
    state: Optional state dictionary to be passed along with the context.
    """

    anchor: AnchorSelector = LAST_ANCHOR
    select: ContextSelector | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def build_query(self, query: TapeQuery) -> TapeQuery:
        if self.anchor is None:
            return query
        if isinstance(self.anchor, _LastAnchor):
            return query.last_anchor()
        return query.after_anchor(self.anchor)


def build_messages(
    entries: Iterable[TapeEntry],
    context: TapeContext,
    tape_format: TapeFormat = DEFAULT_TAPE_FORMAT,
) -> SelectedMessages:
    if context.select is not None:
        return context.select(entries, context)
    return tape_format.select_messages(entries, context)


def select_context_entries(
    entries: Iterable[TapeEntry],
    context: TapeContext,
    tape_format: TapeFormat = DEFAULT_TAPE_FORMAT,
) -> list[TapeEntry]:
    entry_list = list(entries)
    if context.anchor is None:
        return entry_list
    if isinstance(context.anchor, _LastAnchor):
        anchor_index = _find_anchor(entry_list, None, tape_format)
        if anchor_index < 0:
            raise RepublicError(ErrorKind.NOT_FOUND, "No anchors found in tape.")
        return entry_list[anchor_index + 1 :]

    anchor_index = _find_anchor(entry_list, context.anchor, tape_format)
    if anchor_index < 0:
        raise RepublicError(ErrorKind.NOT_FOUND, f"Anchor '{context.anchor}' was not found.")
    return entry_list[anchor_index + 1 :]


def _find_anchor(entries: list[TapeEntry], name: str | None, tape_format: TapeFormat) -> int:
    for index in range(len(entries) - 1, -1, -1):
        anchor_name = tape_format.anchor_name(entries[index])
        if anchor_name is None:
            continue
        if name is not None and anchor_name != name:
            continue
        return index
    return -1
