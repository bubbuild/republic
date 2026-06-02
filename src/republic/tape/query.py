"""Query helpers for tape entries."""

from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, overload

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.view import TAPE_NOW, TAPE_START, offset_id

if TYPE_CHECKING:
    from republic.tape.store import AsyncTapeStore, TapeStore

T = TypeVar("T", bound="TapeStore | AsyncTapeStore", covariant=True)


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
        if isinstance(entries, Coroutine):
            return entries
        return list(entries)

    def _bound(self) -> tuple[str, T]:
        if self.tape is None or self.store is None:
            raise RepublicError(ErrorKind.INVALID_INPUT, "TapeQuery must be bound to a tape and store before all().")
        return self.tape, self.store
