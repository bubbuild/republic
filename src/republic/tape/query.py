"""Query helpers for tape entries."""

from __future__ import annotations

from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from typing import Generic, Self, TypeVar, overload

from republic.tape.entries import TapeEntry
from republic.tape.store import AsyncTapeStore, TapeStore

T = TypeVar("T", TapeStore, AsyncTapeStore)


@dataclass(frozen=True)
class TapeQuery(Generic[T]):
    tape: str
    store: T
    _query: str | None = None
    _after_anchor: str | None = None
    _after_last: bool = False
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

    def between_anchors(self, start: str, end: str) -> Self:
        return replace(self, _between_anchors=(start, end))

    def between_dates(self, start: str | date_type, end: str | date_type) -> Self:
        start_value = start.isoformat() if isinstance(start, date_type) else start
        end_value = end.isoformat() if isinstance(end, date_type) else end
        return replace(self, _between_dates=(start_value, end_value))

    def kinds(self, *kinds: str) -> Self:
        return replace(self, _kinds=kinds)

    def limit(self, value: int) -> Self:
        return replace(self, _limit=value)

    @overload
    def all(self: TapeQuery[TapeStore]) -> Iterable[TapeEntry]: ...

    @overload
    async def all(self: TapeQuery[AsyncTapeStore]) -> Iterable[TapeEntry]: ...

    def all(self) -> Iterable[TapeEntry] | Coroutine[None, None, Iterable[TapeEntry]]:
        return self.store.fetch_all(self)
