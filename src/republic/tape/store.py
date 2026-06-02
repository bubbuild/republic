"""Tape stores for Republic."""

from __future__ import annotations

import asyncio
import copy
import inspect
from typing import NoReturn, Protocol, TypeGuard

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.entries import TapeEntry


class TapeStore(Protocol):
    """Append-only tape persistence interface."""

    def list_tapes(self) -> list[str]: ...

    def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]: ...

    def append(self, tape: str, entry: TapeEntry) -> TapeEntry: ...


class AsyncTapeStore(Protocol):
    """Async append-only tape persistence interface."""

    async def list_tapes(self) -> list[str]: ...

    async def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]: ...

    async def append(self, tape: str, entry: TapeEntry) -> TapeEntry: ...


def is_async_tape_store(store: TapeStore | AsyncTapeStore) -> TypeGuard[AsyncTapeStore]:
    return hasattr(store, "append") and inspect.iscoroutinefunction(store.append)


class InMemoryTapeStore:
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

    def append(self, tape: str, entry: TapeEntry) -> TapeEntry:
        self._raise()
