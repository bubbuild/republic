"""Tape manager helpers for Republic."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import republic.tape.anchor as tape_anchor
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.context import LAST_ANCHOR, TapeContext, build_messages
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery
from republic.tape.store import (
    AsyncTapeStore,
    AsyncTapeStoreAdapter,
    InMemoryTapeStore,
    TapeStore,
    is_async_tape_store,
)
from republic.tape.view import (
    TAPE_NOW,
    TAPE_START,
    TapeInfo,
    TapeView,
    entry_offset,
    offset_id,
)


def _validate_record_kind(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise RepublicError(ErrorKind.INVALID_INPUT, "Tape entry kind must be a non-empty string.")
    if kind in tape_anchor.STRUCTURAL_ENTRY_KINDS:
        raise RepublicError(ErrorKind.INVALID_INPUT, f"'{kind}' is reserved for structural tape entries.")


def _read_view(entries: list[TapeEntry], query: TapeQuery[Any]) -> TapeView:
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


def _anchor_state(entry: TapeEntry) -> dict[str, Any]:
    data = tape_anchor.data(entry)
    if not isinstance(data, dict):
        return {}
    return copy.deepcopy(data)


def _selected_anchor_state(entries: list[TapeEntry], context: TapeContext) -> dict[str, Any]:
    if context.anchor is None:
        return {}
    if context.anchor is LAST_ANCHOR:
        for entry in reversed(entries):
            state = _anchor_state(entry)
            if state:
                return state
            if tape_anchor.name(entry) is not None:
                return {}
        return {}
    for entry in reversed(entries):
        if tape_anchor.name(entry) == context.anchor:
            return _anchor_state(entry)
    return {}


def _context_with_anchor_state(entries: list[TapeEntry], context: TapeContext) -> TapeContext:
    anchor_state = _selected_anchor_state(entries, context)
    if not anchor_state:
        return context
    return replace(context, state={**anchor_state, **copy.deepcopy(context.state)})


class TapeManager:
    """Global tape manager that owns storage and default context."""

    def __init__(
        self,
        *,
        store: TapeStore | None = None,
        default_context: TapeContext | None = None,
    ) -> None:
        self._tape_store = store or InMemoryTapeStore()
        self._global_context = default_context or TapeContext()

    @property
    def default_context(self) -> TapeContext:
        return self._global_context

    @default_context.setter
    def default_context(self, value: TapeContext) -> None:
        self._global_context = value

    def list_tapes(self) -> list[str]:
        return self._tape_store.list_tapes()

    def resolve_context(self, tape: str, *, context: TapeContext | None = None) -> TapeContext:
        active_context = context or self._global_context
        entries = list(self.query_tape(tape).all())
        return _context_with_anchor_state(entries, active_context)

    def read_messages(self, tape: str, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = self.resolve_context(tape, context=context)
        query = self.query_tape(tape)
        query = active_context.build_query(query)
        messages = build_messages(query.all(), active_context)
        if inspect.isawaitable(messages):
            raise ValueError(  # noqa: TRY003
                "Context selector returned a coroutine, but TapeManager is sync. Use AsyncTapeManager for async support."
            )
        return messages

    def query_tape(self, tape: str) -> TapeQuery[TapeStore]:
        return TapeQuery(tape=tape, store=self._tape_store)

    def append_record(
        self,
        tape: str,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> str:
        _validate_record_kind(kind)
        if self.tape_info(tape).closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append to a closed tape.")
        stored = self._tape_store.append(
            tape,
            TapeEntry.record(payload, kind=kind, content_type=content_type, **meta),
        )
        return entry_offset(stored)

    def append_anchor(self, tape: str, name: str, payload: Any | None = None, **meta: Any) -> str:
        tape_anchor.validate(name, custom=True)
        if self.tape_info(tape).closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append anchor entries to a closed tape.")
        stored = self._tape_store.append(tape, TapeEntry.anchor(name, payload, **meta))
        return entry_offset(stored)

    def read_view(self, tape: str, query: TapeQuery[Any] | None = None) -> TapeView:
        active_query = query or TapeQuery()
        if active_query.offset == TAPE_NOW:
            entries = self._tape_store.read(tape)
        else:
            start_id = offset_id(active_query.offset)
            entries = self._tape_store.read(tape, after=start_id)
            if (
                not entries
                and start_id > 0
                and active_query.stops_at_close
                and _is_close_offset(self._tape_store.read(tape, after=start_id - 1, limit=1), start_id)
            ):
                return TapeView((), active_query.offset, up_to_date=True, closed=True)
        return _read_view(entries, active_query)

    def close_tape(self, tape: str, payload: Any | None = None) -> str:
        entries = self._tape_store.read(tape)
        closed = _closed_entry(entries)
        if closed is not None:
            return entry_offset(closed)
        stored = self._tape_store.append(tape, TapeEntry.anchor(tape_anchor.TAPE_CLOSE_ANCHOR, payload))
        return entry_offset(stored)

    def tape_info(self, tape: str) -> TapeInfo:
        return _tape_info(self._tape_store.read(tape))

    def handoff(
        self,
        tape: str,
        name: str,
        payload: Any | None = None,
        **meta: Any,
    ) -> TapeEntry:
        entry = TapeEntry.anchor(name, payload, **meta)
        return self._tape_store.append(tape, entry)

    def record_chat(  # noqa: C901
        self,
        *,
        tape: str,
        run_id: str,
        system_prompt: str | None,
        context_error: RepublicError | None,
        new_messages: list[dict[str, Any]],
        response_text: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        error: RepublicError | None = None,
        response: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        meta = {"run_id": run_id}
        if system_prompt:
            self._tape_store.append(tape, TapeEntry.system(system_prompt, **meta))
        if context_error is not None:
            self._tape_store.append(tape, TapeEntry.error(context_error, **meta))

        for message in new_messages:
            self._tape_store.append(tape, TapeEntry.message(message, **meta))

        if tool_calls:
            self._tape_store.append(tape, TapeEntry.tool_call(tool_calls, **meta))
        if tool_results is not None:
            self._tape_store.append(tape, TapeEntry.tool_result(tool_results, **meta))

        if error is not None and error is not context_error:
            self._tape_store.append(tape, TapeEntry.error(error, **meta))

        if response_text is not None:
            self._tape_store.append(
                tape,
                TapeEntry.message({"role": "assistant", "content": response_text}, **meta),
            )

        data: dict[str, Any] = {"status": "error" if error is not None else "ok"}
        resolved_usage = usage or self._extract_usage(response)
        if resolved_usage is not None:
            data["usage"] = resolved_usage
        if provider:
            data["provider"] = provider
        if model:
            data["model"] = model
        self._tape_store.append(tape, TapeEntry.event("run", data, **meta))

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump(exclude_none=True)
        if hasattr(usage, "dict"):
            return usage.dict(exclude_none=True)
        return dict(getattr(usage, "__dict__", {}) or {}) or None


class AsyncTapeManager:
    """Async tape manager for async chat and tool-call paths."""

    def __init__(
        self,
        *,
        store: AsyncTapeStore | TapeStore | None = None,
        default_context: TapeContext | None = None,
    ) -> None:
        if store is None:
            sync_store = InMemoryTapeStore()
            self._tape_store = AsyncTapeStoreAdapter(sync_store)
        elif is_async_tape_store(store):
            self._tape_store = store
        else:
            self._tape_store = AsyncTapeStoreAdapter(cast(TapeStore, store))
        self._global_context = default_context or TapeContext()

    @property
    def default_context(self) -> TapeContext:
        return self._global_context

    @default_context.setter
    def default_context(self, value: TapeContext) -> None:
        self._global_context = value

    def query_tape(self, tape: str) -> TapeQuery[AsyncTapeStore]:
        return TapeQuery(tape=tape, store=self._tape_store)

    async def list_tapes(self) -> list[str]:
        return await self._tape_store.list_tapes()

    async def append_record(
        self,
        tape: str,
        payload: Any,
        *,
        kind: str = "record",
        content_type: str | None = None,
        **meta: Any,
    ) -> str:
        _validate_record_kind(kind)
        if (await self.tape_info(tape)).closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append to a closed tape.")
        stored = await self._tape_store.append(
            tape,
            TapeEntry.record(payload, kind=kind, content_type=content_type, **meta),
        )
        return entry_offset(stored)

    async def append_anchor(self, tape: str, name: str, payload: Any | None = None, **meta: Any) -> str:
        tape_anchor.validate(name, custom=True)
        if (await self.tape_info(tape)).closed:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Cannot append anchor entries to a closed tape.")
        stored = await self._tape_store.append(tape, TapeEntry.anchor(name, payload, **meta))
        return entry_offset(stored)

    async def read_view(self, tape: str, query: TapeQuery[Any] | None = None) -> TapeView:
        active_query = query or TapeQuery()
        if active_query.offset == TAPE_NOW:
            entries = await self._tape_store.read(tape)
        else:
            start_id = offset_id(active_query.offset)
            entries = await self._tape_store.read(tape, after=start_id)
            if not entries and start_id > 0:
                current = await self._tape_store.read(tape, after=start_id - 1, limit=1)
                if active_query.stops_at_close and _is_close_offset(current, start_id):
                    return TapeView((), active_query.offset, up_to_date=True, closed=True)
        return _read_view(entries, active_query)

    async def close_tape(self, tape: str, payload: Any | None = None) -> str:
        entries = await self._tape_store.read(tape)
        closed = _closed_entry(entries)
        if closed is not None:
            return entry_offset(closed)
        stored = await self._tape_store.append(tape, TapeEntry.anchor(tape_anchor.TAPE_CLOSE_ANCHOR, payload))
        return entry_offset(stored)

    async def tape_info(self, tape: str) -> TapeInfo:
        return _tape_info(await self._tape_store.read(tape))

    async def resolve_context(self, tape: str, *, context: TapeContext | None = None) -> TapeContext:
        active_context = context or self._global_context
        entries = list(await self.query_tape(tape).all())
        return _context_with_anchor_state(entries, active_context)

    async def read_messages(self, tape: str, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = await self.resolve_context(tape, context=context)
        query = self.query_tape(tape)
        query = active_context.build_query(query)
        entries = await query.all()
        messages = build_messages(entries, active_context)
        if inspect.isawaitable(messages):
            messages = await messages
        return messages

    async def handoff(
        self,
        tape: str,
        name: str,
        payload: Any | None = None,
        **meta: Any,
    ) -> TapeEntry:
        entry = TapeEntry.anchor(name, payload, **meta)
        return await self._tape_store.append(tape, entry)

    async def record_chat(  # noqa: C901
        self,
        *,
        tape: str,
        run_id: str,
        system_prompt: str | None,
        context_error: RepublicError | None,
        new_messages: list[dict[str, Any]],
        response_text: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        error: RepublicError | None = None,
        response: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        meta = {"run_id": run_id}
        if system_prompt:
            await self._tape_store.append(tape, TapeEntry.system(system_prompt, **meta))
        if context_error is not None:
            await self._tape_store.append(tape, TapeEntry.error(context_error, **meta))

        for message in new_messages:
            await self._tape_store.append(tape, TapeEntry.message(message, **meta))

        if tool_calls:
            await self._tape_store.append(tape, TapeEntry.tool_call(tool_calls, **meta))
        if tool_results is not None:
            await self._tape_store.append(tape, TapeEntry.tool_result(tool_results, **meta))

        if error is not None and error is not context_error:
            await self._tape_store.append(tape, TapeEntry.error(error, **meta))

        if response_text is not None:
            await self._tape_store.append(
                tape,
                TapeEntry.message({"role": "assistant", "content": response_text}, **meta),
            )

        data: dict[str, Any] = {"status": "error" if error is not None else "ok"}
        resolved_usage = usage or TapeManager._extract_usage(response)
        if resolved_usage is not None:
            data["usage"] = resolved_usage
        if provider:
            data["provider"] = provider
        if model:
            data["model"] = model
        await self._tape_store.append(tape, TapeEntry.event("run", data, **meta))
