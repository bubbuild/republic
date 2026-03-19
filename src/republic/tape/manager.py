"""Tape manager helpers for Republic."""

from __future__ import annotations

import inspect
from typing import Any, cast

from republic.core.results import RepublicError
from republic.tape.context import TapeContext, build_messages
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery
from republic.tape.store import (
    AsyncTapeStore,
    AsyncTapeStoreAdapter,
    InMemoryTapeStore,
    TapeStore,
    is_async_tape_store,
)


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

    def read_messages(self, tape: str, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = context or self._global_context
        query = self.query_tape(tape)
        query = active_context.build_query(query)
        messages = build_messages(self._tape_store.fetch_all(query), active_context)
        if inspect.isawaitable(messages):
            raise ValueError(  # noqa: TRY003
                "Context selector returned a coroutine, but TapeManager is sync. Use AsyncTapeManager for async support."
            )
        return messages

    def append_entry(self, tape: str, entry: TapeEntry) -> None:
        self._tape_store.append(tape, entry)

    def query_tape(self, tape: str) -> TapeQuery[TapeStore]:
        return TapeQuery(tape=tape, store=self._tape_store)

    def reset_tape(self, tape: str) -> None:
        self._tape_store.reset(tape)

    def handoff(
        self,
        tape: str,
        name: str,
        *,
        state: dict[str, Any] | None = None,
        **meta: Any,
    ) -> list[TapeEntry]:
        entry = TapeEntry.anchor(name, state=state, **meta)
        event = TapeEntry.event("handoff", {"name": name, "state": state or {}}, **meta)
        self._tape_store.append(tape, entry)
        self._tape_store.append(tape, event)
        return [entry, event]

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

    async def read_messages(self, tape: str, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = context or self._global_context
        query = self.query_tape(tape)
        query = active_context.build_query(query)
        entries = await self._tape_store.fetch_all(query)
        messages = build_messages(entries, active_context)
        if inspect.isawaitable(messages):
            messages = await messages
        return messages

    async def append_entry(self, tape: str, entry: TapeEntry) -> None:
        await self._tape_store.append(tape, entry)

    async def reset_tape(self, tape: str) -> None:
        await self._tape_store.reset(tape)

    async def handoff(
        self,
        tape: str,
        name: str,
        *,
        state: dict[str, Any] | None = None,
        **meta: Any,
    ) -> list[TapeEntry]:
        entry = TapeEntry.anchor(name, state=state, **meta)
        event = TapeEntry.event("handoff", {"name": name, "state": state or {}}, **meta)
        await self._tape_store.append(tape, entry)
        await self._tape_store.append(tape, event)
        return [entry, event]

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
