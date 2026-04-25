"""Tape manager helpers for Republic."""

from __future__ import annotations

import inspect
from typing import Any, cast

from republic.core.results import RepublicError
from republic.tape.context import TapeContext, build_messages, select_context_entries
from republic.tape.entries import TapeEntry
from republic.tape.format import DEFAULT_TAPE_FORMAT, TapeFormat
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
        tape_format: TapeFormat = DEFAULT_TAPE_FORMAT,
    ) -> None:
        self._tape_store = store or InMemoryTapeStore()
        self._global_context = default_context or TapeContext()
        self._tape_format = tape_format

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
        entries = self._tape_store.fetch_all(self.query_tape(tape))
        entries = select_context_entries(entries, active_context, self._tape_format)
        messages = build_messages(entries, active_context, self._tape_format)
        if inspect.isawaitable(messages):
            raise ValueError(  # noqa: TRY003
                "Context selector returned a coroutine, but TapeManager is sync. Use AsyncTapeManager for async support."
            )
        return messages

    def append_entry(self, tape: str, entry: TapeEntry) -> None:
        self._tape_store.append(tape, entry)

    def query_tape(self, tape: str) -> TapeQuery[TapeStore]:
        return TapeQuery(tape=tape, store=self._tape_store, tape_format=self._tape_format)

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
        entries = _handoff_entries(self._tape_format, name, state=state, **meta)
        for entry in entries:
            self._tape_store.append(tape, entry)
        return entries

    def record_chat(
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
        for entry in _chat_entries(
            self._tape_format,
            run_id=run_id,
            system_prompt=system_prompt,
            context_error=context_error,
            new_messages=new_messages,
            response_text=response_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            error=error,
            response=response,
            provider=provider,
            model=model,
            usage=usage,
        ):
            self._tape_store.append(tape, entry)

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
        tape_format: TapeFormat = DEFAULT_TAPE_FORMAT,
    ) -> None:
        if store is None:
            sync_store = InMemoryTapeStore()
            self._tape_store = AsyncTapeStoreAdapter(sync_store)
        elif is_async_tape_store(store):
            self._tape_store = store
        else:
            self._tape_store = AsyncTapeStoreAdapter(cast(TapeStore, store))
        self._global_context = default_context or TapeContext()
        self._tape_format = tape_format

    @property
    def default_context(self) -> TapeContext:
        return self._global_context

    @default_context.setter
    def default_context(self, value: TapeContext) -> None:
        self._global_context = value

    def query_tape(self, tape: str) -> TapeQuery[AsyncTapeStore]:
        return TapeQuery(tape=tape, store=self._tape_store, tape_format=self._tape_format)

    async def list_tapes(self) -> list[str]:
        return await self._tape_store.list_tapes()

    async def read_messages(self, tape: str, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = context or self._global_context
        entries = await self._tape_store.fetch_all(self.query_tape(tape))
        entries = select_context_entries(entries, active_context, self._tape_format)
        messages = build_messages(entries, active_context, self._tape_format)
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
        entries = _handoff_entries(self._tape_format, name, state=state, **meta)
        for entry in entries:
            await self._tape_store.append(tape, entry)
        return entries

    async def record_chat(
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
        for entry in _chat_entries(
            self._tape_format,
            run_id=run_id,
            system_prompt=system_prompt,
            context_error=context_error,
            new_messages=new_messages,
            response_text=response_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            error=error,
            response=response,
            provider=provider,
            model=model,
            usage=usage,
        ):
            await self._tape_store.append(tape, entry)


def _handoff_entries(
    tape_format: TapeFormat,
    name: str,
    *,
    state: dict[str, Any] | None = None,
    **meta: Any,
) -> list[TapeEntry]:
    return [
        tape_format.anchor(name, state=state, **meta),
        tape_format.event("handoff", {"name": name, "state": state or {}}, **meta),
    ]


def _chat_entries(
    tape_format: TapeFormat,
    *,
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
) -> list[TapeEntry]:
    meta = {"run_id": run_id}
    entries: list[TapeEntry] = []
    if system_prompt:
        entries.append(tape_format.system(system_prompt, **meta))
    if context_error is not None:
        entries.append(tape_format.error(context_error, **meta))
    entries.extend(tape_format.message(message, **meta) for message in new_messages)
    if tool_calls:
        entries.append(tape_format.tool_call(tool_calls, **meta))
    if tool_results is not None:
        entries.append(tape_format.tool_result(tool_results, **meta))
    if error is not None and error is not context_error:
        entries.append(tape_format.error(error, **meta))
    if response_text is not None:
        entries.append(tape_format.message({"role": "assistant", "content": response_text}, **meta))
    entries.append(tape_format.event("run", _run_event_data(error, response, provider, model, usage), **meta))
    return entries


def _run_event_data(
    error: RepublicError | None,
    response: Any | None,
    provider: str | None,
    model: str | None,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"status": "error" if error is not None else "ok"}
    resolved_usage = usage or TapeManager._extract_usage(response)
    if resolved_usage is not None:
        data["usage"] = resolved_usage
    if provider:
        data["provider"] = provider
    if model:
        data["model"] = model
    return data
