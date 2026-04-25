from __future__ import annotations

from datetime import date

import pytest

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.context import LAST_ANCHOR, TapeContext
from republic.tape.entries import TapeEntry
from republic.tape.format import RepublicTapeFormat
from republic.tape.manager import AsyncTapeManager, TapeManager
from republic.tape.query import TapeQuery
from republic.tape.store import AsyncTapeStoreAdapter, InMemoryTapeStore


def _seed_entries() -> list[TapeEntry]:
    return [
        TapeEntry.message({"role": "user", "content": "before"}),
        TapeEntry.anchor("a1"),
        TapeEntry.message({"role": "user", "content": "task 1"}),
        TapeEntry.message({"role": "assistant", "content": "answer 1"}),
        TapeEntry.anchor("a2"),
        TapeEntry.message({"role": "user", "content": "task 2"}),
    ]


@pytest.fixture
def manager() -> TapeManager:
    store = InMemoryTapeStore()
    for entry in _seed_entries():
        store.append("test_tape", entry)
    return TapeManager(store=store)


def test_build_messages_uses_last_anchor_slice(manager) -> None:
    messages = manager.read_messages("test_tape", context=TapeContext(anchor=LAST_ANCHOR))
    assert [message["content"] for message in messages] == ["task 2"]


def test_build_messages_reports_missing_anchor(manager) -> None:
    with pytest.raises(RepublicError) as exc_info:
        manager.read_messages("test_tape", context=TapeContext(anchor="missing"))
    assert exc_info.value.kind == ErrorKind.NOT_FOUND


class _AwaitableMessages:
    def __init__(self, messages: list[dict[str, str]]) -> None:
        self._messages = messages

    def __await__(self):
        async def _resolve() -> list[dict[str, str]]:
            return self._messages

        return _resolve().__await__()


def test_sync_manager_rejects_async_context_selector(manager) -> None:
    def select(entries, context):
        return _AwaitableMessages([{"role": "assistant", "content": str(len(list(entries)))}])

    context = TapeContext(anchor=LAST_ANCHOR, select=select)

    with pytest.raises(ValueError, match="Use AsyncTapeManager for async support"):
        manager.read_messages("test_tape", context=context)


class EventTapeFormat(RepublicTapeFormat):
    name = "test.events"
    version = "1"

    def message(self, message: dict, **meta) -> TapeEntry:
        return TapeEntry.event("chat.message", {"message": dict(message)}, **meta)

    def anchor(self, name: str, state: dict | None = None, **meta) -> TapeEntry:
        return TapeEntry.event("chat.anchor", {"name": name, "state": state or {}}, **meta)

    def anchor_name(self, entry: TapeEntry) -> str | None:
        name = _event_data(entry, "chat.anchor").get("name")
        return name if isinstance(name, str) else None

    def entry_kind(self, entry: TapeEntry) -> str:
        if entry.kind != "event":
            return entry.kind
        match entry.payload.get("name"):
            case "chat.message":
                return "message"
            case "chat.anchor":
                return "anchor"
            case _:
                return "event"

    def matches(self, entry: TapeEntry, query: str) -> bool:
        message = _event_data(entry, "chat.message").get("message")
        if isinstance(message, dict):
            return query.casefold() in str(message.get("content", "")).casefold()
        return super().matches(entry, query)

    def select_messages(self, entries, context):
        del context
        messages = []
        for entry in entries:
            message = _event_data(entry, "chat.message").get("message")
            if isinstance(message, dict):
                messages.append(dict(message))
        return messages


def _event_data(entry: TapeEntry, name: str) -> dict:
    if entry.kind != "event" or entry.payload.get("name") != name:
        return {}
    data = entry.payload.get("data")
    return data if isinstance(data, dict) else {}


def _record_chat(manager: TapeManager, content: str, *, tape: str = "custom", run_id: str = "run") -> None:
    manager.record_chat(
        tape=tape,
        run_id=run_id,
        system_prompt=None,
        context_error=None,
        new_messages=[{"role": "user", "content": content}],
        response_text=None,
    )


async def _record_chat_async(manager: AsyncTapeManager, content: str, *, tape: str = "custom") -> None:
    await manager.record_chat(
        tape=tape,
        run_id="run",
        system_prompt=None,
        context_error=None,
        new_messages=[{"role": "user", "content": content}],
        response_text=None,
    )


def test_tape_format_owns_entry_shape_and_default_injection() -> None:
    store = InMemoryTapeStore()
    manager = TapeManager(store=store, default_context=TapeContext(anchor=None), tape_format=EventTapeFormat())

    manager.record_chat(
        tape="custom",
        run_id="run_1",
        system_prompt=None,
        context_error=None,
        new_messages=[{"role": "user", "content": "hello"}],
        response_text="hi",
    )

    entries = store.read("custom") or []
    assert [entry.kind for entry in entries] == ["event", "event", "event"]
    assert entries[0].payload["name"] == "chat.message"
    assert manager.read_messages("custom") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_tape_format_owns_anchor_shape_for_context_injection() -> None:
    store = InMemoryTapeStore()
    manager = TapeManager(store=store, tape_format=EventTapeFormat())

    _record_chat(manager, "before", run_id="run_1")
    manager.handoff("custom", "phase")
    _record_chat(manager, "after", run_id="run_2")

    assert manager.read_messages("custom") == [{"role": "user", "content": "after"}]


def test_tape_query_uses_format_for_anchor_kind_and_text() -> None:
    store = InMemoryTapeStore()
    manager = TapeManager(store=store, tape_format=EventTapeFormat())

    _record_chat(manager, "before", run_id="run_1")
    manager.handoff("custom", "phase")
    _record_chat(manager, "after", run_id="run_2")

    entries = list(manager.query_tape("custom").last_anchor().kinds("message").query("after").all())

    assert len(entries) == 1
    assert entries[0].payload["data"]["message"]["content"] == "after"


@pytest.mark.asyncio
async def test_async_manager_awaits_context_selector_after_anchor_slice() -> None:
    sync_store = InMemoryTapeStore()
    for entry in _seed_entries():
        sync_store.append("test_tape", entry)
    manager = AsyncTapeManager(store=AsyncTapeStoreAdapter(sync_store))

    seen: dict[str, object] = {}

    async def select(entries, context):
        entry_list = list(entries)
        seen["contents"] = [entry.payload["content"] for entry in entry_list]
        seen["state"] = dict(context.state)
        return [{"role": "system", "content": f"{context.state['prefix']}:{entry_list[0].payload['content']}"}]

    context = TapeContext(anchor=LAST_ANCHOR, select=select, state={"prefix": "summary"})
    messages = await manager.read_messages("test_tape", context=context)

    assert messages == [{"role": "system", "content": "summary:task 2"}]
    assert seen == {
        "contents": ["task 2"],
        "state": {"prefix": "summary"},
    }


@pytest.mark.asyncio
async def test_async_tape_format_owns_entry_shape_and_default_injection() -> None:
    store = InMemoryTapeStore()
    manager = AsyncTapeManager(
        store=AsyncTapeStoreAdapter(store),
        default_context=TapeContext(anchor=None),
        tape_format=EventTapeFormat(),
    )

    await _record_chat_async(manager, "hello")

    entries = store.read("custom") or []
    assert [entry.kind for entry in entries] == ["event", "event"]
    assert await manager.read_messages("custom") == [{"role": "user", "content": "hello"}]


def test_query_between_anchors_and_limit() -> None:
    store = InMemoryTapeStore()
    tape = "session"

    for entry in _seed_entries():
        store.append(tape, entry)

    entries = list(TapeQuery(tape=tape, store=store).between_anchors("a1", "a2").kinds("message").limit(1).all())
    assert len(entries) == 1
    assert entries[0].payload["content"] == "task 1"


def test_query_text_matches_payload_and_meta() -> None:
    store = InMemoryTapeStore()
    tape = "searchable"

    store.append(tape, TapeEntry.message({"role": "user", "content": "Database timeout on checkout"}, scope="db"))
    store.append(tape, TapeEntry.event("run", {"status": "ok"}, scope="system"))

    entries = list(TapeQuery(tape=tape, store=store).query("timeout").all())
    assert len(entries) == 1
    assert entries[0].kind == "message"

    meta_entries = list(TapeQuery(tape=tape, store=store).query("system").all())
    assert len(meta_entries) == 1
    assert meta_entries[0].kind == "event"


def test_query_between_dates_filters_inclusive_range() -> None:
    store = InMemoryTapeStore()
    tape = "dated"

    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="message",
            payload={"role": "user", "content": "before"},
            date="2026-03-01T08:00:00+00:00",
        ),
    )
    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="message",
            payload={"role": "user", "content": "during"},
            date="2026-03-02T09:30:00+00:00",
        ),
    )
    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="message",
            payload={"role": "user", "content": "after"},
            date="2026-03-04T18:45:00+00:00",
        ),
    )

    entries = list(TapeQuery(tape=tape, store=store).between_dates(date(2026, 3, 2), "2026-03-03").all())
    assert [entry.payload["content"] for entry in entries] == ["during"]


def test_query_combines_anchor_date_and_text_filters() -> None:
    store = InMemoryTapeStore()
    tape = "combined"

    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="anchor",
            payload={"name": "a1"},
            date="2026-03-01T00:00:00+00:00",
        ),
    )
    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="message",
            payload={"role": "user", "content": "old timeout"},
            date="2026-03-01T12:00:00+00:00",
        ),
    )
    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="anchor",
            payload={"name": "a2"},
            date="2026-03-02T00:00:00+00:00",
        ),
    )
    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="message",
            payload={"role": "user", "content": "new timeout"},
            meta={"source": "ops"},
            date="2026-03-02T12:00:00+00:00",
        ),
    )
    store.append(
        tape,
        TapeEntry(
            id=0,
            kind="message",
            payload={"role": "user", "content": "new success"},
            meta={"source": "ops"},
            date="2026-03-03T12:00:00+00:00",
        ),
    )

    entries = list(
        TapeQuery(tape=tape, store=store)
        .after_anchor("a2")
        .between_dates("2026-03-02", "2026-03-02")
        .query("timeout")
        .all()
    )
    assert [entry.payload["content"] for entry in entries] == ["new timeout"]
