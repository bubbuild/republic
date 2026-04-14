from __future__ import annotations

from datetime import date

import pytest

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.context import LAST_ANCHOR, TapeContext
from republic.tape.entries import TapeEntry
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
