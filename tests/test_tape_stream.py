from __future__ import annotations

import pytest

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape import (
    TAPE_ANCHOR_KIND,
    TAPE_ANCHOR_NAME_KEY,
    TAPE_CLOSE_ANCHOR,
    AsyncTapeManager,
    TapeEntry,
    TapeManager,
    TapeQuery,
    entry_offset,
)
from republic.tape.store import InMemoryTapeStore


def test_tape_entries_append_and_read_schemaless_payloads() -> None:
    manager = TapeManager()

    first = manager.append_record("artifacts", {"event": "created"})
    second = manager.append_record("artifacts", b"\x89PNG", content_type="image/png", modality="image")

    view = manager.read_view("artifacts")

    assert [entry.payload for entry in view.entries] == [{"event": "created"}, b"\x89PNG"]
    assert [entry_offset(entry) for entry in view.entries] == [first, second]
    assert view.next_offset == second
    assert view.up_to_date is True


def test_tape_entries_resume_from_opaque_offset() -> None:
    manager = TapeManager()

    offset = manager.append_record("events", {"step": 1})
    manager.append_record("events", {"step": 2})

    view = manager.read_view("events", TapeQuery().after_offset(offset))

    assert [entry.payload for entry in view.entries] == [{"step": 2}]


@pytest.mark.parametrize("kind", [TAPE_ANCHOR_KIND])
def test_tape_entries_reject_structural_entry_kinds(kind: str) -> None:
    manager = TapeManager()

    with pytest.raises(RepublicError) as exc_info:
        manager.append_record("events", {"name": "not-a-real-anchor"}, kind=kind)

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_tape_entries_resume_uses_store_offset_read() -> None:
    class TrackingTapeStore(InMemoryTapeStore):
        def __init__(self) -> None:
            super().__init__()
            self.read_calls: list[tuple[int, int | None]] = []

        def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]:
            self.read_calls.append((after, limit))
            return super().read(tape, after=after, limit=limit)

    store = TrackingTapeStore()
    first = store.append("events", TapeEntry.record({"step": 1}))
    store.append("events", TapeEntry.record({"step": 2}))
    manager = TapeManager(store=store)

    view = manager.read_view("events", TapeQuery().after_offset(entry_offset(first)))

    assert [entry.payload for entry in view.entries] == [{"step": 2}]
    assert store.read_calls == [(first.id, None)]


def test_tape_entries_close_is_append_only_lifecycle_entry() -> None:
    manager = TapeManager()

    manager.append_record("events", {"step": 1})
    close_offset = manager.close_tape("events", {"status": "done"})

    info = manager.tape_info("events")
    assert info.closed is True
    assert info.closed_offset == close_offset
    assert info.entry_count == 1

    view = manager.read_view("events")
    assert [entry.payload for entry in view.entries] == [{"step": 1}]
    assert view.closed is True

    anchor_view = manager.read_view("events", TapeQuery().include_anchors())
    close_entry = anchor_view.entries[-1]
    assert close_entry.kind == TAPE_ANCHOR_KIND
    assert close_entry.payload == {"status": "done"}
    assert close_entry.meta[TAPE_ANCHOR_NAME_KEY] == TAPE_CLOSE_ANCHOR

    after_close = manager.read_view("events", TapeQuery().after_offset(close_offset))
    assert after_close.entries == ()
    assert after_close.next_offset == close_offset
    assert after_close.closed is True

    with pytest.raises(RepublicError) as exc_info:
        manager.append_record("events", {"step": 2})
    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_tape_entries_can_append_custom_anchors() -> None:
    manager = TapeManager()

    offset = manager.append_anchor("events", "checkpoint", {"consumer": "indexer"}, consumer="search")
    view = manager.read_view("events", TapeQuery().include_anchors())

    assert view.next_offset == offset
    assert [(entry.kind, entry.payload, entry.meta) for entry in view.entries] == [
        (TAPE_ANCHOR_KIND, {"consumer": "indexer"}, {TAPE_ANCHOR_NAME_KEY: "checkpoint", "consumer": "search"})
    ]


@pytest.mark.parametrize("name", [TAPE_CLOSE_ANCHOR])
def test_tape_entries_anchor_rejects_builtin_names(name: str) -> None:
    manager = TapeManager()

    with pytest.raises(RepublicError) as exc_info:
        manager.append_anchor("events", name, {"name": "incident"})

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


@pytest.mark.parametrize("name", ["", " ", " checkpoint"])
def test_tape_entries_anchor_rejects_invalid_names(name: str) -> None:
    manager = TapeManager()

    with pytest.raises(RepublicError) as exc_info:
        manager.append_anchor("events", name)

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_tape_entries_now_starts_at_current_tail() -> None:
    manager = TapeManager()

    first = manager.append_record("events", {"step": 1})

    view = manager.read_view("events", TapeQuery().now())

    assert view.entries == ()
    assert view.next_offset == first
    assert view.up_to_date is True


@pytest.mark.asyncio
async def test_async_tape_entries_append_and_read() -> None:
    manager = AsyncTapeManager()

    offset = await manager.append_record("events", {"step": 1})
    view = await manager.read_view("events")

    assert [entry.payload for entry in view.entries] == [{"step": 1}]
    assert view.next_offset == offset


def test_tape_query_can_include_anchor_entries_in_tape_views() -> None:
    manager = TapeManager()

    manager.append_record("events", {"step": 1})
    manager.close_tape("events", {"status": "done"})

    view = manager.read_view("events", TapeQuery().include_anchors())

    assert [entry.kind for entry in view.entries] == ["record", TAPE_ANCHOR_KIND]
    assert view.closed is True


def test_tape_query_can_read_past_close_when_requested() -> None:
    store = InMemoryTapeStore()
    manager = TapeManager(store=store)

    manager.append_record("events", {"step": 1})
    manager.close_tape("events", {"status": "done"})
    after_close = store.append("events", TapeEntry.record({"step": 2}))

    default_view = manager.read_view("events")
    full_view = manager.read_view("events", TapeQuery().stop_at_close(False))

    assert [entry.payload for entry in default_view.entries] == [{"step": 1}]
    assert [entry.payload for entry in full_view.entries] == [{"step": 1}, {"step": 2}]
    assert full_view.next_offset == entry_offset(after_close)
    assert full_view.closed is True
