from __future__ import annotations

import pytest

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.llm import LLM
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
    entries = manager.stream_tape("artifacts")

    first = entries.append({"event": "created"})
    second = entries.append(b"\x89PNG", content_type="image/png", modality="image")

    view = entries.read()

    assert [entry.payload for entry in view.entries] == [{"event": "created"}, b"\x89PNG"]
    assert [entry_offset(entry) for entry in view.entries] == [first, second]
    assert view.next_offset == second
    assert view.up_to_date is True


def test_tape_session_exposes_entries_stream() -> None:
    tape = LLM(model="openai:gpt-4o-mini", api_key="dummy").tape("artifacts")

    offset = tape.entries.append({"event": "created"})
    view = tape.entries.read(TapeQuery().after_offset(offset))

    assert view.entries == ()
    assert view.next_offset == offset


def test_tape_entries_resume_from_opaque_offset() -> None:
    manager = TapeManager()
    entries = manager.stream_tape("events")

    offset = entries.append({"step": 1})
    entries.append({"step": 2})

    view = entries.read(TapeQuery().after_offset(offset))

    assert [entry.payload for entry in view.entries] == [{"step": 2}]


@pytest.mark.parametrize("kind", [TAPE_ANCHOR_KIND])
def test_tape_entries_reject_structural_entry_kinds(kind: str) -> None:
    entries = TapeManager().stream_tape("events")

    with pytest.raises(RepublicError) as exc_info:
        entries.append({"name": "not-a-real-anchor"}, kind=kind)

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
    entries = TapeManager(store=store).stream_tape("events")

    view = entries.read(TapeQuery().after_offset(entry_offset(first)))

    assert [entry.payload for entry in view.entries] == [{"step": 2}]
    assert store.read_calls == [(first.id, None)]


def test_tape_entries_close_is_append_only_lifecycle_entry() -> None:
    manager = TapeManager()
    entries = manager.stream_tape("events")

    entries.append({"step": 1})
    close_offset = entries.close({"status": "done"})

    info = entries.info()
    assert info.closed is True
    assert info.closed_offset == close_offset
    assert info.entry_count == 1

    view = entries.read()
    assert [entry.payload for entry in view.entries] == [{"step": 1}]
    assert view.closed is True

    anchor_view = entries.read(TapeQuery().include_anchors())
    close_entry = anchor_view.entries[-1]
    assert close_entry.kind == TAPE_ANCHOR_KIND
    assert close_entry.payload == {"status": "done"}
    assert close_entry.meta[TAPE_ANCHOR_NAME_KEY] == TAPE_CLOSE_ANCHOR

    after_close = entries.read(TapeQuery().after_offset(close_offset))
    assert after_close.entries == ()
    assert after_close.next_offset == close_offset
    assert after_close.closed is True

    with pytest.raises(RepublicError) as exc_info:
        entries.append({"step": 2})
    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_tape_entries_can_append_custom_anchors() -> None:
    entries = TapeManager().stream_tape("events")

    offset = entries.anchor("checkpoint", {"consumer": "indexer"}, consumer="search")
    view = entries.read(TapeQuery().include_anchors())

    assert view.next_offset == offset
    assert [(entry.kind, entry.payload, entry.meta) for entry in view.entries] == [
        (TAPE_ANCHOR_KIND, {"consumer": "indexer"}, {TAPE_ANCHOR_NAME_KEY: "checkpoint", "consumer": "search"})
    ]


def test_tape_entries_anchor_can_have_empty_payload() -> None:
    entries = TapeManager().stream_tape("events")

    entries.anchor("checkpoint")
    view = entries.read(TapeQuery().include_anchors())

    assert view.entries[0].kind == TAPE_ANCHOR_KIND
    assert view.entries[0].payload is None
    assert view.entries[0].meta == {TAPE_ANCHOR_NAME_KEY: "checkpoint"}


@pytest.mark.parametrize("name", [TAPE_CLOSE_ANCHOR])
def test_tape_entries_anchor_rejects_builtin_names(name: str) -> None:
    entries = TapeManager().stream_tape("events")

    with pytest.raises(RepublicError) as exc_info:
        entries.anchor(name, {"name": "incident"})

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


@pytest.mark.parametrize("name", ["", " ", " checkpoint"])
def test_tape_entries_anchor_rejects_invalid_names(name: str) -> None:
    entries = TapeManager().stream_tape("events")

    with pytest.raises(RepublicError) as exc_info:
        entries.anchor(name)

    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_tape_entries_now_starts_at_current_tail() -> None:
    manager = TapeManager()
    entries = manager.stream_tape("events")

    first = entries.append({"step": 1})

    view = entries.read(TapeQuery().now())

    assert view.entries == ()
    assert view.next_offset == first
    assert view.up_to_date is True


def test_tape_entry_factories_deep_copy_payloads() -> None:
    payload = {"items": [{"name": "before"}]}
    record_entry = TapeEntry.record(payload)
    message_entry = TapeEntry.message({"role": "user", "content": payload})
    anchor_entry = TapeEntry.anchor("checkpoint", payload)

    payload["items"][0]["name"] = "after"

    assert record_entry.payload == {"items": [{"name": "before"}]}
    assert message_entry.payload["content"] == {"items": [{"name": "before"}]}
    assert anchor_entry.payload == {"items": [{"name": "before"}]}


@pytest.mark.asyncio
async def test_async_tape_entries_append_and_read() -> None:
    manager = AsyncTapeManager()
    entries = manager.stream_tape("events")

    offset = await entries.append({"step": 1})
    view = await entries.read()

    assert [entry.payload for entry in view.entries] == [{"step": 1}]
    assert view.next_offset == offset


@pytest.mark.asyncio
async def test_async_tape_query_now_is_awaitable() -> None:
    manager = AsyncTapeManager()
    entries = manager.stream_tape("events")

    await entries.append({"step": 1})
    selected = await manager.query_tape("events").now().all()

    assert selected == []


def test_tape_query_can_include_anchor_entries_in_stream_views() -> None:
    manager = TapeManager()
    entries = manager.stream_tape("events")

    entries.append({"step": 1})
    entries.close({"status": "done"})

    view = entries.read(TapeQuery().include_anchors())

    assert [entry.kind for entry in view.entries] == ["record", TAPE_ANCHOR_KIND]
    assert view.closed is True


def test_tape_query_can_read_past_close_when_requested() -> None:
    store = InMemoryTapeStore()
    entries = TapeManager(store=store).stream_tape("events")

    entries.append({"step": 1})
    entries.close({"status": "done"})
    after_close = store.append("events", TapeEntry.record({"step": 2}))

    default_view = entries.read()
    full_view = entries.read(TapeQuery().stop_at_close(False))

    assert [entry.payload for entry in default_view.entries] == [{"step": 1}]
    assert [entry.payload for entry in full_view.entries] == [{"step": 1}, {"step": 2}]
    assert full_view.next_offset == entry_offset(after_close)
    assert full_view.closed is True
