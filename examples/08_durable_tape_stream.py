from __future__ import annotations

from typing import Any

from republic import TAPE_ANCHOR_NAME_KEY, TAPE_START, TapeManager, TapeStreamQuery
from republic.tape.stream import TapeStream


def build_stream() -> TapeStream:
    manager = TapeManager()
    stream = manager.stream_tape("ops-durable")

    stream.append({"event": "created", "id": "ticket-1"})
    stream.anchor("checkpoint", {"consumer": "indexer"})
    stream.append({"event": "labeled", "label": "database"})
    stream.close({"status": "done"})

    return stream


def consume_visible_entries(stream: TapeStream) -> dict[str, Any]:
    cursor = TAPE_START
    payloads: list[Any] = []
    offsets: list[str] = []

    while True:
        view = stream.read(TapeStreamQuery().after_offset(cursor).limit(1))
        payloads.extend(entry.payload for entry in view.entries)
        offsets.append(view.next_offset)
        cursor = view.next_offset

        if view.closed:
            return {
                "payloads": payloads,
                "offsets": offsets,
                "closed": True,
            }
        if view.up_to_date:
            return {
                "payloads": payloads,
                "offsets": offsets,
                "closed": False,
            }


def inspect_anchor_names(stream: TapeStream) -> list[str]:
    view = stream.read(TapeStreamQuery().include_anchors())
    return [entry.meta[TAPE_ANCHOR_NAME_KEY] for entry in view.entries if entry.kind == "anchor"]


def run_example() -> dict[str, Any]:
    stream = build_stream()
    return {
        "visible": consume_visible_entries(stream),
        "anchors": inspect_anchor_names(stream),
    }


def main() -> None:
    result = run_example()
    print("payloads:", result["visible"]["payloads"])
    print("offsets:", result["visible"]["offsets"])
    print("closed:", result["visible"]["closed"])
    print("anchors:", result["anchors"])


if __name__ == "__main__":
    main()
