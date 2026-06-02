from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from republic import TAPE_ANCHOR_NAME_KEY, TapeContext, TapeEntry, TapeManager
from republic.core.errors import ErrorKind
from republic.core.results import RepublicError


@dataclass(frozen=True)
class OTelEvent:
    tape: str
    name: str
    body: Any
    attributes: dict[str, Any] = field(default_factory=dict)


class OTelCollector:
    """Tiny OTel-shaped collector used by the example.

    Real deployments would replace this with an OpenTelemetry exporter or backend.
    """

    def __init__(self) -> None:
        self._events: list[OTelEvent] = []

    def start_span(self, name: str, *, tape: str, trace_id: str, **attributes: Any) -> OTelSpan:
        return OTelSpan(
            collector=self,
            name=name,
            tape=tape,
            trace_id=trace_id,
            attributes=attributes,
        )

    def emit(
        self,
        *,
        tape: str,
        name: str,
        body: Any,
        attributes: dict[str, Any],
    ) -> None:
        self._events.append(
            OTelEvent(
                tape=tape,
                name=name,
                body=copy.deepcopy(body),
                attributes=copy.deepcopy(attributes),
            )
        )

    def events(self) -> list[OTelEvent]:
        return copy.deepcopy(self._events)


class OTelSpan:
    def __init__(
        self,
        *,
        collector: OTelCollector,
        name: str,
        tape: str,
        trace_id: str,
        attributes: dict[str, Any],
    ) -> None:
        self._collector = collector
        self._name = name
        self._tape = tape
        self._trace_id = trace_id
        self._attributes = dict(attributes)

    def event(self, name: str, body: Any, **attributes: Any) -> None:
        event_attributes = {
            **self._attributes,
            **attributes,
            "span": self._name,
            "trace_id": self._trace_id,
        }
        self._collector.emit(
            tape=self._tape,
            name=name,
            body=body,
            attributes=event_attributes,
        )


def record_incident_trace() -> OTelCollector:
    collector = OTelCollector()
    span = collector.start_span("llm.chat", tape="ops", trace_id="trace-1", run_id="run-1")

    span.event("anchor", {"owner": "tier1"}, **{TAPE_ANCHOR_NAME_KEY: "incident_42"})
    span.event("debug", {"component": "db.pool", "duration_ms": 813})
    span.event("message", {"role": "user", "content": "Investigate DB timeout."})
    span.event("tool_call", {"calls": [{"name": "lookup_db", "arguments": "{}"}]})
    span.event("tool_result", {"results": [{"pool": "exhausted"}]})
    span.event("message", {"role": "assistant", "content": "Check pool saturation and slow queries."})

    return collector


class OTelTapeStore:
    """TapeStore view over events produced by an OTel collector."""

    def __init__(self, collector: OTelCollector) -> None:
        self._collector = collector

    def list_tapes(self) -> list[str]:
        return sorted({event.tape for event in self._collector.events()})

    def read(self, tape: str, *, after: int = 0, limit: int | None = None) -> list[TapeEntry]:
        entries = [
            self._entry_from_event(entry_id, event)
            for entry_id, event in enumerate(self._collector.events(), start=1)
            if event.tape == tape and entry_id > after
        ]
        if limit is None:
            return entries
        return entries[:limit]

    def append(self, tape: str, entry: TapeEntry) -> TapeEntry:
        del tape, entry
        raise RepublicError(ErrorKind.INVALID_INPUT, "OTelTapeStore is a read-only tape projection.")

    @staticmethod
    def _entry_from_event(entry_id: int, event: OTelEvent) -> TapeEntry:
        return TapeEntry(
            id=entry_id,
            kind=event.name,
            payload=copy.deepcopy(event.body),
            meta=copy.deepcopy(event.attributes),
        )


def select_model_messages(entries, context: TapeContext) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": f"owner={context.state['owner']}"},
    ]
    messages.extend(
        copy.deepcopy(entry.payload) for entry in entries if entry.kind == "message" and isinstance(entry.payload, dict)
    )
    return messages


def build_view() -> dict[str, Any]:
    collector = record_incident_trace()
    manager = TapeManager(store=OTelTapeStore(collector))
    context = TapeContext(anchor="incident_42", select=select_model_messages)
    messages = manager.read_messages("ops", context=context)
    query_entries = manager.query_tape("ops").after_anchor("incident_42").kinds("message").all()

    return {
        "otel_event_names": [event.name for event in collector.events()],
        "tapes": manager.list_tapes(),
        "messages": messages,
        "query_kinds": [entry.kind for entry in query_entries],
        "query_payloads": [entry.payload for entry in query_entries],
    }


def main() -> None:
    view = build_view()
    print("tapes:", view["tapes"])
    print("messages:", view["messages"])
    print("query_kinds:", view["query_kinds"])


if __name__ == "__main__":
    main()
