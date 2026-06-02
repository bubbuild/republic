# Tape

Tape is an append-only execution log. It stores `TapeEntry` records and provides small helpers for context slicing, stream reads, and replay.

## Core Actions

- `handoff(name, payload=...)`: Append a named anchor entry.
- `chat(...)`: Continue on the current tape and record the run.
- `append(TapeEntry.record(...))`: Append a schemaless entry and receive the stored entry.
- `query.after_offset(...).view()`: Build a view from a stored stream offset.
- `query.all()`: Read all entries.
- `query.*()`: Run slice queries.

## Minimal Session

```python
from republic import LLM

llm = LLM(model="openrouter:openrouter/free", api_key="<API_KEY>")
tape = llm.tape("ops")

tape.handoff("incident_42", {"owner": "tier1"})
out = tape.chat("Connection pool is exhausted. Give triage steps.", max_tokens=96)

print(out)
print([entry.kind for entry in tape.query.all()])
```

## Anchor-Based Context Slicing

```python
tape.handoff("incident_43")
_ = tape.chat("This time the issue is cache penetration.")

previous = tape.query.after_anchor("incident_42").all()
print([entry.kind for entry in previous])
```

## Query Filters

```python
matches = tape.query.query("timeout").kinds("message").all()
recent = tape.query.between_dates("2026-03-01", "2026-03-07").all()
resumed = tape.query.after_offset(saved_offset).all()
```

## Schemaless Records

Tape entries are not limited to chat messages. Use explicit `TapeEntry` factories for payloads owned by a downstream system, such as events, artifacts, labels, or binary data.

```python
from republic import TAPE_CLOSE_ANCHOR, TapeEntry, entry_offset

image = tape.append(TapeEntry.record(b"...png bytes...", content_type="image/png", modality="image"))
image_offset = entry_offset(image)
tape.append(TapeEntry.record({"event": "label", "target": image_offset, "value": "diagram"}))

view = tape.query.after_offset(image_offset).view()
print(view.next_offset)
print([entry.payload for entry in view.entries])

done = tape.append(TapeEntry.anchor(TAPE_CLOSE_ANCHOR, {"status": "complete"}))
print(entry_offset(done), tape.query.info().closed)
```

Offsets are opaque strings derived from stored tape entries. Store returned offsets and pass them back to `TapeQuery().after_offset(...)`; do not construct offsets from entry ids.

## Entry Model

Every stored item is a `TapeEntry`:

- `id`: assigned by the store and used to derive stream offsets.
- `kind`: the entry category, such as `record`, `message`, `tool_call`, `event`, or `anchor`.
- `payload`: the application-owned value.
- `meta`: descriptive metadata about the entry, not a replacement for payload.
- `date`: the entry timestamp used by date queries.

`anchor` is the only structural entry kind reserved by the tape. An anchor is still a normal `TapeEntry`: its payload is not wrapped, and the anchor name is stored in `meta["anchor"]`. `handoff(...)` writes named anchors. The built-in `close` anchor is expressed with `TapeEntry.anchor(TAPE_CLOSE_ANCHOR, ...)`.

```python
from republic import TAPE_ANCHOR_NAME_KEY

checkpoint = tape.handoff("checkpoint", {"consumer": "indexer"})
view = tape.query.include_anchors().view()

entry = view.entries[-1]
assert entry.kind == "anchor"
assert entry.payload == {"consumer": "indexer"}
assert entry.meta[TAPE_ANCHOR_NAME_KEY] == "checkpoint"
```

Schemaless records cannot use `kind="anchor"` through `TapeEntry.record(...)`; use `TapeEntry.anchor(...)` or `handoff(...)` for anchors. Downstream readers can import constants and helpers from `republic.tape.anchor`.

## Store Boundary

`TapeStore` is the durable, streamable tape boundary. It is not a file format. Storage backends implement the append-only protocol: list tapes, read by id, execute `TapeQuery`, and append entries. Simple stores can inherit `InMemoryQueryMixin` to get the standard in-memory query behavior from `read(...)`.

Read rules live in `TapeQuery` for both `tape.query.all()` and `tape.query.view()`. A `TapeView` hides anchors by default, because anchors normally mark structure rather than user data. Use `include_anchors()` when a consumer needs to interpret anchors. `stop_at_close(False)` only changes read paging; appending through `TapeManager` or `Tape` still rejects later record and anchor writes after the built-in close anchor.

Downstream systems can define their own anchor names, such as `reset`, `checkpoint`, or `compact`. Republic records those anchors but does not prescribe their effects. A consumer that treats `reset` as a new logical epoch owns that interpretation.

## External Sources

A tape does not have to be produced through a `Tape` session. A `TapeStore` can project an external fact source, such as OpenTelemetry span events, into `TapeEntry` objects when it is read. In that model:

- the telemetry backend owns the raw events;
- the `TapeStore` adapter maps those events to `TapeEntry`;
- `InMemoryQueryMixin` can provide standard query execution for projected entries;
- `TapeQuery` and `TapeContext` decide which entries become a model-facing view.

This keeps debug spans, tool traces, usage, and provider metadata available for replay without forcing all of it into prompt context.

## Conventions

- Tape entries are append-only and never overwrite history.
- Anchors are named `TapeEntry` records. They mark reconstruction points, not deletion points.
- Dict anchor payloads are passed to context selectors as state. Explicit `TapeContext.state` values override same-name anchor payload keys.
- Query/Context depend on entry order, not external indexes.
- Persistent backends implement the `TapeStore` protocol; Republic core does not prescribe a file or database format.
- Errors are recorded as first-class entries for replay.

## Async Tape Store

When `tape_store` is configured as an `AsyncTapeStore` (or its adapter), async calls with `tape=...` read and write context through `AsyncTapeManager`.

```python
from republic import LLM, TapeContext
from republic.tape.store import AsyncTapeStoreAdapter, InMemoryTapeStore

llm = LLM(
    model="openai:gpt-4o-mini",
    api_key="<API_KEY>",
    tape_store=AsyncTapeStoreAdapter(InMemoryTapeStore()),
    context=TapeContext(anchor=None),
)

first = await llm.chat_async("Investigate DB timeout", tape="ops")
second = await llm.chat_async("Include rollback criteria", tape="ops")
print(first, second)
```

## Sync vs Async Rules

When `tape_store` is an `AsyncTapeStore`:

- Sync APIs with `tape=...` are unavailable (they raise `RepublicError`).
- Use async APIs instead: `chat_async`, `tool_calls_async`, `run_tools_async`, `stream_async`, and `stream_events_async`.
- `llm.tape("...")` returns a session object that exposes both sync and async methods; in this mode, use the `*_async` methods.
