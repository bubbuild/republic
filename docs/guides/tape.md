# Tape

Tape is an append-only execution log and a context selector.

## Core Actions

- `handoff(name, state=...)`: Create a new task anchor.
- `chat(...)`: Continue on the current tape and record the run.
- `query.all()`: Read all entries (message/tool/error/event).
- `query.*()`: Run slice queries.

## Deprecation

`read_entries()` is deprecated. Use `tape.query.all()` instead.

```python
# before (deprecated)
entries = tape.read_entries()

# after
entries = list(tape.query.all())
```

## Minimal Session

```python
from republic import LLM

llm = LLM(model="openrouter:openrouter/free", api_key="<API_KEY>")
tape = llm.tape("ops")

tape.handoff("incident_42", state={"owner": "tier1"})
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
```

## Conventions

- Tape entries are append-only and never overwrite history.
- Query/Context depend on entry order, not external indexes.
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
