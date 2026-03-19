# Breaking Changes (2026-02-11)

This document summarizes the breaking changes introduced in this branch and shows how to migrate with minimal effort.

## Summary

Error handling is now fully aligned with standard Python style:

- Success paths return plain values.
- Failure paths raise `RepublicError`.
- `StructuredOutput(value, error)` is no longer used as a general return wrapper.

If your code still depends on `.value` / `.error`, migrate using the patterns below.

## Scope

### 1. `StructuredOutput` Removed

`StructuredOutput` has been removed from core types and public exports.

- `src/republic/core/results.py`
- `src/republic/core/__init__.py`
- `src/republic/__init__.py`

### 2. Non-Streaming APIs Now Use "Return Value + Exception"

The following APIs no longer return `StructuredOutput`:

- `LLM.chat(...) -> str`
- `LLM.chat_async(...) -> str`
- `LLM.tool_calls(...) -> list[dict[str, Any]]`
- `LLM.tool_calls_async(...) -> list[dict[str, Any]]`
- `LLM.if_(...) -> bool`
- `LLM.if_async(...) -> bool`
- `LLM.classify(...) -> str`
- `LLM.classify_async(...) -> str`
- `LLM.embed(...) -> Any`
- `LLM.embed_async(...) -> Any`

`Tape` session shortcuts changed in the same way:

- `Tape.chat(...) -> str`
- `Tape.chat_async(...) -> str`
- `Tape.tool_calls(...) -> list[dict[str, Any]]`
- `Tape.tool_calls_async(...) -> list[dict[str, Any]]`

### 3. `ToolExecutor.execute` Error Semantics Changed

`ToolExecutor.execute(...)` now raises `RepublicError` for invalid input, validation failures, missing context, unknown tools, and similar failures. It no longer returns a result object with an `error` field for these cases.

### 4. Tape Return-Type Simplification

This branch also finalizes:

- `ContextSelection` removed. `read_messages(...)` now returns `list[dict[str, Any]]`.
- `QueryResult` removed. `TapeQuery.all()` now returns `list[TapeEntry]`, and errors are raised as `RepublicError`.
- `read_entries()` is deprecated. Use `tape.query.all()` for full entry reads.

## Migration Examples

### Chat

Before:

```python
out = llm.chat("Ping")
if out.error:
    handle_error(out.error)
else:
    print(out.value)
```

After:

```python
from republic.core.results import RepublicError

try:
    text = llm.chat("Ping")
    print(text)
except RepublicError as exc:
    handle_error(exc)
```

### Text Decision / Classify

Before:

```python
decision = llm.if_("service down", "should page?")
if decision.error is None and decision.value:
    page_oncall()
```

After:

```python
from republic.core.results import RepublicError

try:
    decision = llm.if_("service down", "should page?")
    if decision:
        page_oncall()
except RepublicError as exc:
    handle_error(exc)
```

### Embedding

Before:

```python
out = llm.embed("incident summary")
if out.error:
    handle_error(out.error)
else:
    vectors = out.value
```

After:

```python
from republic.core.results import RepublicError

try:
    vectors = llm.embed("incident summary")
except RepublicError as exc:
    handle_error(exc)
```

### ToolExecutor

Before:

```python
result = executor.execute(calls, tools=tools)
if result.error:
    handle_error(result.error)
else:
    print(result.tool_results)
```

After:

```python
from republic.core.results import RepublicError

try:
    result = executor.execute(calls, tools=tools)
    print(result.tool_results)
except RepublicError as exc:
    handle_error(exc)
```

## Fast Detection

Use this command to locate old calling patterns:

```bash
rg -n "StructuredOutput|\\.value\\b|\\.error\\b" src tests
```

Then migrate matches to "return value + `try/except RepublicError`".

## Release Guidance

- This is a clear API breaking change. Use a major version bump, or at minimum a minor bump with explicit release notes.
- If you maintain downstream SDK consumers, provide a migration note or codemod focused on replacing `.value` / `.error` branches.
