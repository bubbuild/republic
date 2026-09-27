# Single-call contract

Step 1 provides data models and a small asynchronous boundary. Step 2's
[OpenAI Chat Completions adapter](openai-chat.md) is its first concrete consumer;
Step 3 adds [OpenAI Responses](openai-responses.md) with native reasoning metadata
and full-history replay. Step 4 adds [Anthropic Messages](anthropic-messages.md),
including signed/redacted thinking and inclusive cache usage. Step 5 adds
[ChatGPT/Codex OAuth](codex-oauth.md) and its explicit Responses provider. Names and
signatures remain provisional as more protocols are added.

## Run an offline call

Install this checkout with `uv sync`. This complete example implements the
structural `Provider` protocol without inheritance, credentials or network calls:

```python
import asyncio
from collections.abc import AsyncGenerator

from republic import Message, Request, Response, TextPart, events, generate, stream


class ExampleProvider:
    async def generate(self, request: Request) -> Response:
        return Response(
            message=Message(role="assistant", parts=[TextPart(text="Hello")]),
            response_id="offline-1",
            response_model=request.model,
            finish_reason="stop",
        )

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        yield events.TextStart(block_id="text-1")
        yield events.TextDelta(block_id="text-1", chunk="Hello")
        yield events.TextEnd(block_id="text-1")
        yield events.StreamEnd(
            response_id="offline-2", response_model=request.model, finish_reason="stop"
        )


async def main() -> None:
    provider = ExampleProvider()
    request = Request(
        model="offline",
        messages=[Message(role="user", parts=[TextPart(text="Hi")])],
    )
    response = await generate(provider, request)
    print(response.message.text)
    async with stream(provider, request) as output:
        async for event in output:
            if isinstance(event, events.TextDelta):
                print(event.chunk)
        assert output.response is not None
        saved = output.response.model_dump_json()
    restored = Response.model_validate_json(saved)
    assert restored.message.text == "Hello"


asyncio.run(main())
```

These are two explicit calls, one per API entry point. `generate` delegates once
to the adapter's generation method. API-key adapters use non-streaming requests;
Codex deliberately aggregates one SSE request. There is no automatic fallback.
`stream` starts lazily on iteration and owns that response's stream until exit.

## Data and persistence

The public data types are exported from `republic`; events live in
`republic.events`. Each data model supports Pydantic's `model_dump_json()` and
`model_validate_json()`. Unknown fields and non-JSON metadata/results are rejected
during validation. Treat instances as data and use validated fields when editing
them; Pydantic does not intercept in-place edits to nested lists/dictionaries.

| Type | Contract |
| --- | --- |
| `Message` | `role` (system/user/assistant/tool), ordered `parts`, optional `provider_metadata`. `text` concatenates text only; `tool_calls` returns call data. |
| `TextPart`, `ReasoningPart` | Text plus optional provider metadata. Reasoning is separate from display text. |
| `FilePart` | `data`, `media_type`, explicit `encoding` (url/base64), optional filename and metadata. `from_bytes` stores standard base64. No fetch, file access or media detection. |
| `Tool` | Name, optional description, JSON Schema `parameters`, optional metadata. No callable or execution state. |
| `ToolCallPart` | Call ID, tool name, raw `tool_args` string, optional metadata. Even malformed/partial JSON remains verbatim. |
| `ToolResultPart` | Caller-provided call ID, tool name, JSON `result`, `is_error`, optional metadata. |
| `Request` | Model ID, message history, tools, `RequestOptions`. Client, endpoint and credentials belong to the runtime provider. |
| `RequestOptions` | Temperature, top-p, max output tokens, stops, tool choice, parallel tool calls, JSON provider options. None means omitted. Named choices use `ToolChoice(name=...)`. |
| `Response` | Message, optional usage, finish reason, response ID and actual response model. Unknown identity stays None. |
| `Usage` | Optional input/output/reasoning/cache read/cache write counts and raw JSON usage. Unknown stays None; zero stays zero. Total is unknown if input or output is unknown. |

Common finish reasons are `stop`, `length`, `content_filter`, `tool_call`, `error`
and `other`. An adapter maps unknown native reasons to `other` and preserves the
raw value in message provider metadata. A terminal response may still be
truncated (`length`), filtered, or report an error; terminal does not mean the
output is suitable for executing a tool. The caller decides how to handle it.

Metadata uses JSON objects, conventionally keyed by provider name. The SDK never
interprets reasoning signatures, encrypted reasoning or provider item IDs.
Adapters retain those values on the relevant part/message for subsequent calls.
Provider clients cannot be fields of persisted messages or nested JSON metadata.
This is a data boundary, not a credential-redaction service.

No role/part combination or tool history is silently normalized. A provider
adapter must explicitly reject combinations/options it cannot represent.
The shared API passes a deep copy of the request, preserving order and contents
while isolating the caller from adapter mutation. Appending the response, running
tools, adding results, and choosing the next call are caller operations.

## Events and aggregation

Text and reasoning each use `Start`, `Delta`, `End` events with an explicit
`block_id`. Tools use `ToolStart(tool_call_id, tool_name)`,
`ToolDelta(tool_call_id, chunk)` and `ToolEnd(tool_call_id)`.
All event constructors use keyword arguments. `FileEvent(part=...)` adds one
complete file. `StreamEnd` carries response identity, finish reason, final usage
and message metadata.

Different blocks may interleave. Each kind/ID pair must start once, receive its
own deltas, and end once. Parts retain start order; tool argument strings append
only to the matching call ID. IDs are unique within a kind for that response;
stream block IDs are routing keys, not persisted message IDs. Provider IDs needed
for later requests belong in metadata (tool call IDs are always persisted).
Unknown, reused or already closed IDs and a terminal event with open blocks raise
`StreamProtocolError`. The SDK does not invent starts, ends, arguments or results.

Part metadata on start/delta/end merges recursively by object key. Supplied
scalar/list/null leaves replace older values; omitted keys remain. Strings are
snapshots, not fragments: an adapter receiving signature fragments must assemble
them before emitting metadata. Reasoning text deltas concatenate independently.
Events carry data only; inspect `output.message` for a detached partial snapshot
and `output.response` for a detached terminal response.

`StreamEnd` is final. Adapters must consume late usage-only wire chunks before
emitting it, then release the transport in `finally`. Exhausting a wire stream
does not by itself justify a terminal event: adapters must verify the provider's
termination marker. The shared stream closes immediately on `StreamEnd`, without
pulling another event. No events after it are consumed.

## Errors and resource ownership

| Outcome | Observable behavior |
| --- | --- |
| Valid terminal event | `status == "completed"`, terminal `response` available, source closed. |
| Source exhausts without terminal | `IncompleteStreamError`, `status == "incomplete"`, partial `message` available, no response. |
| Early break, context exit or `aclose()` | `status == "closed"`, source closed without draining, no synthetic terminal response. |
| Caller task cancellation before terminal | `CancelledError` propagates, `status == "cancelled"`, partial output remains. |
| Adapter or event-protocol failure | Exception propagates, `status == "failed"`, partial output remains. |

`RepublicError` is the SDK error base. `ProviderError` carries optional provider,
HTTP status, service error code and request ID. API-key adapters preserve native
errors as causes; the Codex OAuth adapter omits native error objects and untrusted
diagnostics to protect credentials. `UnsupportedRequestError` reports unrepresentable input
before a request is sent.
`IncompleteStreamError` and `StreamProtocolError` describe streaming failures.
Cancellation is never converted into a provider error or retried. The SDK does
not automatically retry any request.

Always use `async with stream(...)`. Breaking an `async for` alone does not close
an iterator; leaving the enclosing context does. `aclose()` is idempotent and
can be called inside that context. An unconsumed stream opens no request.
One task consumes a stream at a time; cancel and await that task before closing
from another task. Concurrent iteration/close is outside this contract.

The adapter's stream is an async generator. Acquire request resources inside
its body and release them in `finally`; never depend on code after a final yield
running. Explicit closure is awaited in a shielded cleanup task, so cancellation
during closure waits for cleanup and then propagates. Adapter cleanup must allow
execution in that task and must eventually return. Client lifetime is separate:
the adapter/caller owns a reusable client; closing one response must not close a
borrowed client. Step 2 tests these ownership rules through the official OpenAI
client with controlled HTTP responses.

## Source and intentional changes

The message/usage shapes, block events, aggregation and incomplete-stream
behavior are adapted from [Vercel AI Python at c788059d](https://github.com/vercel-labs/ai-python/tree/c788059dd1db2d93ae1c3da6daffb660eca07dbb).
`NOTICE` in the repository and distributions maps upstream files and copyright.
This subset drops agent contexts, replay, hooks, UI, telemetry, execution state,
random message IDs and the executor/provider registry. Python 3.12 type syntax
is rewritten for 3.11. Republic adds recursive metadata merging, explicit file
encoding, strict block ordering, unknown usage counts and separate response data.
Upstream API compatibility is not claimed.

The test-only fake transport records exact calls, event consumption and resource
release. Tests exercise serialized history, interleaved arguments, metadata,
normal termination, missing termination, early close, cancellation and errors.
They establish local contracts; they do not establish real-service compatibility,
OAuth access or Bub integration. See the [rebuild plan](rebuild-plan.md) for the
recorded commands and the remaining increments.
