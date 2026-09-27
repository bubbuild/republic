# OpenAI Chat Completions

`OpenAIChatCompletions` implements the shared `Provider` protocol using the
official `openai.AsyncOpenAI` client. Both entry points issue a single
`POST /chat/completions`; `generate` requests a non-streaming response and
`stream` requests SSE. The current evidence is HTTP/SSE fixtures through the
real client, not live OpenAI or OpenRouter acceptance.

## Install and make one call

From this checkout, use `pip install .` (runtime) or `uv sync` (development).
OpenAI 2.x, HTTPX and Pydantic are runtime dependencies; there is no separate
provider extra. These instructions target `dev`, not an already published release.

```python
import asyncio
import os

from republic import Message, Request, TextPart, generate
from republic.providers.openai import OpenAIChatCompletions


async def main() -> None:
    async with OpenAIChatCompletions(api_key=os.environ["OPENAI_API_KEY"]) as provider:
        response = await generate(
            provider,
            Request(
                model=os.environ["CHAT_MODEL"],
                messages=[Message(role="user", parts=[TextPart(text="Hello")])],
            ),
        )
        print(response.message.text)


asyncio.run(main())
```

Running this example sends a real request. Choose an accessible Chat Completions
model yourself; Republic does not resolve model names or consult a model catalog.
If `api_key` or `base_url` is omitted, the official client's defaults/environment
configuration applies. Republic does not load token files or perform login.

For an OpenAI-compatible endpoint, pass its base URL explicitly. For example,
[OpenRouter documents](https://openrouter.ai/docs/quickstart)
`base_url="https://openrouter.ai/api/v1"` with an OpenRouter API key. Pass the
service's model ID in `Request.model`. Republic appends `chat/completions` through
the official client and does not route or fall back to another model. This path
is fixture-tested; endpoint/model-specific behavior still needs live validation.

## Stream and close

Inside an async function with `provider` and `request` already constructed:

```python
from republic import events, stream

async with stream(provider, request) as output:
    async for event in output:
        if isinstance(event, events.TextDelta):
            print(event.chunk, end="", flush=True)
    response = output.response
```

Always leave the `async with` block after breaking iteration. Early exit closes
the response without draining it or making another request. A task cancellation
propagates as `asyncio.CancelledError`; the response closes and partial emitted
output remains on `output.message`. Close the provider after its active calls
have finished or been cancelled and awaited.

The adapter requires a nonempty provider finish reason. `[DONE]` alone, or a
transport EOF without a finish reason, raises `IncompleteStreamError`. After a
finish reason it keeps reading usage-only chunks; only then does it emit
`StreamEnd`, with the HTTP response already closed. A finish reason followed by
a transport/API error still fails. A final finish reason with EOF and no usage
is accepted, with unknown usage left as None. The SDK's SSE decoder owns framing
and `[DONE]`; Republic does not implement a parallel SSE parser.

Tool argument deltas are tracked by the provider's index. They can arrive before
the ID/name; the adapter buffers them until both complete, nonempty header fields
arrive, then emits the original argument fragments. Earlier pending headers hold
back later tool starts so tool order is preserved. ID/name fields are atomic
headers: repeating the same header is accepted; changing/fragmenting a header
is rejected rather than guessed or concatenated. No call IDs are invented.
Pending unidentified tool data has no common part until its header is available.

Argument JSON is never parsed or repaired, including malformed JSON and a
`length` response containing partial arguments. A call with no argument fragments
has the empty string as its arguments. Callers decide whether a terminal result
is usable; a terminal event does not authorize tool execution.

## Client ownership and retries

Without `client=`, the adapter creates an `AsyncOpenAI(max_retries=0)` and owns
it. Use `async with OpenAIChatCompletions(...)` or call `await provider.aclose()`.
Closing a response stream leaves this reusable client open; closing the provider
closes it. A closed provider rejects further calls. Closure is idempotent.

You can borrow a client, including one with custom HTTP configuration:

```python
from openai import AsyncOpenAI
from republic.providers.openai import OpenAIChatCompletions

async with AsyncOpenAI(api_key="your-key", base_url="https://your-service.example/v1") as client:
    async with OpenAIChatCompletions(client=client) as provider:
        # await generate(provider, request)
        pass
    # client is still open here and belongs to this outer context.
```

`client=` cannot be combined with `api_key=` or `base_url=`. The adapter borrows
the transport through the official client's `with_options(max_retries=0)`;
it does not mutate the original client's retry setting or close its transport.
It snapshots the other client settings at construction.

Both owned and borrowed clients therefore disable the official SDK's automatic
retry loop for Republic calls. Fixtures assert one HTTP attempt for 429/500,
connection and timeout failures, as well as failures after streamed output.
An injected custom transport or remote gateway may independently retry; the
caller must configure those layers if exactly one end-to-end attempt is needed.
Republic has no generate-to-stream fallback or automatic next turn.

## Messages, tools and options

Supported history consists of system text; user text/images; assistant text,
reasoning text and function calls; and tool-result messages. Images can be
HTTP(S)/image data URLs or standard base64 via `FilePart.from_bytes`. No content
is downloaded. Image `provider_metadata={"openai": {"detail": "low"}}` supports
auto/low/high. Images with a filename, other file types, audio output, provider
built-in tools and deprecated `function_call` data are not supported.

`Tool.parameters` is sent as the function's JSON Schema; description is optional.
`Tool.provider_metadata={"openai": {"strict": True}}` forwards strict mode.
The caller supplies tool results. String results are sent as strings; other JSON
values are JSON-encoded (`None` becomes `"null"`). Results are linked by call ID;
the descriptive `tool_name` remains in stored data, not the tool-result wire
message. Multiple results become consecutive tool messages. There is no error
flag in this wire format, so `is_error=True` is rejected; explicitly represent
an error in the result if that is what the model should receive.

For a caller-controlled continuation, after receiving a tool call:

```python
from republic import Message, ToolResultPart, generate

call = response.message.tool_calls[0]
# The caller validates arguments and runs its own tool outside Republic.
caller_result = {"temperature": 20}
request.messages.extend([
    response.message,
    Message(role="tool", parts=[ToolResultPart(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        result=caller_result,
    )]),
])
next_response = await generate(provider, request)
```

Nothing validates or repairs the relationship between history calls and results.
Unmatched history is passed through and can be rejected by the service. The
shared entry points copy the request to protect caller-owned history.

`RequestOptions` maps temperature, top-p, stop and parallel tool calls directly;
`max_output_tokens` becomes `max_completion_tokens`. Tool choice accepts auto,
none, required or `ToolChoice(name=...)`. Only one response choice is supported.

`provider_options` accepts these official-client keyword options: `seed`,
`frequency_penalty`, `presence_penalty`, `logit_bias`, `logprobs`, `top_logprobs`,
`response_format`, `reasoning_effort`, `user`, `store`, `service_tier`, `metadata`,
`extra_headers`, `extra_body`, and positive `timeout` seconds. They are wire
options, not a schema-validation or typed-output layer. The endpoint validates
its supported values. For example, `response_format` does not make Republic
parse the returned text.

Unknown provider option keys are rejected. Managed fields such as model,
messages, tools, stream, stream_options, n, common options and token-limit aliases
cannot be overridden. `extra_body` allows service extensions (for example an
OpenRouter `provider` object), but cannot override managed or listed standard
fields. It must be a JSON object; `extra_headers` must have string values.

## Results, metadata and errors

Usage preserves missing counts as None, reported zeroes, reasoning/cache-read
breakdowns, and the full raw usage object. Results retain the provider response
ID and actual model. Finish reasons map stop/length/content_filter/tool_calls to
the common vocabulary; other values map to `other` with the original reason
under message metadata `openai.finish_reason`.

Plain-text `reasoning` and `reasoning_content` extensions are accepted and kept
as separate `ReasoningPart` values. Their `openai.field` metadata preserves the
wire key when history is sent again. Refusal text is assembled into message
metadata and sent back as the assistant's refusal field. Log probabilities,
system fingerprint and service tier are persisted as response records and are
not sent back as history input.

Unknown input metadata is rejected rather than discarded. Encrypted/signed
`reasoning_details`, annotations and generated media are explicitly unsupported
output fields in this increment. In particular, not every OpenRouter model or
reasoning mode fits this subset. Native Responses reasoning items require the
separate [Responses adapter](openai-responses.md). [Anthropic Messages](anthropic-messages.md)
and [ChatGPT/Codex OAuth](codex-oauth.md) also have explicit adapters.

Unrepresentable input raises `UnsupportedRequestError` before HTTP. Provider/API
failures raise `ProviderError` with available status, code and request ID; the
original SDK or HTTPX exception remains `__cause__`. Malformed wire output raises
`ProviderError(code="invalid_response")`. No error triggers a retry or history
repair. Missing completion and caller cancellation keep their distinct shared
errors/status described in the [single-call contract](contracts.md).

The implementation adapts the pinned upstream
[Chat Completions protocol](https://github.com/vercel-labs/ai-python/blob/c788059dd1db2d93ae1c3da6daffb660eca07dbb/src/ai/providers/openai/protocol.py).
`NOTICE` records provenance and changes. The
[official Chat API reference](https://developers.openai.com/api/reference/resources/chat)
describes the wire protocol; supported Republic behavior and local evidence are
recorded here and in the [rebuild plan](rebuild-plan.md).
