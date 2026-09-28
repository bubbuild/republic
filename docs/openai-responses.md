# OpenAI Responses

`OpenAIResponses` implements one `POST /responses` operation using the official
asynchronous OpenAI client. The adapter has offline HTTP/SSE fixture evidence;
no live OpenAI or compatible Responses endpoint has been validated. Chat
Completions compatibility alone does not establish Responses compatibility.
[ChatGPT/Codex OAuth](codex-oauth.md) and [Anthropic Messages](anthropic-messages.md)
have separate explicit adapters.

## Install and make one call

Install this checkout with `pip install .` (runtime) or `uv sync` (development).
OpenAI 2.x, HTTPX and Pydantic are included; there is no extra to install.
Python 3.11+ remains supported. This branch has not been published.

```python
import asyncio
import os

from republic import Message, Request, TextPart, generate
from republic.providers.openai import OpenAIResponses


async def main():
    request = Request(
        model="your-responses-model",
        messages=[Message(role="user", parts=[TextPart(text="Hello")])],
    )
    async with OpenAIResponses(api_key=os.environ["OPENAI_API_KEY"]) as provider:
        response = await generate(provider, request)
        print(response.message.text)


# Runs a real request only when you explicitly call it:
# asyncio.run(main())
```

Use `base_url="https://your-endpoint.example/v1"` for an explicit endpoint. The
provider creates and owns its client unless you pass `client=AsyncOpenAI(...)`.
Use `async with` or `await provider.aclose()` to close an owned client. An injected
client is borrowed: Republic closes each response stream, but never that client.
Providers accept `client`, `base_url`, `headers`, `timeout` and `max_retries`.
Owned clients default to zero SDK retries. Borrowed clients retain retries,
redirects, headers, query, organization/project and transport configuration on a
private SDK copy; Republic neither mutates nor closes the caller's client.
Explicit constructor values override borrowed settings, which override service
defaults. Request `provider_options["extra_headers"]` overrides constructor
headers. An OAuth access token supplies the bearer credential; an explicit
Authorization header can override it. Choose endpoints and redirect policy
appropriate for your credentials. With a borrowed client, its base URL is used
unless `base_url` is passed explicitly.

One generate/stream is one logical model operation, without login, refresh,
agent/tool execution or follow-up inference. Caller-selected SDK/transport retries
may make multiple HTTP attempts. Set `max_retries=0` and configure the HTTP
transport accordingly when a single HTTP attempt is required. Individual streams
release their response without closing a reusable client. After refresh, build a
new provider with the returned access token; lifecycle policy belongs to the caller.

## Full history and native reasoning

The default payload has `store=False`, `truncation="disabled"` and
`include=["reasoning.encrypted_content"]`. These choices make the caller supply
complete history and request encrypted reasoning explicitly. An explicit
`include=[]` is accepted, but Republic cannot manufacture encrypted content if
the endpoint omits it. See the official guidance on
[preserving reasoning without stored responses](https://developers.openai.com/api/docs/guides/reasoning).

These are defaults, not required caller policies. Native `store`, `truncation`,
`include`, `previous_response_id` and `instructions` are forwarded. `extra_body`
provides a native extension path without overriding managed/common fields. A
previous response ID relies on the server's actual retention and access rules;
it does not guarantee recovery of arbitrary history. Republic does not poll
background jobs, recover conversations or interpret unsupported output items.

Each native output item becomes one Republic part, in output order:

| Responses item | Republic representation |
| --- | --- |
| Assistant message | `TextPart`; concatenated `output_text` for display |
| Reasoning | `ReasoningPart`; concatenated summary text for display |
| Function call | `ToolCallPart`; actual `call_id`, name and raw arguments |

Every returned part retains `provider_metadata["openai"]` with `item_id` and
`raw_item`. The latter preserves native content boundaries, item status, message
phase, annotations, logprobs, refusal text, reasoning summary, native reasoning
`content` and `encrypted_content`, when present. Refusal and native reasoning
content stay in metadata; they are not mixed into display text. Reasoning without
summary text produces an empty `ReasoningPart`, retaining its encrypted data.

`item_id` identifies the output item. A function's `call_id` identifies the tool
call and the corresponding `function_call_output`; these IDs are never substituted
for one another. Arguments remain unchanged even if they are malformed JSON.

Persist the **whole message**, including metadata, for a later caller-controlled
request. For example, inside an async function with a provider and request:

```python
from republic import Message, ToolResultPart, generate

response = await generate(provider, request)
saved_message = response.message.model_dump_json()
restored = Message.model_validate_json(saved_message)
request.messages.append(restored)

# After checking the response outcome and independently deciding to execute:
# call = restored.tool_calls[0]
# result = await your_tool_executor(call)
# request.messages.append(Message(role="tool", parts=[ToolResultPart(
#     tool_call_id=call.tool_call_id, tool_name=call.tool_name, result=result,
# )]))
# next_response = await generate(provider, request)
```

Replay sends retained native items inline, including encrypted-only reasoning.
A visible part changed independently of its retained item is rejected; Republic
does not silently prefer stale metadata or rewrite history. To supply new text
or a new function call, construct a new part without retained native metadata.
Unbacked reasoning text cannot recreate a native reasoning item and is rejected.
Tool results map strings verbatim and other JSON values to JSON text. The common
`is_error=True` flag is rejected: encode an error explicitly in the result data.

## Streaming and completion

```python
from republic import stream

async with stream(provider, request) as output:
    async for event in output:
        if event.kind == "text_delta":
            print(event.chunk, end="")
    response = output.response
```

Native item-added events establish part order and identity. Text/reasoning blocks
use the item ID; function events use the call ID. Interleaved function arguments
are accumulated independently. Incomplete function headers wait for real IDs and
names, holding later starts as necessary to preserve order. Content and summary
slots retain their native indices; a later slot waits until its predecessor is
done before becoming display text.

Deltas append once. A text/argument/content/item done snapshot can supply a
missing suffix when all received data is its prefix. Equal overlapping snapshots
are accepted without duplication. Once a value is done it cannot change or grow.
The terminal response may supply omitted items or optional metadata, including
late encrypted reasoning. It must contain every received output item and agree
with received IDs, types and content. Contradictory snapshots, missing headers,
changed identity and unsupported output kinds raise `ProviderError` with
`code="invalid_response"`; partial output remains inspectable.

Blocks remain open until the terminal snapshot is reconciled, so their end events
carry complete native metadata. The HTTP stream is closed before `StreamEnd` is
yielded. No additional EOF or `[DONE]` is required after a Responses terminal.

| Provider outcome | Republic outcome |
| --- | --- |
| `response.completed` | `stop`, or `tool_call` if output includes function calls |
| `response.incomplete` / `max_output_tokens` | `length`, retaining partial output |
| `response.incomplete` / `content_filter` | `content_filter` |
| Other incomplete reason | `other`, with raw incomplete details |
| `response.failed` | Terminal result with `finish_reason="error"` and native error metadata |
| SSE `error`, SDK/HTTP failure | `ProviderError`; SDK/HTTP exceptions remain `__cause__` |
| EOF or `[DONE]` without terminal | `IncompleteStreamError`; no successful response |
| Caller cancellation | Unchanged `asyncio.CancelledError`; response resource closed |

A non-streaming response uses the same status mapping. `queued`, `in_progress`
and unknown statuses cannot become a final result; Republic does not poll them.
Response ID, model, status, incomplete details, error and service tier are
retained in message metadata when present. Usage maps input/output/cached/reasoning
tokens and retains the raw usage object; missing usage stays unknown.

The public stream's `completed` status means a terminal result was received,
including a native failed or incomplete result. A tool end event is a data block
boundary, not permission to run a tool. Check the response finish reason, native
response/item status and arguments before deciding whether a call is executable.
Leaving the stream context early closes the HTTP response and retains partial
output, without returning a final response or closing a borrowed client.

## Options and structured output

Common options supported here are `temperature`, `top_p`, `max_output_tokens`,
`parallel_tool_calls`, and `tool_choice` (`auto`, `none`, `required`, or
`ToolChoice(name="...")`). Function tools use the flat Responses schema; optional
`provider_metadata={"openai": {"strict": True}}` sets native strict mode.

Accepted `provider_options` keys are `text`, `reasoning`, `store`, `include`,
`truncation`, `metadata`, `user`, `service_tier`, `safety_identifier`,
`prompt_cache_key`, `extra_headers`, and a positive numeric `timeout` in seconds.
`include` accepts encrypted reasoning and output-text logprobs. Native `text`
and `reasoning` configuration is sent as supplied; model-specific acceptance is
up to the endpoint. Managed keys such as `model`, `input`, `tools`, `stream`,
`tool_choice` and common options cannot be overridden. `extra_body` is not
supported. Unknown options and incompatible metadata fail before HTTP.

Responses uses [`text.format` for structured output](https://developers.openai.com/api/docs/guides/migrate-to-responses).
Republic returns model text without parsing it or issuing a repair call. The
caller can validate it explicitly with Pydantic:

```python
from pydantic import BaseModel, ConfigDict, ValidationError
from republic import RequestOptions, generate


class Count(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    count: int


request.options = RequestOptions(provider_options={"text": {"format": {
    "type": "json_schema", "name": "Count", "strict": True,
    "schema": Count.model_json_schema(),
}}})
response = await generate(provider, request)
# Check terminal status/refusal metadata before treating text as an answer.
answer = Count.model_validate_json(response.message.text)

# This fails locally; it does not make a model request:
try:
    Count.model_validate_json('{"count":"not an integer"}')
except ValidationError:
    pass  # The caller decides what happens next.
```

## Current limits and evidence

Supported input includes system text and user text/images/PDF, assistant text,
retained native reasoning, function calls and function results. See
[media inputs](media-inputs.md) for URLs/base64/file IDs and text-file conversion.
It rejects audio/video input, generated media output, hosted/built-in/custom tools, MCP, compaction, unknown reasoning
content kinds, unsupported roles/parts and stop sequences. No files are fetched,
no tools are executed, and no hidden history is reconstructed.

Tests use the real official async client with `httpx.MockTransport` and SSE bytes.
They inspect payloads, request counts, native metadata round trips, snapshot
reconciliation, structured-output validation, terminal outcomes, cancellation
and owned/borrowed resource release. They do not establish online model or account
access. Source revision and adaptations are recorded in `NOTICE` and the
[rebuild plan](rebuild-plan.md).
