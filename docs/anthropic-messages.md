# Anthropic Messages

`AnthropicMessages` implements one non-streaming or streaming `POST /v1/messages`
through the official asynchronous Anthropic client. Verification uses offline
HTTP/SSE fixtures, not a live account. No OAuth, hosted tools or agent runtime is
included.

## Install and make one request

Install this checkout with `pip install .` for runtime use or `uv sync` for
development. Anthropic `>=0.83.0,<1`, HTTPX and Pydantic are included dependencies;
Python 3.11+ remains the compatibility policy. No package extra is required.
This rebuild branch has not been published.

```python
import os

from republic import Message, Request, RequestOptions, TextPart, generate
from republic.providers.anthropic import AnthropicMessages


async def example():
    request = Request(
        model="your-anthropic-model",
        messages=[
            Message(role="system", parts=[TextPart(text="Be concise.")]),
            Message(role="user", parts=[TextPart(text="Hello")]),
        ],
        options=RequestOptions(max_output_tokens=1024),
    )
    async with AnthropicMessages(api_key=os.environ["ANTHROPIC_API_KEY"]) as provider:
        response = await generate(provider, request)
        return response.message.text
```

`max_output_tokens` is required by this adapter and maps directly to Anthropic's
`max_tokens`. Omitting it raises `UnsupportedRequestError` before HTTP. Republic
does not guess a model limit. `generate` uses `messages.create(stream=False)`;
`stream` uses `messages.create(stream=True)` and consumes raw events. There is no
fallback from non-streaming to streaming if the SDK rejects a large non-streaming
limit; the caller can choose streaming or an explicit request timeout.

## Clients and lifetime

Pass `api_key` explicitly to create an owned client. Reading an environment
variable, as above, is the caller's choice; Republic does not discover credential
files or log in. `base_url="https://your-endpoint.example"` sets an explicit
endpoint; the SDK appends `/v1/messages`.

Alternatively, pass `client=anthropic.AsyncAnthropic(...)`.
Explicit constructor `api_key`, `base_url`, `headers`, `timeout` and `max_retries`
override borrowed settings without mutating or closing the caller's client.
Owned clients default to zero SDK retries; borrowed retry/redirect policy is
retained. A response stream closes independently of its reusable client. See
[client configuration](client-configuration.md) for logical operation versus HTTP
attempt semantics. No tool execution, history repair or next turn is added.

## Messages, tools and options

Leading system messages become top-level `system` text blocks, preserving their
order and boundaries. A system message after any conversation message is
rejected instead of being hoisted to the front. Conversation messages and parts
retain input order; Republic does not merge consecutive roles or repair tool
history. The endpoint may apply its own role normalization.

Supported parts are:

| Role | Parts |
| --- | --- |
| Leading system | Text |
| User | Text, tool results |
| Assistant | Text, signed thinking, redacted thinking, function calls |
| Tool | Tool results, mapped to a user message |

`Tool` maps to `name`, optional `description`, and `input_schema`.
`ToolCallPart.tool_call_id` maps to `tool_use.id`; a result's ID maps to
`tool_result.tool_use_id`. Results preserve `is_error`; string results are sent
verbatim and other JSON values become JSON text. The `tool_name` on a result
remains caller data because Anthropic identifies results by ID.

The caller decides when and whether to execute a tool:

```python
from republic import Message, ToolResultPart

# After independently validating the outcome, call and arguments:
# call = response.message.tool_calls[0]
# result = await your_tool_executor(call)
# request.messages.extend([
#     response.message,
#     Message(role="tool", parts=[ToolResultPart(
#         tool_call_id=call.tool_call_id,
#         tool_name=call.tool_name,
#         result=result,
#         is_error=False,
#     )]),
# ])
# next_response = await generate(provider, request)
```

Anthropic's history `tool_use.input` is an object. Therefore input `tool_args`
must parse to a JSON object without duplicate keys or non-finite numbers;
malformed JSON is rejected before HTTP. Non-streaming output objects are encoded
as JSON strings. Streamed argument fragments remain verbatim, including truncated
or malformed JSON. Reusing those fragments as history requires a valid object;
Republic never fills missing braces or performs a repair request.

Common options: `temperature`, `top_p`, `max_output_tokens`, `stop` (mapped to
`stop_sequences`), and `tool_choice`. Choices map `auto` to `auto`, `none` to
`none`, `required` to `any`, and `ToolChoice(name=...)` to a named `tool` choice.
`parallel_tool_calls` becomes the inverse `disable_parallel_tool_use` flag;
when supplied alone it uses `auto`. Combining it with `none` is rejected because
Anthropic's none choice has no parallel flag.

Accepted `provider_options`: `thinking`, `top_k`, `metadata`, `service_tier`,
`output_config`, `cache_control`, `extra_headers`, and positive numeric `timeout`
in seconds. Native inference configuration is forwarded as supplied; the SDK or
endpoint validates its model-specific values. `output_config` does not parse or
validate model output for the caller. Tools may use
`provider_metadata={"anthropic": {"strict": True}}`.

Managed fields such as `model`, `messages`, `system`, `tools`, `max_tokens` and
`stream` cannot be overridden. Unknown options, `extra_body`, `extra_query`,
unsupported parts and incompatible metadata are rejected explicitly.

## Thinking and persistence

Signed thinking maps to `ReasoningPart(text=..., provider_metadata={"anthropic":
{"signature": ...}})`. Signature fragments are concatenated within their block;
the end event carries the complete signature, so generic metadata merging never
replaces one fragment with another. Initial signature/text data is also retained.
An empty visible thinking string with a signature remains a reasoning part.

`redacted_thinking` maps to an empty `ReasoningPart` with
`provider_metadata={"anthropic": {"redacted_data": ...}}`. Its data is opaque;
Republic neither decodes it nor invents display text.

Persist the whole response or message, including metadata:

```python
from republic import Response

saved = response.model_dump_json()
restored = Response.model_validate_json(saved)
request.messages.append(restored.message)
```

The next request reconstructs signed and redacted blocks in their original order.
Keep both thinking text and signature unchanged; signatures cannot be used to
validate edited reasoning. Unsigned/empty-signature thinking and redacted data
with visible text cannot be sent as history. A truncated result may not contain
reusable thinking. See the official
[thinking guidance](https://platform.claude.com/docs/en/build-with-claude/extended-thinking).

## Cache control and token counts

Set explicit breakpoints on text parts (including system text), tool-call parts,
tool results, or tool definitions using
`provider_metadata={"anthropic": {"cache_control": {"type": "ephemeral"}}}`.
An optional `ttl` is `5m` or `1h`. Thinking and redacted blocks reject explicit
cache markers. Top-level `provider_options["cache_control"]` supports the same
shape for native automatic caching; Republic does not insert its own breakpoints.

Anthropic reports uncached input, cache reads and cache writes separately.
Republic uses an inclusive `Usage.input_tokens`:

```text
input_tokens = native input_tokens
             + native cache_read_input_tokens
             + native cache_creation_input_tokens
```

`cache_read_tokens` and `cache_write_tokens` expose the corresponding components;
do not add them again to Republic's input/total tokens. TTL-specific creation
counts are a breakdown of cache writes and are retained in `Usage.raw`, not
added again. Output counts are already inclusive of thinking; where supplied,
`output_tokens_details.thinking_tokens` maps to `reasoning_tokens`.

If any of the three input components is absent/null, inclusive input and total
tokens remain unknown. Independently known output/cache counts and the raw usage
object remain available. Explicit zeroes are preserved. Cumulative usage fields
in `message_delta` replace previous values from `message_start` or earlier deltas;
omitted/null fields do not erase known counts. They are never added together.
This follows Anthropic's [cache token definitions](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
and [cumulative streaming usage](https://platform.claude.com/docs/en/build-with-claude/streaming).

## Stream outcomes and limits

Use `async with stream(provider, request) as output`, iterate its events, and then
inspect `output.response`. Block IDs are the native indices converted to strings;
tool events use actual call IDs. Interleaved blocks aggregate independently.
An initial `tool_use.input={}` is a placeholder and is not prepended to JSON
deltas. With no nonempty fragments, the initial input object is emitted at block
stop. Deltas following a nonempty initial object are rejected as ambiguous.

The adapter requires `message_start`, valid block lifetimes, a nonempty stop
reason, and `message_stop`. Pings are accepted (normally filtered by the SDK).
The response resource closes before `StreamEnd`; there is no need to wait for
EOF after message_stop. Partial output remains inspectable after early close,
cancellation or failure, without a final response.

| Native stop reason | Finish reason |
| --- | --- |
| `end_turn`, `stop_sequence` | `stop` |
| `max_tokens`, `model_context_window_exceeded` | `length` |
| `tool_use` | `tool_call` |
| `refusal` | `content_filter` |
| `pause_turn`, unknown future reasons | `other` |

Native stop reason, stop sequence and refusal details are retained in message
metadata. ID/model and usage come from the actual response. No continuation is
automatic. A tool end event is a block boundary; the caller must inspect the
terminal outcome and argument data before deciding whether to run anything.

Missing message_stop raises `IncompleteStreamError`. HTTP, connection and SSE
errors raise `ProviderError`, preserving the SDK/HTTP exception as `__cause__`,
with available status, error type and request ID. For an SSE error the HTTP status
may still be 200; Republic does not fabricate a 529 status from the error type.
Cancellation remains `asyncio.CancelledError`. Malformed blocks/order are explicit
`ProviderError(code="invalid_response")` failures, with no retry.

User images, PDF and plain-text documents are supported as described in
[media inputs](media-inputs.md), including native cache controls and file references.
Audio/video, generated media/citations, hosted/server tools,
programmatic callers, compaction, containers, MCP, beta-specific workflows and
mid-conversation system messages are outside its supported conversion. Input
unsupported media and incompatible metadata fail before HTTP; unsupported output blocks or
deltas fail visibly. No files are fetched and no tools are executed.

Real-SDK MockTransport/SSE tests cover payloads, request counts, signed/redacted
reasoning replay, cache accounting, stop reasons and resource release. They do
not establish live model/account access. See the [rebuild plan](rebuild-plan.md)
for verification evidence and `NOTICE` for the pinned upstream source.
