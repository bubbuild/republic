# Provider correctness cases

Run `uv run pytest tests/test_provider_golden.py`.

The fixed JSON/SSE inputs come from pinned Goose and Fantasy cases. Expected
behavior is written as ordinary Python `assert` statements, following the
upstream tests. Both sources use the same arrange, act, assert pattern:

1. Queue the case's numbered response bodies on the existing `FakeService`.
2. Call Republic's public `chat()` or `stream()` API.
3. Assert the case's expected text, decoded calls, usage, or round-trip fields.

Goose's format tests use constructed protocol inputs and explicit assertions.
Fantasy uses VCR recordings as HTTP inputs and checks Portuguese greetings,
weather calls and results, and parallel arithmetic tools. Its original checks
are kept in the corresponding tests, with additional checks for decoded args,
finish reasons, unique call IDs, and the outgoing tool-result fields.

Fantasy's VCR also matches requests. Native assertions retain checks for HTTP
method and URL, prompts, model, token limit, streaming options, tool declarations
and choice, and the complete replayed assistant calls and caller tool outputs.
Gemini call signatures are checked against the recorded input, so a signature
lost from both the stream events and final response cannot pass unnoticed.

The three Fantasy tests are parametrized with native pytest fixtures over four
API formats and ordinary/streaming calls, giving 24 cases. Goose contributes 14
focused cases; usage variants use native `pytest.mark.parametrize`.
All 38 migrated cases use ordinary assertions, including signature fragments
and Responses keepalive events followed by the `[DONE]` terminal marker.

`respond()` shares the chat/stream mechanics. For streaming it checks the
terminal `Completed`, accumulated text and reasoning, ready tool calls, and
that usage deltas sum to the final usage. Each test supplies its own expected
values from the source assertions. Token totals are explicit expected numbers,
and signatures and decoded tool arguments are asserted directly. Generated
Gemini call IDs are checked for identity and uniqueness within the conversation.

Tool outputs are supplied by the test caller, as required by Republic's API.
Thinking and redacted-thinking cases make a second request to check that the
preserved provider blocks are sent back correctly. These steps are explicit
in the tests. No provider keys or network requests are needed.

## Sources and adaptations

Fantasy is pinned to
[`b0fadc9ae2e7da81a5d0262563d3ced9b594bd75`](https://github.com/charmbracelet/fantasy/tree/b0fadc9ae2e7da81a5d0262563d3ced9b594bd75/providertests).
The 24 cases extract **all interactions** from each of `simple.yaml`,
`tool.yaml`, `multi_tool.yaml`, and their `_streaming` counterparts:

| Format | Original cassette directory under `providertests/testdata/` |
| --- | --- |
| `chat` | `TestOpenAICommon/openai-gpt-4o-mini` |
| `responses` | `TestOpenAIResponsesCommon/openai-gpt-4o-mini` |
| `messages` | `TestAnthropicCommon/claude-sonnet-4` |
| `gemini` | `TestGoogleCommon/gemini-2.5-flash` |


Goose is pinned to
[`9560429ff982bbfecec5094963e5f34696b63a1c`](https://github.com/block/goose/tree/9560429ff982bbfecec5094963e5f34696b63a1c/crates/goose-provider-types/src/formats).
The 14 cases adapt tests in that directory:

| Case suffix | Source file | Original test |
| --- | --- | --- |
| `chat_length` | `openai.rs` | `test_response_to_message_marks_length_finish_reason` |
| `chat_empty_arguments` | `openai.rs` | `test_response_to_message_empty_argument` |
| `messages_cache_write` | `anthropic.rs` | `test_parse_text_response` |
| `messages_unsigned_thinking` | `anthropic.rs` | `test_parse_unsigned_thinking_response` |
| `messages_thinking_streaming` | `anthropic.rs` | `test_streaming_thinking_and_text` |
| `messages_redacted_streaming` | `anthropic.rs` | `test_streaming_redacted_thinking` |
| `messages_parallel_streaming` | `anthropic.rs` | `test_streaming_reassembles_interleaved_parallel_tool_calls` |
| `messages_cache_streaming` | `anthropic.rs` | `test_streaming_preserves_cache_tokens_through_delta_merge` |
| `gemini_cached` | `google.rs` | `test_get_usage_with_cached_content` |
| `gemini_thinking_usage` | `google.rs` | `test_get_usage_includes_thinking_tokens` |
| `gemini_signature` | `google.rs` | `test_thought_signature_roundtrip` |
| `responses_keepalive_streaming` | `openai_responses.rs` | `test_responses_stream_ignores_keepalive_event` |
| `responses_refusal` | `openai_responses.rs` | `test_refusal_content_block_deserializes_in_non_streaming_response` |
| `responses_call_id` | `openai_responses.rs` | `test_responses_api_to_message_uses_call_id_for_tool_request_id` |


JSON bodies are pretty-printed; SSE response bodies retain the recorded data
and event order. Cassette timings and transport headers are omitted. Requests
use Republic's native field mappings, such as Chat's `max_completion_tokens`
and Anthropic's `is_error: false` on tool results. Chat omits `strict: false`,
and Responses does not add `store: false`. Gemini uses `parametersJsonSchema`,
omits the optional system-instruction role and generated call IDs, and wraps
tool output as `output` rather than Fantasy's `result`.

Goose's line-based streams gain SSE blank-line separators. The Gemini usage
cases keep their original usage-only payloads. The signature case replays the
source's shell/read calls, caller tool outputs, the later `echo` call with
`sig_456`, and the final `Done!` response. Assertions check the two distinct
signatures in the third outgoing request. Republic retains signatures attached
to individual calls; Goose's inherited-signature policy is outside this case.
Refusals use Republic's `refusal` field.

Two Goose cases cover behavior that required implementation fixes:

- `messages_thinking_streaming` constructs two signature fragments and expects
  `sig_abc123`. The Messages parser accumulates signatures within each thinking
  block, preserving both fragmented and single-event signatures for replay.
  This follows Goose and the official
  [Go SDK v1.75.0](https://github.com/anthropics/anthropic-sdk-go/blob/v1.75.0/messageutil.go)
  accumulator used by Fantasy. Anthropic's Python and TypeScript accumulators
  instead retain the last signature delta; the migrated case explicitly chooses
  fragment support.
- `responses_keepalive_streaming` includes a non-JSON `[DONE]` terminal marker.
  The [OpenAI SDK consumes that marker in its stream layer](https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py)
  before JSON decoding. Goose also handles it before event decoding, and Fantasy
  delegates it to its OpenAI SDK stream. Republic's provider stream configures
  the existing SSE reader to stop at that marker for `chat` and `responses`,
  before their JSON parsers run. Other formats retain ordinary SSE behavior.

The raw inputs and Goose expectations remain intact. Both cases pass without
expected-failure marks or rewritten response bodies. The Responses case checks
keepalive handling, text, ID, model, and token usage after the original terminal
marker; the signature case checks reasoning, usage, and opaque-block replay.

Change expected assertions only when the intended Republic behavior changes,
checking the source case and API contract. There is no generated output baseline
or automatic baseline update. The EOF formatting hook excludes raw SSE inputs
to preserve their event separators.

Both sources use Apache-2.0. Original licenses and applicable notices are retained
as `LICENSE.fantasy`, `NOTICE.fantasy`, and `LICENSE.goose`.
