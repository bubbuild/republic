# Anthropic Messages usage fixtures — provenance

The two `*.txt` files under `cases/` are verbatim, unmodified byte copies of upstream
Anthropic Python SDK streaming test fixtures. They are vendored only so that the
per-format usage tests can replay real provider SSE bytes through Republic's public
streaming API with a mocked transport. No network access is needed and no live
provider call is made.

| Field | Value |
| --- | --- |
| Repository | [`anthropics/anthropic-sdk-python`](https://github.com/anthropics/anthropic-sdk-python) |
| Pinned commit | `5139cf2e7d07aa618a6c30b7e0c712f10ccb5ad9` |
| Immutable path prefix | `tests/lib/streaming/fixtures/` |
| Upstream license file | `LICENSE` at the pinned commit (`Apache-2.0`) |
| Local license copy | `LICENSE.anthropic` (byte-identical) |
| Transformations | none |

Byte-exact URLs are
`https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/5139cf2e7d07aa618a6c30b7e0c712f10ccb5ad9/tests/lib/streaming/fixtures/<name>.txt`.

File sizes and hashes measured on the bytes now on disk:

| File | Bytes | SHA256 | Git blob SHA-1 |
| --- | --- | --- | --- |
| `cases/message_delta_fields_response.txt` | 1249 | `194442cb0bb34fa71ebfe7779221ed0761f6ca1714e626c01178e81dd3b79600` | `3b28f34cb0716aac2968d6d24722154d82087f15` |
| `cases/message_delta_omitted_usage_response.txt` | 980 | `163b57c8fbb83a2bd74c474b8afa130159a00c8fbce96d7d24e42c082f1fb76f` | `0e619cf6c9bef37404d9ec0e5a3b070aeabf977e` |
| `LICENSE.anthropic` | 11345 | `7fc343be7ec189b803c1872267cbaf0381b12ffdb22d972b8ea7b21b262ba563` | `98d5648323c30d7f62a7d9244baf0bd3e5d429a2` |

Both fixtures end with a single `\n`, so no SSE framing normalization is applied and the
replayed event payloads are the upstream ones. The reader under test dispatches the final
frame at end of stream, which is the same bytes an upstream client sees.

Upstream tests mirrored:

- `cases/message_delta_fields_response.txt` → `TestAsyncMessages::test_message_delta_fields_propagated`.
  A `message_delta` re-sends **cumulative** usage (`input_tokens` 40,
  `cache_creation_input_tokens` 12, `cache_read_input_tokens` 7, `output_tokens` 8,
  `output_tokens_details.thinking_tokens` 3, `server_tool_use.web_search_requests` 1) that
  supersedes the `message_start` values (25 / 10 / 5 / 1).
- `cases/message_delta_omitted_usage_response.txt` →
  `TestAsyncMessages::test_message_delta_omitted_usage_keeps_message_start`. The
  `message_delta` re-sends only `output_tokens` 8, so every other count must survive from
  `message_start`.

## Republic contract applied to these bytes

Republic's `TokenUsage.input_tokens` is documented as "all prompt tokens, including those
read from or written to a cache". Anthropic reports cache reads and writes apart from the
uncached input, so the final input count for the fields fixture is
`40 + 7 + 12 = 59`. `cached_tokens` and `cache_write_tokens` stay 7 and 12; they are not
double counted and the wire `input_tokens` is not re-reported as 40.
`output_tokens` stays 8 because Anthropic's `output_tokens` already includes thinking.

Republic keeps only the provider's own fields, so `server_tool_use` and the `container`
object on `message_delta` do not appear in `TokenUsage`. Republic is not endorsed by or
affiliated with Anthropic; the fixtures are redistributed under Apache-2.0.
