# Bub consumer acceptance

Step 8 replaces Bub's model SDK on `feat/republic-provider-sdk`. The initial optional
integration (`a3c45120`) was superseded by the user's direct-migration requirement:
Republic is mandatory, with no alternate SDK or backend selector. The replacement
commit is `bb89a96db9c13034aa8de87f30d8ac9880755f9c`
(`refactor: use Republic as the sole model SDK`). The original
Bub integration baseline is `357901db1a3f82d7f696024574225e595b09d4ac`.
Republic remains a provider SDK: no Bub dependency, agent loop, tool execution,
hook, model router or tape implementation was added here. The caller-policy correction now makes client configuration and OAuth lifecycle
policy composable; see [client configuration](client-configuration.md).

## Install the local baseline

Historical direct-migration artifact (before the caller-policy correction): clean Republic commit
`5dff4aa7bdbf6f81411f16d64f74307d4d83f167`:

- File: `republic-0.5.9.dev17+g5dff4aa7b-py3-none-any.whl`.
- Version: `0.5.9.dev17+g5dff4aa7b`.
- SHA256: `217d45d0c354b83916412296ac2efdeb018d3399117e30f158d223c4bca6e06c`.
- Build: `uv build --wheel` in the clean Republic checkout. This artifact predates the capability correction and must not be used for the
  updated Bub credential path. Build the new clean SDK commit instead. Hashes identify exact artifacts, not reproducible-build guarantees.

For source development, place `bub-republic-dev` beside `republic-dev`, then run
`uv sync --locked --extra trace` in Bub. Its normal runtime dependency is
`republic>=0.5.9.dev20,<0.6`; the local uv source is the repository-relative
`../republic-dev`, not an absolute path or unavailable remote commit. The sibling
checkout may include later documentation commits; the minimum runtime must include the caller-policy correction.
The trace extra is optional telemetry; Republic is required for imports and tests.

For wheel deployment, build both wheels and install their explicit paths together:

```bash
uv pip install --python /explicit/venv/bin/python \
  /explicit/path/to/bub-VERSION-py3-none-any.whl \
  /explicit/path/to/republic-VERSION-py3-none-any.whl
uv pip check --python /explicit/venv/bin/python
```

Bub's wheel declares the normal dependency and does not embed the uv source. Do
not substitute the old PyPI Republic package. These commits/wheels have not been
published or pushed; current source development requires both local checkouts.

## Configure the consumer

Bub YAML configuration for an API-key Responses model:

```yaml
model: openai:your-responses-model
api_key: YOUR_EXPLICIT_API_KEY
republic_protocols:
  openai: responses
max_tokens: 16384
```

The equivalent protocol setting is `BUB_REPUBLIC_PROTOCOLS='{"openai":"responses"}'`.
Supported combinations are OpenAI Chat/Responses/Codex, OpenRouter Chat and
Anthropic Messages. Provider/model parsing preserves everything after the first
colon. A configured `api_base` is explicit for API-key adapters; OpenRouter
otherwise uses `https://openrouter.ai/api/v1`. Unsupported providers are rejected,
not routed to an alternate SDK. Onboarding offers precisely the supported API-key
connections and custom OpenAI-compatible endpoints; listing models is not inference
acceptance. Standard provider API-key variables are lower-priority defaults.

Bub maps `max_tokens` to `RequestOptions.max_output_tokens`. An unset limit
defaults to 16384 for API-key protocols and omits the limit for Codex (an explicit
limit is rejected there). `completion_args` uses Republic common options and a nested `provider_options` object. Unsupported
settings or managed-field overrides fail. Text input and function tools/results
are enabled; SDK media capabilities do not imply Bub media integration.

The default factory owns an adapter/client per attempt and closes both on exit.
An embedding application can override async `ModelRunner.create_provider`
to return an adapter borrowing its official async client. The caller still owns
that client; individual response streams close on completion, failure, early exit
and cancellation. Republic preserves caller-selected retries and never refreshes inference; Bub owns the
pre-call refresh point described below.

## Bub Codex authentication

`bub login openai` delegates PKCE/state, exchange and refresh to Republic's
Authlib functions. Bub owns browser/local callback reception and manual URL/code
input. A complete URL is state-validated; a manually received bare code uses the
lower-level PKCE exchange. Login and refresh use the original nested `tokens`
format in `CODEX_HOME/auth.json` (default `~/.codex/auth.json`). `--codex-home`
selects the login directory; runtime `codex_home` / `BUB_CODEX_HOME` can select
that same directory. Existing credentials work directly; reads never write or
move them, and no credential migration command/file is required.

Automatic Codex selection applies only to `openai:` without explicit base/protocol.
Normal API keys use the API; an access token with a ChatGPT account hint selects
Codex without refresh/expiry. Without an explicit key/base, a parseable auth.json
selects Codex. Neither filename existence nor model name alone determines this.
Explicit protocol and endpoint/credential configuration remain available.

Bub retains its original expiry policy: `tokens.expires_at` (numeric or RFC3339),
then access JWT `exp`, then `last_refresh + 3600`, or current time plus 3600 when
no hints exist. JWT hints and fallback estimates are not verified identity or
server guarantees. The SDK does not impose these estimates. Bub refreshes 120
seconds early before inference; failure can use a still-valid old token, but an
expired token fails. Successful refresh updates the same auth.json, retaining
unknown fields and prior account if no replacement is supplied. Atomic 0600 writes
precede inference; persistence failure after rotation prevents inference. There
is no implicit refresh/replay after 401, login UI during inference or concurrent
refresh locking. Only leading system messages become Codex instructions, without
reordering native history.

Copilot/Grok login UX is not selectable here. Their live acceptance and all real
Codex account login/inference/refresh remain open in the [matrix](support-matrix.md).

## Native tape and tool behavior

Bub consumes Republic `Request`, `Response` and events directly, without intermediate completion DTOs. Text/reasoning use existing UI events.
Full message and part metadata live in a versioned `_republic` envelope with a
provider/protocol tag, alongside the existing visible tape message. Tool-call
entries add a native `message`, tool-result entries add native `messages`.
Original Responses item IDs/call IDs, encrypted-only reasoning and Anthropic
thinking signatures are durable JSON, not an in-memory cache.

Legacy text and complete function histories still read. Incompatible/unknown
legacy fields, missing IDs, cross-protocol native history fail explicitly. Hooks cannot change only the visible projection
while leaving stale native content. Older Bub releases cannot promise lossless
continuation of these new tapes. Start a new tape or explicitly migrate history
when changing protocol.

Bub validates the terminal outcome and function arguments before its ToolExecutor
runs a tool. Incomplete/failed/truncated outcomes and invalid JSON never execute
calls. Native terminal output is saved for inspection with `context=false` and
excluded from later request history. Cancelled/interrupted streams retain Bub's
non-terminal behavior and execute no tools. Tool failure/denial status survives
result hooks: Anthropic receives `is_error`; Chat/Responses/Codex receive explicit
error result JSON. Usage/raw fields and finish reason remain in Bub run records.

The agent loop, hooks and configured model fallback belong to Bub. Fallback may
try another candidate only on a provider error before any native event; it does
not replay after output, invalid input or an incomplete result. The default Bub factory explicitly sets zero SDK retries, so each candidate
makes one HTTP attempt. Custom factories choose their own transport policy. Tape write errors still propagate without retrying tools.
External tool effects and tape persistence are not an atomic transaction; no
process-crash exactly-once guarantee is added.

## Reproduce the evidence

The Bub branch includes a standalone acceptance entry point:

```bash
python scripts/check_republic_wheel.py \
  /explicit/path/to/republic-VERSION-py3-none-any.whl \
  --source-commit FULL_REPUBLIC_COMMIT
```

It creates a clean Python 3.12 environment, builds and installs the current Bub
wheel non-editably, installs the exact Republic wheel under Bub's locked
constraints (excluding the sibling source), and runs `uv pip check`. It asserts
the removed SDK is absent and Republic is a declared Bub dependency. Isolated imports must resolve to
site-packages; installed package files are compared with the two archives.
The JSON report records artifact hashes, source revision, version and import
paths. Building an explicit Bub wheel prevents uv's project cache from accepting
an earlier working-tree build.

The integration cases use real OpenAI/Anthropic SDK methods with synthetic
HTTP/SSE MockTransport fixtures. Separate OS processes prove, for **Responses, Anthropic and Codex**, the following sequence:

1. Actual Bub ModelRunner requests a model response, receives a tool call,
   executes it through ToolExecutor and merges a FileTapeStore fork to JSONL.
2. A new Python process opens that tape, constructs another real SDK request
   containing the original call ID plus encrypted reasoning or thinking signature,
   then receives the final answer.
3. Each process asserts one model HTTP request and released resources. The
   on-disk execution log contains exactly one tool effect across both processes.

Additional cases exercise the real Bub agent loop, Chat/OpenRouter factory,
Codex authorization/exchange/refresh/auth.json compatibility, CLI/config, tracing,
tool failure/denial, hooks, explicit fallback, incomplete outcomes, history
rejection, tape-write failure, cancellation and borrowed client reuse. These are
not isolated conversion tests and do not substitute for live service acceptance.
The final suite blocks unmocked HTTP and uses temporary synthetic credential files.
During development, a missing Codex transport patch allowed synthetic requests to
reach the real endpoint and receive 401; these are recorded as a test-isolation
failure, not live acceptance. No real credentials were used. The helper now also
patches the OAuth client constructor; unexpected HTTP fails locally.

The [plan](rebuild-plan.md#step-8-implementation) records both repository commits,
full check/test/docs results, installed dependency versions and remaining work.
Lower-bound SDK and interpreter checks from prior steps remain valid; this step
also verifies the real Bub lock combination without ignoring dependency conflicts.
