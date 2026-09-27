# Bub consumer acceptance

Step 8 adds an explicit Republic backend in the separate Bub branch
`feat/republic-provider-sdk`: commit `a3c45120de4878f7167247c360721c5d628477a2`,
starting at `357901db1a3f82d7f696024574225e595b09d4ac`.
Republic remains a provider SDK: no Bub dependency, agent loop, tool execution,
hook, model router or tape implementation was added here. No public SDK contract
change was required by this consumer.

## Install the local baseline

The validated SDK artifact was built from clean Republic commit
`5dff4aa7bdbf6f81411f16d64f74307d4d83f167`:

- File: `republic-0.5.9.dev17+g5dff4aa7b-py3-none-any.whl`.
- Version: `0.5.9.dev17+g5dff4aa7b`.
- SHA256: `217d45d0c354b83916412296ac2efdeb018d3399117e30f158d223c4bca6e06c`.
- Build: `uv build --wheel` in the clean Republic checkout. This Step 8 Republic
  increment changes documentation only; the validated SDK implementation is that
  source commit. Hashes identify exact artifacts, not reproducible-build guarantees.

In the authorized Bub checkout, use an explicit path to this artifact:

```bash
uv sync --locked
uv pip install --python .venv/bin/python /explicit/path/to/republic-0.5.9.dev17+g5dff4aa7b-py3-none-any.whl
uv pip check --python .venv/bin/python
```

Do not install the old published `republic` package as a substitute. These commits
have not been published and are not promised to be available remotely. Bub's
project/lock retain any-llm and contain no machine-specific Republic path. Exact
`uv sync` may remove the extra wheel; reinstall it or use `--inexact`.

## Select the consumer path

Bub YAML configuration for an API-key Responses model:

```yaml
model_backend: republic
model: openai:your-responses-model
api_key: YOUR_EXPLICIT_API_KEY
republic_protocols:
  openai: responses
max_tokens: 16384
```

Equivalent selectors are `BUB_MODEL_BACKEND=republic` and
`BUB_REPUBLIC_PROTOCOLS='{"openai":"responses"}'`. The default stays `any_llm`.
Supported combinations are OpenAI Chat/Responses, OpenRouter Chat and Anthropic
Messages, expressed using Bub's existing `provider:model` names and the separate
protocol map. A configured `api_base` is explicit; OpenRouter otherwise uses
`https://openrouter.ai/api/v1`.

Bub maps `max_tokens` to `RequestOptions.max_output_tokens`. Its `completion_args`
uses Republic common options and a nested `provider_options` object. Unsupported
settings or managed-field overrides fail. Text input and function tools/results
are enabled; SDK media capabilities do not imply Bub media integration.

The default factory owns an adapter/client per attempt and closes both on exit.
An embedding application can override `ModelRunner.create_republic_provider`
to return an adapter borrowing its official async client. The caller still owns
that client; individual response streams close on completion, failure, early exit
and cancellation. No retry or automatic refresh is introduced.

Existing onboarding and `bub login openai` remain on Bub's prior any-llm/Codex
path. The Republic backend requires explicit Bub API-key configuration and never
falls back to that path. Migrating OAuth presentation, existing credentials and
token renewal coordination is future Bub work; Copilot/Grok are not selectable
here. Their live acceptance remains open in the [matrix](support-matrix.md).

## Native tape and tool behavior

Bub consumes Republic `Request`, `Response` and events directly, without an
any-llm ChatCompletion intermediary. Text/reasoning use existing UI events.
Full message and part metadata live in a versioned `_republic` envelope with a
provider/protocol tag, alongside the existing visible tape message. Tool-call
entries add a native `message`, tool-result entries add native `messages`.
Original Responses item IDs/call IDs, encrypted-only reasoning and Anthropic
thinking signatures are durable JSON, not an in-memory cache.

Legacy text and complete function histories still read. Incompatible/unknown
legacy fields, missing IDs, cross-protocol native history and native history sent
through any-llm fail explicitly. Hooks cannot change only the visible projection
while leaving stale native content. Older Bub releases cannot promise lossless
continuation of these new tapes. Start a new tape or explicitly migrate history
when changing protocol/backend.

Bub validates the terminal outcome and function arguments before its ToolExecutor
runs a tool. Incomplete/failed/truncated outcomes and invalid JSON never execute
calls. Native terminal output is saved for inspection with `context=false` and
excluded from later request history. Cancelled/interrupted streams retain Bub's
non-terminal behavior and execute no tools. Tool failure/denial status survives
result hooks: Anthropic receives `is_error`; Chat/Responses receive explicit
error result JSON. Usage/raw fields and finish reason remain in Bub run records.

The agent loop, hooks and configured model fallback belong to Bub. Fallback may
try another candidate only on a provider error before any native event; it does
not replay after output, invalid input or an incomplete result. Each candidate
makes one HTTP request. Tape write errors still propagate without retrying tools.
External tool effects and tape persistence are not an atomic transaction; no
process-crash exactly-once guarantee is added.

## Reproduce the evidence

The Bub branch includes a standalone acceptance entry point:

```bash
python scripts/check_republic_wheel.py \
  /explicit/path/to/republic-0.5.9.dev17+g5dff4aa7b-py3-none-any.whl \
  --source-commit 5dff4aa7bdbf6f81411f16d64f74307d4d83f167
```

It creates a clean Python 3.12 environment, builds and installs the current Bub
wheel non-editably, installs the exact Republic wheel under Bub's locked
constraints, and runs `uv pip check`. Isolated imports must resolve to
site-packages; installed package files are compared with the two archives.
The JSON report records artifact hashes, source revision, version and import
paths. Building an explicit Bub wheel prevents uv's project cache from accepting
an earlier working-tree build.

The integration cases use real OpenAI/Anthropic SDK methods with synthetic
HTTP/SSE MockTransport fixtures. Separate OS processes prove, for **both Responses
and Anthropic**, the following sequence:

1. Actual Bub ModelRunner requests a model response, receives a tool call,
   executes it through ToolExecutor and merges a FileTapeStore fork to JSONL.
2. A new Python process opens that tape, constructs another real SDK request
   containing the original call ID plus encrypted reasoning or thinking signature,
   then receives the final answer.
3. Each process asserts one model HTTP request and released resources. The
   on-disk execution log contains exactly one tool effect across both processes.

Additional cases exercise the real Bub agent loop, Chat/OpenRouter factory,
tool failure/denial, hooks, explicit fallback, incomplete outcomes, history
rejection, tape-write failure, cancellation and borrowed client reuse. These are
not isolated conversion tests and do not substitute for live service acceptance.
Tests neither read actual token files nor make live inference calls.

The [plan](rebuild-plan.md#step-8-implementation) records both repository commits,
full check/test/docs results, installed dependency versions and remaining work.
Lower-bound SDK and interpreter checks from prior steps remain valid; this step
also verifies the real Bub lock combination without ignoring dependency conflicts.
