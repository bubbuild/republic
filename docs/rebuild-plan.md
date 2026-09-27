# Provider SDK rebuild plan

Status: Steps 0-7 have local code and deterministic HTTP/SSE evidence, including
concrete ChatGPT/Codex, GitHub Copilot and Grok OAuth protocol adaptations.
No live service or real-account OAuth acceptance has been validated. Step 7
implementation is complete for the documented subset; live acceptance remains
open, including Grok client/scopes/entitlement. Step 8 adds an explicit Bub backend
and installed-wheel, fresh-process integration evidence. The local code/offline
baseline is complete for its documented subset; OAuth UX migration and all live
acceptance remain open. Verification evidence and supported subsets follow.

## Goal and boundaries

Build a readable reference implementation of a provider SDK that Bub can use
without adopting another agent runtime. A call supplies messages and tool
declarations, then returns one model response or an asynchronous event stream.
The caller executes tools and decides whether to make another model call.

Keep the public surface small: messages and content parts, tool data, request
options, results, usage, provider errors, and `generate`/`stream`. Names and exact
signatures remain provisional until the first working provider uses them.
Pydantic is suitable for data modeling. Official OpenAI and Anthropic clients
can handle transport. Do not carry over agent loops, tool execution, tape,
hooks, approval/resume state, MCP, UI, gateway routing, or unrelated media
generation and embedding operations.

OAuth is Authlib plus concrete provider logic. Use a small module per service
with login, refresh, and credential read/write functions as needed. Extract
shared helpers only after actual duplication appears. Do not introduce a
general authentication framework. Bub owns the interactive login presentation.

## Source baseline

- Republic `main`: `216098ef8f71afa5fc164284504036d81f4706c4`.
- Vercel AI Python reference: `c788059dd1db2d93ae1c3da6daffb660eca07dbb`.
- Bub consumer inspected at `357901db`.

Relevant upstream sources:

- [Model API and stream aggregation](https://github.com/vercel-labs/ai-python/blob/c788059dd1db2d93ae1c3da6daffb660eca07dbb/src/ai/models/core/api.py).
- [Message types](https://github.com/vercel-labs/ai-python/blob/c788059dd1db2d93ae1c3da6daffb660eca07dbb/src/ai/types/messages.py) and [events](https://github.com/vercel-labs/ai-python/blob/c788059dd1db2d93ae1c3da6daffb660eca07dbb/src/ai/types/events.py).
- [OpenAI protocols](https://github.com/vercel-labs/ai-python/blob/c788059dd1db2d93ae1c3da6daffb660eca07dbb/src/ai/providers/openai/protocol.py).
- [Anthropic protocol](https://github.com/vercel-labs/ai-python/blob/c788059dd1db2d93ae1c3da6daffb660eca07dbb/src/ai/providers/anthropic/protocol.py).
- [Republic GitHub authentication](https://github.com/bubbuild/republic/blob/216098ef8f71afa5fc164284504036d81f4706c4/src/republic/auth/github_copilot.py) and [Codex authentication](https://github.com/bubbuild/republic/blob/216098ef8f71afa5fc164284504036d81f4706c4/src/republic/auth/openai_codex.py).
- [Bub model runner](https://github.com/bubbuild/bub/blob/357901db/src/bub/builtin/model_runner.py), [authentication](https://github.com/bubbuild/bub/blob/357901db/src/bub/builtin/auth.py), and [Codex adapter](https://github.com/bubbuild/bub/blob/357901db/src/bub/builtin/codex_provider.py).

Extract necessary code and the matching behavior tests, not whole directories.
Upstream `types` also contain agent state; remove that coupling. Retain applicable
copyright and Apache-2.0 notices and record provenance for copied code. Do not
claim unchanged upstream compatibility after modifying its contracts.

## Delivery sequence

Implement in order on `dev`. Each step owns its behavior, tests, documentation,
and the smallest necessary dependency/configuration changes. Use separate
Conventional Commits; split a step further when its behavior warrants it.
The suggested subjects below are a sequence, not a requirement to create one
large commit per step.

### Step 0: Clear the legacy implementation

- **Prerequisite:** a clean worktree branched from `main`.
- **Change:** remove old `src`, `tests`, product documentation, and examples.
  Preserve `.github`, `.gitignore`, `.pre-commit-config.yaml`, `pyproject.toml`,
  `uv.lock`, `Makefile`, `tox.ini`, `LICENSE`, `CONTRIBUTING.md`, and `docs/CNAME`.
  Keep MkDocs and its theme/plugins; update only its description and navigation
  for the new pages. Add an honest README and this plan.
- **Failure boundary:** never reset or clean the original checkout's uncommitted
  work. An archived remote does not prevent local preparation; leave archive
  settings, publication, and remote branches untouched.
- **Acceptance:** `dev` descends from the recorded `main`; the removal diff is
  limited to the agreed content; preserved infrastructure matches `main` except
  the documented MkDocs description/navigation edits; no SDK support is claimed.
- **Commits:** `chore: clear legacy implementation for provider sdk rebuild`,
  then `docs: define provider sdk rebuild increments`.
- **Temporary limit:** the intentionally empty package/test tree cannot yet
  satisfy installation and runtime CI. Restore it in Step 1 rather than
  disabling checks or adding meaningless tests.

### Step 1: Establish the single-call data contract

- **Prerequisite:** Step 0.
- **Behavior:** a deterministic fake model can emit and aggregate one response;
  messages and results can be serialized and restored without an agent runtime.
- **Contract:** text/file/reasoning content, tool declarations/calls/results,
  common request options, usage, errors, completion reason, and response identity.
  Retain provider metadata needed for subsequent requests. Tool arguments remain
  reconstructable from partial JSON. Keep provider clients out of persisted data.
- **Failure boundary:** distinguish completion, incomplete streams, and caller
  cancellation; close owned streams on exit. The SDK never executes a tool or
  automatically starts a follow-up model turn. History repair must be explicit.
- **Acceptance:** serialization round trips, interleaved tool-call aggregation,
  missing terminal event, and early cancellation tests. Restore package imports,
  typing, wheel construction, and the existing checks with meaningful tests.
- **Dependency decision:** retain the existing Python 3.11+ policy unless adopting
  upstream code requires a deliberate compatibility change; ai-python uses
  Python 3.12+ syntax. Do not silently break the CI matrix. Remove obsolete
  runtime dependencies when no implementation uses them and refresh `uv.lock`.
- **Commit:** `feat: establish single-call model contracts`.

### Step 2: Implement OpenAI Chat Completions

- **Prerequisite:** Step 1; this is the first real consumer of the contracts.
- **Behavior:** generate/stream text and function calls against OpenAI-compatible
  endpoints, including an explicit custom base URL. Preserve the path needed by
  Bub's default OpenRouter configuration.
- **Contract:** conversion between common messages and Chat Completions wire
  payloads, tool choice, streaming usage, errors, and client ownership.
- **Failure boundary:** malformed stream/tool data and unsupported options are
  visible; once output has been emitted, do not silently replay the request.
- **Acceptance:** fixture-driven request/response tests for text, multiple tool
  calls, fragmented arguments, usage-only chunks, API failures, and cancellation.
  A caller-controlled example feeds a tool result into a second request.
  Record a real-service smoke test separately when an authorized test account
  is available; fixtures alone do not establish live compatibility.
- **Commit:** `feat: add OpenAI Chat Completions provider`.

### Step 3: Implement OpenAI Responses

- **Prerequisite:** Steps 1-2.
- **Behavior:** the same model-facing API uses the Responses protocol.
- **Contract:** Responses input/output items, call IDs, response identity,
  structured output, reasoning metadata, and complete/incomplete outcomes.
- **Failure boundary:** retained reasoning data must not be silently flattened
  into display text; unsupported input/options produce explicit errors.
- **Acceptance:** stream aggregation, function call/result round trips,
  serialized reasoning metadata reused in the next request, structured-output
  validation, and interrupted-stream regression fixtures.
- **Commit:** `feat: add OpenAI Responses protocol`.

### Step 4: Implement Anthropic Messages

- **Prerequisite:** Steps 1-3; extract common helpers only where both providers
  demonstrate the same responsibility.
- **Behavior:** generate/stream text, thinking, and tool calls through Anthropic.
- **Contract:** message and tool-result conversion, thinking signatures, cache
  options and usage breakdown, stop reasons, and provider-specific parameters.
- **Failure boundary:** invalid/incompatible blocks or options must not disappear
  silently; preserve missing usage breakdowns as unknown rather than invented
  zeroes where the common contract supports that distinction.
- **Acceptance:** text/tool fixtures, signature deltas and history round trips,
  cache usage, error mapping, and the shared stream lifecycle tests.
- **Commit:** `feat: add Anthropic Messages provider`.

### Step 5: Add ChatGPT/Codex OAuth

- **Prerequisite:** Step 3 and the shared provider contracts; implemented after
  Step 4 in this initial sequence.
- **Behavior:** a user can log in, reuse credentials, refresh them, and call the
  Codex-backed model endpoint through Republic.
- **Contract:** Authlib plus concrete login/refresh/read/write functions and
  the necessary Codex endpoint, header, and request adaptations. Start from
  Bub's existing implementation and tests. Keep login UX with the caller.
- **Failure boundary:** cancelled login, bad state, expired credentials, refresh
  rejection, and inference permission failures remain distinguishable. A model
  request must not unexpectedly open a browser or wait for terminal input.
- **Acceptance:** deterministic auth/expiry/refresh tests and protocol fixtures;
  login -> inference -> refreshed inference is the live acceptance path.
- **Commit:** `feat: add ChatGPT Codex OAuth support`.

### Step 6: Add GitHub Copilot OAuth

- **Prerequisite:** Steps 2 and 5.
- **Behavior:** a GitHub Copilot account can authenticate and make a single model
  request without installing another agent runtime.
- **Contract:** Authlib plus concrete GitHub login/token logic, provider headers,
  and any required token exchange. Revisit legacy Republic logic against current
  service behavior; do not assume old token handling still suffices.
- **Failure boundary:** device polling respects pending/slow-down/denied/expired
  outcomes. Successful GitHub login does not by itself establish Copilot access.
- **Acceptance:** device-flow and token tests, model request/stream fixtures, and
  a separately recorded account-backed inference check.
- **Commit:** `feat: add GitHub Copilot OAuth support`.

### Step 7: Add Grok OAuth

- **Prerequisite:** shared contracts and the concrete auth patterns established
  by Steps 5-6. Investigate protocol feasibility early if it threatens scope.
- **Behavior:** authenticate a Grok account and perform inference through the
  endpoint available to that authenticated account.
- **Contract:** Authlib plus Grok-specific login, refresh, credential persistence,
  and request adaptation. Reuse an existing wire protocol only after confirming
  it fits; authentication can select a different endpoint from API-key access.
- **Failure boundary:** token acquisition, account entitlement, and inference are
  separate outcomes. Do not label this supported based solely on login success.
- **Implementation checkpoint:** following the Step 7 authorization, establish
  concrete public protocol evidence first, then implement and test that subset
  offline. No live login or paid inference is authorized in this increment.
  Record source-derived facts separately from adapter choices and account results.
- **Live acceptance (still open):** login -> inference -> refreshed inference
  with the intended client/account, including client/scopes/model entitlement.
  Fixtures cannot close this item. Missing account access does not block the code
  checkpoint or other independent work; protocol gaps must remain explicit.
- **Commit:** `feat: add Grok OAuth support` for the sourced protocol subset;
  use an auth-only title if inference wire cannot be established.

### Step 8: Validate the reference implementation with Bub

- **Prerequisite:** implemented protocol/auth steps; any unverified auth service
  remains explicitly experimental and must not be included in support claims.
- **Behavior:** Bub can consume the SDK while retaining its own agent loop,
  ToolExecutor, hooks, model selection, and tape implementation.
- **Contract:** translate SDK results/events at the Bub boundary, persist the
  message metadata needed for the next call, and preserve existing cancellation
  and tool-result behavior. Plan settings/onboarding/Codex migration and the
  treatment of Bub's other existing providers before removing any-llm.
- **Acceptance:** build/install the wheel in a clean environment; run existing
  lint/typing/tests/docs checks. In a separate Bub integration worktree, exercise
  model -> caller-executed tool -> tape persistence -> fresh-process continuation.
  Report deterministic fixtures and live-provider evidence separately.
- **Commits:** `docs: document provider sdk usage` and
  `test: validate Bub integration contracts`; Bub runtime changes belong in Bub.

## Completion evidence

Update this document after each increment with its commit, commands run, results,
and remaining limits. Do not claim the plan or a green generic test run proves
live account access.

### Step 1 implementation

- Commit: `fc10b257ca362c70741dc5f96cd4f038516462f3`
  (`feat: establish single-call model contracts`).
- Restored `src/republic`, `py.typed`, and deterministic tests. Public entry points
  are `generate(provider, request)` and `stream(provider, request)`; the sole
  runtime adapter boundary is a structural `Provider` protocol with two methods.
- Added JSON message/content/tool/request/response/usage types. Provider clients
  stay outside persisted data. Request copies isolate caller history; the API
  does not repair it, execute tools, retry, fall back or append another turn.
- Retained upstream text/reasoning/tool start/delta/end semantics, with strict
  ID/order validation and interleaved argument aggregation. Metadata merges
  recursively with latest supplied leaves, preserving reasoning data. Tool JSON
  remains verbatim. Response identity/finish reason survive serialization.
- Stream exhaustion without `StreamEnd` raises; early close/cancellation retain
  partial output without inventing a response. Normal completion, errors and
  context exit close the source; explicit close is idempotent and waits through
  cancellation during cleanup. The contract is single-consumer asyncio.
- `NOTICE` records the fixed upstream revision, Vercel copyright, extracted
  files and changes. No agent/UI/MCP/replay/approval models were carried over.
- Retained Python 3.11+ and all existing quality/release tooling. Runtime now
  needs only Pydantic; removed unused any-llm-sdk, Authlib and HTTPX and refreshed
  `uv.lock`. Authlib and official transport clients return with their actual
  implementation increments.
- Verification (2026-09-28, local Linux):
  - `make check`: passed lock consistency, all applicable prek hooks, and ty
    without diagnostics. Existing rules and hook versions were not disabled.
  - `make test`: 30 passed on Python 3.11.15 with locked Pydantic 2.12.5.
  - `make docs-test`: strict MkDocs build passed; the complete offline example
    from `docs/contracts.md` also ran successfully.
  - `uv run tox`: py311 (3.11.15), py312 (3.12.13), py313 (3.13.13) and py314
    (3.14.7) each passed 30 tests and ty. The retained tox commands select each
    interpreter into the project `.venv`; uv reports the existing `VIRTUAL_ENV`
    mismatch warning. Test headers confirmed the intended Python versions.
    Restored `.venv` to Python 3.11 afterward.
  - `uv build --wheel`: succeeded. Inspected the wheel for `py.typed`, LICENSE
    and NOTICE. Created a fresh Python 3.11 environment under `/tmp`, installed
    only the wheel and its runtime dependencies, and used `python -I` outside
    the repository to confirm the import came from site-packages, round-trip a
    request, and run the offline example. The clean resolver chose Pydantic
    2.13.5; the wheel declares only `pydantic>=2.7.0` as a runtime dependency.
  - In that isolated wheel environment, installed Pydantic 2.7.0 and pytest /
    pytest-asyncio, then ran `python -I -m pytest <checkout>/tests -q` from `/tmp`:
    30 passed, verifying the declared dependency floor against installed code.
  - `git diff --check` and `git diff --cached --check`: passed.
- Limits: no real OpenAI/Anthropic provider, OAuth service, live account access,
  or Bub integration has been implemented or validated. The API remains
  provisional for the next real adapter; client ownership will be exercised then.

### Step 2 implementation

- Commit: `d00428477c166009bf312d2d06e43ddba4bd09ac`
  (`feat: add OpenAI Chat Completions provider`).
- Added `republic.providers.openai.OpenAIChatCompletions`, using official
  `openai.AsyncOpenAI` for native non-streaming generation and streaming.
  Supports explicit API key/base URL or an injected client; owned clients have
  context/close methods, while borrowed transports remain caller-owned.
- Both ownership paths disable SDK retries; injected settings are not mutated.
  Actual HTTP-attempt assertions cover success, retryable HTTP errors, timeout,
  connection and mid-stream failures. No automatic turns, execution or repair.
- Added Chat message/tool/options conversion, user images, text reasoning
  extensions, response identity, usage and finish-reason mapping. Unknown or
  conflicting options and unrepresentable input fail explicitly before HTTP.
- Streaming buffers argument fragments until tool ID/name are available,
  aggregates interleaved indices and consumes tail usage before StreamEnd.
  Missing finish evidence fails; malformed argument JSON remains verbatim.
- Small shared-contract changes: `ProviderError` also carries optional service
  code/request ID; `UnsupportedRequestError` describes unrepresentable input;
  the shared Stream marks adapter-raised `IncompleteStreamError` as incomplete.
  Existing data/event signatures are unchanged.
- Added OpenAI 2.x and HTTPX runtime dependencies and updated `uv.lock`. Updated
  NOTICE with the pinned upstream Chat protocol and selected test provenance.
- Verification (2026-09-28, local Linux):
  - `make check`: lock consistency, applicable prek hooks and ty passed without
    disabling or relaxing checks.
  - `make test`: 110 passed on Python 3.11.15, including all 30 Step 1 cases
    and 80 provider cases. Locked versions: OpenAI 2.54.0, HTTPX 0.28.1,
    Pydantic 2.12.5. The provider cases use the real SDK with MockTransport and
    controlled SSE, including byte/UTF-8 fragmentation and request counting.
  - `make docs-test`: strict MkDocs build passed.
  - `uv run tox -e py314`: 110 passed and ty passed on Python 3.14.7. Only the
    upper supported interpreter was added to this step's 3.11 evidence; the
    whole Step 1 matrix was not repeated. Restored `.venv` to Python 3.11.
  - `uv build --wheel`: succeeded. Inspected provider files, `py.typed`, LICENSE
    and NOTICE inside the wheel. Installed it into a fresh Python 3.11 venv in
    `/tmp`; `python -I` outside the repository imported it from site-packages
    and completed an SDK/MockTransport call, checking one HTTP attempt and
    borrowed-client reuse. Clean resolution used OpenAI 2.54.0/Pydantic 2.13.5.
  - In the isolated wheel environment, installed the declared lower bounds
    OpenAI 2.16.0 and Pydantic 2.7.0, plus pytest/pytest-asyncio; running
    `python -I -m pytest <checkout>/tests -q` passed all 110 tests.
  - `git diff --check` and `git diff --cached --check`: passed.
- Environment evidence: the first mocked SDK request stalled in the restricted
  execution sandbox. A 5-second faulthandler dump showed the asyncio selector
  waiting and a thread worker idle; the focused test timed out at 15 seconds
  (exit 124). A sandbox thread-wakeup limitation is suspected, not established. The unchanged fixtures
  passed in a normal permitted process, as did all Makefile/tox checks. No SDK
  methods or thread helpers were patched to make the checks pass, and no live
  request was used to work around this sandbox behavior.
- Limits: no live OpenAI/OpenRouter inference, account access, OAuth or Bub
  integration was attempted. Fixtures prove local protocol behavior only.
  Unsupported: Responses, Anthropic, audio/PDF/generated media, built-in/legacy
  function-call formats, annotations and encrypted/signed reasoning_details.
  Tool ID/name fields must be atomic (late arrival is supported); changing
  header values fails rather than guessing. See the provider guide for details.

### Step 3 implementation

- Commit: `04c724f2b089c771b4af5f2ffffbdf5c424042bc`
  (`feat: add OpenAI Responses protocol`).
- Added `republic.providers.openai.OpenAIResponses` with native non-streaming
  `responses.create` and streaming through the same public `generate`/`stream`
  API. No common data, event or error signatures changed. Only the actually
  shared client lifetime/error mapping moved into `_openai_client.py`.
- Full history is sent inline with `store=False` and `truncation="disabled"`;
  encrypted reasoning is requested by default. Server-side state options are
  explicitly rejected. Output item IDs remain distinct from function call IDs.
  Each output item maps to one part, retaining its complete native item in
  metadata. Empty-display encrypted reasoning, summary/content boundaries,
  status, message phase, annotations and refusal survive JSON persistence.
  Replay checks visible parts against retained items and never repairs history.
- Stream conversion reconciles text, summary, native reasoning content, function
  argument, content-part and output-item events. Snapshots can fill missing
  suffixes before values close; they cannot rewrite received content. Overlapping
  done/terminal data is not appended twice. Interleaved output and content indices
  retain order; pending function identity never uses a fabricated call ID.
- Completed, incomplete/length, native failed and SSE errors remain distinct.
  Native failed responses preserve partial output and error metadata with finish
  reason `error`; connection exhaustion without a terminal raises
  `IncompleteStreamError`. Streams close before StreamEnd, and early exit,
  cancellation or exceptions release only their response resource. Owned clients
  close with the provider; borrowed clients remain open and keep their settings.
- Native `text.format` structured-output configuration is supported. Parsing and
  Pydantic validation are explicit caller operations, including schema mismatch;
  neither validation failure nor provider error causes another request. Input
  roles/parts and managed/unsupported options fail explicitly. Tool arguments
  remain verbatim, including malformed JSON. No tools are executed locally.
- Updated README, protocol/contract guides, MkDocs navigation and NOTICE using
  fixed ai-python revision `c788059dd1db2d93ae1c3da6daffb660eca07dbb`. Consulted
  the installed official SDK event/input schemas and official Responses guidance
  as a cross-check; this is a modified subset, not upstream compatibility.
- Verification (2026-09-28, local Linux):
  - `make check`: lock consistency, prek hooks and ty passed, without relaxing
    checks. Runtime dependencies and `uv.lock` did not need changes.
  - `make test`: **194 passed** on Python 3.11.15, including all 110 prior tests
    and 84 Responses cases. Locked OpenAI 2.54.0, HTTPX 0.28.1 and Pydantic 2.12.5.
    Tests use the real SDK, MockTransport and controlled SSE bytes; assertions
    cover payloads, request counts, serialization/replay, snapshot consistency,
    terminal outcomes, client ownership and cleanup. Two intentionally invalid
    output fixtures produce SDK/Pydantic serialization warnings before Republic
    rejects them; the warnings were not suppressed.
  - `make docs-test`: strict MkDocs build passed.
  - `uv build --wheel`: passed. Installed that wheel in a fresh Python 3.11 venv
    under `/tmp/republic-step3-install.Ch15tb`; isolated `python -I` imports resolve
    to site-packages, including both provider classes. Verified packaged NOTICE
    and `py.typed`. Ran all **84 Responses tests** against the installed wheel
    with declared lower bounds OpenAI 2.16.0 and Pydantic 2.7.0: passed, with the
    same two expected invalid-output warnings. The full Python matrix was not
    repeated; new code runs on 3.11 and lint/typing retain that syntax target.
  - `git diff --check` and `git diff --cached --check`: passed.
- Offline SDK tests ran in a normal permitted process with 90-second command
  timeouts, following the Step 2 restricted-sandbox wakeup evidence. No mocks
  were changed to bypass the SDK, and no real credentials or live inference were
  used. Fixtures do not establish live OpenAI/compatible endpoint acceptance.
- Remaining limits: text-only input; no hosted/custom tools, media, MCP,
  compaction, previous-response references, stored conversations or background
  polling. Unknown reasoning content kinds are rejected. Native reasoning text
  is retained in metadata; only summaries are exposed as display reasoning.
  Anthropic, OAuth and Bub integration remain later steps.

### Step 4 implementation

- Commit: `716b654d56eea24608eb4c0a71ab187823742f3c`
  (`feat: add Anthropic Messages provider`).
- Added `republic.providers.anthropic.AnthropicMessages`, using official
  `anthropic.AsyncAnthropic.messages.create` for one native non-streaming request
  or one raw event stream. Explicit API key/base URL and borrowed clients are
  supported; no credential-file discovery or login is performed by Republic.
  Both client paths disable SDK retries, preserve caller settings and close only
  resources they own. Anthropic does not inherit the OpenAI-specific helper.
- Converts leading system text blocks, ordered user/assistant history, function
  declarations/calls/results (including is_error), common options and a bounded
  set of native options. `max_output_tokens` is required and maps to `max_tokens`;
  there is no guessed default. Later system messages, unsupported parts/metadata,
  managed overrides and invalid tool-history JSON fail explicitly. No history
  repair, consecutive-role merging, tool execution or follow-up inference.
- Thinking text and signatures, including multiple signature fragments, survive
  Response JSON round trips and reconstruct subsequent request blocks. Signature
  fragments are assembled before complete metadata is emitted. Redacted thinking
  and signed thinking without display text remain empty reasoning parts with
  their opaque provider data intact. Public part/event/error fields did not need
  extension; `Usage` documentation now states the inclusive input-token meaning.
- Streaming validates indexed block lifetimes, handles interleaved text/thinking/
  tool deltas, ignores the initial empty tool input placeholder when fragments
  arrive, and retains malformed/truncated argument text. A nonempty stop reason
  plus message_stop is required. Pings are accepted, SDK/HTTP causes survive error
  mapping, missing terminals fail and early exit/cancellation release responses.
  `pause_turn` and unknown reasons map to `other` with raw reason metadata and
  never trigger continuation. Owned/borrowed client reuse and closure are tested.
- Cache control is explicit on supported blocks/tools or at request level.
  Normalized input is **uncached + cache-read + cache-creation** tokens; cache
  breakdowns and TTL details are not added twice. This corrects the fixed
  upstream's omission of cache-creation tokens. Missing components leave total
  input unknown; raw counters remain available. Cumulative usage patches replace
  earlier values instead of being summed. Consulted the official
  [cache definitions](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
  [streaming protocol](https://platform.claude.com/docs/en/build-with-claude/streaming),
  and installed SDK source for Messages, thinking/redacted block types, usage,
  stream filtering/errors, client copy and retry behavior.
- Added `anthropic>=0.83.0,<1` and refreshed `uv.lock` (locked SDK 0.125.0;
  docstring-parser is its additional transitive dependency). Preserved Python
  3.11+ and all existing check settings. Shared only the existing controllable
  HTTP byte-body fixture between provider tests. Updated README, protocol guides,
  MkDocs navigation and NOTICE; Vercel source remains revision
  `c788059dd1db2d93ae1c3da6daffb660eca07dbb`.
- Verification (2026-09-28, local Linux):
  - `make check`: lock consistency, applicable prek hooks and ty passed.
  - `make test`: **302 passed**, preserving all 194 Steps 1-3 cases and adding
    108 Anthropic cases. Python 3.11.15, Anthropic 0.125.0, HTTPX 0.28.1 and
    Pydantic 2.12.5. The same two expected invalid-output serialization warnings
    from Step 3 remain; no new warnings or disabled checks.
  - `make docs-test`: strict MkDocs build passed.
  - `uv build --wheel`: passed. Installed the wheel in a fresh Python 3.11
    environment under `/tmp/republic-step4-install.F1exin`. Isolated `python -I`
    imports resolved to site-packages; AnthropicMessages, packaged NOTICE and
    `py.typed` were verified. All **108 Anthropic tests** passed against that
    installed wheel with Anthropic **0.83.0** and Pydantic **2.7.0**.
  - `git diff --check` and `git diff --cached --check`: passed.
- All inference tests use the real official SDK plus MockTransport/SSE fixtures,
  with request-count assertions and no real tokens or paid requests. Tests ran
  in the already permitted normal process with 90-second command timeouts, given
  the earlier restricted-sandbox wakeup issue. No SDK request method or thread
  helper was patched to bypass the transport. The prior full Python matrix was
  not mechanically repeated; new code executes on the 3.11 lower bound.
- Limits: text-only; no media/files, citations, hosted tools, programmatic callers,
  containers, compaction, MCP or beta-specific workflows. Native inference option
  values remain subject to endpoint/model validation. No live Anthropic account
  or model acceptance, OAuth or Bub integration was attempted. At this point,
  Steps 5-8 were unimplemented; the increment stopped at the Messages provider.

### Step 5 implementation

- Commit: `38367c6d233f2aba86c988002835b4c6be914d22`
  (`feat: add ChatGPT Codex OAuth support`).
  Started from accepted Step 4 `716b654d` on the clean `dev` worktree at
  `/home/psiace/bubbuild/republic-dev`. Other checkouts were not modified.
- Added concrete `republic.auth.codex` helpers: `create_authorization`,
  `exchange_code`, `refresh_tokens`, `read_tokens`, `write_tokens`, immutable
  `CodexAuthorization`/`CodexTokens`, `CodexTokens.is_expired`, and `CodexAuthError`.
  Authlib owns S256 PKCE, authorization-code parsing and public-client token
  exchange/refresh. Local validation rejects missing/bad state, ambiguous or
  denied callbacks, malformed token/expiry values and invalid PKCE. No implicit
  login, callback server, browser, CLI, credential discovery or auth platform.
- Expiry fields are checked before Authlib can coerce them; there is no guessed
  lifetime. A JWT exp/account claim is only an unverified freshness/routing hint.
  Refresh returns new data, preserves a non-rotated refresh token and updates
  account hints. Repr hides secrets; mapped errors omit native token-bearing
  causes/diagnostics. Explicit file writes use mode 0600, fsync and atomic replace;
  parent directories must exist. Files are Republic's format, not a silent import
  of Codex/Bub auth.json. Failed refresh never falls back to stale credentials.
- Added `republic.providers.codex.OpenAICodex(tokens, client=None)`. Both generate
  and stream use one SSE POST to the fixed ChatGPT Codex Responses endpoint with
  bearer/account/originator headers. generate aggregates that stream; it never
  tries non-streaming first. The pinned official client uses streaming wire;
  whether all backend versions require it was not tested live. Shared Request,
  Response, part and event fields are unchanged; the Provider.generate docstring
  now permits this explicitly documented transport mode.
- Reuses the native Responses converter/parser, preserving item/call IDs,
  encrypted-only reasoning, JSON history round trips, tool results, usage and
  delta/done/terminal reconciliation. Leading system text becomes instructions in
  order; later system messages are rejected, including after empty history items.
  store=False/full history and encrypted reasoning include are enforced. No local
  tools, history repair, model routing or automatic continuation. A bounded native
  option set includes reasoning/text format; unsupported/managed options fail.
- Owned clients close with the provider; single streams close without closing
  borrowed clients. Borrowed SDK settings stay unchanged, retries are actually
  disabled, and their API-key endpoint/header/query/account routing is replaced
  only on Republic's private copy. Owned HTTP clients disable redirects; borrowed
  clients must already disable them. Expiry fails before HTTP, and 401/403 never
  refresh or replay inference. Caller cancellation remains CancelledError.
- Read Bub source/tests only, pinned at
  `357901db1a3f82d7f696024574225e595b09d4ac`; inspected official Codex source at
  `21eb35513df478a2a090bfc2c0293caaf435b36d` and the
  [official authentication documentation](https://learn.chatgpt.com/docs/auth).
  NOTICE records both sources, Apache-2.0 attribution and removed behaviors.
  Existing Vercel Responses provenance stays at
  `c788059dd1db2d93ae1c3da6daffb660eca07dbb`. No real credential files were read.
- Added only the direct runtime dependency `authlib>=1.6.5,<1.8`; locked 1.7.2 and
  its crypto dependencies. The upper bound retains the existing HTTPX integration
  rather than adopting Authlib 1.8's HTTPX2 transition. Existing dependency pins,
  Python 3.11 minimum and quality-check configuration remain intact.
- Verification (2026-09-28, local Linux):
  - `make check`: lock consistency, applicable prek hooks and ty passed. A test
    regex lint finding was corrected; no checks were disabled.
  - `make test`: **398 passed / 2 existing expected warnings**, preserving all
    302 prior cases and adding 96 Codex/OAuth cases. Python 3.11.15, Authlib 1.7.2,
    OpenAI 2.54.0, HTTPX 0.28.1, Pydantic 2.12.5. Fixtures exercise real clients,
    full authorization → save/read → inference → explicit refresh → inference,
    failure/denial/cancellation, expiry/rotation, headers/payload/request counts,
    secret-safe repr/errors/logging, native replay and stream/client cleanup.
  - `make docs-test`: strict MkDocs build passed; README, guides, navigation and
    the previously stale docs landing-page status now distinguish Steps 1-5.
  - `uv build --wheel`: passed. Clean installation in
    `/tmp/republic-step5-install.k68cl5di` on Python 3.11. Isolated imports resolve
    to site-packages, including auth/Codex public APIs; NOTICE and py.typed are
    packaged. All **96 new tests passed against the installed wheel** with
    Authlib **1.6.5**, OpenAI **2.16.0** and Pydantic **2.7.0**. No matrix rerun.
  - `git diff --check` and `git diff --cached --check`: passed.
- Tests ran with 90-second command timeouts in the previously authorized normal
  process, following Step 2's sandbox thread-wakeup evidence. No SDK/Authlib
  request method was mocked; only HTTP transports and owned-client transport
  construction are injected. No live login, token refresh or paid inference.
- **Code and offline Step 5 evidence complete; live account acceptance pending.**
  A real account still needs login → inference → refreshed inference, including
  entitlement/model/redirect acceptance. Fixtures do not establish that access.
  This does not block later independent steps. GitHub Copilot/Grok OAuth, device
  login, credential migration, hosted tools and Bub integration remain out of
  this increment; no subsequent step was started.


### Step 6 implementation

- Commit: `feat: add GitHub Copilot OAuth support` (the commit containing this
  evidence; resolve with `git log --grep='^feat: add GitHub Copilot OAuth support$'`).
  Started from accepted Step 5 `38367c6d233f2aba86c988002835b4c6be914d22` on clean
  `dev` at `/home/psiace/bubbuild/republic-dev`. Other checkouts remain untouched.
- Added `republic.auth.github_copilot`: `start_device_authorization`,
  `wait_for_token`, `exchange_copilot_token`, conditional `refresh_github_token`,
  `read_token`/`write_token`, immutable `DeviceAuthorization`, `GitHubToken`,
  `CopilotToken` and `CopilotAuthError`. Standard device/refresh token requests
  and OAuth parsing use async Authlib HTTPX with public-client authentication.
  GitHub-specific polling waits before the first attempt, honors increasing
  intervals/slow-down, stops on denial/expiry/unknown errors, and enforces caller
  and device deadlines, including requests in flight. Cancellation releases the
  auth client. No UI, subprocess, gh/environment/home discovery or profile calls.
- The source-derived default client ID is VS Code's `01ab8ac9400c4e429b23`, with
  minimal Copilot scope `user:email`; an explicit client ID is accepted without
  fallback. GitHub login does not prove Copilot entitlement. OAuth tokens may
  legitimately omit expiry/refresh data; there is no guessed lifetime or invented
  refresh grant. Current official docs support optional expiring device tokens,
  so refresh is available only when the server actually supplied a refresh token.
- Copilot access is a separate explicit GET to
  `https://api.github.com/copilot_internal/v2/token`, using the GitHub token.
  It returns an opaque inference token with actual expiry, optional advisory
  renewal time, and validated API origin. Renewal explicitly repeats this
  exchange; inference never does it automatically. Exactly four known HTTPS
  Copilot API origins are accepted; other domains/paths/ports/query/userinfo and
  all redirects are rejected. The default origin follows the inspected client.
- Added `republic.providers.github_copilot.GitHubCopilot(token, integration_id=...,
  client=None)`, reusing the existing Chat conversion/stream parser and official
  OpenAI SDK. One JSON `generate` or SSE `stream` POST goes to the validated
  Copilot origin plus `/chat/completions`. Actual model/tool data, interleaved
  argument fragments, tail usage, identity and finish metadata retain the prior
  single-call behavior. `max_output_tokens` maps to the source's `max_tokens`.
  Tools are never executed; no next turn, fallback or history repair is added.
- The sourced Chat subset supports text, function calls/results, temperature,
  top-p, stop and auto/none/named choice. Explicit parallel-tool setting,
  `required` tool choice, structured response format/reasoning effort, media,
  native reasoning/opaque fields and references are not supported. Such inputs
  or recognized unsupported outputs fail instead of being dropped. Copilot
  Responses/Messages, model routing and agent features remain outside this step.
- Client identity is Republic's own installed version, with an explicit caller
  integration ID; no first-party editor identity is supplied by default. Its
  server acceptance is unverified. Owned/borrowed clients disable actual SDK
  retries and redirects; borrowed configuration/lifetime remains unchanged.
  401/403/429, stream errors and disconnects never trigger exchange or replay.
  Fixed errors omit token-bearing SDK causes, server text/codes and request IDs.
  Repr hides credentials; explicit tagged JSON files use atomic 0600 writes.
- Only genuinely shared private helpers were extracted: explicit JSON file I/O
  and OpenAI OAuth client/error handling from Step 5, plus an HTTP test transport.
  Public Request/Response/part/event contracts did not change. No new dependency
  or lockfile change was needed; original Python 3.11+ and checks remain intact.
- Read historical Republic auth/client/tests at `216098ef` via git show. The old
  client targeted GitHub Models, not Copilot; it was not retained. Public sources
  pinned in [the guide](copilot-oauth.md) and NOTICE: MIT Microsoft VS Code
  `216fe2adc8e0f4436829e40004307e4098fcb478`, Copilot Chat
  `5863f5a7088958050792b5dccbe8b46c6e13eccc`, GitHub OAuth docs and RFC 8628.
  URL/header facts were also inspected in the referenced `@vscode/copilot-api`
  0.2.19 archive, whose custom restrictive license is recorded separately. None
  of that package's code was copied, executed, bundled or added as a dependency.
  Existing Vercel provenance stays at `c788059dd1db2d93ae1c3da6daffb660eca07dbb`.
- Verification (2026-09-28, local Linux):
  - `make check`: lock consistency, all applicable prek hooks and ty passed.
    Type/lint findings were corrected without disabling or relaxing checks.
  - `make test`: **530 passed / 2 existing expected warnings**, preserving all
    398 prior cases and adding 132 Copilot cases. Python 3.11.15, Authlib 1.7.2,
    OpenAI 2.54.0, HTTPX 0.28.1 and Pydantic 2.12.5. Real clients plus controlled
    HTTP/SSE/time fixtures cover the full explicit login → storage → exchange →
    inference → renewed token → inference sequence, payload/header/request counts,
    pending/slow-down/deadlines, errors, malformed input, persistence and cleanup.
  - `make docs-test`: strict MkDocs build passed. README, contract/landing pages,
    new Copilot guide, navigation and this plan distinguish implemented code from
    unverified service acceptance; the plan's previously stale top status is fixed.
  - `uv build --wheel`: passed. Installed into a new Python 3.11 environment at
    `/tmp/republic-step6-install.yjwZb8` with declared lower bounds Authlib
    **1.6.5**, OpenAI **2.16.0**, Pydantic **2.7.0** and HTTPX **0.28.1**. Isolated
    `python -I` imports resolve to site-packages; public auth/provider imports,
    packaged NOTICE and py.typed verified. All **228 affected Copilot/Codex tests**
    passed against the installed wheel. No unnecessary interpreter matrix rerun.
  - `git diff --check` and `git diff --cached --check`: passed.
- Tests ran in the already authorized normal process with 90-second timeouts,
  following the earlier restricted-sandbox SDK wakeup evidence. Authlib and SDK
  request methods were not mocked; transports, fake polling time and owned-client
  transport construction were controlled. No actual token file, .env, login or
  live inference was accessed.
- **Code and offline Step 6 evidence complete; account-backed acceptance pending.**
  This editor protocol is not established as a stable public third-party API.
  A real device login → entitled Copilot inference → explicitly renewed inference,
  accepted integration/editor identity, app/model availability and endpoint
  behavior remain unverified. Fixtures do not prove any account entitlement.
  Grok and Bub integration remain Steps 7-8; neither was started.

### Step 7 implementation

- Commit: `feat: add Grok OAuth support` (the commit containing this evidence;
  resolve with `git log --grep='^feat: add Grok OAuth support$'`). Started from
  accepted Step 6 `efde348eab25230b9da49a64da7609c2bf5450a6`, clean `dev` at
  `/home/psiace/bubbuild/republic-dev`. Other checkouts were not modified.
- Established concrete protocol evidence before adding the adapter. Official
  enterprise documentation separates `auth.x.ai` and the Grok Build proxy from
  `api.x.ai` API-key access. Public discovery confirms the device/token endpoints,
  device/refresh grants, public-client authentication and scopes. Official
  `xai-org/grok-build` revision `f0e3be1100ef5252488e3be8bb0e91cf68d8c305`, source
  revision `036a5d8348cd744767cd0b08518ab17bf608fa7f`, version **1.0.41**, supplies
  the concrete client ID, headers and Responses wire. Apache-2.0 / Copyright
  2023-2026 SpaceXAI provenance, exact paths and changes are in NOTICE and the
  [Grok guide](grok-oauth.md). No CLI was installed or executed.
- Added `republic.auth.grok`: immutable `GrokDeviceAuthorization`, `GrokTokens`,
  `GrokAuthError`, `start_device_authorization`, `wait_for_tokens`,
  `refresh_tokens`, `read_tokens` and `write_tokens`. The single supported login
  path is RFC 8628 device authorization. Authlib async HTTPX performs standard
  public-client grants; concrete polling honors the first interval, pending,
  slow-down, denial, expiry, caller deadlines and cancellation, including in-flight
  requests. Unknown/malformed responses stop. No invented expiry or refresh grant.
- Tokens retain actual expiry and refresh rotation. Unverified JWT principal
  claims are only refresh-routing hints; no identity/team/permission check uses
  them. There is no userinfo/profile call or ID-token authentication. Auth has
  no home/env/CLI credential discovery, browser/callback server, enterprise issuer
  abstraction or external broker. File helpers reuse existing private atomic-0600
  JSON operations at caller-selected paths; secrets are hidden from repr and
  fixed diagnostics omit server text/native causes.
- The sourced public client ID is `b1a00492-073a-47ea-816f-4c329264a828`.
  Republic requests only `offline_access grok-cli:access api:access`, uses its
  own identity/referrer and requires an explicit `client_version` (reference
  1.0.41). Reduced scopes and Republic identity are adapter choices, **not proven
  server acceptance**. Public source does not grant arbitrary third-party access.
- Added `republic.providers.grok.GrokOAuth(tokens, client_version=..., client=None)`.
  Both generate/stream issue one SSE POST to
  `https://cli-chat-proxy.grok.com/v1/responses`, using the sourced bearer/auth,
  version, headless-mode and model-override headers. Generate aggregates directly;
  no initial JSON attempt, fallback, retry, implicit refresh, 401 replay, tool
  execution, next model turn or history repair. Owned/borrowed client and stream
  cleanup follow the prior OAuth boundary; redirects are disabled or rejected.
- Reuses the existing Responses converter/parser, OAuth client/error helper and
  public Request/Response/part/event contracts. Ordered text/function input and
  output, separate item/call IDs, interleaved deltas, complete snapshots, reasoning
  summaries/text and encrypted-only items retain native metadata through JSON
  replay. Complete/incomplete/failed outcomes, length truncation and absent
  terminal behavior remain explicit. Raw usage retains cost/context details;
  normalized totals stay input plus output, without agent context-total rewriting.
- Supports the documented temperature/top-p/output-limit/tool-choice subset and
  native Responses text-format/reasoning options. Store is false, encrypted
  reasoning is included, and history is self-contained. Media, hosted tools,
  explicit parallel control, agent-control events, server-side history and
  managed-field/header overrides are not supported; no silent lossy conversion.
  Structured output remains caller-validated text with no repair call.
- No dependency, lockfile or public core contract changes. The only new shared
  extraction is the identical deterministic Clock test helper from Copilot.
  Python 3.11+ and the original quality tooling remain unchanged.
- Verification (2026-09-28, Linux):
  - `make check`: locked dependency resolution, all applicable prek hooks and ty
    passed. Lint/type findings were corrected without disabling checks.
  - `make test`: **647 passed / 2 existing expected warnings**, retaining all
    530 prior cases and adding 117 Grok cases. Python 3.11.15, Authlib 1.7.2,
    OpenAI 2.54.0, HTTPX 0.28.1, Pydantic 2.12.5. Real Authlib/OpenAI clients and
    controlled HTTP/SSE/time fixtures exercise device login → explicit persistence
    → inference → explicit refresh → refreshed inference, request counts, payloads,
    reasoning replay, usage, malformed/unsupported data, errors and cleanup.
  - `make docs-test`: strict MkDocs build passed. README, guide, contract/index,
    navigation, NOTICE and this plan record exact implementation/live boundaries.
  - `uv build --wheel`: passed. Clean install at
    `/tmp/republic-step7-install.wbdrDP`, Python 3.11.15, with declared lower bounds
    Authlib **1.6.5**, OpenAI **2.16.0**, Pydantic **2.7.0**, HTTPX **0.28.1**.
    Isolated `python -I` imports resolve to installed site-packages; public APIs,
    packaged NOTICE and py.typed verified. All **117 Grok tests passed** against
    that wheel. No unrelated interpreter/version matrix was repeated.
  - `git diff --check` and `git diff --cached --check`: passed.
- Tests used the already authorized normal process with 90-second command limits
  because of the earlier sandbox SDK thread-wakeup issue. SDK/Authlib request
  methods were not mocked; only HTTP transport, test time and owned-client
  transport construction were controlled. No real credential/token file, live
  login, refresh or inference was accessed.
- **Step 7 code and offline evidence checkpoint complete; live acceptance remains
  incomplete.** The minimum outstanding evidence is an authorized account/client
  accepting this device client/scope set and Republic/version headers, followed by
  permitted-model inference and refreshed inference. Fixtures prove neither
  entitlement nor protocol stability. Browser PKCE and custom enterprise issuer
  paths remain out of scope. Bub integration (Step 8) was not started.

### Step 8 implementation

- Bub commit: `a3c45120de4878f7167247c360721c5d628477a2` (`feat: integrate Republic provider SDK`), on
  `feat/republic-provider-sdk` in `/home/psiace/bubbuild/bub-republic-dev`, starting
  from clean `357901db1a3f82d7f696024574225e595b09d4ac`.
- Republic commit: `docs: document provider SDK integration and acceptance`
  (the commit containing this evidence; resolve by its exact subject). Started
  from accepted clean Step 7 `5dff4aa7bdbf6f81411f16d64f74307d4d83f167` on `dev`
  in `/home/psiace/bubbuild/republic-dev`. Original Republic/Bub checkouts and
  other worktrees were not changed; no push, publication or PR was performed.
- Read Bub's settings/onboarding/auth/Codex, runner, tools, hooks and tape/store
  implementation before fixing the migration boundary. Added explicit
  `model_backend="republic"` and a small `republic_protocols` map. Any-llm remains
  the default dependency/backend with its prior provider coverage. Supported Bub
  Republic selections are OpenAI Chat/Responses, OpenRouter Chat and Anthropic
  Messages with API keys and explicit base URLs. Unsupported combinations fail;
  no implicit any-llm fallback, OAuth discovery or onboarding rewrite was added.
- Bub builds Republic Request/Response/events directly. Its original ToolExecutor,
  hooks, model selection/fallback and agent loop remain caller behavior. Only
  complete, identified calls with object JSON execute. Incomplete/failed/unknown
  outcomes and malformed arguments execute no tools; terminal native output stays
  on tape with context disabled. Errors/denials preserve per-call error flags
  across hooks; Anthropic uses its bit, Chat/Responses use explicit result JSON.
- Native messages persist in a versioned protocol-tagged envelope alongside the
  existing tape view; tool-call/result entries carry full native messages. Real
  JSONL restart preserves Responses item IDs and encrypted-only reasoning,
  Anthropic signature fragments and original call IDs. Legacy text/function tape
  records still read. Protocol changes, lossy view edits, unsupported legacy fields
  and sending native history through any-llm fail explicitly. Run records retain
  actual response identity, finish reason and normalized/raw usage. Tape errors
  propagate; no transactional exactly-once promise was added for external tools.
- Owned/borrowed clients and stream cleanup retain the SDK's no-retry contract.
  Bub may explicitly try its configured next model only after a provider error
  before any native event. It never switches after partial output or input errors.
  Each candidate makes one HTTP request. Cancellation/consumer close bypasses
  terminal hooks as before and releases the response without executing tools.
- No Republic public contract, provider implementation, dependency or lock change
  was required. Bub adds no mandatory Republic dependency or absolute wheel path.
  The local artifact is built from clean Step 7: version
  `0.5.9.dev17+g5dff4aa7b`, SHA256
  `217d45d0c354b83916412296ac2efdeb018d3399117e30f158d223c4bca6e06c`.
  [The consumer guide](bub-integration.md) records explicit installation and
  `scripts/check_republic_wheel.py`, which builds a fresh Bub wheel, installs both
  non-editably under locked constraints and verifies actual package file contents.
- Verification (2026-09-28, local Linux):
  - Republic `make check`, `make test`, `make docs-test`: lock/hooks/ty and strict
    MkDocs pass; **647 passed / 2 existing expected warnings**, Python 3.11.15.
    No baseline test or safety check was removed or disabled.
  - Bub `make check`, `make test`, `make docs-test`: lock/hooks/mypy and Astro build
    pass; **558 passed / 1 skipped**, Python 3.12.13 with the existing trace extra
    installed. The sole skip requires unavailable PowerShell. There are **54 new
    integration cases**; the prior default-backend, hook, CLI, turn and tape tests
    remain intact. Website build uses pnpm and public snapshots without GitHub
    credentials. pnpm 11.13.1 required explicit local `approve-builds esbuild
    sharp workerd` for already locked dependencies; the generated approval file
    was removed after the successful build. No web toolchain change was committed.
  - Clean-environment installed-wheel acceptance: **54 integration cases passed**,
    including independent Python process pairs for Responses and Anthropic.
    First process: actual Bub runner -> model tool call -> ToolExecutor -> JSONL
    merge. Second process: new store/runner -> actual SDK request with retained
    opaque metadata -> final text. One HTTP per process and exactly one on-disk
    tool effect are asserted. Normal/failed/early-exit/cancel cleanup is checked.
    Final report: `/tmp/bub-republic-wheel-gu417qdl/report.json`; both packages
    are installed wheels, including committed Bub `0.4.5.dev16+ga3c45120d`.
    Responses process IDs were 2405850/2405859; Anthropic 2405868/2405875.
    Each pair records HTTP counts `[1, 1]` and one tool effect in its JSONL test
    directory. The report also records the exact Bub wheel SHA256.
  - The clean consumer uses Bub's actual lock: any-llm-sdk 1.22.1, OpenAI 2.31.0,
    Anthropic 0.94.0, Authlib 1.7.2, HTTPX 0.28.1 and Pydantic 2.12.5.
    `uv pip check` passes; both imports resolve to installed site-packages.
    No `--no-deps`, forced incompatible resolution or old PyPI Republic was used.
  - Existing SDK lower-bound checks from Steps 2-7 and Python 3.11 policy remain
    unchanged; no unrelated version matrix was mechanically repeated. Runtime
    source did not change in Republic. `git diff --check` and cached checks pass.
- Tests use real official SDK methods and synthetic HTTP/SSE transports, not live
  service calls. Normal processes with bounded test commands follow the already
  documented sandbox SDK thread-wakeup workaround. An initial uv project-cache
  stale build was detected by installed-wheel negative tests; explicit Bub wheel
  construction now prevents accepting that stale artifact.
- README, contract landing, navigation, [support matrix](support-matrix.md), Bub
  consumer guide and Bub's English/Chinese usage/settings pages distinguish
  implemented protocol/consumer behavior from account-backed acceptance.
- **Local implementation and offline consumer acceptance complete for this
  subset.** All live provider checks remain open. Codex/Copilot/Grok still require
  actual authorized login -> inference -> explicitly renewed inference. Grok's
  reduced scopes and Republic/version headers and Copilot integration entitlement
  remain unverified. Bub OAuth UX/credential migration is not implemented; original
  onboarding/Codex behavior remains on the default backend. Bub media input and
  arbitrary provider/history interchange are unsupported. No new agent/auth
  platform or release/publication decision is part of this completion.

Do not publish a package or replace `main` as part of preparing this baseline.
The repository was archived when inspected; local commits can proceed while
remote write/publication decisions remain separate.
