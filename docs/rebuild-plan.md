# Provider SDK rebuild plan

Status: Step 0 prepared; Steps 1-8 are planned, not implemented or validated.

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
- **Acceptance:** first prove login -> inference -> refreshed inference with the
  intended client/account. Add deterministic fixtures for the verified behavior,
  including failure responses. If account/client access remains unavailable,
  record the concrete limitation and leave this step incomplete.
- **Commit:** `feat: add Grok OAuth support` after the behavior is established.

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
and remaining limits. At the cleared baseline, no runtime tests or provider/auth
calls have passed because no implementation exists yet. Do not claim the plan
or a green generic test run proves live account access.

Do not publish a package or replace `main` as part of preparing this baseline.
The repository was archived when inspected; local commits can proceed while
remote write/publication decisions remain separate.
