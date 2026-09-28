# Implemented protocols and acceptance limits

This matrix describes the unreleased `dev` implementation, not the previously
published Republic API. Python 3.11+ remains the SDK requirement; the Bub consumer
requires Python 3.12+. All service evidence below is **offline protocol fixtures**
through real official clients. No provider or OAuth account has live acceptance.

## Provider entry points

| Import / class | Wire and credentials | Implemented subset | Bub selection |
| --- | --- | --- | --- |
| `republic.providers.openai.OpenAIChatCompletions` | API key; `/chat/completions`; explicit compatible base URL | Text, user images, plain reasoning extensions, function tools/results, common/native options | `openai:`, `openrouter:` with `chat`; text input only in Bub |
| `republic.providers.openai.OpenAIResponses` | API key; `/responses`; full inline history by default | Text, native reasoning/summary/encrypted items, functions, structured output wire schema | `openai:` with explicit `responses` |
| `republic.providers.anthropic.AnthropicMessages` | API key; `/v1/messages`; explicit required output limit | Text, signed/redacted thinking, functions/error results, cache controls/usage | `anthropic:` with `messages` |
| `republic.providers.codex.OpenAICodex` | Existing ChatGPT access token or token data; Codex Responses SSE endpoint | Sourced Codex text/reasoning/function subset; generate aggregates one SSE request | `openai:` with `codex`, or saved Bub Codex login without API-key/base/protocol overrides |
| `republic.providers.github_copilot.GitHubCopilot` | Existing Copilot inference token and integration ID; service endpoint default or explicit caller route | Sourced editor Chat text/function subset; separate GitHub login and inference tokens | Not connected |
| `republic.providers.grok.GrokOAuth` | Existing Grok access token or token data/version; Grok Build OAuth Responses proxy | Sourced text/native reasoning/function subset; generate aggregates one SSE request | Not connected |

An OpenAI-compatible Chat endpoint is not evidence of Responses compatibility.
GitHub Models is not GitHub Copilot. `api.x.ai` API-key access is not Grok Build
OAuth proxy access. The SDK provides no model registry, account entitlement discovery or automatic
protocol selection. Bub owns its explicit protocol map and saved-Codex-login selection.

See each guide for exact allowed options and source revisions:
[Chat](openai-chat.md), [Responses](openai-responses.md),
[Messages](anthropic-messages.md), [Codex](codex-oauth.md),
[Copilot](copilot-oauth.md), [Grok](grok-oauth.md).

## Independent model operations

`republic.providers.openai_embeddings.OpenAIEmbeddings` provides async `embed`
through the official OpenAI `/embeddings` client, including explicit compatible
base URLs. Text batches, dimensions, ordered original indexes, float/base64
vectors, usage and native metadata are covered by offline HTTP fixtures. No
implicit batching, retrieval or agent facade is included. See [embeddings](embeddings.md)
and the [capability restoration inventory](capability-restoration.md). Reranking
and message media restoration are the next increments in that inventory.

## Authentication helpers

| Module | Explicit login path | Renewal | Remaining acceptance |
| --- | --- | --- | --- |
| `republic.auth.codex` | Authlib S256 PKCE authorization URL/state and callback exchange | Explicit refresh grant; returned immutable tokens; rebuild provider | Real account login → inference → refreshed inference; accepted client/scope/account routing |
| `republic.auth.github_copilot` | Authlib GitHub device authorization and bounded polling | Explicit GitHub-to-Copilot exchange/renewal; optional GitHub refresh only when a refresh token exists | Device login, Copilot entitlement, accepted integration identity/endpoints/models and renewed inference |
| `republic.auth.grok` | Authlib Grok device authorization and bounded polling | Explicit refresh grant; rebuild provider | Reduced scope set, Republic/version headers, client/account entitlement, login → inference → refreshed inference |

Helpers accept explicit file paths and use atomic restricted-permission writes.
They do not scan home directories, environment credential files or CLI profiles;
there is no browser, callback server, terminal UI or credential manager in Republic.
Bub supplies Codex browser/callback/manual URL-or-code UX, original `auth.json`
compatibility and its own pre-call refresh/fallback policy; Copilot/Grok UX remains unconnected. Inference
never initiates login, refreshes on 401 or replays the model request. Login/token
success alone cannot establish inference permission. Copilot's editor protocol
and Grok's client integration are not stable public third-party API promises.

## Shared contract and deliberate limits

- `generate(provider, request)` and `stream(provider, request)` each perform one
  model operation. owned clients default to zero retries; borrowed and explicitly configured retry/redirect
  settings are retained. See [caller configuration](client-configuration.md).
  The SDK does not execute tools, append rounds, repair history or choose models.
- Persist the whole `Message`/`Response`, including part metadata. Responses item
  IDs differ from tool call IDs; encrypted reasoning and Anthropic signatures are
  opaque data. Native histories are not freely interchangeable between providers.
- A terminal result is not permission to execute tools. Inspect finish reason,
  native status and raw arguments. Truncation keeps partial output; missing
  termination raises `IncompleteStreamError`; cancellation propagates and closes
  the response. Always close streams when leaving iteration early.
- Usage input includes cached input, and cache/reasoning values are breakdowns.
  Missing values stay unknown. Anthropic inclusive input is uncached input plus
  cache-read plus cache-created input, when all required counts are present;
  cumulative start/delta snapshots are not added together. Raw usage is retained.
- Structured output configuration forwards the supported native schema. Output
  remains text; the caller explicitly validates it with Pydantic. Validation
  failure does not trigger repair inference. See the Responses guide's example.
- Chat has a bounded user-image subset. Responses, Messages and OAuth adapters
  currently reject media. Hosted tools, agents, MCP, replay/approval state, model
  catalogs, server-side conversation recovery and background polling are outside the implementation. Protocol guides also list rejected
  reasoning forms, role combinations and managed-field overrides.

## Evidence levels

The [rebuild plan](rebuild-plan.md) records per-step source references, SDK versions,
unit/HTTP/SSE evidence and lower-bound installation results. Step 8 additionally
validates [actual Bub consumption](bub-integration.md): installed wheels, real
runner/tool/tape processing and native history restored by new Python processes.
This establishes the documented local consumer contract. It does not close live
service, OAuth entitlement or endpoint/model availability acceptance.
