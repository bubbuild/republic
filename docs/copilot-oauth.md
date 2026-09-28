# GitHub Copilot OAuth

Step 6 provides explicit async device login and token exchange in
`republic.auth.github_copilot`, and `GitHubCopilot` in
`republic.providers.github_copilot`. Install this checkout with `pip install .`
or `uv sync`. It reuses the existing Authlib HTTPX integration and official
OpenAI async client; no additional runtime dependency or agent SDK is needed.

**This is an offline-tested adaptation of the Copilot editor protocol, not a
public, stable third-party inference API guarantee.** No real account device
login → Copilot inference → renewed inference has been tested. A GitHub login,
a Copilot subscription, and acceptance of a particular integration/model are
separate facts. The adapter requires an explicit, service-recognized
`integration_id`. There is no live-validated Republic integration identity or
promise that the service accepts Republic's own editor headers.

## Service path and source evidence

The legacy Republic `github-copilot` client at `216098ef` used
`https://models.github.ai/inference`. That was **GitHub Models**, not the Copilot
editor inference path. GitHub's [Models SDK](https://github.com/github/models-ai-sdk)
identifies that URL separately; GitHub [retired Models on July 30, 2026](https://github.blog/changelog/2026-07-30-github-models-is-now-retired/).
This adapter never uses that URL or falls back to it after an entitlement failure.

The implemented path is:

| Explicit operation | Endpoint and credential |
| --- | --- |
| `start_device_authorization()` | POST `https://github.com/login/device/code`; public client ID and `user:email`. |
| `wait_for_token(device, timeout=...)` | POST `https://github.com/login/oauth/access_token`; standard device grant, no client secret or HTTP Basic auth. |
| `exchange_copilot_token(github_token)` | GET `https://api.github.com/copilot_internal/v2/token`; `Authorization: token <GitHub token>`. |
| `GitHubCopilot.generate` / `.stream` | POST `<validated Copilot API origin>/chat/completions`; `Authorization: Bearer <Copilot token>`. |

Protocol references inspected on **2026-09-28**:

- GitHub's [device/OAuth documentation](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps)
  and [RFC 8628](https://www.rfc-editor.org/rfc/rfc8628.html) define token requests,
  polling and the conditional refresh grant.
- Microsoft VS Code [config.ts](https://github.com/microsoft/vscode/blob/216fe2adc8e0f4436829e40004307e4098fcb478/extensions/github-authentication/src/config.ts)
  and `flows.ts` at that revision supply the public GitHub application ID
  `01ab8ac9400c4e429b23` and its device flow. Both are MIT licensed.
- Microsoft Copilot Chat at
  [5863f5a7088958050792b5dccbe8b46c6e13eccc](https://github.com/microsoft/vscode-copilot-chat/tree/5863f5a7088958050792b5dccbe8b46c6e13eccc),
  MIT licensed: `authentication/common/authentication.ts` and
  `authentication/vscode-node/session.ts` establish the minimal `user:email`
  scope through VS Code GitHub authentication; `authentication/node/copilotTokenManager.ts`
  establishes the separate exchange, with `X-GitHub-Api-Version: 2025-04-01`;
  `authentication/common/copilotToken.ts` defines expiry and endpoint fields.
  `networking/common/fetch.ts`, `networking.ts`, and `endpoint/node/chatEndpoint.ts`
  define the Chat request, JSON/SSE handling and option limitations. Paths are
  under `src/platform`; `src/extension/prompt/node/chatMLFetcher.ts` supplies
  `max_tokens`. Copilot's native reasoning fields differ from generic Chat.
- That source pins `@vscode/copilot-api` **0.2.19**. Its
  [published archive](https://registry.npmjs.org/@vscode/copilot-api/-/copilot-api-0.2.19.tgz)
  (SHA-256 `8b96d8da5349db6d2ebfe37c01a6f721c2d48f5e9be31c79c02af41a0f0294dc`)
  confirms token/inference URL construction and integration/editor headers,
  including inference API version `2025-10-01`. This package has **custom,
  restricted terms**, not the surrounding repository's MIT license. It was
  inspected only for wire facts: no package code was copied, run, bundled or
  added as a dependency. Its first-party integration selection is not reproduced.

These sources establish a concrete editor protocol; they do not establish a
supported third-party product contract. The default client ID is source-derived,
not a Republic-owned registration. `client_id=` can select the caller's explicitly
registered device-enabled OAuth app, but Copilot's acceptance of that app is also
unverified. No alternative app IDs or broader scopes are tried automatically.
[NOTICE](https://github.com/bubbuild/republic/blob/dev/NOTICE) records provenance
and applicable notices. The reusable Chat converter remains based on Vercel
AI Python revision `c788059dd1db2d93ae1c3da6daffb660eca07dbb`.

## Caller-owned device login

```python
from republic.auth.github_copilot import (
    start_device_authorization, wait_for_token, write_token,
)

# Inside an async function:
device = await start_device_authorization()
# Caller displays device.verification_uri and device.user_code, opens a browser
# if desired, and offers cancellation. Republic starts no UI or subprocess.
github_token = await wait_for_token(device, timeout=600)
write_token("/explicit/private/directory/github.json", github_token)
```

`DeviceAuthorization` contains private `device_code`/`user_code`, the exact
GitHub.com verification URL, `expires_at`, `interval` and `client_id`. The first
poll waits the advertised interval (five seconds only when the server omits the
optional interval). Pending responses retain the interval; `slow_down` raises it
by at least five seconds, respecting a larger server value. Intervals never
shrink. A monotonic deadline enforces both the required caller `timeout` and the
device's declared lifetime, including requests in flight. There is no busy loop.

Only `authorization_pending` and `slow_down` continue polling. Denial, device
expiry, an unknown OAuth error, missing/malformed token data, HTTP failure or
connection failure stops the operation. HTTP failures are not retried as if they
were pending login. Cancellation propagates `asyncio.CancelledError`; discard the
pending object. `CopilotAuthError.code` distinguishes `access_denied`,
`expired_token`, `deadline_exceeded`, `oauth_error`, invalid response/token/expiry,
HTTP status categories and `transport_error`. Untrusted server descriptions and
native secret-bearing causes are omitted.

Each auth operation owns and closes its HTTP client. `transport=` permits an
explicit HTTPX transport, owned once the operation opens its client; rejected
input before that does not take ownership. `timeout` defaults to 30 seconds for
individual auth calls; the waiter uses `request_timeout=30` plus its overall
required budget. Redirects and automatic HTTP retries are disabled. No environment,
home, `gh` credentials, profile endpoint or existing auth file is consulted.

## Two token types and explicit renewal

```python
from republic.auth.github_copilot import GitHubToken, read_token, exchange_copilot_token

stored = read_token("/explicit/private/directory/github.json")
if not isinstance(stored, GitHubToken):
    raise TypeError("This path must contain a GitHub login token")
# Inside an async function:
copilot_token = await exchange_copilot_token(stored)
# Later, when the caller chooses to renew, explicitly exchange again:
# renewed = await exchange_copilot_token(stored)
```

`GitHubToken(access_token, expires_at=None, scope=None, client_id=...,`
`refresh_token=None, refresh_expires_at=None)` is immutable login data. Tokens
without a declared expiry are valid data; `is_expired()` returns false for an
unknown expiry, not proof that the token is still accepted. No lifetime is guessed.
All supplied expiry fields must be numeric, finite and positive; strings and
booleans are rejected before Authlib can coerce them.

Current GitHub documentation also supports expiring device-flow tokens. If GitHub
actually supplies a refresh token, `refresh_github_token(token)` explicitly makes
one standard Authlib refresh request with public-client authentication. New tokens
replace old data; an omitted refresh token retains the previous value. No refresh
grant is attempted for a legacy token without one, and no `offline_access` scope
is implicitly added. This optional **GitHub OAuth refresh** is different from
**Copilot inference-token renewal**, which re-runs `exchange_copilot_token`.

`CopilotToken(token, expires_at, api_endpoint, refresh_at=None)` contains an opaque
inference credential. The exchange preserves the server's actual `expires_at`;
`refresh_in`, when present, becomes an advisory `refresh_at` timestamp. It never
replaces actual expiry with a guessed renewal buffer. No profile, account record,
SKU or quota manager is built. A 403 at exchange is not successful Copilot access.

Only these exact HTTPS origins are allowed, as used by the inspected client and
listed in GitHub's [Copilot network allowlist](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-environment):
`api.githubcopilot.com`, `api.individual.githubcopilot.com`,
`api.business.githubcopilot.com`, `api.enterprise.githubcopilot.com`.
The documented client default `https://api.githubcopilot.com` is used only when
no API endpoint is supplied. Unexpected domains, paths, ports, queries, userinfo,
non-HTTPS URLs and redirects fail. Enterprise Server/custom-host configurations
are outside this increment; an unfamiliar service endpoint requires explicit code
review, not token forwarding.

`read_token(path)`/`write_token(path, token)` use a tagged Republic JSON file for
either token type, only at the exact caller path. The parent directory must exist.
Writes use a 0600 temporary file, flush/fsync and atomic replacement; pre-replacement
failure preserves the original and cleans the temporary file. No vault, locking,
automatic saving, file migration or default-path discovery is included. Credential
fields are hidden in repr, but accessing fields or writing JSON deliberately reveals
them; keep the explicit directory private. Files never contain SDK clients.

## One inference operation

```python
from republic import Message, Request, TextPart, generate, stream
from republic.providers.github_copilot import GitHubCopilot

request = Request(
    model="your-copilot-chat-model",  # Must support Chat Completions for the account.
    messages=[Message(role="user", parts=[TextPart(text="Hello")])],
)
# Inside an async function; integration_id must come from your service integration:
async with GitHubCopilot(copilot_token, integration_id=integration_id) as provider:
    response = await generate(provider, request)
    # A second inference only because the caller explicitly requests it:
    async with stream(provider, request) as output:
        async for event in output:
            pass
        streamed_response = output.response
```

`generate` makes one non-streaming JSON request; `stream` makes one SSE request
with `stream_options.include_usage=True`. Neither tries another transport on
failure. Models available only via Responses/Messages are outside this adapter.
There is no model catalog, routing, tool execution, automatic next turn or history
repair. Parallel tool *outputs* are supported without executing them locally.

| Input | Supported subset |
| --- | --- |
| Messages | Ordered system/user/assistant text and assistant function calls; explicit tool results with matching call IDs. Later system messages stay in place. |
| Tools | Function JSON schemas, description, optional strict metadata; auto/none/named tool choice. Raw model arguments may be malformed JSON and remain unchanged. |
| Common options | Temperature, top-p, stop sequences, `max_output_tokens` mapped to Copilot's `max_tokens`. No guessed token limit. |
| Native options | Shared Chat options, including parallel control, native response format, `store`, `extra_headers` and `extra_body`. Service/model acceptance is caller-verified. |
| Rejected options | `tool_choice="required"` (explicitly unsupported in the pinned Chat definition), overrides of managed model/messages/tools/stream/common fields, and unrecognized SDK keyword arguments (use `extra_body` for native extensions). |
| Output | Text, function-call data, identity/model, finish reason, raw/normalized usage, cached/reasoning token counts, and the existing Chat response-record metadata. |
| Outside this subset | Media, native reasoning/opaque reasoning, citations/references, hosted tools, legacy functions, Copilot agents, CLI, Responses/Messages and model-specific automatic history transformations. |

Native reasoning/reference output fails visibly instead of silently dropping
opaque data needed for the next turn. The common Chat converter's `openai`
metadata namespace is retained for response records such as raw unknown finish
reasons and system fingerprint; it denotes the wire schema, not API-key access.
Unknown finish reasons map to `other`. `length` retains truncated output and does
not send a continuation. Interleaved indexed tool fragments wait for actual ID/name
fields, retain raw arguments, and consume trailing usage before `StreamEnd`.
Missing completion evidence raises `IncompleteStreamError` with partial stream data.

Headers identify the client as `republic/<installed version>` in Editor-Version
and Editor-Plugin-Version, with User-Agent `republic`, the caller's
Copilot-Integration-Id, OpenAI-Intent `conversation-panel`, and the sourced API
version. Republic does not default to a first-party editor's integration ID,
machine/session IDs or telemetry. Acceptance of this identity remains unverified.

## Client lifetime and failures

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

A raw access string here must already be a Copilot inference token; ordinary
GitHub OAuth login is not interchangeable. Expiry can be unknown and is not
checked by the provider. Server-issued endpoints from token exchange remain
validated against known Copilot origins; an explicit caller `base_url` is a
separate, trusted configuration decision.

`ProviderError` identifies `github-copilot`, HTTP status and fixed codes such as
`unauthorized`, `forbidden`, `rate_limit`, `request_failed`, `invalid_response`
or `stream_error`. Server diagnostics, raw SDK error objects, request IDs and
native causes are omitted for this OAuth path to avoid echoed credential leaks.
A 401/403/429 sends no retry, refresh, token exchange or fallback model request.

## Verification limits

Tests use real Authlib and OpenAI clients with HTTPX MockTransport, SSE fixtures
and controlled polling time. They assert actual URLs, headers, form/JSON bodies,
request counts, pending/slow-down timing, denial/expiry/deadline/cancellation,
secret-safe diagnostics, explicit token files, exchange/renewal, usage, tool
history, stream termination and resource ownership. No real credential file,
login or live inference was used.

Still required for account-backed acceptance: device login with an accepted app,
Copilot entitlement, acceptance of the supplied integration/editor identity,
available Chat model/option behavior, inference, explicit token renewal, then
another inference. These are unverified service facts, not fixture outcomes.
Grok and Bub integration remain later steps.
