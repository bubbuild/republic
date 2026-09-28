# ChatGPT / Codex OAuth

Step 5 adds concrete async Authlib helpers in `republic.auth.codex` and
`OpenAICodex` in `republic.providers.codex`. Install this checkout with
`pip install .` (or `uv sync` for development). Authlib is included; the supported
range is `>=1.6.5,<1.8`, using its HTTPX integration. Authlib 1.8's HTTPX2 transition
is outside this increment.

These are **offline-tested protocol adaptations**. A real account's login →
inference → refresh → inference path has not been validated. ChatGPT subscription
authentication and Platform API-key authentication are distinct access paths;
account/workspace entitlements still apply. See the
[official authentication guide](https://learn.chatgpt.com/docs/auth).

## Login belongs to the caller

```python
from pathlib import Path

from republic.auth.codex import create_authorization, exchange_code, write_tokens

# Inside an async function:
authorization = await create_authorization()
# Caller opens authorization.url and receives the full callback URL at
# http://localhost:1455/auth/callback. Republic starts no browser or server.
# callback_url = await your_callback_receiver(...)
# credentials = await exchange_code(authorization, callback_url)
# write_tokens(Path("/explicit/private/directory/republic-codex.json"), credentials)
```

`create_authorization()` does no network I/O. It returns immutable
`CodexAuthorization(url, state, code_verifier, redirect_uri)` data with a hidden
repr. S256, random state and the code exchange use Authlib; no client secret or
HTTP Basic authentication is sent. The public client ID is
`app_EMoamEEZ73f0CkXaXp7hrann`, with `openid profile email offline_access` scopes.
Connector scopes used by the larger Codex application are deliberately excluded.
Endpoints are fixed at `https://auth.openai.com/oauth/authorize` and `/oauth/token`.
An optional `redirect_uri` must be an HTTP localhost/127.0.0.1 URL without query or
fragment; actual redirect registration/acceptance remains the server's decision.

Keep the authorization object private and consume each pending login once. `exchange_code` accepts a **full callback URL** and validates its destination,
state, code, duplicate parameters, errors and denial; missing state fails.
For an existing manual code UX, `exchange_authorization_code(authorization, code)`
performs the same Authlib PKCE exchange; the caller owns callback/state handling. On
user cancellation, discard the pending object. Cancelling an exchange task
propagates `asyncio.CancelledError` and closes its response/client. No background
login task, callback server, device-code flow or CLI UX is included.

Each exchange/refresh creates and closes one `AsyncOAuth2Client`, with a positive
`timeout` (default 30 seconds), no redirects and no automatic retries. The optional
HTTPX `transport` argument is for explicit transport configuration/testing; once
an operation opens its client, it owns and closes that transport. A rejected
callback sends no request and does not take ownership of an unused transport.

## Explicit storage, expiry and refresh

```python
from republic.auth.codex import read_tokens, refresh_tokens, write_tokens

path = "/explicit/private/directory/republic-codex.json"
credentials = read_tokens(path)
# Inside an async function, at a point chosen by the caller:
if credentials.is_expired(leeway=120):
    credentials = await refresh_tokens(credentials)
    write_tokens(path, credentials)
```

`CodexTokens(access_token, refresh_token=None, expires_at=None, account_id=None)` is plain,
immutable data. `expires_at`, when supplied, is a positive, finite Unix timestamp. Token fields
are nonempty strings without whitespace/control characters; bearer access tokens
must be ASCII. Repr hides credentials/account IDs. Accessing fields or serializing
them deliberately reveals their values: do not put them into messages or logs.

Token responses accept numeric `expires_at` or `expires_in`; every supplied
expiry field must be valid (booleans, strings, nonfinite and nonpositive values
fail). If both are absent, an access JWT's numeric `exp` may supply a freshness
hint. If no expiry is available, it stays `None`; there is no guessed lifetime. `is_expired(leeway=0)` is only a local check,
not token validation. JWT signatures/issuer/audience are **not verified** here;
claims are never proof of identity, permission or entitlement.

Refresh is one explicit exchange. It returns new tokens, preserves a refresh
token when the server omits rotation, and rejects an explicitly empty replacement.
It uses a new access token's account claim first, then an ID token claim, then the
previous account ID. ID tokens themselves are not persisted. An opaque access token is usable without refresh or expiry data. An optional
account ID adds `chatgpt-account-id`; the service decides whether a particular
credential requires that routing hint. `OpenAICodex(access_token, account_id=...)`
also accepts a string directly.

`read_tokens(path)` and `write_tokens(path, tokens)` accept only an explicit path.
They use Republic's small JSON format, not a Codex/Bub credential-file importer.
No home-directory/environment discovery occurs. The parent directory must exist;
use a private directory. Writes create a 0600 temporary file in the same directory,
flush/fsync and atomically replace the destination. Failure before replacement
preserves the old file and attempts temporary-file cleanup. No vault, locking,
concurrent refresh coordination or power-loss durability guarantee is provided.

`CodexAuthError.code` distinguishes `state_mismatch`, `denied`, `callback_error`,
`missing_code`, `invalid_callback`, `invalid_pkce`/`invalid_authorization`,
`no_refresh_token`, malformed token/expiry, exchange/refresh rejection
or transport failure, and credential read/write failure. Exceptions contain fixed
diagnostics, not callback URLs, token responses or secret-bearing native causes.
A refresh failure is returned to the caller, which decides whether to keep using
a still-valid old token. Providers never reject tokens based on local expiry.

## One model operation

```python
from republic import Message, Request, TextPart, generate, stream
from republic.auth.codex import read_tokens
from republic.providers.codex import OpenAICodex

credentials = read_tokens("/explicit/private/directory/republic-codex.json")
request = Request(
    model="your-codex-model",  # Caller selects a model available to its account.
    messages=[
        Message(role="system", parts=[TextPart(text="Answer concisely.")]),
        Message(role="user", parts=[TextPart(text="Hello")]),
    ],
)
# Inside an async function:
async with OpenAICodex(credentials) as provider:
    response = await generate(provider, request)
    # This is a separate, explicitly requested inference:
    async with stream(provider, request) as output:
        async for event in output:
            pass
        streamed_response = output.response
```

Both operations use the SSE endpoint (one POST with the default client policy) at
`https://chatgpt.com/backend-api/codex/responses`, with bearer authorization,
optional `chatgpt-account-id` and `originator: republic`. Constructor `base_url`
and `headers` override routing defaults; request `extra_headers` has final priority. No `OpenAI-Beta: responses=experimental` is added: the
pinned current official HTTP path no longer uses that older Bub header.

The adapter follows the official Codex HTTP client's `stream=True` wire mode.
`generate` collects that one SSE stream into a normal `Response`; it never tries
non-streaming first or performs a fallback. Source inspection establishes the
official client's streaming shape, not a live proof that every backend version
rejects non-streaming. Ordinary `OpenAIResponses.generate` stays non-streaming.

`store=False` and `include=["reasoning.encrypted_content"]` are overridable defaults.
Supply complete history yourself. Leading system text parts become `instructions`,
in original order separated by two newlines; with none, instructions is empty.
Later system messages are rejected, never hoisted. The remaining items keep their
order, native IDs, function-call IDs, raw arguments and encrypted reasoning.
Serialization/replay uses the same `openai.raw_item` metadata as the
[Responses adapter](openai-responses.md). No tool runs locally and no reasoning,
tool result or malformed JSON triggers a repair/continuation request.

| Input | Supported behavior |
| --- | --- |
| Messages/parts | Leading system text, user/assistant text, native assistant reasoning/function calls, tool results; same part validation as Responses. |
| Tools | Function schemas, optional strict metadata, auto/none/required/named choice, parallel calls. Default: empty tools, auto choice, parallel enabled. |
| Native `provider_options` | `reasoning`, `text` (including structured `format`), `service_tier`, `prompt_cache_key`, positive `timeout`. Native values remain model/endpoint-dependent. |
| Native extensions | `store`, `include`, `truncation`, `previous_response_id`, `instructions`, `extra_headers`, `extra_body`; shared Responses options apply. |
| Rejected options | Temperature, top-p, output-token limit and stop sequences in common fields; conflicting managed fields. |
| Media input | Native user images via URL/base64/file_id; see [media inputs](media-inputs.md). |
| Outside this adapter | PDF/audio/video and generated media, hosted/custom tools, agents, model catalogs/routing, WebSocket sessions and background polling. |

The adapter rejects options outside its supported Codex subset, even if a general
Responses endpoint accepts them. It does not silently discard caller parameters.
Structured-output validation is an explicit caller operation, as in the Responses
guide; no repair inference is sent on schema mismatch.

## Ownership and failure behavior

By default the provider owns its client; `aclose()` or its async context closes
it. To borrow a reusable SDK client:

```python
import httpx
from openai import AsyncOpenAI

# Inside an async function:
async with httpx.AsyncClient(follow_redirects=False) as http:
    async with AsyncOpenAI(api_key="unused-by-codex", base_url="https://chatgpt.com/backend-api/codex",
                           max_retries=0, http_client=http) as client:
        async with OpenAICodex(credentials, client=client) as provider:
            response = await generate(provider, request)
```

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

401/403 responses surface sanitized provider errors; there is no automatic
refresh or re-authentication. Account entitlement still requires live evidence.

Delta/done/terminal snapshots use the existing Responses reconciliation rules,
so overlapping text/arguments/reasoning are not duplicated. Incomplete/length
retains partial output; failed responses retain output with finish reason `error`.
Error diagnostics are replaced by a fixed `response_failed` entry. SSE errors
raise `ProviderError`, and missing terminals raise `IncompleteStreamError`.
For this OAuth adapter, native SDK causes, raw server error messages/codes and
request IDs are deliberately omitted to prevent credentials echoed by a server
from leaking through exceptions. The API-key adapters retain their earlier
native-cause behavior.

## Evidence and sources

The tests use real Authlib and official OpenAI SDK clients with controlled HTTP
and SSE transports, including the full explicit login/storage/inference/refresh
sequence. No real credential file, login or live inference was used. Real-account
acceptance remains outstanding and does not block the independent next steps.

Protocol sources inspected on 2026-09-28:

- [Bub auth](https://github.com/bubbuild/bub/blob/357901db1a3f82d7f696024574225e595b09d4ac/src/bub/builtin/auth.py),
  its Codex provider and tests at that revision.
- [Official Codex login](https://github.com/openai/codex/blob/21eb35513df478a2a090bfc2c0293caaf435b36d/codex-rs/login/src/server.rs),
  OAuth public-client exchange, account routing and Responses request source at
  that revision. Its automatic recovery, agent state, connectors and UX are excluded.
- The installed Authlib HTTPX source (PKCE, state, public client auth, compliance
  hooks and expiry coercion) and official OpenAI SDK copy/stream/error code.

[NOTICE](https://github.com/bubbuild/republic/blob/dev/NOTICE) records revisions,
licenses and the modified subset. OAuth uses Authlib; the unverified JWT helper
only decodes routing/freshness hints and is not an authentication verifier.
