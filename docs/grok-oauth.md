# Grok Build OAuth

Step 7 implements a concrete device login, explicit token refresh and a single-call
Responses adapter for the Grok Build OAuth proxy. Install this checkout with
`pip install .` or `uv sync`; the existing Authlib, HTTPX and OpenAI dependencies
are sufficient. Python 3.11+ remains supported.

This is a source-backed protocol adaptation with synthetic HTTP/SSE evidence.
No real account login, entitlement, inference or refreshed inference has been
validated. Public source does not establish that arbitrary third-party clients
or accounts may use this proxy. Successful token issuance alone is insufficient.
`api.x.ai` API-key access is a separate service path and is not a fallback.

## Device login owned by the caller

```python
from collections.abc import Awaitable, Callable
from pathlib import Path

from republic.auth.grok import (
    GrokTokens,
    start_device_authorization,
    wait_for_tokens,
    write_tokens,
)


async def login(
    path: Path,
    display: Callable[[str, str], Awaitable[None]],
) -> GrokTokens:
    pending = await start_device_authorization(client_version="1.0.41")
    # Caller UI shows this first-party URL and code; Republic opens no browser.
    await display(pending.verification_uri, pending.user_code)
    tokens = await wait_for_tokens(pending, timeout=300)
    write_tokens(path, tokens)
    return tokens
```

`client_version` is required: it is the Grok Build protocol version selected by
the caller, not Republic's package version. `1.0.41` is the inspected source
version. It is sent in `x-grok-client-version`; the proxy uses version gating.
There is no automatic latest-version lookup or CLI installation. Acceptance of
this version with Republic's identity remains unverified.

The fixed public OAuth client ID is `b1a00492-073a-47ea-816f-4c329264a828`, observed
in the official source. Authlib uses token endpoint authentication `none`, with
the client ID in the form body and no client secret. The endpoints are:

| Operation | Endpoint |
| --- | --- |
| Start device authorization | `POST https://auth.x.ai/oauth2/device/code` |
| Poll device grant / refresh grant | `POST https://auth.x.ai/oauth2/token` |
| One inference request | `POST https://cli-chat-proxy.grok.com/v1/responses` |

Republic requests `offline_access grok-cli:access api:access`. All three scopes
are in discovery and the official client's scope set; source explicitly connects
`grok-cli:access` to proxy access. Omitting the client's identity, conversation
and workspace scopes is a Republic design choice. Server acceptance of this
reduced scope set is **not proven**. The device request uses `referrer=republic`
and the source-defined `headless` client surface. No profile/userinfo request is
made, and no identity authentication is claimed.

The device response must contain codes, a valid first-party HTTPS verification
URL, and a positive `expires_in`. `interval` defaults to RFC 8628's five seconds
only when omitted. Polling waits before the first request, honors increasing
server intervals, and increases the interval by at least five seconds after
`slow_down`. It stops at the earlier of device expiry and the caller's required
timeout, including requests in flight. It does not invent a minimum device
lifetime. Denial, expiry, unknown errors, missing tokens and transport failures
stop polling. Cancelling the task closes its Authlib client and propagates
`CancelledError`.

Each auth helper owns and closes its HTTPX client and any explicitly supplied
`transport`. The auth client disables redirects and adds no retry loop; an injected transport
retains its caller-selected retry behavior. Polling is the only
deliberate repeated auth operation; it does not replay inference. Verification
URLs accept only `auth.x.ai` or `accounts.x.ai`, without credentials, ports or
fragments. Browser PKCE/callback/state handling, custom enterprise issuers,
external token brokers and login UX are outside this implementation. Device
authorization has no browser redirect URI or callback state to validate.

## Tokens, files and explicit refresh

`GrokTokens` is immutable data: access token, optional refresh token, optional
`expires_at`, and optional principal routing hints. `is_expired()` checks a known
expiry locally; `False` with an unknown expiry does not prove validity. Numeric
expiry must be finite and positive. Missing expiry remains `None`; neither a
guessed lifetime nor an unverified JWT `exp` is used.

`read_tokens(path)` / `write_tokens(path, tokens)` access only the explicit file.
The parent directory must exist. Writes use a mode-0600 temporary file and atomic
replacement; a failed replacement leaves the previous file intact. This is a
small Republic JSON format, not a Grok CLI credential-file importer. There is no
home/environment scan, auth.json migration, subprocess or credential manager.

```python
from pathlib import Path
from republic.auth.grok import read_tokens, refresh_tokens, write_tokens


async def renew(path: Path) -> None:
    updated = await refresh_tokens(read_tokens(path))
    write_tokens(path, updated)
```

`refresh_tokens` performs one explicit Authlib refresh grant. A missing refresh
token fails before HTTP. Omission of a rotated refresh token preserves the old
one; a returned token replaces it. Expiry is taken only from the new response.
The access token's unverified `principal_type`/`principal_id` (also camelCase)
claims, when present, are retained as refresh routing hints. Fresh hints replace
old hints; absent hints retain the previous pair. They are never used to validate
identity, team membership, permissions or account entitlement. ID tokens and
profile data are not persisted or used for authentication decisions.

Credential and device-code fields are hidden from repr. `GrokAuthError` exposes a
fixed `code` and optional HTTP status without server text or native exception
causes. HTTP 401/403/429 remain distinguishable. A refusal or revoked grant does
not start login automatically. Applications must also avoid logging token data,
authorization headers or their explicit JSON files.

## A single Responses call

```python
from republic import Message, Request, TextPart, generate
from republic.auth.grok import GrokTokens
from republic.providers.grok import GrokOAuth


async def answer(tokens: GrokTokens) -> str:
    async with GrokOAuth(tokens, client_version="1.0.41") as provider:
        result = await generate(provider, Request(
            model="grok-build",  # Source example; account/model availability unverified.
            messages=[Message(role="user", parts=[TextPart(text="Hello")])],
        ))
        return result.message.text
```

Both `generate` and `stream` use an SSE inference operation. `generate`
aggregates that stream. This follows the official source's streaming Responses
path; it does not claim that the server cannot support JSON responses. There is
no JSON-first fallback, implicit token refresh, 401 replay, tool execution or
next-turn decision. Known expiry is an optional caller hint, not a local gate.

The request carries `Authorization: Bearer …`, `X-XAI-Token-Auth: xai-grok-cli`,
`x-authenticateresponse: authenticate-response`, the explicit
`x-grok-client-version`, `x-grok-client-mode: headless`, and
`x-grok-model-override` matching `Request.model`. `Accept` requests SSE.
`x-grok-client-identifier` and `User-Agent` identify Republic. No first-party CLI
runtime or telemetry/session/agent identifiers are created. Version/identity
acceptance is an account-backed validation item, not inferred from these headers.

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

## Supported protocol subset

- Ordered system/user/assistant text, native reasoning and function calls, function
  definitions and caller-supplied `function_call_output`. System messages remain
  in place. Function call IDs and response item IDs stay distinct. No unmatched
  history is repaired; tool arguments, including malformed JSON, remain data.
- Native reasoning summaries, reasoning text and `encrypted_content`, including
  items with no visible summary, survive Response JSON serialization and replay.
  The shared wire metadata namespace remains `provider_metadata["openai"]` because
  it stores Responses items, not OpenAI account credentials. Its `raw_item` is
  retained; visibly editing a part without updating the native item is rejected.
- `temperature`, `top_p`, `max_output_tokens`, and auto/none/required/named tool
  choice. Native `provider_options` accepts `text` (including Responses JSON-schema
  format), `reasoning`, `prompt_cache_key`, positive `timeout`, `store=False`, and
  `include=["reasoning.encrypted_content"]`. These last two are overridable defaults.
  Parallel-tool control and shared Responses native options/extensions are forwarded;
  acceptance of caller-selected options remains service/model-dependent.
  Structured output is unparsed text; the caller uses Pydantic validation as in
  the [Responses guide](openai-responses.md), with no model repair request.
- Responses lifecycle/item/content/text/reasoning/function-argument events and
  terminal completed/incomplete/failed outcomes reuse the existing parser.
  Interleaved blocks preserve order and identity. A complete snapshot extends a
  matching delta prefix; contradictions fail, and overlap is never appended twice.
  Missing terminal events fail with partial output inspectable on `Stream.message`.
  Token-limit incomplete output is retained with `finish_reason="length"`.
- Actual response identity/model and usage are preserved. Input/output counts are
  cumulative wire counts; cache/reasoning counts are breakdowns. Total is input
  plus output, or unknown if either is missing. Raw usage keeps Grok's optional
  `cost_in_usd_ticks` and `context_details`; the latter does not replace the total
  with a live agent context length. Missing values are never filled with zero.

Unsupported: media, hosted search/code tools, arbitrary tool types, stop sequences
and managed-field conflicts. Native `store`, `include`, `previous_response_id`,
`truncation`, headers/body and explicit constructor endpoint are configurable.
No server-side recovery or agent behavior is implemented.
Nonstandard agent control/check events are not requested or interpreted; if sent,
they fail as unsupported protocol events instead of initiating recovery. Unknown
output items are not silently dropped. Native failed-response diagnostic text is
replaced with a fixed error; status and partial output remain. OAuth transport/SSE
errors omit credential-bearing native causes and do not trigger another request.

## Source evidence and remaining acceptance

Inspected **2026-09-28**:

- [Official enterprise documentation](https://docs.x.ai/build/enterprise), page
  updated 2026-08-10, distinguishes the OAuth issuer/proxy from the API-key path
  and documents RFC 8628, refresh grants and SSE use. It does not promise general
  third-party account access.
- [Public discovery](https://auth.x.ai/.well-known/openid-configuration) confirms
  the issuer, device/token endpoints, grants, `none` public-client authentication,
  and the requested scopes. It advertises S256 browser PKCE too; that separate
  login path is not implemented here. No authenticated discovery was needed.
- Official [grok-build source](https://github.com/xai-org/grok-build/tree/f0e3be1100ef5252488e3be8bb0e91cf68d8c305),
  revision `f0e3be1100ef5252488e3be8bb0e91cf68d8c305`, records source-monorepo
  revision `036a5d8348cd744767cd0b08518ab17bf608fa7f` and version **1.0.41**.
  `xai-grok-login/src/{config,device_code,oidc/protocol}.rs` establishes the client,
  scopes, device parameters and refresh principal hints. Under the same
  `crates/codegen/` root, `xai-grok-shell/src/agent/{config,proxy_headers}.rs`,
  `xai-grok-http/src/lib.rs`, `xai-grok-sampler/src/client.rs` and
  `xai-grok-sampling-types/src/conversation/responses.rs` establish the proxy,
  headers, request fields, Responses events, encrypted replay and raw usage.
  Source is Apache-2.0, Copyright 2023-2026 SpaceXAI; see [NOTICE](https://github.com/bubbuild/republic/blob/dev/NOTICE).
- [Official npm metadata](https://registry.npmjs.org/@xai-official/grok/1.0.41)
  independently reports version 1.0.41 and Apache-2.0. No npm/CLI code was installed,
  executed, imported or bundled. The Rust protocol subset was rewritten with
  Authlib/OpenAI clients; retries, auth discovery, agent execution and telemetry
  were excluded.

Fixtures demonstrate request construction and parsing through real Authlib/OpenAI
clients, not service acceptance. They cover device login → explicit file → one
SSE inference → explicit refresh → one new inference, plus failure and resource
paths. The remaining minimum live evidence is an authorized account/client with
device-flow access, acceptance of the reduced scopes and Republic/version headers,
and a permitted model completing inference before and after refresh. No such
account evidence is available in this increment; Step 7 live acceptance remains
open. This does not start Bub integration or require buying access, registering
an application, or changing enterprise policy.
