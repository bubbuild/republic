# Provider configuration

These options apply to `get_provider()` and the model factories, including `get_model()`.

## Construction options

| Option | Behavior |
| --- | --- |
| `api_key` | Service key; otherwise read `<PREFIX>_API_KEY` |
| `api_base` | Service base URL; otherwise read `<PREFIX>_API_BASE`, then use the provider default. Trailing slashes are removed. |
| `auth` | `httpx2.Auth` object; takes precedence over API-key auth |
| `env_prefix` | Environment namespace; defaults to `REPUBLIC_` plus the uppercase registered provider name, with hyphens replaced by underscores |
| `api_format` | Preferred supported format for its model kind; other model kinds retain their defaults |
| `headers` | Headers sent on every request, overriding the format's default headers |
| `extra_body` | Extra chat request fields, merged with each call's `extra_body` |
| `http_client` | Caller-owned `httpx2.AsyncClient` |
| `timeout` | Timeout for clients Republic creates; default is 600 seconds, with 10 seconds for connection establishment |
| `max_retries` | Additional attempts after a retryable failure; default is 2, and 0 disables retries |
| `retry_delay` | Initial exponential backoff in seconds; default is 0.5, with jitter |
| `max_retry_delay` | Maximum wait between attempts, including `Retry-After`; default is 60 seconds |

`api_key`, `api_base`, and `env_prefix` treat `None` and an empty string as absent. An empty string does not disable environment lookup.

## Credential precedence

For the built-in providers, authentication is selected in this order:

1. Explicit `auth=`.
2. Explicit, nonempty `api_key=`.
3. Nonempty `<PREFIX>_API_KEY` from the environment.
4. Authentication on the supplied `http_client`.
5. The provider's credential default, when one exists: Codex or Grok file credentials, or GitHub CLI auth for Copilot.

API-key providers without any of these send an unauthenticated request and let the service respond. A configured but invalid credential does not trigger a search through lower-priority choices.

Endpoint selection is separate: explicit, nonempty `api_base=`, then `<PREFIX>_API_BASE`, then the provider default. Setting `auth=` does not disable endpoint lookup. An auth flow can subsequently change the request origin; Copilot Plugin exchange follows the service's returned API origin.

## Environment variables

For `get_model("openai:MODEL_ID")`, the variables are `REPUBLIC_OPENAI_API_KEY` and `REPUBLIC_OPENAI_API_BASE`. To use application-specific names:

```python
from republic import get_model

model = get_model("openai:gpt-6-sol", env_prefix="MY_APP_OPENAI")
```

This reads `MY_APP_OPENAI_API_KEY` and `MY_APP_OPENAI_API_BASE`. It replaces the prefix; it does not add another fallback namespace.

The default prefix replaces hyphens in a registered name with underscores. For `github-copilot`, it is `REPUBLIC_GITHUB_COPILOT`; for `azure-openai`, it is `REPUBLIC_AZURE_OPENAI`.

Republic reads these provider values at construction. Changing them later affects newly created providers, not existing ones. Republic does not load `.env` files or write provider values to the process environment. Load application configuration before creating a provider if you use a dotenv loader.

Credential sources can have different read timing:

| Source | When it is read |
| --- | --- |
| Provider API-key and base-URL variables | Provider construction |
| `CodexAuth.from_file()` | File path selected at construction; contents read then and before each request |
| `GrokAuth.from_file()` | File path selected at construction; contents read then and before each request |
| `GitHubCLIAuth()` | `gh auth token` runs for each request and follows the CLI's credential selection |
| `CopilotAuth(token)` | Original credential supplied by the caller; inference token exchanged on demand and renewed before expiry |
| `OpenRouterAuth(key)` | Fixed key supplied by the caller; requests do not refresh it |

Transport settings belong to the HTTP client. Setting `httpx2.AsyncClient(trust_env=False)` controls that client's environment handling; it does not disable Republic's API-key or base-URL lookup. Republic has no separate switch to turn off provider environment lookup.

## HTTP clients

Without `http_client=`, Republic creates and closes a client for each call. Reusing a model alone does not share a connection pool across calls.

Supply a client to share connections and control its lifetime. Configure timeouts on that client; the provider's `timeout=` option only configures clients Republic creates.

```python
import asyncio

import httpx2

import republic


async def main():
    async with httpx2.AsyncClient(timeout=30) as client:
        model = republic.get_model("openai:gpt-6-sol", http_client=client)
        response = await model.chat("Say hello.")
        print(response.text)


asyncio.run(main())
```

Republic leaves a supplied client open. In the example, leaving the application's `async with` block closes it. Exiting a model stream closes that response while leaving the supplied client available for later requests.

This client is used for inference requests and Copilot Plugin token exchange. Codex and Grok token refresh, Copilot device login, and OpenRouter authorization-code exchange create their own auth clients; an injected inference client's transport options do not configure those clients. CLI login methods use the selected CLI's network configuration.

## Request body merging

Provider-level `extra_body` applies to every chat request. Each call's `extra_body` merges over it recursively for mappings; other values replace the earlier value. The resulting fields merge after the API format builds the request and can override generated fields.

Use named options such as `max_tokens` when the format can express them. Use `extra_body` for fields specific to the selected service. An option the format cannot express raises `errors.UnsupportedFeatureError`; a format that can encode an option does not guarantee every model accepts it.

## Retries

Republic retries HTTP 408, 409, 429 and 5xx responses, timeouts, network errors and remote protocol errors. Other HTTP errors, invalid response payloads and local authentication errors are not retried. The policy applies to chat, embeddings, decisions, and each page of model listing, including when you supply an HTTP client.

```python
model = republic.get_model(
    "openai:gpt-6-sol",
    max_retries=3,
    retry_delay=0.5,
    max_retry_delay=30,
)
```

Retries wait with exponential backoff and jitter. A valid `retry-after-ms` or `Retry-After` header takes precedence; `Retry-After` accepts seconds or an HTTP date. All waits are capped by `max_retry_delay`. The failed response is closed before waiting. `max_retries=2` means at most three attempts; configure timeouts separately because each attempt has its own timeout.

For a stream, only the initial request can be retried. Once a successful HTTP response is opened, Republic never replays it, even if reading fails before the first event. Transport failures raise `errors.APIConnectionError` or its `errors.APITimeoutError` subclass, with the original exception as `__cause__`. Cancellation propagates without a retry.

A timeout can occur after the service accepted a request. Retrying can therefore produce another generation or repeat a provider-run tool operation; set `max_retries=0` when your application requires a single attempt.

## Response diagnostics

Chat, embedding and decision responses expose `request_id` and `headers`. Header names are lowercase. The request ID comes from `x-request-id`, `request-id` or `x-goog-request-id`, when present, and is separate from the generated response's `id`.

```python
response = await model.chat("Hello")
print(response.request_id)
```

A stream exposes `stream.request_id` and `stream.headers` as soon as its context is entered, before the final response is available. These values belong to that call, so concurrent requests do not overwrite one another's metadata.

`errors.APIStatusError` preserves `status_code`, `body`, `headers` and `request_id` from the final failed attempt. `errors.APIResponseError` and `errors.StreamIncompleteError` preserve response headers and request IDs too. Connection and timeout errors retain headers when a response was already opened; connection failures before a response have no request ID.

## Error imports

Exception classes are exported from `republic.errors`, rather than the package root. Import the module to catch Republic errors:

```python
from republic import errors

try:
    response = await model.chat("Hello")
except errors.APIStatusError as exc:
    print(exc.status_code, exc.request_id)
except errors.APIConnectionError as exc:
    print(exc.__cause__)
```

`errors.RepublicError` is the common base class. Direct imports such as `from republic.errors import APIStatusError` are also supported.
