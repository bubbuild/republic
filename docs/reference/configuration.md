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
| `extensions` | Optional callback bundles from `republic.extensions`; empty by default |

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

Without `http_client=`, a Provider lazily creates one client on its first request. Models created by the same Provider and model listing share its connection pool. Separate calls to the top-level model factories create separate Providers.

Supply a client to share connections across Providers and control its lifetime yourself. Configure timeouts on that client; the Provider's `timeout=` option only configures clients Republic creates.

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

This client is used for inference requests and Copilot Plugin token exchange. Codex and Grok token refresh, Copilot device login, and OpenRouter authorization-code exchange create their own auth clients; an injected inference client's transport options do not configure those clients. CLI login methods use the selected CLI's network configuration.

## Request body merging

Provider-level `extra_body` applies to every chat request. Each call's `extra_body` merges over it recursively for mappings; other values replace the earlier value. The resulting fields merge after the API format builds the request and can override generated fields.

Use named options such as `max_tokens` when the format can express them. Use `extra_body` for fields specific to the selected service. An option the format cannot express raises `errors.UnsupportedFeatureError`; a format that can encode an option does not guarantee every model accepts it.

## Retries

Republic retries HTTP 408, 409, 429 and 5xx responses, timeouts, network errors and remote protocol errors. Other HTTP errors, invalid response payloads and local authentication errors are not retried. The policy applies to chat, embeddings, images, decisions, and each page of model listing, including when you supply an HTTP client.

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

## Extensions

Pass an `Extension` when you need to observe Republic's calls or adjust retry
waiting. It is a collection of optional callbacks; no registration or subclass
is needed. For HTTP request headers, transport logging and authentication,
continue using the supplied HTTP client's hooks, transport and `auth=`.

```python
import logging

import republic
from republic.extensions import Call, Extension

logger = logging.getLogger(__name__)


def log_call(call: Call) -> None:
    logger.info(
        "%s %s:%s status=%s attempts=%s seconds=%.3f request_id=%s",
        call.operation, call.provider, call.model,
        call.status, call.attempts, call.elapsed, call.request_id,
    )


model = republic.get_model(
    "openai:gpt-6-sol",
    extensions=[Extension(on_finish=log_call)],
)
```

The same `extensions=` option applies to Providers and all model factories.
Callbacks may be regular or async functions. A package can expose a function
that builds an `Extension` with its own configuration and callbacks.

| Callback | When it runs |
| --- | --- |
| `on_start(call)` | Before preparing the request |
| `retry_delay(call, error, delay)` | After an HTTP status or transport failure, when additional attempts remain; returns a wait in seconds or `None` to stop retrying |
| `on_retry(call, error, delay)` | Before waiting for the next attempt, after the failed HTTP response is closed |
| `on_finish(call)` | Once the call finishes, including parsing or output-validation failures; for streams, after the stream context exits and releases the response |

`Call` is unique to each call, including concurrent calls sharing an Extension.
It carries `provider`, `model`, `operation`, `api_format`, `streaming`,
`attempts`, elapsed seconds, response headers and `request_id`. At finish,
`response` is the public chat, embedding, image or decision response, or the
list returned by `list_models()`. `error` is the original exception when one
occurred. A fully parsed chat response is retained when output validation or
history writing fails, so its usage remains observable. Extensions may keep
their own per-call data in `call.state`; Republic
maintains the other fields. Response text and usage remain available through
the public response; no prompt or credential fields are added to `Call`.

The final `status` is `success`, `error`, `cancelled`, or `closed`. A stream
closed before full consumption is `closed` and has no final response. A
failure caught inside the stream context is still observed as `error`.
Consumer code raising an unrelated exception closes an unfinished stream;
it is not reported as a provider error. Cancellation of an unfinished call
is reported as `cancelled` and propagates to the caller. Finish notification
does not require consuming the stream, but it does require exiting its context.

Observer return values are ignored. Observer exceptions are logged and skipped
so later observers still run; cancellation propagates. Callbacks are awaited
in extension order and should return promptly. An Extension shared between
calls must keep per-call state on `Call`, rather than overwriting shared fields.

To adjust the existing retry behavior:

```python
from republic import errors


def retry_delay(call: Call, error: Exception, delay: float | None) -> float | None:
    if isinstance(error, errors.APIStatusError) and error.status_code == 409:
        return None
    return delay


model = republic.get_model(
    "openai:gpt-6-sol",
    extensions=[Extension(retry_delay=retry_delay, on_finish=log_call)],
)
```

The proposed `delay` is the existing backoff, or `None` for a failure Republic
would not retry by default. Returning a delay can opt such a failure into
retries; returning `None` stops retries. Multiple callbacks transform the
proposal in extension order. `max_retries` remains the limit per request
(per page for model listing), and every wait is capped by `max_retry_delay`.
`attempts` counts all attempts across a call, including all model-list pages.
The `error` passed to retry callbacks is an `APIStatusError` or normalized
connection error. An unread streaming error response has an empty `body` in
this callback; its status and headers are available without reading it.

Because `retry_delay` controls execution, its exceptions and invalid return
values stop the call after closing the failed response. Values must be finite,
nonnegative numbers or `None`. Cancellation, local authentication failures,
payload errors, and successfully opened streams never enter retry callbacks.
Extensions do not change HTTP client ownership or configure auth-client calls.

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
