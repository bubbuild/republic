# Provider configuration

These options apply to `get_provider()` and the model factories, including `get_model()`.

## Construction options

| Option | Behavior |
| --- | --- |
| `api_key` | Service key; otherwise read `<PREFIX>_API_KEY` |
| `api_base` | Service base URL; otherwise read `<PREFIX>_API_BASE`, then use the provider default. Trailing slashes are removed. |
| `auth` | `httpx2.Auth` object; takes precedence over API-key auth |
| `env_prefix` | Environment namespace; defaults to `REPUBLIC_` plus the uppercase registered provider name |
| `api_format` | Preferred supported format for its model kind; other model kinds retain their defaults |
| `headers` | Headers sent on every request, overriding the format's default headers |
| `extra_body` | Extra chat request fields, merged with each call's `extra_body` |
| `http_client` | Caller-owned `httpx2.AsyncClient` |
| `timeout` | Timeout for clients Republic creates; default is 600 seconds, with 10 seconds for connection establishment |

`api_key`, `api_base`, and `env_prefix` treat `None` and an empty string as absent. An empty string does not disable environment lookup.

## Credential precedence

For the built-in providers, authentication is selected in this order:

1. Explicit `auth=`.
2. Explicit, nonempty `api_key=`.
3. Nonempty `<PREFIX>_API_KEY` from the environment.
4. Authentication on the supplied `http_client`.
5. The provider's credential default, when one exists: Codex file credentials or GitHub CLI auth for Copilot.

API-key providers without any of these send an unauthenticated request and let the service respond. A configured but invalid credential does not trigger a search through lower-priority choices.

Endpoint selection is separate: explicit, nonempty `api_base=`, then `<PREFIX>_API_BASE`, then the provider default. Setting `auth=` does not disable endpoint lookup. An auth flow can subsequently change the request origin; Copilot Plugin exchange follows the service's returned API origin.

## Environment variables

For `get_model("openai:MODEL_ID")`, the variables are `REPUBLIC_OPENAI_API_KEY` and `REPUBLIC_OPENAI_API_BASE`. To use application-specific names:

```python
from republic import get_model

model = get_model("openai:gpt-6-sol", env_prefix="MY_APP_OPENAI")
```

This reads `MY_APP_OPENAI_API_KEY` and `MY_APP_OPENAI_API_BASE`. It replaces the prefix; it does not add another fallback namespace.

The default prefix preserves punctuation in a registered name. For `github-copilot`, it is `REPUBLIC_GITHUB-COPILOT`. Use `env_prefix="REPUBLIC_GITHUB_COPILOT"` if you want shell-friendly names with underscores.

Republic reads these provider values at construction. Changing them later affects newly created providers, not existing ones. Republic does not load `.env` files or write provider values to the process environment. Load application configuration before creating a provider if you use a dotenv loader.

Credential sources can have different read timing:

| Source | When it is read |
| --- | --- |
| Provider API-key and base-URL variables | Provider construction |
| `CodexAuth.from_file()` | File path selected at construction; contents read then and before each request |
| `GitHubCLIAuth()` | `gh auth token` runs for each request and follows the CLI's credential selection |
| `CopilotAuth(token)` | Original credential supplied by the caller; inference token exchanged on demand and renewed before expiry |

Transport settings belong to the HTTP client. Setting `httpx2.AsyncClient(trust_env=False)` controls that client's environment handling; it does not disable Republic's API-key or base-URL lookup. Republic has no separate switch to turn off provider environment lookup.

## HTTP client ownership

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

This client is used for inference requests and Copilot Plugin token exchange. Codex token refresh and Copilot device login create their own auth clients; an injected inference client's transport options do not configure those clients. CLI login methods use the selected CLI's network configuration.

## Request body merging

Provider-level `extra_body` applies to every chat request. Each call's `extra_body` merges over it recursively for mappings; other values replace the earlier value. The resulting fields merge after the API format builds the request and can override generated fields.

Use named options such as `max_tokens` when the format can express them. Use `extra_body` for fields specific to the selected service. An option the format cannot express raises `UnsupportedFeatureError`; a format that can encode an option does not guarantee every model accepts it.
