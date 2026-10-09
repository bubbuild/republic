# Codex with ChatGPT login

Use the `codex` provider with a Codex ChatGPT account login. Chat and streaming use the Responses endpoint.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic). With an existing Codex file login, replace `MODEL_ID` with a model available to your account and run:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("codex:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

### Use an existing login

The default credential file is `$CODEX_HOME/auth.json`, or `~/.codex/auth.json` when `CODEX_HOME` is unset. Credentials must be available in that file; a CLI login stored only in the keyring cannot be read through this path.

To choose another file, replace the model construction with `get_model(..., auth=CodexAuth.from_file(path))`. Import `CodexAuth` from `republic.providers`.

### Log in explicitly

Inside an async function, obtain an auth object and pass it to the model:

```python
import republic
from republic.providers import CodexAuth

auth = await CodexAuth.login()
model = republic.get_model("codex:MODEL_ID", auth=auth)
```

This runs the installed Codex CLI and selects file storage for this invocation. Use `device_auth=True` for device authorization or `executable=` to select the CLI executable. Model requests do not start this login flow themselves.

### Store and refresh credentials

`CodexAuth.from_file()` reads the file immediately and before every request. Expiring tokens are refreshed when a refresh token is available, then saved to the same file.

For application-managed credentials, use `CodexAuth(token, account_id=...)`. Reuse the auth object across requests and persist its updated `auth.token` in your own credential store.

## Model support

Use `chat()` for a complete response or `stream()` for events. Responses is the only supported format; `max_tokens` is unsupported. Model access depends on your ChatGPT account.

For OpenAI API-key access, use the [OpenAI provider](openai.md). See [configuration](../reference/configuration.md) for credential precedence and HTTP client ownership.
