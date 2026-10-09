# Grok API and CLI login

Use the `grok` provider to call xAI with the official Grok CLI's OAuth login or an API key. Chat requests default to Responses; Chat Completions is also available.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic). With an existing Grok CLI login, replace `MODEL_ID` with a model available to your account and run:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("grok:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

### Use an existing login

Credential-file selection is explicit `GrokAuth.from_file(path)`, then `GROK_AUTH_PATH`, then `$GROK_HOME/auth.json`. When `GROK_HOME` is unset, its default is `~/.grok`. Import `GrokAuth` from `republic.providers` and pass it as `auth=` to choose a file yourself.

### Log in explicitly

Inside an async function:

```python
import republic
from republic.providers import GrokAuth

auth = await GrokAuth.login()
model = republic.get_model("grok:MODEL_ID", auth=auth)
```

This runs the official CLI's `grok login`. Set `device_auth=True` for device authorization or `executable=` to select the CLI executable. Normal model requests do not launch login.

### Use an API key

Set `REPUBLIC_GROK_API_KEY` before constructing the model, or pass a key loaded by your application:

```python
import os

import republic

model = republic.get_model("grok:MODEL_ID", api_key=os.environ["MY_APP_API_KEY"])
```

An explicit key or one in `REPUBLIC_GROK_API_KEY` takes precedence over the default CLI login. This path uses API-key authentication without reading or refreshing CLI credentials.

### Store and refresh credentials

File credentials are read at construction and before each request. Expiring tokens are refreshed through Authlib, and rotated credentials are saved under the CLI's file lock.

If your application manages credentials, use `GrokAuth(token)` and persist the updated `auth.token` yourself. The token needs an access token and expiry; refresh also needs a refresh token. Reuse the auth object across requests.

## Model support

Use `chat()` for a complete response or `stream()` for events. To use Chat Completions, replace the model construction in the example with:

```python
model = republic.get_model("grok:MODEL_ID", api_format="chat")
```

Model access depends on your account and credentials. See [configuration](../reference/configuration.md) for credential precedence, endpoint overrides, and HTTP clients.
