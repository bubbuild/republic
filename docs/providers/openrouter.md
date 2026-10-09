# OpenRouter API keys and OAuth PKCE

Use the `openrouter` provider with an API key supplied directly or obtained through OAuth PKCE authorization. It supports chat, embeddings, and decisions. Chat requests default to Responses; Chat Completions and Messages are also available.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic) and set `REPUBLIC_OPENROUTER_API_KEY`. Replace `MODEL_ID` with an OpenRouter model ID, including its provider prefix, and run:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("openrouter:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

### Use an API key

Republic reads `REPUBLIC_OPENROUTER_API_KEY` when the provider is created. You can pass an existing key as `api_key=` or `auth=OpenRouterAuth(saved_key)`, with `OpenRouterAuth` imported from `republic.providers`.

### Authorize with OAuth PKCE

`OpenRouterAuth.login()` creates an authorization URL and exchanges the returned code for an API key. Your callback displays the URL and returns the code the user copies from OpenRouter. This complete terminal example provides that callback:

```python
import asyncio

import republic
from republic.providers import OpenRouterAuth


async def authorize(url: str) -> str:
    print(f"Open this URL and authorize access: {url}")
    return await asyncio.to_thread(input, "Authorization code: ")


async def main():
    auth = await OpenRouterAuth.login(on_authorize=authorize)
    model = republic.get_model("openrouter:MODEL_ID", auth=auth)
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

The flow uses PKCE with S256 to bind the code exchange to the login request. A desktop or web application can supply its own callback instead of terminal input. Your callback handles the authorization UI and cancellation.

### Store credentials

OpenRouter issues an API key, not a refreshable OAuth token. Save `auth.api_key` in your application's credential store and restore it with `OpenRouterAuth(saved_key)`. Login does not persist the key automatically, and model requests do not refresh it.

## Model support

Choose a model available to your account. Model names use `openrouter:provider/model`; the model ID determines which service OpenRouter routes to.

Use `chat()` or `stream()` for chat models. Select `api_format="chat"` or `api_format="messages"` when needed. Use `get_embedding_model()` or `get_decision_model()` for models of those kinds.

See [configuration](../reference/configuration.md) for credential precedence, endpoint overrides, and request options.
