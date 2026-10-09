# OpenRouter API

Use the `openrouter` provider with an OpenRouter API key. It supports chat, embeddings, and decisions. Chat requests default to Responses; Chat Completions and Messages are also available.

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

Republic reads `REPUBLIC_OPENROUTER_API_KEY` when the provider is created. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

Choose a model available to your account. Model names use `openrouter:provider/model`; the model ID determines which service OpenRouter routes to.

Use `chat()` or `stream()` for chat models. Select `api_format="chat"` or `api_format="messages"` when needed. Use `get_embedding_model()` or `get_decision_model()` for models of those kinds.

See [configuration](../reference/configuration.md) for credential precedence, endpoint overrides, and request options.
