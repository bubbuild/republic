# Together AI

Use the `together` provider to call Together AI with an API key. It supports chat and embeddings. Chat uses Chat Completions.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_TOGETHER_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("together:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_TOGETHER_API_KEY` when the provider is created and sends the key in the `Authorization: Bearer` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

The default endpoint is `https://api.together.ai/v1`. Chat requests send `max_tokens`. Model names include the organization, such as `together:ORG/MODEL`. Use `get_embedding_model()` for embedding models.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
