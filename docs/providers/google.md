# Google Gemini API

Use the `google` provider to call the Gemini API with an API key. It supports chat through the Gemini format and embeddings through the Embed Content format.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_GOOGLE_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("google:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_GOOGLE_API_KEY` when the provider is created and sends the key in the `x-goog-api-key` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

Use `chat()` for a complete response or `stream()` for events. For an embedding model, use `get_embedding_model("google:MODEL_ID")`, then await `embed(text)` or `embed_many(texts)` inside an async function.

Choose a model that supports the requested operation and inputs. See [configuration](../reference/configuration.md) for endpoint overrides and request options.
