# Mistral AI

Use the `mistral` provider to call Mistral with an API key. It supports chat and embeddings. Chat uses Chat Completions.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_MISTRAL_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("mistral:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_MISTRAL_API_KEY` when the provider is created and sends the key in the `Authorization: Bearer` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

Chat requests send `max_tokens`, map `seed` to `random_seed`, and omit `stream_options` because Mistral reports usage in the last chunk. `top_k` raises `UnsupportedFeatureError`.

With `reasoning_effort`, reasoning models return thinking chunks, which Republic exposes as `response.reasoning`. Earlier thinking is sent back as a thinking chunk; keep `response.message` in the conversation.

For embedding models, use `get_embedding_model()`; `dimensions=` is sent as `output_dimension`.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
