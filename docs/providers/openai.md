# OpenAI API

Use the `openai` provider to call OpenAI with an API key. It supports chat and embeddings. Chat requests default to Responses; Chat Completions is also available.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_OPENAI_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("openai:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_OPENAI_API_KEY` when the provider is created. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

To use a ChatGPT account login, select the [Codex provider](codex.md).

## Model support

To use Chat Completions, replace the model construction above with:

```python
model = republic.get_model("openai:MODEL_ID", api_format="chat")
```

For an embedding model, use `get_embedding_model("openai:MODEL_ID")`, then await `embed(text)` or `embed_many(texts)` inside an async function. Choose a model that supports the selected operation.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
