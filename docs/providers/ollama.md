# Ollama

Use the `ollama` provider to call a local Ollama server or Ollama Cloud. It supports chat and embeddings. Chat requests default to Chat Completions; stateless Responses requires Ollama 0.13.3 or later.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), start Ollama, and replace `MODEL_ID` with a pulled model:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("ollama:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

A local server needs no key, and Republic sends no `Authorization` header unless `REPUBLIC_OLLAMA_API_KEY` or `api_key=` is set. For Ollama Cloud, set `REPUBLIC_OLLAMA_API_BASE=https://ollama.com/v1` and an API key from your Ollama account.

## Model support

The default endpoint is `http://localhost:11434/v1`. Chat requests send `max_tokens`. Select `api_format="responses"` for stateless Responses. Use `get_embedding_model()` for embedding models.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
