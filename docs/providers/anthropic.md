# Anthropic API

Use the `anthropic` provider to call Claude with an API key. Chat and streaming use the Messages API format.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_ANTHROPIC_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("anthropic:MODEL_ID")
    response = await model.chat("Say hello in one sentence.", max_tokens=256)
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_ANTHROPIC_API_KEY` when the provider is created and sends the key in the `x-api-key` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

Use `chat()` for a complete response or `stream()` for events. Messages is the only supported format for this provider. Set `max_tokens` to the output limit you need; reasoning and tool capabilities depend on the selected model.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
