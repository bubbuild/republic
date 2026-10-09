# Moonshot AI (Kimi)

Use the `moonshot` provider to call Kimi models with an API key. Chat uses Chat Completions.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_MOONSHOT_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("moonshot:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_MOONSHOT_API_KEY` when the provider is created and sends the key in the `Authorization: Bearer` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

The default endpoint is `https://api.moonshot.ai/v1`; set `api_base="https://api.moonshot.cn/v1"` for the China platform. Thinking models return reasoning in `response.reasoning`. Republic sends `reasoning_content` back with earlier assistant messages, which these models require across tool calls; keep `response.message` in the conversation. Pass model-specific thinking options, such as `thinking`, through `extra_body`.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
