# Z.ai (GLM)

Use the `zai` provider to call GLM models with an API key. Chat uses Chat Completions.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_ZAI_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("zai:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_ZAI_API_KEY` when the provider is created and sends the key in the `Authorization: Bearer` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

The default endpoint is `https://api.z.ai/api/paas/v4`; set `api_base="https://open.bigmodel.cn/api/paas/v4"` for the China platform. Chat requests send `max_tokens`. Thinking models return reasoning in `response.reasoning`. Republic sends `reasoning_content` back with earlier assistant messages, which these models require across tool calls; keep `response.message` in the conversation. Z.ai accepts only `tool_choice="auto"`.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
