# MiniMax

Use the `minimax` provider to call MiniMax with an API key. Chat requests default to MiniMax's Anthropic-compatible Messages API; Chat Completions is also available.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_MINIMAX_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("minimax:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_MINIMAX_API_KEY` when the provider is created and sends the key in the `Authorization: Bearer` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

The base URL is the host `https://api.minimax.io`; set `api_base="https://api.minimaxi.com"` for the China platform. Messages requests go to `/anthropic/v1/messages` and Chat Completions requests to `/v1/chat/completions`.

Select `api_format="chat"` for Chat Completions. Chat requests send `reasoning_split: true`, so thinking arrives in `response.reasoning` instead of `<think>` tags. Republic sends `reasoning_content` back with earlier assistant messages, which these models require across tool calls; keep `response.message` in the conversation.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
