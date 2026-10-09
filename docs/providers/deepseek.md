# DeepSeek

Use the `deepseek` provider to call DeepSeek with an API key. Chat requests default to Chat Completions; Responses and Anthropic Messages are also available.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_DEEPSEEK_API_KEY`, and replace `MODEL_ID` with a model available to your account:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("deepseek:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_DEEPSEEK_API_KEY` when the provider is created and sends the key in the `Authorization: Bearer` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

Chat requests send `max_tokens`, and `reasoning_effort="none"` turns thinking off. Thinking models return reasoning in `response.reasoning`. Republic sends `reasoning_content` back with earlier assistant messages, which these models require across tool calls; keep `response.message` in the conversation.

Select `api_format="responses"` or `api_format="messages"` to use another format. Messages requests go to DeepSeek's Anthropic-compatible endpoint, `https://api.deepseek.com/anthropic`, while `api_base` stays `https://api.deepseek.com`.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
