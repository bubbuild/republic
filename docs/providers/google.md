# Google Gemini API

Use the `google` provider to call the Gemini API with an API key. It supports chat through the Gemini format, embeddings through the Embed Content format, and images through the Gemini Image format.

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

For an image model, use `get_image_model("google:MODEL_ID")`, then await `generate(prompt)` or `edit(prompt, images)`. The `gemini_image` format calls `generateContent` and asks only for image output. Set `aspect_ratio=` (such as `"16:9"`) and `size=` (`"1K"`, `"2K"`, or `"4K"`) to shape the image. Masks and the OpenAI-only options `quality`, `background`, `output_format`, `output_compression`, and `moderation` raise `errors.UnsupportedFeatureError`. A reply without images, such as one stopped by a safety filter, raises `errors.APIResponseError`. See [image models](../reference/provider-api.md#image-models).

Choose a model that supports the requested operation and inputs. See [configuration](../reference/configuration.md) for endpoint overrides and request options.
