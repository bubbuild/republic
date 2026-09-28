# Embeddings

`EmbeddingProvider` is independent of chat. `EmbeddingRequest` accepts a nonempty
list of texts (`model_validate` also normalizes a single string), optional
`dimensions`, and native `provider_options`. `embed` delegates once, without
splitting inputs, concurrency, model selection or additional retry policy.

```python
from republic import EmbeddingRequest, embed
from republic.providers.openai_embeddings import OpenAIEmbeddings

async def vectors(api_key: str):
    async with OpenAIEmbeddings(api_key=api_key) as provider:
        return await embed(provider, EmbeddingRequest(
            model="text-embedding-3-small", inputs=["hello", "world"], dimensions=256,
        ))
```

`OpenAIEmbeddings` uses the official asynchronous client. `base_url` selects an
explicit compatible endpoint; `client`, `headers`, `timeout`, `max_retries` and
ownership follow [client configuration](client-configuration.md). Inputs remain
in order. Returned embeddings are ordered by original input index, and retain
that index plus native metadata. Missing/duplicate/out-of-range indexes, invalid
vectors and inconsistent dimensions fail rather than silently pairing the wrong
vector with a document. Text length/model limits remain service decisions.

Native `user`, `encoding_format` (`float` or `base64`), `extra_headers`,
`extra_query`, `extra_body`, and `timeout` are supported. Base64 float32 vectors
are decoded to the same numeric result contract. Managed fields cannot be
overridden. Tokenized input, embedding images, local vector search and automatic
batching are outside this text embedding increment.

Usage retains raw fields; known `prompt_tokens` becomes input tokens. Embeddings
produce no generated output tokens, so output is zero when usage is present;
missing input remains unknown. `response_model` and per-vector/response metadata
remain serializable. Cancellation and SDK errors follow the existing client
contract, and borrowed clients remain open.

Sources inspected 2026-09-28: the pinned source inventory above, installed OpenAI
2.54.0 `resources/embeddings.py` and typed request/results, and the
[official embeddings reference](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create).
Evidence is synthetic HTTP through the real SDK, not live model availability.
