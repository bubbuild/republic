# Direct Cohere reranking

Reranking is an independent async operation. Republic sends candidates to the
actual rerank endpoint; it does not retrieve documents or simulate scores with a
chat prompt. No Cohere SDK dependency or AI Gateway is required.

```python
from republic import RerankRequest, rerank
from republic.providers.cohere import CohereRerank

async def rank(api_key: str):
    async with CohereRerank(api_key=api_key) as provider:
        return await rerank(provider, RerankRequest(
            model="rerank-v3.5",
            query="capital of Japan",
            documents=["Paris is in France.", "Tokyo is in Japan."],
            top_n=1,
            provider_options={"max_tokens_per_doc": 512},
        ))
```

The default route is `https://api.cohere.com/v2/rerank`. The response keeps service
ranking order, original document indexes, unscaled `relevance_score` as `score`,
and the response `id`. `provider_metadata["cohere"]` retains `meta` and other native
fields, including `meta.billed_units.search_units`. Search units are billing
units, not tokens; `usage` stays unknown. Per-result native fields are retained on
each `RankedDocument`. Duplicate/out-of-range indexes and excess top_n results
are errors rather than silently rewritten output.

The common data type can represent homogeneous string or object-document lists,
following upstream ai-python. **This v2 adapter accepts strings only.** It never
serializes caller objects as YAML or JSON automatically. Object-document support
in another version/protocol needs its own actual wire adapter. Empty documents
return an empty ranking locally. No implicit batching or concurrent calls occur.

`top_n` is a common request field. Native options include `max_tokens_per_doc`,
`priority`, `extra_body`, `extra_headers` and `timeout`. `extra_body` forwards
compatible-endpoint extensions; managed/common fields cannot be overridden there.
Long-document truncation is the service's behavior controlled by its native
option; Republic does not truncate input itself.

Inject `httpx.AsyncClient` to select transport, proxy, redirects or retries. The
adapter preserves and never closes a borrowed client. `base_url` overrides the
client base URL, which otherwise precedes the Cohere default. Explicit API key
and adapter headers override client headers; per-request headers override adapter
headers, case-insensitively. Explicit timeout overrides the client timeout;
per-request timeout takes precedence. Owned clients close with `aclose()` or
`async with`, default to no retries, and release responses on errors/cancellation.
HTTP errors retain status/request ID and their native exception as cause.

Data semantics originate from Vercel ai-python `c788059dd1` `ops/reranking.py`.
The direct wire follows [Cohere's v2 reference](https://docs.cohere.com/reference/rerank),
checked 2026-09-28; it is distinct from upstream's AI Gateway adapter. Tests use
HTTPX MockTransport, with real HTTP blocked. Compatible services, live models
and account access have not been verified online.
