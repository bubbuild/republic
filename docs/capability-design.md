# Capability and provider API proposal

Status: **design for review, not an implemented API** (2026-09-28).
This supersedes the Gateway delivery choice in the initial restoration inventory.
Republic will not integrate AI Gateway. The uncommitted Gateway/media prototypes
were archived outside the repository before this proposal. The subsequent direct
Cohere and protocol-specific media increments follow the confirmed delivery scope;
provider registration and the broader API refinements below remain proposals.
The local embedding checkpoint `d690c81` is also subject to this API review.

## What to borrow from the references

- [Vercel ProviderV4](https://github.com/vercel/ai/blob/main/packages/provider/src/provider/v4/provider-v4.ts): distinct language, embedding, reranking, image, speech and transcription capabilities. A provider interface does not require using Vercel's hosted gateway.
- [Pydantic AI providers](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/providers/__init__.py): connection/client configuration is distinct from protocol conversion. Reuse an OpenAI-compatible interface across endpoints where its wire actually matches.
- [Pydantic AI embeddings](https://pydantic.dev/docs/ai/api/pydantic-ai/embeddings/): explicit embedding request/results, including query/document intent for models that distinguish them.
- [Tetos Speaker](https://github.com/frostming/tetos/blob/main/src/tetos/base.py) and [OpenAISpeaker](https://github.com/frostming/tetos/blob/main/src/tetos/openai.py): a small operation interface and concrete provider-specific constructor. Its CLI, implicit file output and synchronous convenience layer are not needed here. Tetos currently uses a built-in class list, not an external plugin registry.
- [Marko Markdown.use](https://github.com/frostming/marko/blob/master/marko/__init__.py): explicit extension registration on an instance. Borrow explicit registration, not parser mixins, dynamically composed inheritance or automatic module loading.

These public sources were inspected on 2026-09-28. The existing semantic source
remains ai-python `c788059dd1db2d93ae1c3da6daffb660eca07dbb` (Apache-2.0).
New source revisions must be pinned if implementation is extracted later.
The following interfaces are Republic proposals, not claims that all four
projects share the same design.

## Keep operations independent

Keep existing `Request.model`, `Response`, parts and stream events. Do not add a
second model-bound object that repeats the model ID already held by Request.
The current language `Provider` protocol can remain under its existing name;
`LanguageProvider` below is a descriptive name for this proposal.

```python
from collections.abc import AsyncIterator
from typing import Protocol

class LanguageProvider(Protocol):
    async def generate(self, request: Request) -> Response: ...
    def stream(self, request: Request) -> AsyncIterator[Event]: ...

class EmbeddingProvider(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

class RerankProvider(Protocol):
    async def rerank(self, request: RerankRequest) -> RerankResponse: ...
```

These are structural interfaces. A third-party class implements only the methods
for its capability; it need not inherit an SDK implementation or implement fake
chat methods. Existing `generate(provider, request)` / context-managed `stream`
remain the language entry points. Each logical call leaves retry, batching,
refresh, fallback, retrieval and tool execution policy with its caller.

Serializable request/results use the existing Pydantic data conventions:

```python
class EmbeddingRequest(Data):
    model: str
    inputs: list[str]
    dimensions: int | None = None
    input_type: Literal["query", "document"] | None = None
    provider_options: dict[str, JsonValue] = Field(default_factory=dict)

class Embedding(Data):
    index: int
    vector: list[float]
    provider_metadata: dict[str, JsonValue] | None = None

class EmbeddingResponse(Data):
    embeddings: list[Embedding]
    response_model: str | None = None
    usage: Usage | None = None
    provider_metadata: dict[str, JsonValue] | None = None

class RerankRequest(Data):
    model: str
    query: str
    documents: list[str] | list[dict[str, JsonValue]]
    top_n: int | None = None
    provider_options: dict[str, JsonValue] = Field(default_factory=dict)

class RankedDocument(Data):
    index: int
    score: float
    provider_metadata: dict[str, JsonValue] | None = None

class RerankResponse(Data):
    ranking: list[RankedDocument]
    response_id: str | None = None
    usage: Usage | None = None
    provider_metadata: dict[str, JsonValue] | None = None
```

`Data` above means the current JSON-only Pydantic base; validators are omitted
from the sketch. Embeddings retain input alignment and original indexes;
reranking retains the service's ranking order, original document indexes and raw
scores. No fabricated IDs, costs or usage. Object-document support must be checked
by each adapter. Query/document intent has a documented provider mapping;
symmetric OpenAI text embeddings use the same wire for both. No implicit chunking.

## Registration is optional wiring

Prefer concrete constructors for typed credentials/client/settings. Register the
configured objects, not a universal constructor with `**kwargs` for every service.

```python
class ProviderRegistry:
    def register(
        self,
        name: str,
        *,
        language: LanguageProvider | None = None,
        embedding: EmbeddingProvider | None = None,
        reranking: RerankProvider | None = None,
        replace: bool = False,
    ) -> None: ...

    def language(self, name: str) -> LanguageProvider: ...
    def embedding(self, name: str) -> EmbeddingProvider: ...
    def reranking(self, name: str) -> RerankProvider: ...
```

The registry is an instance-owned name table. Duplicate names fail unless an
explicit replacement is requested. Unknown names or unavailable capabilities
raise a configuration error before network I/O. It performs no discovery,
imports, credential loading, fallback, requests or client closing.
Registration aliases are separate from wire/provider metadata namespaces.

```python
async with AsyncOpenAI(api_key=key, max_retries=0) as client:
    providers = ProviderRegistry()
    providers.register(
        "openai",
        language=OpenAIResponses(client=client),
        embedding=OpenAIEmbeddings(client=client),
    )
    providers.register("acme", reranking=AcmeReranker(client=acme_http))

    result = await providers.embedding("openai").embed(
        EmbeddingRequest(model="text-embedding-3-small", inputs=["hello"])
    )
```

A caller can instead pass `OpenAIEmbeddings(...)` directly; registry use is not
required. An external `AcmeReranker` only implements the typed `rerank` method and
can live in another package. No Republic source changes, global enum or entry
point installation is required. The caller owns injected clients; concrete
adapters retain their existing owned-client context management. No new common
client wrapper is required. Bub can split its configured name once at `:` and
use the remaining model ID verbatim.

## Multimodal input belongs in message parts

Retain the existing FilePart representation and add an explicit file reference:

```python
class FilePart(Data):
    kind: Literal["file"] = "file"
    data: str
    media_type: str
    encoding: Literal["url", "base64", "file_id"] = "url"
    filename: str | None = None
    provider_metadata: dict[str, JsonValue] | None = None

message = Message(role="user", parts=[
    TextPart(text="Describe these inputs"),
    FilePart(data="https://example.test/photo.png", media_type="image/png"),
    FilePart.from_bytes(audio_bytes, media_type="audio/wav"),
    FilePart(data="https://example.test/clip.mp4", media_type="video/mp4"),
])
```

Data and native metadata remain serializable in order. File IDs are provider
references, not portable cross-provider assets. Bytes are normalized explicitly;
Republic does not download URLs or open local paths.

The protocol adapter maps these parts using the actual service schema. Common
media does not need a synthetic `content_type` metadata flag. An OpenRouter
adapter/explicit compatible protocol extension can handle documented video and
audio forms without claiming that native OpenAI or Anthropic supports them.
Native detail/cache/reasoning fields stay in namespaced metadata and must round
trip. Unsupported wire forms fail explicitly; no media-to-text placeholders.

Independent image generation, speech synthesis, transcription and video jobs are
separate capabilities, not more message roles or methods forced onto every
language provider. Their detailed result/stream/job types will be established
with their first real adapter. Tetos is a useful speech-interface reference;
its save-to-file policy is not imposed on callers.

## Delivery after API review

1. Finalize these request/result and registration interfaces, including native
   metadata, capability absence and ownership contracts. Exercise a tiny external
   provider module through the same public calls as built-ins.
2. Complete basic direct adapters: existing OpenAI Chat/Responses and Anthropic;
   OpenAI embeddings; direct Cohere v2 reranking (the caller-selected default). No AI Gateway dependency or endpoint.
3. Restore the actual Bub image/audio/video input pipeline using those types:
   build_prompt → runner → official SDK/HTTP fixture → tape → new process →
   next request, retaining tool IDs and reasoning metadata. Keep restored OAuth
   behavior and Republic as the only SDK.
4. Add separate image generation, speech and transcription increments; define
   video job lifecycle from a real backend before implementing polling helpers.

Each implementation increment still requires offline behavior tests, source
attribution, meaningful compatibility checks and focused local commits. Existing
parser/stream/auth behavior remains a regression baseline. Live provider/OAuth
entitlement is not established by these design sketches or fixtures.
