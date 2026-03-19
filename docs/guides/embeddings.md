# Embeddings

The embedding interface shares the same `LLM` facade as chat.

```python
from republic import ErrorPayload, LLM

llm = LLM(model="openrouter:openai/text-embedding-3-small", api_key="<API_KEY>")
try:
    vectors = llm.embed(["republic", "tape-first"])
    print(type(vectors).__name__)
except ErrorPayload as exc:
    print(exc.kind, exc.message)
```

You can also override the model per call:

```python
vector = llm.embed(
    "incident root cause analysis",
    model="openrouter:openai/text-embedding-3-small",
)
```
