# Embeddings

The embedding interface shares the same `LLM` facade as chat.

```python
from republic import LLM, RepublicError

llm = LLM(model="openrouter:openai/text-embedding-3-small", api_key="<API_KEY>")
try:
    out = llm.embed(["republic", "tape-first"])
    print(out)
except RepublicError as error:
    print(error.kind, error.message)
```

You can also override the model per call:

```python
out = llm.embed(
    "incident root cause analysis",
    model="openrouter:openai/text-embedding-3-small",
)
```
