# Chat

`llm.chat(...)` is the smallest entry point and fits most text-only workloads.

## Prompt Mode

```python
from republic import LLM

llm = LLM(model="openrouter:openrouter/free", api_key="<API_KEY>")
out = llm.chat("Output exactly one word: ready", max_tokens=8)
print(out)
```

## Messages Mode

```python
messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain tape-first in one sentence."},
]
out = llm.chat(messages=messages, max_tokens=48)
```

## Structured Error Handling

```python
from republic import RepublicError, LLM

llm = LLM(model="openrouter:openrouter/free", api_key="<API_KEY>")

try:
    out = llm.chat("Write one sentence.", max_tokens=32)
    print(out)
except RepublicError as error:
    if error.kind == "temporary":
        print("retry later")
    else:
        print("fail fast:", error.message)
```

## Retries and Fallback

Note: `max_retries` is the number of retries after the first attempt (total attempts per model is `1 + max_retries`).

```python
llm = LLM(
    model="openai:gpt-4o-mini",
    fallback_models=["anthropic:claude-3-5-sonnet-latest"],
    max_retries=3,
    api_key={"openai": "<OPENAI_KEY>", "anthropic": "<ANTHROPIC_KEY>"},
)

out = llm.chat("Give me one deployment checklist item.")
```

Recommendation: keep `max_retries` small (for example 2-4), and pick fallback models that are slightly more stable while still meeting quality requirements.
