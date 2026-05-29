# Text Decisions

`if_` and `classify` are useful when you want model decisions in a clear structured form.

## if_

```python
from republic import LLM, RepublicError

llm = LLM(model="openrouter:openai/gpt-4o-mini", api_key="<API_KEY>")
try:
    decision = llm.if_("The release is blocked by a migration failure.", "Should we page on-call now?")
    print(decision)  # bool
except RepublicError as error:
    print(error.kind, error.message)
```

## classify

```python
try:
    label = llm.classify(
        "User asks for invoice and tax receipt.",
        ["sales", "support", "finance"],
    )
    print(label)  # one of choices
except RepublicError as error:
    print(error.kind, error.message)
```

## Usage Tips

- Treat these as shortcut entry points for agentic `if` and classification.
- Keep business logic in regular Python branches for testability and audits.
