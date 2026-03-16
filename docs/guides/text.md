# Text Decisions

`if_` and `classify` are useful when you want model decisions without manually parsing text output.

## if_

```python
from republic import LLM

llm = LLM(model="openrouter:openai/gpt-4o-mini", api_key="<API_KEY>")
decision = llm.if_("The release is blocked by a migration failure.", "Should we page on-call now?")

print(decision)  # bool
```

## classify

```python
label = llm.classify(
    "User asks for invoice and tax receipt.",
    ["sales", "support", "finance"],
)

print(label)  # one of choices
```

## Usage Tips

- Treat these as shortcut entry points for agentic `if` and classification.
- Keep business logic in regular Python branches for testability and audits.
- Handle failures with `try/except ErrorPayload` instead of checking `.error`.
