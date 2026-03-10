# Quickstart

## Install

```bash
pip install republic
```

If you need provider-specific OAuth instead of static tokens, continue with
[Authentication](guides/authentication.md) after this page.

## Step 1: Create an LLM

Republic uses the `provider:model` format.

```python
from republic import LLM

llm = LLM(model="openrouter:openrouter/free", api_key="<API_KEY>")
```

## Step 2: Send one request

```python
out = llm.chat("Write one short release note.", max_tokens=48)
print("text:", out)
```

## Step 3: Add an auditable trace to the session

`tape` organizes context around anchors by default, so start with one `handoff`.

```python
tape = llm.tape("release-notes")
tape.handoff("draft_v1", state={"owner": "assistant"})

reply = tape.chat("Summarize the version changes in three bullets.", system_prompt="Keep it concise.")
print(reply)
```

## Step 4: Handle failures and fallback

Note: `max_retries` is the number of retries after the first attempt (total attempts per model is `1 + max_retries`).

```python
from republic import ErrorPayload, LLM

llm = LLM(
    model="openai:gpt-4o-mini",
    fallback_models=["openrouter:openrouter/free"],
    max_retries=2,
    api_key={"openai": "<OPENAI_KEY>", "openrouter": "<OPENROUTER_KEY>"},
)

try:
    result = llm.chat("say hello", max_tokens=8)
    print(result)
except ErrorPayload as error:
    print(error.kind, error.message)
```
