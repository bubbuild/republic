# Republic

[![Release](https://img.shields.io/github/v/release/bubbuild/republic)](https://img.shields.io/github/v/release/bubbuild/republic)
[![Build status](https://img.shields.io/github/actions/workflow/status/bubbuild/republic/main.yml?branch=main)](https://github.com/bubbuild/republic/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/bubbuild/republic/branch/main/graph/badge.svg)](https://codecov.io/gh/bubbuild/republic)
[![Commit activity](https://img.shields.io/github/commit-activity/m/bubbuild/republic)](https://img.shields.io/github/commit-activity/m/bubbuild/republic)
[![License](https://img.shields.io/github/license/bubbuild/republic)](https://img.shields.io/github/license/bubbuild/republic)

Build LLM workflows like normal Python while keeping a full audit trail by default.

> Visit https://getrepublic.org for concepts, guides, and API reference.

Republic is a **tape-first** LLM client: messages, tool calls, tool results, errors, and usage are all recorded as structured data. You can make the workflow explicit first, then decide where intelligence should be added.

## Quick Start

```bash
pip install republic
```

```python
from __future__ import annotations

import os

from republic import LLM

api_key = os.getenv("LLM_API_KEY")
if not api_key:
    raise RuntimeError("Set LLM_API_KEY before running this example.")

llm = LLM(model="openrouter:openrouter/free", api_key=api_key)
result = llm.chat("Describe Republic in one sentence.", max_tokens=48)
print(result)
```

## Why It Feels Natural

- **Plain Python**: The main flow is regular functions and branches, no extra DSL.
- **Structured error handling**: Errors are explicit and typed, so retry and fallback logic stays deterministic.
- **Tools without magic**: Supports both automatic and manual tool execution with clear debugging and auditing.
- **Tape-first memory**: Use anchor/handoff to bound context windows and replay full evidence.
- **Event streaming**: Subscribe to text deltas, tool calls, tool results, usage, and final state.

## Provider Auth Resolver

Republic can resolve provider keys dynamically via `api_key_resolver`.

```python
from republic import LLM, login_openai_codex_oauth, openai_codex_oauth_resolver

# First-time login (paste redirect URL when prompted by your app/CLI wrapper).
# You can wire `prompt_for_redirect` to your own input UI.
login_openai_codex_oauth(
    prompt_for_redirect=lambda authorize_url: input(f"Open this URL and paste callback URL:\n{authorize_url}\n> "),
)

llm = LLM(
    model="openai:gpt-5.3-codex",
    api_key_resolver=openai_codex_oauth_resolver(),
)
print(llm.chat("Say hello in one sentence."))
```

`openai_codex_oauth_resolver()` reads `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) and returns
the current access token for `openai`, refreshing it automatically when it is near expiry.
If you omit `prompt_for_redirect`, login will try to capture the callback from `redirect_uri` automatically.
## Development

```bash
make check
make test
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for local setup, testing, and release guidance.

## License

[Apache 2.0](./LICENSE)

---

> This project is derived from [lightning-ai/litai](https://github.com/lightning-ai/litai) and inspired by [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai); we hope you like them too.
