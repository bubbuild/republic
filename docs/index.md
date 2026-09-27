# Republic provider SDK rebuild

Republic is being rebuilt as a small LLM provider SDK with typed messages,
single-call generation and streaming, OpenAI and Anthropic adapters, and
Authlib-based OAuth helpers for GitHub Copilot, ChatGPT/Codex, and Grok.

The `dev` branch implements Step 1: serializable request/response data and a
single-call asynchronous API, exercised with deterministic fake providers.
Python 3.11+ remains the compatibility policy. Real OpenAI/Anthropic adapters
and all OAuth helpers are still planned. These pages describe the rebuild,
not the API of the previously published package.

Start with the [single-call contract](contracts.md) for a runnable offline
example and the data, streaming, and ownership rules.

See the [rebuild plan](rebuild-plan.md) for scope, increment order, and acceptance
criteria. Bub remains responsible for agent loops, local tool execution, and
tape/context management.
