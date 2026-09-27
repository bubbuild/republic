# Republic provider SDK rebuild

Republic is being rebuilt as a small LLM provider SDK with typed messages,
single-call generation and streaming, OpenAI and Anthropic adapters, and
Authlib-based OAuth helpers for GitHub Copilot, ChatGPT/Codex, and Grok.

The `dev` branch currently contains the cleared baseline and its delivery plan.
It has no usable SDK implementation yet. These pages describe the rebuild,
not the API of the previously published package.

See the [rebuild plan](rebuild-plan.md) for scope, increment order, and acceptance
criteria. Bub remains responsible for agent loops, local tool execution, and
tape/context management.
