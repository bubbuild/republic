# Republic provider SDK rebuild

Republic is being rebuilt as a small LLM provider SDK with typed messages,
single-call generation and streaming, OpenAI and Anthropic adapters, and
Authlib-based OAuth helpers for GitHub Copilot, ChatGPT/Codex, and Grok.

The `dev` branch implements Steps 1-6: serializable request/response data and a
single-call asynchronous API, OpenAI Chat Completions/Responses, Anthropic Messages,
and concrete ChatGPT/Codex and GitHub Copilot OAuth helpers/providers. Real async clients are tested
with HTTP/SSE fixtures. Python 3.11+ remains the compatibility policy. Live service
access and real-account OAuth acceptance have not been validated. Copilot remains an editor-protocol adaptation with unverified integration/entitlement
acceptance. Grok OAuth remains unimplemented. These pages describe the rebuild,
not the API of the previously published package.

Start with the [single-call contract](contracts.md) for a runnable offline
example and the data, streaming, and ownership rules.
Protocol guides cover [Chat Completions](openai-chat.md),
[Responses](openai-responses.md), [Anthropic Messages](anthropic-messages.md),
[ChatGPT/Codex OAuth](codex-oauth.md), and [GitHub Copilot OAuth](copilot-oauth.md).

See the [rebuild plan](rebuild-plan.md) for scope, increment order, and acceptance
criteria. Bub remains responsible for agent loops, local tool execution, and
tape/context management.
