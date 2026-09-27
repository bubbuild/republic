# Republic

Republic is being rebuilt on `dev` as a small LLM provider SDK for Python,
with Bub as its first integration target.

The intended scope is typed messages and results, single-call generation and
streaming, OpenAI and Anthropic protocol adapters, and concrete OAuth helpers
for GitHub Copilot, ChatGPT/Codex, and Grok using Authlib.

Agent loops, local tool execution, tape persistence, and context orchestration
belong to the caller. Tool definitions, calls, and results are model data.

## Current status

Steps 1-4 implement a Python 3.11+ single-call SDK with OpenAI Chat Completions,
OpenAI Responses and Anthropic Messages adapters using official async clients.
Deterministic tests exercise the real clients with HTTP/SSE fixtures, including custom base URLs. No live OpenAI,
OpenRouter or Anthropic service has been validated. OAuth remains planned.
This branch does not describe the currently published package.

Install this checkout with `uv sync` for development or `pip install .` for
runtime use. Pydantic, OpenAI 2.x, Anthropic 0.x and HTTPX are included runtime
dependencies.
Existing packaging, versioning, CI and quality tooling are retained.

Follow the [rebuild plan](docs/rebuild-plan.md) for the ordered increments,
acceptance evidence, and source references. Every increment should be a focused
Conventional Commit with its behavior tests and documentation.

## Single-call API

```python
from republic import Message, Request, TextPart, generate, stream

request = Request(
    model="your-model",
    messages=[Message(role="user", parts=[TextPart(text="Hello")])],
)

# Inside an async function, with a caller-supplied Provider implementation:
# response = await generate(provider, request)
# async with stream(provider, request) as output:
#     async for event in output:
#         ...
#     response = output.response
```

Use `from republic.providers.openai import OpenAIChatCompletions` for the concrete
adapter. The [Chat Completions guide](docs/openai-chat.md) covers API keys,
OpenRouter/custom base URLs, streaming, tool results and client ownership.

Use `from republic.providers.openai import OpenAIResponses` for Responses. The
[Responses guide](docs/openai-responses.md) covers complete-history requests, native
reasoning and encrypted metadata, structured output, and terminal outcomes.

Use `from republic.providers.anthropic import AnthropicMessages` for Anthropic.
The [Messages guide](docs/anthropic-messages.md) covers explicit output limits,
thinking signatures, redacted reasoning, cache control and inclusive usage.

See the [single-call contract](docs/contracts.md) for a runnable offline example,
data types, metadata rules, event ordering, and stream ownership. Each call
performs one provider operation. Republic does not execute tools, start another
turn, retry requests, or repair history. Caller cancellation remains
`asyncio.CancelledError`; premature stream exhaustion raises
`IncompleteStreamError` and leaves partial output inspectable.

## Development

The existing workflow is described in [CONTRIBUTING.md](CONTRIBUTING.md):

```bash
uv sync
make check
make test
make docs-test
```

Tests use deterministic providers and HTTP/SSE fixtures, with no service credentials.
See the plan's completion evidence for verification results and remaining limits.

## License

[Apache License 2.0](LICENSE). [NOTICE](NOTICE) records the Vercel AI Python
revision, copyright, extracted files, and Republic's changes.
