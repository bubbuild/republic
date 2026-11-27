# Prompt caching and session affinity

Multi-turn conversations and agent loops resend a long shared prefix on every step. Pass `session_id` and `cache_retention` to `chat()` or `stream()` to keep turns routed to a warm cache and control explicit cache retention where the service supports it.

Both options are advisory. Providers and formats that do not use them ignore them without raising `UnsupportedFeatureError`, so the same call works across providers. Per-request headers attach to each `HttpRequest` and never mutate a shared `Provider`, making concurrent sessions on one provider instance safe.

## Basic usage

Use the [quickstart installation and API key](../quickstart.md) and pass a stable opaque `session_id` for the lifetime of a conversation:

```python
import asyncio
import uuid

import republic


async def main():
    session_id = uuid.uuid4().hex
    system_prompt = republic.system("You are a concise assistant. " + "Context. " * 400)
    messages = [system_prompt, republic.user("Reply with PONG1.")]

    async with republic.get_model("openai:gpt-6-sol") as model:
        first = await model.chat(messages, session_id=session_id, cache_retention="short")
        print("turn 1:", first.token_usage.cached_tokens, first.token_usage.cache_write_tokens)

        messages.extend([first.message, republic.user("Reply with PONG2.")])
        second = await model.chat(messages, session_id=session_id, cache_retention="short")
        print("turn 2:", second.token_usage.cached_tokens, second.token_usage.cache_write_tokens)


asyncio.run(main())
```

The same options work with `model.stream()`. Read `response.token_usage.cached_tokens` and `response.token_usage.cache_write_tokens` (available on Chat Completions, Responses, and Messages when reported by the service) to observe cache hits and writes.

## Retention modes

`cache_retention` accepts a `republic.CacheRetention` value:

| Value | Behavior |
| --- | --- |
| Omitted (`None`) | Sends routing affinity headers and cache keys where supported, without opting into explicit paid cache-write directives such as `cache_control`. |
| `"short"` | Enables default ephemeral caching (`cache_control: {"type": "ephemeral"}` on Messages and OpenRouter) and provider cache keys. Recommended for multi-turn loops. |
| `"long"` | Requests extended retention where supported: `cache_control: {"type": "ephemeral", "ttl": "1h"}` on Messages and OpenRouter, and `prompt_cache_retention: "24h"` on OpenAI and Azure OpenAI. |
| `"none"` | Omits explicit cache directives and `prompt_cache_key`, while still sending pure routing affinity headers (such as `x-session-id` or `x-affinity`) when `session_id` is set. |

## Provider mapping

Republic maps `session_id` and `cache_retention` by provider and API format rather than by model name:

| Provider | Formats | Affinity headers | Request body fields |
| --- | --- | --- | --- |
| `openai`, `azure-openai` | `chat`, `responses` | — | `prompt_cache_key`, plus `prompt_cache_retention: "24h"` for `"long"` |
| `anthropic` | `messages` | — | `cache_control` for `"short"` / `"long"` |
| `openrouter` | `chat`, `responses`, `messages` | `x-session-id` | `cache_control` for `"short"` / `"long"` |
| `codex` | `responses` | `session-id`, `thread-id` | `prompt_cache_key` |
| `magpie` | `chat`, `responses`, `messages` | `x-session-id` | `prompt_cache_key` (`chat`, `responses`); `cache_control` (`messages`) |
| `grok` | `chat`, `responses` | `x-grok-conv-id` | `prompt_cache_key` (`responses`) |
| `mistral` | `chat` | `x-affinity` | `prompt_cache_key` |
| `moonshot` | `chat` | — | `prompt_cache_key` |

Other built-in providers (such as `deepseek`, `google`, `minimax`, `ollama`, `together`, `zai`, `github-copilot`, and `OpenAICompatible`) rely on implicit prefix caching or ignore these options.

## Overriding generated fields

Explicit `extra_body` fields take precedence over any field derived from `session_id` or `cache_retention`:

```python
response = await model.chat(
    messages,
    session_id=session_id,
    cache_retention="short",
    extra_body={"prompt_cache_key": "custom-cache-key"},
)
```

## Custom providers and formats

Two orthogonal hooks let custom providers and service dialects add their own caching or affinity behavior without mutating shared provider state:

- Override `ChatApiFormat.cache_fields(request)` for wire-protocol cache fields merged before `extra_body`.
- Override `Provider._prepare_request(api_format, chat_request, http_request)` to attach per-request headers or provider-specific body fields by returning a new `HttpRequest` with `dataclasses.replace()`.

```python
from dataclasses import replace

from republic.formats import ApiFormat, ChatRequest, HttpRequest
from republic.providers import OpenAICompatible


class MyGateway(OpenAICompatible):
    name = "my-gateway"
    DEFAULT_API_BASE = "https://gateway.example/v1"

    def _prepare_request(
        self, api_format: ApiFormat, chat_request: ChatRequest, http_request: HttpRequest
    ) -> HttpRequest:
        session_id = chat_request.options.get("session_id")
        if not session_id:
            return http_request
        return replace(http_request, headers={**http_request.headers, "x-session-id": session_id})
```

See [service dialects](../reference/provider-api.md#service-dialects) for registering custom formats and providers.
