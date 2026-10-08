# Republic

A Python library for AI providers.

> [!IMPORTANT]
> Under reconstruction. The API below may still change before the next release.

```python
import republic

model = republic.get_model("openai:gpt-6-sol")  # reads REPUBLIC_OPENAI_API_KEY
response = await model.chat("Hello, how are you?")
print(response.text, response.token_usage)

async with model.stream("Tell me a story") as stream:
    async for event in stream:
        if isinstance(event, republic.events.TextDelta):
            print(event.chunk, end="")
```

Built-in providers are `openai`, `anthropic`, `google`, `openrouter`, `typesafe`, `codex`, and `github-copilot`. Each provider lists the API formats it speaks in `SUPPORTED_API_FORMATS`. Chat models use `responses`, `messages`, `gemini`, or `chat`; embedding models use `embeddings` or `embed_content`; decision models use `system_one`. Within each kind, the first supported format in that order is used unless `api_format=` names another. Copilot defaults to `chat`; select another format according to the model's supported endpoints. Subclass a provider and call `republic.register_provider(MyProvider, "custom")` to add your own.

Tools are schemas only. Execute the calls yourself and send the results back, keeping the assistant message so reasoning state survives the round trip:

```python
weather = republic.Tool("get_weather", "Look up the weather", {"type": "object", "properties": {"city": {"type": "string"}}})
response = await model.chat("Weather in Paris?", tools=[weather])
results = [republic.tool_result(call, run_tool(call.name, call.args)) for call in response.tool_calls]
response = await model.chat(["Weather in Paris?", response.message, republic.assistant(tool_results=results)])
```

Built-in tools run on the provider's side and go in the same `tools=` list:

```python
from republic.tools import CodeExecution, WebSearch

response = await model.chat("What changed in Python 3.14?", tools=[WebSearch(), CodeExecution()])
print(response.text, response.citations, response.builtin_tool_calls)
```

`WebSearch`, `WebFetch`, `CodeExecution`, and `ImageGeneration` map to each API format's native tool; `NativeTool(api_format, definition)` passes any other provider tool through. Tool activity and results are kept in `response.message`, so sending it back continues the conversation, including after a `"pause"` finish.

`chat()` and `stream()` accept `tools`, `tool_choice`, `parallel_tool_calls`, `max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `stop`, `seed`, `reasoning_effort`, and `include_reasoning` (ask for readable reasoning where providers hide it). Each API format maps them to its own fields; options it cannot express raise `UnsupportedFeatureError`, and `extra_body` passes anything provider-specific. Providers accept `headers=` for beta flags or gateway attribution.

A response carries `text`, `reasoning`, `refusal`, `tool_calls`, `output` (with `output_schema=`), `images` (install `republic[image]`), `finish_reason`, `token_usage` (including reasoning and cached tokens), `id`, and `model`. Streams yield `TextDelta`, `ReasoningDelta`, `RefusalDelta`, `ToolCallDelta`, `ToolCallReady`, `ImageReady`, `UsageDelta`, and finally `Completed`.

Other entry points: `republic.image()` and `republic.video()` inputs, `republic.history.InMemoryHistory`, and `republic.get_embedding_model()` with `embed()` / `embed_many()`.

Services that speak a known API format with their own dialect plug in through format hooks. Subclass a format from `republic.formats` and return it from `Provider.select_api_format()`, which also receives the model name:

```python
from republic.formats import ChatFormat


class DeepSeekChat(ChatFormat):
    def reasoning_fields(self, effort, *, include_reasoning):
        return {"thinking": {"type": "disabled" if effort == "none" else "enabled"}} if effort else {}


class DeepSeek(republic.providers.OpenAICompatible):
    name = "deepseek"
    DEFAULT_API_BASE = "https://api.deepseek.com"

    def select_api_format(self, format_kind, model):
        chat = DeepSeekChat()
        return chat if isinstance(chat, format_kind) else super().select_api_format(format_kind, model)
```

Every chat format has `reasoning_fields()`; `chat` also has `max_tokens_fields()` and `reasoning_text()`. Providers accept `extra_body=` for fields every request needs; it deep-merges with each call's `extra_body`.

Decision models answer typed questions with calibrated probabilities instead of generating text:

```python
from republic.decisions import Choice, Noul

model = republic.get_decision_model("typesafe:jev-latest")
response = await model.decide(
    "My card was charged twice for one order.",
    questions={
        "department": Choice("Which team handles this?", {"billing": "charges and refunds", "shipping": "delivery"}),
        "wants_refund": Noul("Is the customer asking for money back?"),
    },
)
print(response.department.choice, response.wants_refund.noul)
```

[Pydantic AI](https://pydantic.dev/docs/ai/overview/) and [ai-python](https://github.com/vercel-labs/ai-python) include agent runtimes. [LiteLLM](https://docs.litellm.ai/docs/) and the [any-llm](https://github.com/mozilla-ai/any-llm)/[Otari](https://github.com/mozilla-ai/otari) ecosystem also provide gateways.

Republic stops at providers. Gateways, agent loops, and tool execution stay out. Application logic stays in your code.

## Authentication

`auth=` accepts a standard `httpx2.Auth` object and takes precedence over `api_key=` and API key environment variables. `republic.auth` exports that same `Auth` base, the fixed-header `HeaderAuth`, and Authlib's `OAuth2Auth`. Service-specific auth classes live in their provider modules and are exported from `republic.providers`.

```python
from republic.providers import CodexAuth, GitHubCLIAuth

# Reuse a ChatGPT login from codex login ($CODEX_HOME/auth.json).
codex = republic.get_model("codex:gpt-6-luna", auth=CodexAuth.from_file())
response = await codex.chat("Hello")

# Reuse the active gh auth login. A Copilot subscription and model access are required.
copilot = republic.get_model(
    "github-copilot:gpt-6-luna",
    api_format="responses",
    auth=GitHubCLIAuth(),
)
response = await copilot.chat("Hello")
```

Applications that already hold Codex credentials can pass them directly. The auth object sets both the bearer token and the ChatGPT account header:

```python
auth = CodexAuth(token, account_id=account_id)
model = republic.get_model("codex:gpt-6-luna", auth=auth)
response = await model.chat("Hello")
# auth.token contains the current OAuth token, including any refreshed credentials.
```

`token` is a standard OAuth token mapping with `access_token` and, when available, `refresh_token` and `expires_at`. Fresh token responses may use `expires_in`. The auth object copies the mapping, refreshes expiring tokens through Authlib, and exposes the current token as `auth.token`. It does not read or write credential files in this mode; the application owns persistence, including rotated refresh tokens. An expired token without a refresh token raises `AuthenticationError`.

`CodexAuth.from_file(auth_file=...)` reads Codex's **file** credential store (configure `cli_auth_credentials_store = "file"` before `codex login`). The file is read when the auth object is constructed and before each request. It refreshes expiring tokens through Authlib and atomically updates that same file with private permissions, preserving other fields. Reuse one auth instance for concurrent requests. Keyring and ephemeral Codex credentials are not read. Unknown expiry is not guessed, and authentication failures do not switch to another credential source. Login and consent remain in the existing CLI.

`Codex` supports only Responses. Its endpoint requires `stream=true` and `store=false`; `chat()` collects the ordinary Republic stream, retaining tool calls, opaque reasoning, history and structured output. It defaults `instructions` to `"You are Codex."` (override with `extra_body`) and requests encrypted reasoning for later turns. `max_tokens` raises `UnsupportedFeatureError` because this endpoint does not accept it. These request rules live in the provider, not in auth or the shared Responses format.

`GitHubCLIAuth(executable="gh", hostname="github.com")` asks the selected GitHub CLI for its current token on each request. It does not copy credentials or add a fallback chain. `GitHubCopilot` sends that token directly to Copilot; `chat`, `responses`, and `messages` use `/chat/completions`, `/responses`, and `/v1/messages`. Availability depends on the model and account; select `api_format` explicitly for models requiring Responses or Messages. This is the Copilot service, not the [retired GitHub Models API](https://github.blog/changelog/2026-07-30-github-models-is-now-retired/).

For an existing Copilot access token, pass `auth=OAuth2Auth(token)` using `OAuth2Auth` from `republic.auth`.

Both auth classes work with synchronous and asynchronous HTTPX clients. Plain `OAuth2Auth(token)` signs requests; the application remains responsible for refreshing that token. During requests, Codex refresh and CLI/file I/O run off the asyncio event loop.

Grok's [documented API authentication](https://docs.x.ai/overview) uses an API key. No Grok OAuth integration is included without a documented OAuth contract; its API key works with `OpenAICompatible` and the appropriate API base.

## Development

```sh
uv sync
make check
make test
make build
```

## License

[Apache 2.0](LICENSE).
