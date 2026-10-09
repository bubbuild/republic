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

Built-in providers are `openai`, `anthropic`, `google`, `openrouter`, `typesafe`, `codex`, `github-copilot`, and `grok`. Each provider lists the API formats it speaks in `SUPPORTED_API_FORMATS`. Chat models use `responses`, `messages`, `gemini`, or `chat`; embedding models use `embeddings` or `embed_content`; decision models use `system_one`. Within each kind, the first format listed in `SUPPORTED_API_FORMATS` is used unless `api_format=` names another; the built-in providers list `responses`, then `messages`, then `chat`, except Copilot, which defaults to `chat`. Subclass a provider and call `republic.register_provider(MyProvider, "custom")` to add your own.

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

`auth=` accepts a standard `httpx2.Auth` object and overrides API-key authentication. `republic.auth` exports `Auth`, `HeaderAuth`, and Authlib's `OAuth2Auth`.

```python
codex = republic.get_model("codex:gpt-6-luna")
copilot = republic.get_model("github-copilot:gpt-6-luna", api_format="responses")
```

Codex defaults to its file login (`$CODEX_HOME/auth.json` or `~/.codex/auth.json`). Copilot defaults to `GitHubCLIAuth()`, which reads the current `gh auth login` for each request and sends it directly to the API. Explicit credentials take precedence.

`gh auth token` also honors `GH_TOKEN` and `GITHUB_TOKEN`. For GitHub App installation credentials, pass the issuing app's Copilot integration ID with `headers={"Copilot-Integration-Id": "<integration-id>"}`. These credentials use the direct path, not `CopilotAuth`'s Plugin token exchange.

To log in explicitly, use one of these methods from `republic.providers` and pass the returned object as `auth=`:

```python
from republic.providers import CodexAuth, GitHubCLIAuth, CopilotAuth

auth = await CodexAuth.login()      # Codex CLI; selects file storage for this login
auth = await GitHubCLIAuth.login()  # GitHub CLI; gh manages credential storage
auth = await CopilotAuth.login()    # Copilot Plugin device authorization
```

Codex accepts `device_auth=True` for device authorization. Both CLI methods accept `executable=`. Copilot displays a URL and code, or calls an async `on_authorize(url, code)` callback supplied by your application. Save `auth.github_token` in your credential store and reuse it with `CopilotAuth(saved_token)`. Normal model requests never start an interactive login.

For custom Codex credentials, pass `auth=CodexAuth.from_file(path)` or `auth=CodexAuth(token, account_id=...)` from `republic.providers`; persist `auth.token` yourself when supplying tokens.

For existing Plugin credentials, pass `auth=CopilotAuth(github_token)`; the caller owns their storage and renewal. Reuse the auth object to exchange, cache and renew Copilot inference tokens. Requests follow `endpoints.api`; `headers=` overrides Plugin headers. GitHub CLI credentials use the direct path above.

`grok:model` supports Chat and Responses using the official Grok CLI's xAI OAuth login by default. Use `await GrokAuth.login()` (optionally `device_auth=True`) or `GrokAuth.from_file(path)` from `republic.providers`. The default file is `$GROK_HOME/auth.json` or `~/.grok/auth.json`; `GROK_AUTH_PATH` overrides it. Expiring tokens are refreshed through Authlib and saved under the CLI's file lock. For caller-managed credentials, use `GrokAuth(token)` and persist `auth.token`; explicit `api_key=` and `auth=` take precedence.

OpenRouter supports `await OpenRouterAuth.login(on_authorize=authorize)` from `republic.providers`. Your async `authorize(url)` callback displays the URL and returns the code the user copies from OpenRouter. Pass the result as `auth=`; save `auth.api_key` and restore it with `OpenRouterAuth(saved_key)`. This [PKCE flow](https://openrouter.ai/docs/guides/overview/auth/oauth) issues an ordinary API key; existing keys also work with `api_key=`.

Codex supports Responses and rejects `max_tokens`. Copilot supports Chat, Responses, and Messages; select a format available to your model and account.

## Development

```sh
uv sync
make check
make test
make build
```

## License

[Apache 2.0](LICENSE).
