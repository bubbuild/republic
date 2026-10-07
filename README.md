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

Built-in providers are `openai`, `anthropic`, `google`, `openrouter`, and `typesafe`. Each provider lists the API formats it speaks in `SUPPORTED_API_FORMATS`. Chat models use `responses`, `messages`, `gemini`, or `chat`; embedding models use `embeddings` or `embed_content`; decision models use `system_one`. Within each kind, the first supported format in that order is used unless `api_format=` names another. Subclass a provider and call `republic.register_provider(MyProvider, "custom")` to add your own.

Tools are schemas only. Execute the calls yourself and send the results back, keeping the assistant message so reasoning state survives the round trip:

```python
weather = republic.Tool("get_weather", "Look up the weather", {"type": "object", "properties": {"city": {"type": "string"}}})
response = await model.chat("Weather in Paris?", tools=[weather])
results = [republic.tool_result(call, run_tool(call.name, call.args)) for call in response.tool_calls]
response = await model.chat(["Weather in Paris?", response.message, republic.assistant(tool_results=results)])
```

`chat()` and `stream()` accept `tools`, `tool_choice`, `parallel_tool_calls`, `max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `stop`, `seed`, `reasoning_effort`, and `include_reasoning` (ask for readable reasoning where providers hide it). Each API format maps them to its own fields; options it cannot express raise `UnsupportedFeatureError`, and `extra_body` passes anything provider-specific. Providers accept `headers=` for beta flags or gateway attribution.

A response carries `text`, `reasoning`, `refusal`, `tool_calls`, `output` (with `output_schema=`), `images` (install `republic[image]`), `finish_reason`, `token_usage` (including reasoning and cached tokens), `id`, and `model`. Streams yield `TextDelta`, `ReasoningDelta`, `RefusalDelta`, `ToolCallDelta`, `ToolCallReady`, `ImageReady`, `UsageDelta`, and finally `Completed`.

Other entry points: `republic.image()` and `republic.video()` inputs, `republic.history.InMemoryHistory`, and `republic.get_embedding_model()` with `embed()` / `embed_many()`.

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

## Development

```sh
uv sync
make check
make test
make build
```

## License

[Apache 2.0](LICENSE).
