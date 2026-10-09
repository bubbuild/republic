# Provider API specification

## API overview

Examples containing `await` run inside an async function. Choose models available to your account; capabilities depend on the provider, model, and API format.

### Basic usage

```python
import republic
from republic.events import TextDelta

model = republic.get_model("openai:gpt-6-sol")

# Chat
response = await model.chat("Hello, how are you?")
print(response.text)

# Stream
async with model.stream("Hello, how are you?") as stream:
    async for event in stream:
        if isinstance(event, TextDelta):
            print(event.chunk)
print(stream.response.text)
```

Consume the stream before reading its final response or shortcuts such as `stream.output`. Reading them early raises `errors.StreamNotFinishedError`.

A clean EOF alone does not complete a stream. Chat Completions requires `[DONE]`; Responses requires `response.completed`, `response.incomplete`, or the `[DONE]` marker used by compatible services; Messages requires `message_stop` (Copilot also accepts `[DONE]`); Gemini requires a candidate's `finishReason`. A provider-reported output limit or paused turn is a terminal result, not a transport truncation.

If the stream ends without its completion signal, Republic raises `errors.StreamIncompleteError`, emits no `Completed` event and writes no conversation history. Any partial events already yielded remain available to the caller, but `stream.response` stays unavailable. Transport failures and provider error events also leave history unchanged. Custom `StreamParser` implementations must set `self.completed = True` on their terminal event, or override `finish()` to validate completion according to their protocol.

### Initialization

```python
# Use the API key and base URL from REPUBLIC_OPENAI_* environment variables.
model = republic.get_model("openai:gpt-6-sol")

# Change the environment variable prefix.
provider = republic.get_provider("openai", env_prefix="MY_CUSTOM_PREFIX")
model = provider.get_model("gpt-6-sol")

# Pass an API key and base URL loaded by the application.
model = republic.get_model("openai:gpt-6-sol", api_base=api_base, api_key=api_key)

# Pass an application-defined HTTPX2-compatible auth object.
model = republic.get_model("openai:gpt-6-sol", auth=MyAuth())

# Use a provider instance.
provider = republic.get_provider("openai", api_key=api_key)
model = provider.get_model("gpt-6-sol")
```

Here, `api_base`, `api_key`, and `MyAuth` are supplied by the application. A top-level constructor accepts `provider:model`; a provider instance accepts only the model name. See [configuration](configuration.md) for options and credential precedence, and [authentication](../guides/authentication.md) for API keys and account logins.

### Tool calls

`tool1` and `tool2` below are `republic.Tool` schemas.

```python
response = await model.chat("Hello, how are you?", tools=[tool1, tool2])
print(response.tool_calls)
```

Each call has an ID, name, raw JSON `arguments`, decoded `args`, and provider metadata. Preserve the call when returning its result.

### Structured input

```python
response = await model.chat([
    republic.system("You are a helpful assistant."),
    republic.user("Hello, how are you?"),
])
print(response.text)
```

Image and video inputs use paths, URLs, data URLs, or bytes. For raw bytes, supply `media_type=`. Select a format and model that accept the media; for example, a Gemini model with video support:

```python
model = republic.get_model("google:MODEL_ID")
response = await model.chat([
    republic.user(
        "Describe these inputs.",
        republic.image("path/to/image.png"),
        republic.video("path/to/video.mp4"),
    ),
])
print(response.text)
```

After executing tool calls from a response, return the results with the full assistant message. Here, `results` contains `republic.tool_result(call, output)` values created by the application:

```python
answer = await model.chat([
    "Hello, how are you?",
    response.message,
    republic.assistant(tool_results=results),
])
print(answer.text)
```

`response.message` preserves tool IDs and opaque provider state. See [tool round trips](../guides/tools.md) for a complete example.

### Structured output

Pass a Pydantic model or another type Pydantic can validate as `output_schema`. `MyOutputSchema` below is defined by the application:

```python
response = await model.chat("Hello, how are you?", output_schema=MyOutputSchema)
print(response.output)

async with model.stream("Hello, how are you?", output_schema=MyOutputSchema) as stream:
    async for event in stream:
        if isinstance(event, TextDelta):
            print(event.chunk)
print(stream.output)
```

The completed result is validated from the response's JSON text. A refusal leaves `output` as `None`; invalid output raises a Pydantic validation error. See [structured output](../guides/structured-output.md).

### Image output

Request image generation using a provider-run tool and a model that supports it:

```python
model = republic.get_model("openai:gpt-6-sol")
response = await model.chat("Generate an image of a tree.", tools=[republic.tools.ImageGeneration()])
print(response.text)
print(response.image_parts)
```

`response.image_parts` contains `republic.Image` values. With `republic[image]` installed, `response.images` decodes inline image data into Pillow images. The response API has no video-output accessor.

### Token usage

```python
response = await model.chat("Hello, how are you?")
print(response.token_usage)
print(response.token_usage.total_tokens)
```

Total tokens are input plus output tokens. Reasoning tokens are included in output; cached and cache-write tokens are included in input when reported.

## Providers

### Built-in providers

The registered names are `openai`, `anthropic`, `google`, `openrouter`, `typesafe`, `codex`, `github-copilot`, `grok`, `azure-openai`, `ollama`, `deepseek`, `moonshot`, `zai`, `together`, `mistral`, `minimax`, and `magpie`. `republic.all_providers()` returns every registered name, including custom providers, sorted. See [supported providers](../providers/index.md) for model kinds, formats, and authentication paths.

### Listing models

```python
provider = republic.get_provider("openai")
for model in await provider.list_models():
    print(model.id, model.display_name)
```

`list_models()` returns `republic.ModelInfo` values for every page of the service's model list. Pass `model.id` to `get_model()` or another model getter; `model.raw` keeps the service's entry, such as context length or pricing. Codex has no model list and raises `errors.UnsupportedFeatureError`.

A custom provider sets `MODELS_PATH` relative to `api_base`, or `None` when the service has no list. Override `_parse_models()` for another response shape and `_models_params()` for pagination.

### Custom providers

Subclass a provider and register a unique name. Set the endpoint and formats to match the service:

```python
from republic.providers import OpenAI


class MyCustomProvider(OpenAI):
    DEFAULT_API_BASE = "https://api.example.com/v1"
    SUPPORTED_API_FORMATS = ("responses", "messages", "chat")


republic.register_provider(MyCustomProvider, "custom")
model = republic.get_model("custom:gpt-6-sol")
```

### Service dialects

Services that speak a known format with their own fields plug in through format hooks. Subclass a format from `republic.formats` and set it as `CHAT_FORMAT` on an `OpenAICompatible` subclass, or return it from `Provider.select_api_format()`, which also receives the model name:

```python
from republic.formats import ChatFormat
from republic.providers import OpenAICompatible


class AcmeChat(ChatFormat):
    def reasoning_fields(self, effort, *, include_reasoning):
        return {"thinking": {"type": "disabled" if effort == "none" else "enabled"}} if effort else {}


class Acme(OpenAICompatible):
    name = "acme"
    DEFAULT_API_BASE = "https://api.acme.example/v1"
    CHAT_FORMAT = AcmeChat()


republic.register_provider(Acme)
```

Every chat format has `reasoning_fields()`. `ChatFormat` also has `max_tokens_fields()`, `reasoning_text()`, `content_text()` for answer text in content parts, and `assistant_fields()` for fields added to assistant messages sent back, such as `reasoning_content`.

## API format support

Republic selects the first format of the requested model kind in the provider's `SUPPORTED_API_FORMATS`. An explicit `api_format=` takes precedence for formats of that kind. Each built-in provider's default is the first chat format in the [provider directory](../providers/index.md).

Use `api_format=` to select a supported format explicitly:

```python
model = republic.get_model("openai:gpt-6-sol", api_format="chat")

provider = republic.get_provider("openai", api_format="chat")
model = provider.get_model("gpt-6-sol")
```

A format not supported by the provider, such as `messages` for OpenAI, raises `errors.UnsupportedApiFormatError`. A preferred chat format does not prevent selecting an embedding or decision format for another model kind.

## Optional conversation history

```python
from republic.history import InMemoryHistory

model = republic.get_model("openai:gpt-6-sol", history=InMemoryHistory(max_entries=10))
response = await model.chat("Hello, how are you?")
print(response.text)
response = await model.chat("How about you?")
print(response.text)
```

History follows this protocol, exported from `republic.history`:

```python
from typing import Protocol

from republic import Message


class HistoryProtocol(Protocol):
    async def read(self) -> list[Message]:
        """Read the conversation history."""
        ...

    async def write(self, messages: list[Message]) -> None:
        """Append messages to the conversation history."""
        ...
```

With attached history, Republic reads earlier messages before each request and appends the new input and assistant response after completion. Without it, the caller supplies conversation history.

## Non-chat models

Embedding models return vectors:

```python
model = republic.get_embedding_model("openai:embedding-model")
response = await model.embed("Hello, how are you?")
print(response.vector)
```

`embed_many(texts)` returns one vector per input in `response.vectors`. Replace `embedding-model` with an available embedding model.

Decision models answer typed questions about state:

```python
from republic.decisions import Noul

model = republic.get_decision_model("typesafe:decision-model")
response = await model.decide(
    "The weather is clear and I have an hour free.",
    questions={"decision": Noul("Should I go for a walk?")},
)
print(response.answers["decision"])
```

Replace `decision-model` with an available decision model. `Choice`, `Noul`, and `Score` describe the questions; answers use the same IDs. Each provider exposes only the model kinds it supports.

Source: [Frost Ming's Republic specification](https://gist.github.com/frostming/1a454601e435016d8f89a17bd68ad1a4).
