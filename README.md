# Republic

A Python library for AI providers, with chat, streaming, tool calls, and structured output.

## Installation

Republic requires Python 3.11 or later.

```sh
python -m pip install "git+https://github.com/bubbuild/republic.git@dev"
```

## Get started

Set `REPUBLIC_OPENAI_API_KEY` to your API key, then run:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("openai:gpt-6-sol")
    response = await model.chat("Explain a tool call in one sentence.")
    print(response.text)


asyncio.run(main())
```

Choose a model available to your account. A model name has the form `provider:model`; `get_model()` creates the provider and selects its default chat API format.

## Choose a provider

With a Google API key in `REPUBLIC_GOOGLE_API_KEY`, replace the model construction line above:

```python
model = republic.get_model("google:gemini-flash-latest")
```

The same `chat()` call returns `response.text`. The [provider directory](docs/providers/index.md) lists supported services, model kinds, and formats. Start with [API-key setup](docs/providers/index.md#use-an-api-key) or [account login](docs/providers/index.md#use-an-account-login), depending on the credentials you have.

## Use Republic

The [quickstart](docs/quickstart.md) covers a complete request and a streaming response. [Build a minimal agent](docs/guides/minimal-agent.md) with a tool function and a loop that returns results to the model. [Tool use](docs/guides/tools.md) explains the messages in each round trip.

Providers accept `api_key=` or an `auth=` object. The [authentication guide](docs/guides/authentication.md) covers API keys, CLI logins, OAuth authorization, and credential storage and renewal, including [OpenRouter PKCE login](docs/providers/openrouter.md#authorize-with-oauth-pkce). See [configuration](docs/reference/configuration.md) for precedence, custom endpoints, and HTTP clients.

[Structured output](docs/guides/structured-output.md) returns values validated against your Python type. Republic also supports images and video as inputs, provider-run tools, embeddings, and decision models. Availability depends on the selected provider, model, and API format.

The [provider API specification](docs/reference/provider-api.md) covers model construction, messages, formats, history, and non-chat models.

## Development

From a repository checkout:

```sh
uv sync
make check
make test
make build
```

See [Contributing](CONTRIBUTING.md) for the development workflow.

## License

[Apache 2.0](LICENSE).
