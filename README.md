# Republic

One Python interface for chat, streaming, tool calls, and structured output across AI providers. Republic stops at providers; agent loops, tool execution, and gateways stay in your code.

## Quickstart

Republic requires Python 3.11 or later.

```sh
python -m pip install "git+https://github.com/bubbuild/republic.git@dev"
export REPUBLIC_OPENAI_API_KEY="your-api-key"
```

```python
import asyncio

import republic
from republic.events import TextDelta


async def main():
    model = republic.get_model("openai:gpt-6-sol")

    response = await model.chat("Explain a tool call in one sentence.")
    print(response.text)

    async with model.stream("Explain how an agent uses a tool result.") as stream:
        async for event in stream:
            if isinstance(event, TextDelta):
                print(event.chunk, end="", flush=True)


asyncio.run(main())
```

A model is named `provider:model`. Switch services by changing the name, such as `anthropic:MODEL_ID`, `google:MODEL_ID`, or `deepseek:MODEL_ID`; each reads its key from `REPUBLIC_<PROVIDER>_API_KEY`. See the [provider directory](docs/providers/index.md) for every supported service.

## Documentation

- [Quickstart](docs/quickstart.md): a first request and a streaming response.
- [Providers](docs/providers/index.md): services, formats, and credentials.
- [Authentication](docs/guides/authentication.md): API keys, CLI logins, and OAuth.
- Guides: [tool use](docs/guides/tools.md), [a minimal agent](docs/guides/minimal-agent.md), and [structured output](docs/guides/structured-output.md).
- Reference: [configuration](docs/reference/configuration.md) and the [provider API](docs/reference/provider-api.md), including media input, embeddings, decision models, history, and custom providers.

## Development

```sh
uv sync
make check
make test
make build
```

See [Contributing](CONTRIBUTING.md) for the development workflow.

## License

[Apache 2.0](LICENSE).
