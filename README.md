# Republic

One Python interface for chat, streaming, tool calls, and structured output across AI providers. Republic stops at providers; agent loops, tool execution, and gateways stay in your code.

## Quickstart

Republic requires Python 3.11 or later.

```sh
python -m pip install republic
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

A model is named `provider:model`. Switch services by changing the name, such as `anthropic:MODEL_ID`, `google:MODEL_ID`, or `deepseek:MODEL_ID`; each reads its key from `REPUBLIC_<PROVIDER>_API_KEY`. See the [provider directory](https://getrepublic.org/providers/) for every supported service.

## Documentation

- [Quickstart](https://getrepublic.org/quickstart/): a first request and a streaming response.
- [Providers](https://getrepublic.org/providers/): services, formats, and credentials.
- [Authentication](https://getrepublic.org/guides/authentication/): API keys, CLI logins, and OAuth.
- Guides: [tool use](https://getrepublic.org/guides/tools/), [a minimal agent](https://getrepublic.org/guides/minimal-agent/), and [structured output](https://getrepublic.org/guides/structured-output/).
- Reference: [configuration](https://getrepublic.org/reference/configuration/) and the [provider API](https://getrepublic.org/reference/provider-api/), including media input, embeddings, decision models, history, and custom providers.

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
