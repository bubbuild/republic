# Republic

A Python library for AI providers.

> [!IMPORTANT]
> Under reconstruction. This branch is a package scaffold with no provider API yet.

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
