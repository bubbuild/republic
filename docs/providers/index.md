# Supported providers

Choose a service, then use an API key or an account login supported by that service. A model specification is `provider:model`; the provider selects the endpoint and credentials, and the model name is passed to the service.

## Use an API key

Start with [OpenAI](openai.md), [Anthropic](anthropic.md), [Google Gemini](google.md), [OpenRouter](openrouter.md), [Grok](grok.md#use-an-api-key), or [TypeSafe](typesafe.md). Each page shows the environment variable and a complete request.

## Use an account login

Reuse a [Codex ChatGPT login](codex.md), [GitHub CLI login for Copilot](github-copilot.md), or [Grok CLI login](grok.md). New authorization is also available through [Copilot Plugin device login](github-copilot.md#authorize-the-copilot-plugin) and [OpenRouter PKCE](openrouter.md#authorize-with-oauth-pkce).

See [authentication](../guides/authentication.md) for credential storage and renewal.

## Provider support

The first chat format in each row is the default. Formats describe the available protocols; feature support also depends on the model and your account.

| Service | Registered name | Model kinds | Chat formats | Default credentials |
| --- | --- | --- | --- | --- |
| [OpenAI](openai.md) | `openai` | Chat, embeddings | `responses`, `chat` | `REPUBLIC_OPENAI_API_KEY` |
| [Anthropic](anthropic.md) | `anthropic` | Chat | `messages` | `REPUBLIC_ANTHROPIC_API_KEY` |
| [Google Gemini](google.md) | `google` | Chat, embeddings | `gemini` | `REPUBLIC_GOOGLE_API_KEY` |
| [OpenRouter](openrouter.md) | `openrouter` | Chat, embeddings, decisions | `responses`, `messages`, `chat` | `REPUBLIC_OPENROUTER_API_KEY` |
| [TypeSafe](typesafe.md) | `typesafe` | Decisions | — | `REPUBLIC_TYPESAFE_API_KEY` |
| [Codex](codex.md) | `codex` | Chat | `responses` | Codex file login |
| [GitHub Copilot](github-copilot.md) | `github-copilot` | Chat | `chat`, `responses`, `messages` | GitHub CLI login |
| [Grok](grok.md) | `grok` | Chat | `responses`, `chat` | Grok file login |

## Select a model and format

Replace `MODEL_ID` with a model available to your account. To select a supported chat format explicitly:

```python
from republic import get_model

model = get_model("openai:MODEL_ID", api_format="chat")
```

Chat models use `get_model()` with `chat()` or `stream()`. Embedding models use `get_embedding_model()` with `embed()` or `embed_many()`. Decision models use `get_decision_model()` with `decide()`; TypeSafe supports this model kind only.

Explicit `auth=` and API keys take precedence over default CLI credentials. See [configuration](../reference/configuration.md) for the complete lookup order, custom endpoints, and HTTP client options.

For the shared API contracts, see the [provider API specification](../reference/provider-api.md).
