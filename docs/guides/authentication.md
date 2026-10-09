# API keys and account authentication

Use an API key, reuse an existing CLI login, or authorize an account explicitly. Authentication is a provider option: set `api_key=` or `auth=` when creating a model, then use its normal request methods.

The [provider directory](../providers/index.md) lists supported services and their default credential sources. Install Republic using the [quickstart](../quickstart.md#install-republic).

## Use an API key

Set the provider's environment variable before creating a model, or pass a key your application has already loaded. Replace `MODEL_ID` with a model available to your account:

```python
import os

import republic

model = republic.get_model("openai:MODEL_ID", api_key=os.environ["MY_APP_API_KEY"])
```

API-key setup is available for [OpenAI](../providers/openai.md), [Anthropic](../providers/anthropic.md), [Google Gemini](../providers/google.md), [OpenRouter](../providers/openrouter.md), and [TypeSafe](../providers/typesafe.md).

## Reuse a CLI login

| Existing login | Provider | Default source |
| --- | --- | --- |
| [Codex ChatGPT login](../providers/codex.md) | `codex` | Codex credential file |
| [GitHub CLI login](../providers/github-copilot.md) | `github-copilot` | `gh auth token` |

For example, with an existing Codex file login:

```python
import republic

model = republic.get_model("codex:MODEL_ID")
```

To select another Codex file, pass `auth=CodexAuth.from_file(path)`, importing the class from `republic.providers`. [Configuration](../reference/configuration.md) describes credential precedence and read timing.

## Authorize an account

Call `.login()` explicitly when you need a new authorization, then pass the returned object as `auth=`. Auth classes are exported from `republic.providers`.

| Service | Login helper | Authorization interface |
| --- | --- | --- |
| [Codex](../providers/codex.md#log-in-explicitly) | `CodexAuth.login()` | Codex CLI; optional device authorization |
| [GitHub CLI](../providers/github-copilot.md#log-in-with-github-cli) | `GitHubCLIAuth.login()` | GitHub CLI |
| [Copilot Plugin](../providers/github-copilot.md#authorize-the-copilot-plugin) | `CopilotAuth.login()` | GitHub device authorization; optional display callback |

Normal model requests reuse credentials without starting an interactive login. Account authorization and model access are separate: the service determines which models your account can use.

## Store and refresh credentials

Reuse auth objects across requests. Storage and renewal depend on how you obtained the credential:

| Credential source | Storage | Renewal |
| --- | --- | --- |
| API key | Your application | Replace the key when needed |
| Codex file login | CLI credential file; Republic saves refreshed tokens there | Republic refreshes tokens when a refresh token is available |
| `CodexAuth(token, account_id=...)` | Your application saves the updated `auth.token` | Republic refreshes tokens when a refresh token is available |
| GitHub CLI login | GitHub CLI | Republic reads `gh auth token` for each request |
| Copilot Plugin login | Your application saves `auth.github_token` | Republic renews the exchanged inference token; your application manages the original GitHub credential |

Restore a saved Plugin credential with `CopilotAuth(saved_token)`. The provider pages contain login examples and service-specific credential paths.

## Supply an auth object

`auth=` accepts the standard `httpx2.Auth` interface, also exported as `republic.auth.Auth`. A fixed bearer credential can use `HeaderAuth`:

```python
import os

from republic import get_model
from republic.auth import HeaderAuth

auth = HeaderAuth("Authorization", f"Bearer {os.environ['MY_APP_API_KEY']}")
model = get_model("openai:MODEL_ID", auth=auth)
```

Explicit `auth=` takes precedence over API-key authentication. `republic.auth` also exports Authlib's `OAuth2Auth`; provider-specific classes handle their own refresh and exchange rules. See [configuration](../reference/configuration.md) for the full precedence order.
