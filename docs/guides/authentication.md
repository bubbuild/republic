# Authentication

Republic supports both static provider keys and dynamic provider token resolution.

## Static API Keys

Use `api_key` when your process already has the provider token.

```python
from republic import LLM

llm = LLM(
    model="openrouter:openrouter/free",
    api_key="<OPENROUTER_KEY>",
)
```

For multi-provider fallback, pass a provider-to-token mapping.

```python
from republic import LLM

llm = LLM(
    model="openai:gpt-4o-mini",
    fallback_models=["anthropic:claude-3-5-sonnet-latest"],
    api_key={
        "openai": "<OPENAI_KEY>",
        "anthropic": "<ANTHROPIC_KEY>",
    },
)
```

## Dynamic Key Resolution

Use `api_key_resolver` when the token should be resolved lazily per provider.

```python
from republic import LLM


def resolve_api_key(provider: str) -> str | None:
    if provider == "openrouter":
        return "<OPENROUTER_KEY>"
    if provider == "openai":
        return "<OPENAI_KEY>"
    return None


llm = LLM(
    model="openrouter:openrouter/free",
    api_key_resolver=resolve_api_key,
)
```

This is the right hook when:

- Tokens come from a local credential store.
- Different providers use different auth flows.
- Tokens may expire and need refresh logic.

## OpenAI Codex OAuth

Republic exposes a dedicated login flow and resolver for OpenAI Codex-style OAuth.

```python
from republic import LLM, login_openai_codex_oauth, openai_codex_oauth_resolver

login_openai_codex_oauth(
    prompt_for_redirect=lambda authorize_url: input(
        f"Open this URL and paste callback URL:\n{authorize_url}\n> "
    ),
)

llm = LLM(
    model="openai:gpt-5.3-codex",
    api_key_resolver=openai_codex_oauth_resolver(),
)
print(llm.chat("Say hello in one sentence."))
```

`openai_codex_oauth_resolver()` reads `~/.codex/auth.json` by default, or
`$CODEX_HOME/auth.json` when `CODEX_HOME` is set. The resolver returns the current
OpenAI access token for provider `openai` and refreshes it automatically when
expiry is near.

See `examples/06_openai_codex_oauth.py` for a runnable example.

## GitHub Copilot OAuth

Republic also supports GitHub device-flow OAuth for the `github-copilot` provider.

```python
from republic import LLM, github_copilot_oauth_resolver, login_github_copilot_oauth

login_github_copilot_oauth(
    device_code_notifier=lambda verification_uri, user_code: print(
        f"Open {verification_uri} and enter code: {user_code}"
    ),
)

llm = LLM(
    model="github-copilot:gpt-4.1",
    api_key_resolver=github_copilot_oauth_resolver(),
)
print(llm.chat("Say hello in one sentence."))
```

`github_copilot_oauth_resolver()` reads the saved GitHub OAuth token from
`~/.config/republic/github_copilot_auth.json` by default, or
`$XDG_CONFIG_HOME/republic/github_copilot_auth.json` when `XDG_CONFIG_HOME` is set.

If no Republic auth file exists, the resolver also tries:

- `COPILOT_GITHUB_TOKEN`
- `GH_TOKEN`
- `GITHUB_TOKEN`
- `~/.config/gh/hosts.yml`
- `gh auth token`

The `github-copilot` backend is implemented through `github-copilot-sdk`, not
through Copilot private HTTP APIs. Republic maps Copilot tool requests into the
same `tool_calls(...)`, `run_tools(...)`, and `stream_events(...)` interfaces used
by other providers.

See `examples/07_github_copilot_oauth.py` for a runnable smoke test.

## Recommendation

Use this split by default:

- Simple scripts: `api_key="<TOKEN>"`
- Multi-provider apps: `api_key={...}` or `api_key_resolver=...`
- Local CLI integrations with existing auth state: provider-specific OAuth resolvers
