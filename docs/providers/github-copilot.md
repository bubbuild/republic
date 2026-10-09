# GitHub Copilot authentication

Use the `github-copilot` provider with a GitHub CLI login or Copilot Plugin credentials. Chat requests default to Chat Completions; Responses and Messages are also supported.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic). With an existing `gh auth login`, replace `MODEL_ID` with a model that supports Chat Completions for your account and run:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("github-copilot:MODEL_ID")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

### Use a GitHub CLI login

The default `GitHubCLIAuth()` reads `gh auth token` for each request and sends that credential directly to Copilot. The CLI also honors `GH_TOKEN` and `GITHUB_TOKEN`.

### Log in with GitHub CLI

To start a new CLI login inside an async function:

```python
from republic.providers import GitHubCLIAuth

auth = await GitHubCLIAuth.login()
```

Pass the result as `auth=` to `get_model()`. The GitHub CLI owns storage. `hostname=` and `executable=` select the GitHub host and executable.

### Use Copilot Plugin credentials

An existing Plugin GitHub token uses `auth=CopilotAuth(saved_token)`, imported from `republic.providers`. This path exchanges the credential for a Copilot inference token. Reuse the auth object to cache that token and renew it before expiry.

### Authorize the Copilot Plugin

For a new Plugin authorization, call this inside an async function:

```python
from republic.providers import CopilotAuth

auth = await CopilotAuth.login()
```

The helper displays a verification URL and code. Supply an async `on_authorize(url, code)` callback to present them in your own UI. Pass the returned object as `auth=` to `get_model()`.

### Store and refresh credentials

The GitHub CLI owns storage for CLI logins. For Plugin authorization, save `auth.github_token` in your credential store and restore it with `CopilotAuth(saved_token)`.

Republic renews the exchanged inference token. Your application owns the original Plugin credential's storage and renewal. GitHub CLI credentials use the direct path above, not this exchange.

## Model support

Use `chat()` for a complete response or `stream()` for events. Choose a format supported by the model. To select Responses, replace the model construction in the example with:

```python
model = republic.get_model("github-copilot:MODEL_ID", api_format="responses")
```

Model access depends on your subscription and organization policy. A successful GitHub login does not establish access to every Copilot model.

### Endpoint and header behavior

Plugin exchange selects the API origin returned by the service, which can replace the origin set through `api_base=`. Plugin client headers have defaults; `headers=` can override them.

For a GitHub App installation credential on the direct path, supply the issuing app's integration ID with `headers={"Copilot-Integration-Id": "<integration-id>"}`. See [configuration](../reference/configuration.md) for auth precedence.
