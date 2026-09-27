# Republic

Republic is being rebuilt on `dev` as a small LLM provider SDK for Python,
with Bub as its first integration target.

The intended scope is typed messages and results, single-call generation and
streaming, OpenAI and Anthropic protocol adapters, and concrete OAuth helpers
for GitHub Copilot, ChatGPT/Codex, and Grok using Authlib.

Agent loops, local tool execution, tape persistence, and context orchestration
belong to the caller. Tool definitions, calls, and results are model data.

## Current status

Only the rebuild baseline and delivery plan exist. The old implementation,
tests, documentation, and examples have been removed from this branch. No new
SDK API or provider support is implemented yet; this branch is not a usable
release and does not describe the currently published package.

Packaging, versioning, CI, linting, typing, test tooling, documentation tooling,
and release configuration are retained. Dependencies and Python compatibility
will be adjusted only when an implementation increment needs them.

Follow the [rebuild plan](docs/rebuild-plan.md) for the ordered increments,
acceptance evidence, and source references. Every increment should be a focused
Conventional Commit with its behavior tests and documentation.

## Development

The existing workflow is described in [CONTRIBUTING.md](CONTRIBUTING.md):

```bash
uv sync
make check
make test
make docs-test
```

At this empty baseline, package installation, typing, and test commands may
fail because `src/republic` and `tests` do not exist yet. Step 1 restores those
paths and verifies the package. No checks have been disabled to hide this state.

## License

[Apache License 2.0](LICENSE). Any code extracted from upstream must retain its
applicable copyright and license notices and record its source revision.
