# Contributing to `republic`

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

You can contribute in many ways:

# Types of Contributions

## Report Bugs

Report bugs at https://github.com/bubbuild/republic/issues

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs.
Anything tagged with "bug" and "help wanted" is open to whoever wants to implement a fix for it.

## Implement Features

Look through the GitHub issues for features.
Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

## Write Documentation

republic could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at https://github.com/bubbuild/republic/issues.

If you are proposing a new feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

# Get Started!

Ready to contribute? Here's how to set up `republic` for local development.
Please note this documentation assumes you already have `uv` and `Git` installed and ready to go.

1. Fork the `republic` repo on GitHub.

2. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/republic.git
```

3. Now we need to install the environment. Navigate into the directory

```bash
cd republic
```

Then, install and activate the environment with:

```bash
uv sync
```

4. Install prek to run linters/formatters at commit time:

```bash
uv run prek install
```

5. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

6. Don't forget to add test cases for your added functionality to the `tests` directory.

7. When you're done making changes, check that your changes pass the formatting tests.

```bash
make check
```

Now, validate that all unit tests are passing:

```bash
make test
```

9. Before raising a pull request you should also run tox.
   This will run the tests across different versions of Python:

```bash
tox
```

This requires you to have multiple versions of python installed.
This step is also triggered in the CI/CD pipeline, so you could also choose to skip this step locally.

10. Commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

11. Submit a pull request through the GitHub website.

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.

## Documentation and releases

Run `make docs-test` to check the site or `make docs` to preview it locally.

Pull requests target `main`. Documentation deploys to https://getrepublic.org on pushes to `main`; the Docs workflow also supports manual dispatch. Publishing a GitHub release builds its tagged revision and uploads it to PyPI using the repository's `PYPI_TOKEN` secret. Package versions come from Git tags.

## Dependency maintenance with Landing

`Landing duty` runs dependency maintenance on Mondays at 02:23 UTC and through manual dispatch of `landing-duty.yml`. It uses the public `bubbuild/landing@main` Action API, following the [Landing duty reference at 9014baa](https://github.com/bubbuild/landing/blob/9014baa64f82893dc194a3f1e31391737f68c426/.github/workflows/landing-duty.yml). This adds maintenance capability; enabling it does not itself change dependency constraints or the lockfile.

Configure `LANDING_MODEL` and the `LANDING_API_KEY` secret, with optional `LANDING_API_BASE`, `LANDING_COMPLETION_ARGS` and `LANDING_TRUST` variables. The workflow prepares uv, project dependencies, gh 2.102.0, pinned documentation-writer, piglet, friendly-python and humanizer skills, and Git authentication. Landing provides its built-in dependency, repair and issue skills. No browser or MCP server is required for Republic's maintenance flow.

`LANDING_GITHUB_TOKEN` selects the publishing credential; otherwise the workflow uses `github.token`. The identity needs contents, issues, pull requests and Actions write access plus Dependabot alerts read access. The workflow requests `vulnerability-alerts: read` for its workflow token; separately supplied credentials need their own alert access. Repository settings must allow that identity to create PRs. Set `LANDING_GIT_NAME` and `LANDING_GIT_EMAIL` to match a custom publisher. `LANDING_ADMISSION_TOKEN` optionally supplies a separate admission credential. These are runner credentials, independent of a maintainer's local gh login.

Maintenance inspects open alerts and bounded Python runtime/development updates with uv, preserving version policy and pinning style. It reuses matching candidates, publishes new drafts on `deps/landing-*` branches, validates changes, and dispatches `main.yml` with `number` and the full candidate `head` SHA as described in `AGENTS.md`. It does not wait for Landing review. Maintainers retain merge and alert-dismissal authority.

Failed `Main` PR runs and non-default-branch manual candidate runs can trigger compatibility repair. Landing admits only one current, open, same-repository PR matching the failed head and an allowed author: Dependabot, Renovate, configured `LANDING_DEPENDENCY_UPDATE_ACTORS` logins (one per line), or the prepared publisher on `deps/landing-*` branches. It rechecks current CI before editing the original branch and preserves draft status. Set `LANDING_BUMP_ON_FAILURE=false` to disable automatic repair while retaining scheduled, manual and explicit `/landing bump` work. Default-branch failures still use the existing triage path; review and release workflows retain their existing behavior.

For real-time alerts, configure an external receiver to verify GitHub webhook signatures and forward the repository's alert number as a repository dispatch:

```json
{
  "event_type": "dependabot_alert",
  "client_payload": {"alert_number": 7}
}
```

`dependabot_alert` is a webhook event, not a native Actions trigger. This workflow does not deploy a receiver or subscribe one. Landing fetches the current alert: open alerts select dependency work; fixed or dismissed alerts reconcile tracked work without authorizing another upgrade. Unavailable alert access is unknown coverage, not evidence of no alerts. Scheduled and manual maintenance remain available without a receiver. Event listeners must be on the default branch before they receive these events.

See the [Landing GitHub contract](https://github.com/bubbuild/landing/blob/9014baa64f82893dc194a3f1e31391737f68c426/docs/guides/github.md), [Action inputs](https://github.com/bubbuild/landing/blob/9014baa64f82893dc194a3f1e31391737f68c426/action.yml), and [GitHub dispatch documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#repository_dispatch) for admission and delivery requirements.
