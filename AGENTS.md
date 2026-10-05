# Project instructions

Republic is a Python provider library under reconstruction. Keep gateways, agent loops, and tool execution outside the package. Use English in repository files and GitHub contributions.

Use the prepared uv environment. Run `uv run prek run --all-files`, `uv run ty check`, and affected behavior tests before publishing implementation changes. Main checks Python 3.11–3.14 and builds the package.

For fixes, use a task-named branch and the prepared Git identity. Publish the candidate PR, then start native CI explicitly when using the workflow token:

```bash
gh workflow run main.yml --ref BRANCH -f number=PR_NUMBER -f head=CANDIDATE_SHA
```

Use the full candidate SHA for dispatch. Link the candidate and pending checks in the requested reply; do not wait for the Landing feedback job or its enclosing workflow. A push made with the workflow token does not start CI automatically.
