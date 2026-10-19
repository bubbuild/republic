#!/usr/bin/env bash
# Prepare Republic's maintenance skills; the runtime only reads the resulting root.
set -euo pipefail

directory=$1
mkdir -p "$directory/humanizer"
gh skill install github/awesome-copilot skills/documentation-writer --pin 143a3d976b3c1603cc8932984d5e1f28501cb5fc --dir "$directory" --force
gh skill install PsiACE/skills skills/piglet --pin 2265aed05caf199426a8062461e2c9901be996d8 --dir "$directory" --force
gh skill install PsiACE/skills skills/friendly-python --pin 2265aed05caf199426a8062461e2c9901be996d8 --dir "$directory" --force
# Native gh discovery does not support a repository-root SKILL.md.
gh api "repos/blader/humanizer/contents/SKILL.md?ref=225a6f39ac85f76ee48dbad772ea4abe4ed6c9d8" --header Accept:application/vnd.github.raw+json > "$directory/humanizer/SKILL.md"
gh api "repos/blader/humanizer/contents/LICENSE?ref=225a6f39ac85f76ee48dbad772ea4abe4ed6c9d8" --header Accept:application/vnd.github.raw+json > "$directory/humanizer/LICENSE"
