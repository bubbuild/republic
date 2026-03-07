from __future__ import annotations

import argparse
import os

from republic import LLM, login_openai_codex_oauth, openai_codex_oauth_resolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate with OpenAI Codex OAuth and run a simple Republic chat.",
    )
    parser.add_argument(
        "--login-only",
        action="store_true",
        help="Run OAuth login and persist tokens without sending a chat request.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("REPUBLIC_CODEX_MODEL", "openai:gpt-5-codex"),
        help="Model to use after login.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain tape-first workflows in one sentence.",
        help="Prompt to send after login.",
    )
    return parser.parse_args()


def prompt_for_redirect(authorize_url: str) -> str:
    print("Open this URL in your browser and complete the sign-in flow:\n")
    print(authorize_url)
    print("\nPaste the full callback URL (or the authorization code) here.")
    return input("> ").strip()


def main() -> None:
    args = parse_args()

    tokens = login_openai_codex_oauth(
        prompt_for_redirect=None,
    )
    print("login: ok")
    print("account_id:", tokens.account_id or "-")

    if args.login_only:
        return

    llm = LLM(
        model=args.model,
        api_key_resolver=openai_codex_oauth_resolver(),
    )
    out = llm.chat(args.prompt)

    if out.error:
        print("error:", out.error.kind, out.error.message)
        return
    print("text:", out.value)


if __name__ == "__main__":
    main()
