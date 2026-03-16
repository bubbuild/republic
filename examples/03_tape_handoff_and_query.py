from __future__ import annotations

import os

from republic import LLM, TapeContext


class MissingEnvVarError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Set {name} before running this example.")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingEnvVarError(name)
    return value


def main() -> None:
    api_key = require_env("LLM_API_KEY")
    model = os.getenv("REPUBLIC_CHAT_MODEL", "openrouter:openai/gpt-4o-mini")
    llm = LLM(model=model, api_key=api_key)

    tape = llm.tape("support-session")

    # Tape default context uses the latest anchor, so handoff first.
    tape.handoff("network_issue", state={"owner": "tier1"})
    out1 = tape.chat("Customer cannot connect to VPN. Give triage steps.", max_tokens=64)
    print("reply1:", out1)

    out2 = tape.chat("Also include DNS checks.", max_tokens=64)
    print("reply2:", out2)

    tape.handoff("billing_issue", state={"owner": "tier2"})
    out3 = tape.chat("Customer asks for refund process.", max_tokens=64)
    print("reply3:", out3)

    network_entries = tape.query.after_anchor("network_issue").all()
    print("after network_issue:", [entry.kind for entry in network_entries])

    # Switch context to read full history from the tape.
    tape.context = TapeContext(anchor=None)
    print("message_count:", len(tape.read_messages()))


if __name__ == "__main__":
    main()
