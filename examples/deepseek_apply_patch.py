"""Set REPUBLIC_DEEPSEEK_API_KEY, then run: uv run examples/deepseek_apply_patch.py."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import republic
from republic.tools import NativeTool

PATCH = """*** Begin Patch
*** Add File: hello.txt
+Hello from Republic.
*** End Patch"""

apply_patch = NativeTool("responses", {"type": "custom", "name": "apply_patch"})


def apply_requested_patch(patch: str) -> str:
    """Apply only the requested patch in a temporary directory."""
    if patch.strip() != PATCH:
        raise ValueError("Unexpected patch")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "hello.txt"
        path.write_text("Hello from Republic.\n")
        return f"Created hello.txt with contents: {path.read_text()}"


async def main():
    task = f"Call apply_patch with this exact patch, then report the tool result:\n{PATCH}"
    async with republic.get_model("deepseek:deepseek-flash", api_format="responses") as model:
        response = await model.chat(task, tools=[apply_patch], tool_choice="required", reasoning_effort="none")
        calls = [
            part.payload
            for part in response.message.parts
            if isinstance(part, republic.ProviderData)
            and part.api_format == "responses"
            and part.payload.get("type") == "custom_tool_call"
        ]
        if not calls:
            print(response.refusal or response.text)
            return

        (call,) = calls
        if call["name"] != "apply_patch":
            raise ValueError(f"Unexpected tool: {call['name']}")
        item = republic.ProviderData(
            "responses",
            {
                "type": "custom_tool_call_output",
                "call_id": call["call_id"],
                "output": apply_requested_patch(call["input"]),
            },
        )
        output = republic.Message("assistant", parts=(item,))
        answer = await model.chat(
            [task, response.message, output], tools=[apply_patch], tool_choice="none", reasoning_effort="none"
        )
        print(answer.refusal or answer.text)


if __name__ == "__main__":
    asyncio.run(main())
