# Build a minimal agent

This example reads a project's README and package metadata, then answers a question using tool results.

Use the [quickstart installation and API key](../quickstart.md). Save the following as `agent.py` in a Python project containing `README.md` and `pyproject.toml`. The model you select must support function tools.

## Define the tool and the loop

```python
import asyncio
from pathlib import Path

import republic

FILES = ("README.md", "pyproject.toml")
read_file = republic.Tool(
    "read_file",
    "Read the project's README or package metadata.",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "enum": list(FILES)}},
        "required": ["path"],
        "additionalProperties": False,
    },
)


def execute(call: republic.ToolCall) -> republic.Message:
    args = call.args
    if call.name != "read_file" or not isinstance(args, dict) or args.get("path") not in FILES:
        return republic.tool(call, "Unknown tool or file", is_error=True)
    try:
        content = Path(args["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return republic.tool(call, "Cannot read that project file", is_error=True)
    return republic.tool(call, content)


async def agent(model: republic.ChatModel, task: str) -> str:
    messages = [
        republic.system("Answer questions about this project. Read its files before making claims about it."),
        republic.user(task),
    ]
    for _ in range(8):
        response = await model.chat(messages, tools=[read_file])
        messages.append(response.message)
        if not response.tool_calls:
            return response.refusal or response.text
        results = [execute(call) for call in response.tool_calls]
        messages.extend(results)
    raise RuntimeError("The agent reached its limit of 8 model calls")


async def main():
    model = republic.get_model("openai:gpt-6-sol")
    answer = await agent(model, "What does this project do, and which Python versions does it support?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from that project directory:

```sh
python agent.py
```

The model can ask for one file, both files, or another read after seeing the first result. The program prints its final answer. Wording and the number of calls vary; a refusal is also a final response.

## Follow one turn

`model.chat()` returns an assistant message. If it contains tool calls, `execute()` reads the requested files and produces a result for every call. The next request contains the complete assistant message followed by those results. The model can now use what the tool returned.

Preserve `response.message`, including call IDs and provider state, before adding tool results. See [tool round trips](tools.md) for the individual messages.

The loop stops when there are no tool calls and raises after eight model calls. It permits only `README.md` and `pyproject.toml`. Run it from the project you want the model to read.

## Change the model without changing the loop

With an existing Codex file login, replace the model construction line in `main()`:

```python
model = republic.get_model("codex:gpt-6-luna")
```

The tool function and loop stay the same. Other services and credential paths are listed in the [provider directory](../providers/index.md). Choose a model that supports the tool requests used here.
