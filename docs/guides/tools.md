# Return tool results to a model

Declare a tool, execute the returned calls, then send the results with the original assistant message.

## Use function tools

The [minimal agent tutorial](minimal-agent.md) defines a `read_file` tool and an `execute()` function in `agent.py`. Save that script first. This example imports them and performs one round, so you can see exactly which messages are sent:

```python
import asyncio

from agent import execute, read_file

import republic


async def main():
    model = republic.get_model("openai:gpt-6-sol")
    task = "Read pyproject.toml and report the supported Python versions."
    response = await model.chat(task, tools=[read_file], tool_choice=read_file)
    if not response.tool_calls:
        print(response.refusal or response.text)
        return

    results = [execute(call) for call in response.tool_calls]
    answer = await model.chat([task, response.message, *results])
    print(answer.refusal or answer.text)


asyncio.run(main())
```

Run this script from the same project directory as `agent.py`. `tool_choice=read_file` asks for that specific tool, so choose a model that supports explicit tool choice. A refusal can still produce a response with no calls.

`call.args` decodes the model's JSON arguments. The executor checks the tool name and permitted filenames before reading anything. A JSON decoding error propagates to the application; a schema alone does not validate the arguments locally.

`republic.tool(call, ...)` builds a `tool` message that keeps the original call with its output, including the ID needed to associate the result with that call. The output may mix text, images, audio, and videos, such as `republic.tool(call, "Captured", republic.image("shot.png"))`; pass `is_error=True` to report a failed call. Media support in tool results depends on the provider and API format: the Gemini format accepts only inline media there, and OpenAI's own Chat Completions service accepts only text in tool messages, although some compatible gateways accept images. `response.message` also preserves reasoning blocks, signed calls, and other provider data needed for the next turn. Replacing it with `response.text` loses that state.

The second request completes this round without offering more tools. To allow another round, supply the schemas again and repeat the sequence, as the minimal agent does. With an attached history object, the first assistant message is already recorded; send only the new results instead of appending the same conversation twice.

During streaming, `ToolCallDelta` carries argument fragments. Use `ToolCallReady` or the completed response when you need a finished call. Preserve its metadata when returning the result.

Provider-run tools, such as `republic.tools.WebSearch`, execute at the service and expose activity through `response.builtin_tool_calls`. They go in `tools=` too, but do not use the application's `execute()` function.

## Use native tools

`NativeTool` sends an API-specific definition unchanged. This OpenRouter example uses a custom tool with a Lark grammar. Set `REPUBLIC_OPENROUTER_API_KEY` and run inside an async function; `run_code(source)` is your application's executor.

```python
import republic
from republic.tools import NativeTool

code_exec = NativeTool(
    "responses",
    {
        "type": "custom",
        "name": "code_exec",
        "description": "Run Python source and return stdout.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": 'start: "print(1 + 1)"',
        },
    },
)

task = "Call code_exec with print(1 + 1), then reply with only its stdout."
async with republic.get_model("openrouter:openai/gpt-5.4") as model:
    response = await model.chat(task, tools=[code_exec], tool_choice="required", parallel_tool_calls=False)
    call = next(
        part.payload
        for part in response.message.parts
        if isinstance(part, republic.ProviderData)
        and part.api_format == "responses"
        and part.payload.get("type") == "custom_tool_call"
    )
    item = republic.ProviderData(
        "responses",
        {
            "type": "custom_tool_call_output",
            "call_id": call["call_id"],
            "output": run_code(call["input"]),
        },
    )
    output = republic.Message("assistant", parts=(item,))
    answer = await model.chat([task, response.message, output], tools=[code_exec], tool_choice="none")
    print(answer.text)
```

Remove `format` for freeform input. The model and service must support custom tools. This example expects one call; handle refusals and validate input in your executor.

Read custom calls from `response.message.parts`, not `response.tool_calls`. Return `custom_tool_call_output` through `ProviderData`; `republic.tool()` encodes a function result. When streaming, consume the stream before reading `stream.response.message.parts`. Custom input does not emit `ToolCallDelta` or `ToolCallReady`.
