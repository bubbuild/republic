# Return tool results to a model

A tool round trip has three parts: declare a schema, execute the calls in the response, then send the results with the original assistant message.

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
