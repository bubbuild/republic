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

`republic.tool(call, ...)` builds a `tool` message that keeps the original call with its output, including the ID needed to associate the result with that call. The output may mix text, images, audio, videos, and documents, such as `republic.tool(call, "Captured", republic.image("shot.png"))`; pass `is_error=True` to report a failed call. Media support in tool results depends on the provider and API format: the Gemini format accepts only inline media there, and OpenAI's own Chat Completions service accepts only text in tool messages, although some compatible gateways accept images. `response.message` also preserves reasoning blocks, signed calls, and other provider data needed for the next turn. Replacing it with `response.text` loses that state.

The second request completes this round without offering more tools. To allow another round, supply the schemas again and repeat the sequence, as the minimal agent does. With an attached history object, the first assistant message is already recorded; send only the new results instead of appending the same conversation twice.

During streaming, `ToolCallDelta` carries argument fragments. Use `ToolCallReady` or the completed response when you need a finished call. Preserve its metadata when returning the result.

## Use built-in tools

Built-in tools in `republic.tools` run at the provider, so there is no call to execute and no result to send back. Pass them in `tools=`, alone or next to function tools:

```python
import asyncio

import republic
from republic.tools import UserLocation, WebSearch


async def main():
    model = republic.get_model("openai:gpt-6-sol")
    search = WebSearch(allowed_domains=["python.org"], user_location=UserLocation(country="US"))
    response = await model.chat("Which Python release is the newest?", tools=[search])
    print(response.text)
    for citation in response.citations:
        print(citation.url, citation.title)
    for call in response.builtin_tool_calls:
        print(call.name, call.input)


asyncio.run(main())
```

Each API format maps a built-in tool to its own equivalent:

| Tool | `responses` | `messages` | `gemini` | `chat` |
| --- | --- | --- | --- | --- |
| `WebSearch` | Yes, without `max_uses` or `blocked_domains` | Yes | Yes, without settings | Yes, `user_location` only |
| `WebFetch` | No | Yes | Yes, without settings | No |
| `CodeExecution` | Yes | Yes | Yes | No |
| `ImageGeneration` | Yes | No | No | No |

An unsupported tool or setting raises `UnsupportedFeatureError` before the request is sent. The model must also support the tool; the service rejects it otherwise.

`response.builtin_tool_calls` lists the `BuiltinToolCall` records the provider reports. Portable tools are named `web_search`, `web_fetch`, `code_execution`, or `image_generation`; `input` and `output` keep the provider's shapes. Sources the provider reports as citations appear in `response.citations`; Gemini's URL context tool reports the retrieved URLs in `builtin_tool_calls` instead. Generated images appear in `response.image_parts`. When streaming, the same data arrives as `BuiltinToolCallReady`, `CitationAdded`, and `ImageReady` events.

A long built-in tool turn may stop with `finish_reason == "pause"`. Send `response.message` back with the same tools to let the model continue. This continues the example above, inside `main()`, with its `model` and `search`:

```python
task = "Which Python release is the newest?"
messages = [task]
response = await model.chat(messages, tools=[search])
while response.finish_reason == "pause":
    messages.append(response.message)
    response = await model.chat(messages, tools=[search])
```

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

For Responses custom tools, read the `ProviderData` items in `response.message.parts`. Execute the input in your application, then return `custom_tool_call_output` with the original `call_id`. Keep the full `response.message` in the conversation, as shown, to preserve provider state. Use `republic.tool()` for function-tool results.

When streaming, consume the stream first, then handle custom calls from `stream.response.message.parts` in the same way.
