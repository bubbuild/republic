# Read a typed result

Use `output_schema=` when application code needs a structured value. Republic asks the provider for the schema and validates the returned JSON with Pydantic. The result is available as `response.output`.

This example turns a short package description into typed project metadata. Use the [quickstart installation and API key](../quickstart.md) and a model that supports structured output.

```python
import asyncio

from pydantic import BaseModel

import republic


class ProjectInfo(BaseModel):
    name: str
    requires_python: str
    dependencies: list[str]


async def main():
    model = republic.get_model("openai:gpt-6-sol")
    response = await model.chat(
        "Extract the project metadata: demo requires Python >=3.11 and depends on httpx2 and pydantic.",
        output_schema=ProjectInfo,
    )
    if response.refusal is not None:
        print(response.refusal)
    elif response.output is not None:
        print(response.output.name)
        print(response.output.requires_python)
        print(response.output.dependencies)


asyncio.run(main())
```

On a successful response, `response.output` is a `ProjectInfo` instance. A refusal leaves it as `None`. JSON that fails the requested type raises `pydantic.ValidationError`; the application decides whether to report the error or retry.

Schema validation checks the returned structure and types, not whether every claim is true. When an answer depends on project files, give the model their contents or use the [minimal agent](minimal-agent.md) to read them.

The same `output_schema=` option works with `stream()`. Consume the stream before reading `stream.output`. The selected API format adapts the schema to its structured-output rules; model support still determines whether the service accepts the request.
