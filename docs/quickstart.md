# Make your first request

Make a model call, read its response, and display text as it arrives from a second request. You need Python 3.11 or later, an OpenAI API key, and access to the model you choose.

## Install Republic

In a virtual environment, install Republic:

```sh
python -m pip install republic
```

## Supply your key

In your shell, set:

```sh
export REPUBLIC_OPENAI_API_KEY="your-api-key"
```

Republic reads this variable when it creates the provider. You can also pass `api_key=` to `get_model()` if your application already loads credentials.

## Send a message and stream a response

Save this as `hello.py`. Replace `gpt-6-sol` if your account uses a different model.

```python
import asyncio

import republic
from republic.events import TextDelta


async def main():
    model = republic.get_model("openai:gpt-6-sol")

    response = await model.chat("Explain a tool call in one sentence.")
    print(response.text)
    print("Tokens:", response.token_usage.total_tokens)

    async with model.stream("Explain how an agent uses a tool result.") as stream:
        async for event in stream:
            if isinstance(event, TextDelta):
                print(event.chunk, end="", flush=True)
        print()
        print("Tokens:", stream.response.token_usage.total_tokens)


asyncio.run(main())
```

Run it:

```sh
python hello.py
```

The first answer appears after `chat()` finishes. The second arrives in text chunks, followed by its token count. The wording and counts will vary. These are two independent requests; a model does not retain earlier messages unless you supply conversation history.

The `async with` block closes the stream. Consume the iterator before reading `stream.response`; reading it early raises `StreamNotFinishedError`. The final response has the same shape as the response from `chat()`.

## Try another provider

With a Google API key in `REPUBLIC_GOOGLE_API_KEY`, replace the model construction line in `hello.py`:

```python
model = republic.get_model("google:gemini-flash-latest")
```

Run the script again. The request methods, text events, and usage fields are the same. Each service still determines which models and features your account can use; see the [provider directory](providers/index.md).

The provider directory offers [API-key setup](providers/index.md#use-an-api-key) and [account login](providers/index.md#use-an-account-login) paths. The [authentication guide](guides/authentication.md) explains explicit authorization, credential storage, and renewal.

To build on a model call, [write a minimal agent](guides/minimal-agent.md) that reads local project files and returns tool results. For a typed response, see [structured output](guides/structured-output.md).
