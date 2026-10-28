# Limit concurrent calls

Services cap how many requests one key may run at once, and a burst of calls can exceed it. A [middleware](../reference/configuration.md#middleware) can hold a slot for each call, so only a fixed number run while the rest wait their turn.

This example caps chat and embedding calls together at four in flight. Use the [quickstart installation and API key](../quickstart.md).

```python
import asyncio

import republic
from republic.providers import Call


class CapacityLimiterMiddleware:
    def __init__(self, limit: int) -> None:
        self._semaphore = asyncio.Semaphore(limit)

    async def __call__(self, call: Call):
        async with self._semaphore:
            async for event in call.events:
                yield event


async def main():
    limiter = CapacityLimiterMiddleware(4)
    model = republic.get_model("openai:gpt-6-sol", middlewares=[limiter])
    embedder = republic.get_embedding_model("openai:embedding-model", middlewares=[limiter])

    topics = ["tides", "eclipses", "auroras", "rainbows", "comets", "monsoons"]
    answers = await asyncio.gather(*(model.chat(f"Explain {topic} in one sentence.") for topic in topics))
    vectors = await asyncio.gather(*(embedder.embed(answer.text) for answer in answers))
    for topic, answer, vector in zip(topics, answers, vectors, strict=True):
        print(topic, len(vector.vector), answer.text)


asyncio.run(main())
```

The six chat calls start together, but only four send requests at a time; the others wait inside the middleware until a slot is free. Sharing one `limiter` across both models gives them one combined limit. Create one per model or provider to limit them separately.

A slot is held from the first request until the call ends, including the waits between retries. A stream keeps its slot until it completes or its context exits, so close streams you stop reading. List the limiter first in `middlewares=` so other middleware, such as timing, starts only once a slot is free.

The semaphore belongs to the event loop that first waits on it. Create the limiter for the loop that runs your calls, as `main()` does here.
