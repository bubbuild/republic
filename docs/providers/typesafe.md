# TypeSafe decision models

Use the `typesafe` provider to answer typed questions about a piece of state. It supports decision models through the System One API format.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_TYPESAFE_API_KEY`, and replace `MODEL_ID` with a decision model available to your account:

```python
import asyncio

import republic
from republic.decisions import Choice


async def main():
    model = republic.get_decision_model("typesafe:MODEL_ID")
    response = await model.decide(
        {"message": "How do I reset my password?"},
        questions={
            "category": Choice(
                instructions="Choose the support category for this message.",
                criteria=["account", "billing", "technical"],
            ),
        },
    )
    print(response.answers["category"])


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_TYPESAFE_API_KEY` when the provider is created. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key).

## Model support

`decide()` returns answers under the question IDs you supplied. Questions can request a choice, a yes/no probability, or a score using `Choice`, `Noul`, or `Score` from `republic.decisions`.

This provider supports decision models only. Use `get_decision_model()`; chat and embedding models are unavailable.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
