import pydantic

import republic


@republic.register_provider
class Magpie(republic.providers.OpenAICompatible):
    SUPPORTED_API_FORMATS = ("responses", "messages", "chat", "system_one")
    DEFAULT_API_BASE = "http://127.0.0.1:3425/v1"

    name = "magpie"


@republic.register_provider
class DeepSeek(republic.providers.OpenAICompatible):
    SUPPORTED_API_FORMATS = ("chat",)
    DEFAULT_API_BASE = "https://api.deepseek.com"

    name = "deepseek"


class Profile(pydantic.BaseModel):
    name: str
    age: int
    email: str


async def test_decision():
    model = republic.get_decision_model("openrouter:~typesafe/jev-latest")

    questions: dict[str, republic.decisions.Question] = {
        "is_urgent": republic.decisions.Noul(
            instructions="Does this message convey urgency?",
            criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        ),
        "department": republic.decisions.Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        ),
        "frustration": republic.decisions.Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    }

    response = await model.decide(
        "Help! My payouts have been failing for 3 days.",
        questions=questions,
    )
    print(response.answers)


async def test_chat():
    model = republic.get_model("google:gemini-flash-latest")

    response = await model.chat("What is the capital of France?")
    print(response.text)
    print("--------")

    async with model.stream("给出五条今天的新闻", tools=[republic.tools.WebSearch()]) as stream:
        async for event in stream:
            if isinstance(event, republic.events.TextDelta):
                print(event.chunk, end="", flush=True)
        print()  # Ensure a newline after streaming output


async def test_reasoning():
    model = republic.get_model("deepseek:deepseek-flash")
    response = await model.chat("Explain the implications of quantum computing.", reasoning_effort="none")
    print(response.text)
    assert not response.reasoning, "Expected no reasoning when reasoning_effort is 'none'"  # noqa: S101

    reasoning_head = False
    answer_head = False
    async with model.stream("Explain the implications of quantum computing.", reasoning_effort="high") as stream:
        async for event in stream:
            if isinstance(event, republic.events.TextDelta):
                if not answer_head:
                    print("\n-----------------")
                    answer_head = True
                print(event.chunk, end="", flush=True)
            elif isinstance(event, republic.events.ReasoningDelta):
                if not reasoning_head:
                    print("Thinking >")
                    reasoning_head = True
                print(event.chunk, end="", flush=True)
        print()  # Ensure a newline after streaming output


async def test_structured_output():
    model = republic.get_model("magpie:openrouter/moonshotai/kimi-k3")
    response = await model.chat("Provide 5 fake users for testing.", output_schema=list[Profile])
    print(response.output)
    assert all(isinstance(item, Profile) for item in response.output), (  # noqa: S101
        "Expected output to be a list of Profile instances"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_structured_output())
