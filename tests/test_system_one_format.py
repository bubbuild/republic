from __future__ import annotations

import copy

import pytest

import republic
from republic.decisions import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from tests.conftest import FakeService


def make_model(service: FakeService) -> republic.DecisionModel:
    return republic.get_decision_model("typesafe:jev-latest", api_key="key", http_client=service.client())


async def test_decide_sends_typed_questions_and_reads_typed_answers(service: FakeService) -> None:
    service.reply_json({
        "model": "jev-1.13.0",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12},
                "confidence": 0.81,
            },
            "urgency": {
                "type": "score",
                "score": 1.3,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3},
                "confidence": 0.54,
            },
            "wants_refund": {"type": "noul", "noul": 0.99},
        },
        "usage": {"input_tokens": 318, "output_tokens": 34},
    })

    response = await make_model(service).decide(
        {"message": "My card was charged twice."},
        questions={
            "department": Choice("Which team handles this?", {"billing": "charges and refunds", "technical": None}),
            "urgency": Score("How urgent is this?", ["low", "medium", "high"]),
            "wants_refund": Noul("Is the customer asking for money back?", {"true": "Asks for a refund"}),
        },
    )

    assert service.requests[0].url == "https://api.typesafe.ai/v1/systemone"
    assert service.requests[0].headers["authorization"] == "Bearer key"
    assert service.body() == {
        "model": "jev-latest",
        "state": {"message": "My card was charged twice."},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team handles this?",
                "criteria": {"billing": "charges and refunds", "technical": None},
            },
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
            "wants_refund": {
                "type": "noul",
                "instructions": "Is the customer asking for money back?",
                "criteria": {"true": "Asks for a refund"},
            },
        },
    }
    assert response.department == ChoiceAnswer("billing", {"billing": 0.88, "technical": 0.12}, 0.81)
    assert isinstance(response.urgency, ScoreAnswer)
    assert response.urgency.legend["2"] == "high"
    assert response.wants_refund == NoulAnswer(0.99)
    assert response.token_usage == republic.TokenUsage(input_tokens=318, output_tokens=34)


async def test_choice_options_may_be_listed_without_descriptions(service: FakeService) -> None:
    service.reply_json({"answers": {}, "usage": {}})

    await make_model(service).decide(
        "Hello",
        questions={"intent": Choice("What does the user want?", ["greet", "complain"]), "spam": Noul("Is it spam?")},
    )

    assert service.body()["questions"] == {
        "intent": {
            "type": "choice",
            "instructions": "What does the user want?",
            "criteria": {"greet": None, "complain": None},
        },
        "spam": {"type": "noul", "instructions": "Is it spam?"},
    }


async def test_unknown_answer_attribute_raises(service: FakeService) -> None:
    service.reply_json({"answers": {"spam": {"type": "noul", "noul": 0.1}}})

    response = await make_model(service).decide("Hello", questions={"spam": Noul("Is it spam?")})

    with pytest.raises(AttributeError):
        _ = response.urgency
    assert copy.deepcopy(response) == response


def test_chat_providers_have_no_decision_format() -> None:
    with pytest.raises(republic.errors.UnsupportedApiFormatError, match="decision"):
        republic.get_decision_model("openai:gpt-6-sol")
