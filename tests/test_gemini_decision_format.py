from __future__ import annotations

import pytest

import republic
from republic.decisions import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from tests.conftest import FakeService


def make_model(service: FakeService) -> republic.DecisionModel:
    return republic.get_decision_model("google:gemini-decision-1", api_key="key", http_client=service.client())


async def test_decide_sends_gemini_questions_and_reads_answers(service: FakeService) -> None:
    service.reply_json({
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12},
                "confidence": 0.88,
            },
            "urgency": {
                "type": "score",
                "score": 1.3,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3},
                "confidence": 0.7,
            },
            "wants_refund": {"type": "noul", "noul": 0.99, "confidence": 0.99},
        },
        "usage": {"total_input_tokens": 318, "total_output_tokens": 0, "total_cached_tokens": 100, "total_tokens": 318},
    })

    response = await make_model(service).decide(
        {"message": "My card was charged twice."},
        questions={
            "department": Choice("Which team handles this?", {"billing": "charges and refunds", "technical": None}),
            "urgency": Score("How urgent is this?", ["low", "medium", "high"]),
            "wants_refund": Noul("Is the customer asking for money back?", {"true": "Asks for a refund"}),
        },
    )

    assert service.requests[0].url == "https://generativelanguage.googleapis.com/v1beta/decisions"
    assert service.requests[0].headers["x-goog-api-key"] == "key"
    assert service.body() == {
        "model": "gemini-decision-1",
        "input": '{"message": "My card was charged twice."}',
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team handles this?",
                "criteria": {"billing": "charges and refunds", "technical": "technical"},
            },
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
            "wants_refund": {
                "type": "noul",
                "instructions": "Is the customer asking for money back?",
                "criteria": {"true_description": "Asks for a refund"},
            },
        },
    }
    assert response.department == ChoiceAnswer("billing", {"billing": 0.88, "technical": 0.12}, 0.88)
    assert isinstance(response.urgency, ScoreAnswer)
    assert response.urgency.legend["2"] == "high"
    assert response.wants_refund == NoulAnswer(0.99, confidence=0.99)
    assert response.token_usage == republic.TokenUsage(input_tokens=318, cached_tokens=100)
    assert response.model is None


async def test_structured_instructions_and_bare_options_become_text(service: FakeService) -> None:
    service.reply_json({"answers": {}, "usage": {}})

    await make_model(service).decide(
        "Hello",
        questions={
            "intent": Choice({"ask": "What does the user want?"}, ["greet", "complain"]),
            "spam": Noul("Is it spam?", {"true": "Unsolicited", "false": {"kind": "normal"}}),
            "tone": Score("How polite?", [{"level": "rude"}, "polite"]),
        },
    )

    assert service.body()["questions"] == {
        "intent": {
            "type": "choice",
            "instructions": '{"ask": "What does the user want?"}',
            "criteria": {"greet": "greet", "complain": "complain"},
        },
        "spam": {
            "type": "noul",
            "instructions": "Is it spam?",
            "criteria": {"true_description": "Unsolicited", "false_description": '{"kind": "normal"}'},
        },
        "tone": {"type": "score", "instructions": "How polite?", "criteria": ['{"level": "rude"}', "polite"]},
    }


async def test_blocked_prompt_raises(service: FakeService) -> None:
    service.reply_json({"answers": {}, "prompt_feedback": {"block_reason": "safety", "block_reason_message": "unsafe"}})

    with pytest.raises(republic.errors.APIResponseError, match="Prompt blocked: safety: unsafe"):
        await make_model(service).decide("Hello", questions={"spam": Noul("Is it spam?")})
