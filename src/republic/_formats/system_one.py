"""The System One decision format introduced by TypeSafe's Jev models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

import pydantic

from republic._response import TokenUsage
from republic.decisions import Answer, Choice, DecisionResponse, JSONValue, Noul, Question, Score

from .base import DecisionApiFormat, HttpRequest

_ANSWERS: pydantic.TypeAdapter[dict[str, Answer]] = pydantic.TypeAdapter(
    dict[str, Annotated[Answer, pydantic.Field(discriminator="type")]]
)


class SystemOneFormat(DecisionApiFormat):
    name = "system_one"

    def decision_request(self, model: str, state: JSONValue, questions: Mapping[str, Question]) -> HttpRequest:
        return HttpRequest(
            "/systemone",
            {
                "model": model,
                "state": state,
                "questions": {question_id: _question(question) for question_id, question in questions.items()},
            },
        )

    def parse_decision(self, data: Mapping[str, Any]) -> DecisionResponse:
        usage = data.get("usage") or {}
        return DecisionResponse(
            answers=_ANSWERS.validate_python(data["answers"]),
            token_usage=TokenUsage(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
            model=data.get("model"),
        )


def _question(question: Question) -> dict[str, Any]:
    body = {"type": question.kind, "instructions": question.instructions}
    match question:
        case Noul(criteria=None):
            pass
        case Choice(criteria=Mapping() as criteria) | Noul(criteria=Mapping() as criteria):
            body["criteria"] = dict(criteria)
        case Choice(criteria=options):
            body["criteria"] = dict.fromkeys(options)
        case Score(criteria=levels):
            body["criteria"] = list(levels)
    return body
