"""The Gemini Decisions API format used by Google's Gemini Decision models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from republic._response import TokenUsage
from republic.decisions import Choice, DecisionResponse, JSONValue, Noul, Question, Score
from republic.errors import APIResponseError

from ._base import DecisionApiFormat, HttpRequest
from .system_one import _ANSWERS


class GeminiDecisionFormat(DecisionApiFormat):
    name = "gemini_decision"

    def decision_request(self, model: str, state: JSONValue, questions: Mapping[str, Question]) -> HttpRequest:
        return HttpRequest(
            "/decisions",
            {
                "model": model,
                "input": _text(state),
                "questions": {question_id: _question(question) for question_id, question in questions.items()},
            },
        )

    def parse_decision(self, data: Mapping[str, Any]) -> DecisionResponse:
        feedback = data.get("prompt_feedback") or {}
        if reason := feedback.get("block_reason"):
            message = feedback.get("block_reason_message")
            raise APIResponseError(f"Prompt blocked: {reason}" + (f": {message}" if message else ""))
        usage = data.get("usage") or {}
        return DecisionResponse(
            answers=_ANSWERS.validate_python(data.get("answers") or {}),
            token_usage=TokenUsage(
                input_tokens=usage.get("total_input_tokens", 0),
                output_tokens=usage.get("total_output_tokens", 0),
                cached_tokens=usage.get("total_cached_tokens", 0),
            ),
            model=data.get("model"),
        )


def _text(value: JSONValue) -> str:
    """Gemini reads strings as text and objects as content blocks, so JSON state is sent as text."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _question(question: Question) -> dict[str, Any]:
    body: dict[str, Any] = {"type": question.kind, "instructions": _text(question.instructions)}
    match question:
        case Noul(criteria=None):
            pass
        case Noul(criteria=criteria):
            body["criteria"] = {
                f"{outcome}_description": _text(criteria[outcome])
                for outcome in ("true", "false")
                if criteria.get(outcome) is not None
            }
        case Choice(criteria=Mapping() as criteria):
            # Every option needs a description; a bare option describes itself.
            body["criteria"] = {option: option if text is None else _text(text) for option, text in criteria.items()}
        case Choice(criteria=options):
            body["criteria"] = {option: option for option in options}
        case Score(criteria=levels):
            body["criteria"] = [_text(level) for level in levels]
    return body
