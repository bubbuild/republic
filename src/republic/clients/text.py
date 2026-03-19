"""Structured text helpers for Republic."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, ValidationError

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError
from republic.tape.context import TapeContext
from republic.tools.schema import schema_from_model


class _DecisionOutput(BaseModel):
    value: bool


class _ClassifyDecision(BaseModel):
    label: str


class TextClient:
    """Structured helpers built on chat tool calls."""

    def __init__(self, chat) -> None:
        self._chat = chat

    @staticmethod
    def _build_if_prompt(input_text: str, question: str) -> str:
        return dedent(
            f"""
            Here is an input:
            <input>
            {input_text.strip()}
            </input>

            And a question:
            <question>
            {question.strip()}
            </question>

            Answer by calling the tool with a boolean `value`.
            """
        ).strip()

    @staticmethod
    def _build_classify_prompt(input_text: str, choices_str: str) -> str:
        return dedent(
            f"""
            You are given this input:
            <input>
            {input_text.strip()}
            </input>

            And the following choices:
            <choices>
            {choices_str}
            </choices>

            Answer by calling the tool with `label` set to one of the choices.
            """
        ).strip()

    @staticmethod
    def _normalize_choices(choices: list[str]) -> list[str]:
        if not choices:
            raise RepublicError(ErrorKind.INVALID_INPUT, "choices must not be empty.")
        normalized = [choice.strip() for choice in choices]
        return normalized

    def if_(
        self,
        input_text: str,
        question: str,
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> bool:
        prompt = self._build_if_prompt(input_text, question)
        tool_schema = schema_from_model(_DecisionOutput, name="if_decision", description="Return a boolean.")
        calls = self._chat.tool_calls(prompt=prompt, tools=[tool_schema], tape=tape, context=context)
        return self._parse_tool_call(calls, _DecisionOutput, field="value")

    async def if_async(
        self,
        input_text: str,
        question: str,
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> bool:
        prompt = self._build_if_prompt(input_text, question)
        tool_schema = schema_from_model(_DecisionOutput, name="if_decision", description="Return a boolean.")
        calls = await self._chat.tool_calls_async(prompt=prompt, tools=[tool_schema], tape=tape, context=context)
        return self._parse_tool_call(calls, _DecisionOutput, field="value")

    def classify(
        self,
        input_text: str,
        choices: list[str],
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> str:
        normalized = self._normalize_choices(choices)
        choices_str = ", ".join(normalized)
        prompt = self._build_classify_prompt(input_text, choices_str)
        tool_schema = schema_from_model(_ClassifyDecision, name="classify_decision", description="Return one label.")
        calls = self._chat.tool_calls(prompt=prompt, tools=[tool_schema], tape=tape, context=context)
        label = self._parse_tool_call(calls, _ClassifyDecision, field="label")
        if label not in normalized:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                "classification label is not in the allowed choices.",
                details={"label": label, "choices": normalized},
            )
        return label

    async def classify_async(
        self,
        input_text: str,
        choices: list[str],
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> str:
        normalized = self._normalize_choices(choices)
        choices_str = ", ".join(normalized)
        prompt = self._build_classify_prompt(input_text, choices_str)
        tool_schema = schema_from_model(_ClassifyDecision, name="classify_decision", description="Return one label.")
        calls = await self._chat.tool_calls_async(prompt=prompt, tools=[tool_schema], tape=tape, context=context)
        label = self._parse_tool_call(calls, _ClassifyDecision, field="label")
        if label not in normalized:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                "classification label is not in the allowed choices.",
                details={"label": label, "choices": normalized},
            )
        return label

    def _parse_tool_call(self, calls: Any, model: type[BaseModel], *, field: str) -> Any:
        if not isinstance(calls, list) or not calls:
            raise RepublicError(ErrorKind.INVALID_INPUT, "tool call is missing.")
        call = calls[0]
        args = call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise RepublicError(
                    ErrorKind.INVALID_INPUT,
                    "tool arguments are not valid JSON.",
                    details={"error": str(exc)},
                ) from exc
        if not isinstance(args, dict):
            raise RepublicError(ErrorKind.INVALID_INPUT, "tool arguments must be an object.")
        try:
            payload = model(**args)
        except ValidationError as exc:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                "tool arguments failed validation.",
                details={"errors": exc.errors()},
            ) from exc
        value = getattr(payload, field, None)
        return value
