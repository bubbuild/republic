"""Typed questions and answers for decision models such as TypeSafe's System One models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias

from republic._response import TokenUsage

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionResponse",
    "JSONValue",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
]

JSONValue: TypeAlias = Any
"""Text, or a JSON object or array, for state, instructions, and criteria."""


@dataclass(frozen=True)
class Noul:
    """A yes/no question answered with the probability of yes.

    ``criteria`` optionally describes the outcomes as ``{"true": ..., "false": ...}``.
    """

    kind: ClassVar[str] = "noul"
    instructions: JSONValue
    criteria: Mapping[str, JSONValue] | None = None


@dataclass(frozen=True)
class Choice:
    """Pick one option. ``criteria`` maps each option to a description, or lists bare option names."""

    kind: ClassVar[str] = "choice"
    instructions: JSONValue
    criteria: Mapping[str, JSONValue] | Sequence[str]


@dataclass(frozen=True)
class Score:
    """Rate along ordered levels, from lowest to highest."""

    kind: ClassVar[str] = "score"
    instructions: JSONValue
    criteria: Sequence[JSONValue]


Question = Noul | Choice | Score


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    """Probability that the answer is yes, from 0 to 1."""
    type: Literal["noul"] = "noul"


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    type: Literal["choice"] = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    """Probability-weighted level index; may land between levels."""
    probabilities: Mapping[str, float]
    confidence: float
    legend: Mapping[str, str] = field(default_factory=dict)
    type: Literal["score"] = "score"


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DecisionResponse:
    """Answers keyed by question id. Each answer is also readable as an attribute."""

    answers: Mapping[str, Answer]
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None

    def __getattr__(self, name: str) -> Answer:
        if name.startswith("_"):
            # Keep copy and pickle protocol lookups from recursing into ``answers``.
            raise AttributeError(name)
        try:
            return self.answers[name]
        except KeyError:
            raise AttributeError(name) from None
