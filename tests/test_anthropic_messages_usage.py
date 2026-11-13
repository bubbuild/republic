"""Anthropic Messages usage normalization over upstream SDK streaming fixtures.

Republic's ``TokenUsage`` contract: ``input_tokens`` counts every prompt token including
cache reads and writes, ``output_tokens`` counts every generated token including reasoning
tokens, and ``reasoning_tokens`` reports the thinking portion when the provider sends it.
``message_delta`` usage is cumulative and replaces ``message_start`` usage field by field.
Streamed ``UsageDelta`` events add up to the final ``token_usage``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import republic
from republic.events import Completed, Event, TextDelta, UsageDelta
from tests.conftest import FakeService

CASES = Path(__file__).with_suffix("") / "cases"


def load_case(service: FakeService, name: str) -> None:
    """Queue one recorded SSE response body, byte for byte as upstream ships it."""
    path = CASES / f"{name}.txt"
    service.reply_bytes(path.read_bytes(), content_type="text/event-stream")


def make_model(service: FakeService) -> republic.ChatModel:
    return republic.get_model("anthropic:claude-sonnet-4-5", api_key="test-key", http_client=service.client())


async def stream(service: FakeService, name: str) -> tuple[republic.Response[None], list[Event]]:
    load_case(service, name)
    async with make_model(service).stream("hi") as stream:
        events = [event async for event in stream]
    return stream.response, events


def usage_deltas(events: list[Event]) -> list[republic.TokenUsage]:
    return [event.usage for event in events if isinstance(event, UsageDelta)]


async def test_message_delta_cumulative_usage_keeps_thinking_tokens(service: FakeService) -> None:
    """``output_tokens_details.thinking_tokens`` reaches ``reasoning_tokens``; nothing is double counted.

    The fixture's ``message_delta`` re-sends cumulative usage that supersedes
    ``message_start``: input 25 -> 40 uncached, cache writes 10 -> 12, cache reads 5 -> 7,
    output 1 -> 8, plus 3 thinking tokens.
    """
    response, events = await stream(service, "message_delta_fields_response")

    assert isinstance(events[-1], Completed)
    assert response.text == "Hello there!"
    assert "".join(event.chunk for event in events if isinstance(event, TextDelta)) == "Hello there!"
    assert response.token_usage == republic.TokenUsage(
        input_tokens=40 + 7 + 12,
        output_tokens=8,
        reasoning_tokens=3,
        cached_tokens=7,
        cache_write_tokens=12,
    )
    assert response.token_usage.total_tokens == 67
    assert usage_deltas(events) == [
        republic.TokenUsage(input_tokens=25 + 5 + 10, output_tokens=1, cached_tokens=5, cache_write_tokens=10),
        republic.TokenUsage(
            input_tokens=19, output_tokens=7, reasoning_tokens=3, cached_tokens=2, cache_write_tokens=2
        ),
    ]


async def test_message_delta_omitted_usage_preserves_message_start_counts(service: FakeService) -> None:
    """A ``message_delta`` that sends only ``output_tokens`` leaves the other counts untouched."""
    response, events = await stream(service, "message_delta_omitted_usage_response")

    assert isinstance(events[-1], Completed)
    assert response.text == "Hello there!"
    assert response.token_usage == republic.TokenUsage(
        input_tokens=25 + 5 + 10,
        output_tokens=8,
        cached_tokens=5,
        cache_write_tokens=10,
    )
    assert usage_deltas(events) == [
        republic.TokenUsage(input_tokens=25 + 5 + 10, output_tokens=1, cached_tokens=5, cache_write_tokens=10),
        republic.TokenUsage(output_tokens=7),
    ]


@pytest.mark.parametrize("case", ["message_delta_fields_response", "message_delta_omitted_usage_response"])
async def test_streamed_usage_deltas_sum_to_final_totals(service: FakeService, case: str) -> None:
    response, events = await stream(service, case)

    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens"):
        assert sum(getattr(usage, field) for usage in usage_deltas(events)) == getattr(response.token_usage, field)
