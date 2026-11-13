"""Anthropic SDK regressions for public usage totals and streaming accounting.

Input tokens include cache reads/writes; output tokens include thinking tokens. Cumulative
message_delta counts replace earlier fields, while omitted fields retain message_start counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import republic
from republic.events import TextDelta, UsageDelta
from tests.conftest import FakeService

CASES = Path(__file__).with_suffix("") / "cases"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "message_delta_fields_response",
            republic.TokenUsage(
                input_tokens=59, output_tokens=8, reasoning_tokens=3, cached_tokens=7, cache_write_tokens=12
            ),
        ),
        (
            "message_delta_omitted_usage_response",
            republic.TokenUsage(input_tokens=40, output_tokens=8, cached_tokens=5, cache_write_tokens=10),
        ),
    ],
)
async def test_cumulative_usage_preserves_thinking_and_omitted_counts(
    service: FakeService, case: str, expected: republic.TokenUsage
) -> None:
    service.reply_bytes((CASES / f"{case}.txt").read_bytes(), content_type="text/event-stream")
    model = republic.get_model("anthropic:claude-sonnet-4-5", api_key="test-key", http_client=service.client())

    async with model.stream("hi") as stream:
        events = [event async for event in stream]

    assert stream.response.text == "Hello there!"
    assert "".join(event.chunk for event in events if isinstance(event, TextDelta)) == stream.response.text
    assert stream.response.token_usage == expected
    # A consumer accounting from public usage events must get the same totals, regardless
    # of how many events the parser emits or how it partitions those increments.
    usages = [event.usage for event in events if isinstance(event, UsageDelta)]
    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens"):
        assert sum(getattr(usage, field) for usage in usages) == getattr(expected, field)
