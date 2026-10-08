from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from republic.formats._sse import ServerSentEvent, iter_events


@pytest.mark.parametrize("separator", [False, True], ids=["eof", "blank-line"])
@pytest.mark.parametrize("end_marker", [None, "[DONE]"], ids=["ordinary-data", "end-marker"])
async def test_optional_end_marker(separator: bool, end_marker: str | None) -> None:
    async def lines() -> AsyncIterator[str]:
        yield "data: [DONE]"
        if separator:
            yield ""

    events = [event async for event in iter_events(lines(), end_marker=end_marker)]

    assert events == ([] if end_marker else [ServerSentEvent("message", "[DONE]")])
