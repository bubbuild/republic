from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx
import pytest


class FakeService:
    """Answers every request with a canned body and records what was sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._responses: list[httpx.Response] = []

    def reply_json(self, body: Any, *, status_code: int = 200) -> None:
        self._responses.append(httpx.Response(status_code, json=body))

    def reply_events(self, events: Iterable[Any]) -> None:
        lines = []
        for event in events:
            lines.append(f"data: {event if isinstance(event, str) else json.dumps(event)}")
            lines.append("")
        body = "\n".join(lines) + "\n"
        self._responses.append(httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


@pytest.fixture
def service() -> FakeService:
    return FakeService()
