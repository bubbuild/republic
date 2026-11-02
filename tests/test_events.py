from __future__ import annotations

from typing import Any, assert_type

import republic
from republic.decisions import DecisionResponse
from republic.events import CallEvent, Completed, Event


def test_call_events_accept_non_chat_completions() -> None:
    embeddings = republic.EmbeddingResponse([[0.1]])
    images = republic.ImageResponse([])
    decision = DecisionResponse({})
    models = [republic.ModelInfo("test")]
    events: list[CallEvent] = [
        Completed(embeddings),
        Completed(images),
        Completed(decision),
        Completed(models),
    ]

    assert [event.response for event in events if isinstance(event, Completed)] == [
        embeddings,
        images,
        decision,
        models,
    ]


def test_stream_completion_keeps_chat_response_type() -> None:
    def completed_response(event: Event) -> republic.Response[Any]:
        assert isinstance(event, Completed)
        assert_type(event.response, republic.Response[Any])
        return event.response

    response: republic.Response[None] = republic.Response(republic.assistant("ok"))
    assert completed_response(Completed(response)) is response
