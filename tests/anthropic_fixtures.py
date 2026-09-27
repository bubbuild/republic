"""Anthropic HTTP/SSE fixtures using the real async client and no credentials."""

import json
from types import TracebackType
from typing import Any, Self

import anthropic
import httpx

from republic import Message, Request, RequestOptions, TextPart
from republic.providers.anthropic import AnthropicMessages


def request() -> Request:
    return Request(
        model="fixture-model",
        messages=[Message(role="user", parts=[TextPart(text="Hello")])],
        options=RequestOptions(max_output_tokens=256),
    )


def message(content: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "resolved-model",
        "content": [{"type": "text", "text": "Hello"}] if content is None else content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        **extra,
    }


def sse(kind: str, **data: Any) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n".encode()


def start(**extra: Any) -> bytes:
    return sse("message_start", message=message([], stop_reason=None, **extra))


def block(content: dict[str, Any], index: int = 0) -> bytes:
    return sse("content_block_start", index=index, content_block=content)


def delta(kind: str, index: int = 0, **extra: Any) -> bytes:
    return sse("content_block_delta", index=index, delta={"type": kind, **extra})


def stop(index: int = 0) -> bytes:
    return sse("content_block_stop", index=index)


def finish(reason: str = "end_turn", **extra: Any) -> bytes:
    return sse(
        "message_delta", delta={"stop_reason": reason, "stop_sequence": None}, usage={"output_tokens": 3}, **extra
    )


class Wire:
    def __init__(self, replies: list[httpx.Response | Exception]) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        self.client = anthropic.AsyncAnthropic(
            api_key="fixture-key", base_url="https://unit.test/api", max_retries=3, http_client=self.http
        )
        self.provider = AnthropicMessages(client=self.client)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def payload(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.provider.aclose()
        await self.client.close()
