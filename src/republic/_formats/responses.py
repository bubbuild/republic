"""The OpenAI Responses format."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Mapping
from typing import Any

from republic._content import Image, Message, ProviderData, Text, Tool, Video
from republic._errors import APIResponseError
from republic._response import FinishReason
from republic.events import ImageReady, ReasoningDelta, RefusalDelta, TextDelta

from .base import (
    ChatApiFormat,
    ChatRequest,
    Delta,
    HttpRequest,
    ResponseInfo,
    StreamParser,
    ToolCallFragment,
    UsageReport,
    provider_payloads,
    unsupported_media,
)

_REASONING_TEXT_TYPES = frozenset({"summary_text", "reasoning_text"})
_INCOMPLETE_REASONS: dict[str | None, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


class ResponsesFormat(ChatApiFormat):
    name = "responses"

    def chat_request(self, request: ChatRequest, *, stream: bool) -> HttpRequest:
        request.reject(self.name, "stop", "seed", "top_k", "presence_penalty", "frequency_penalty")
        body: dict[str, Any] = {
            "model": request.model,
            "input": [item for message in request.messages for item in self._input_items(message)],
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    # The Responses API defaults to strict, which rejects most ordinary schemas.
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
        if (tool_choice := request.options.get("tool_choice")) is not None:
            body["tool_choice"] = (
                {"type": "function", "name": tool_choice.name} if isinstance(tool_choice, Tool) else tool_choice
            )
        if request.output_schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema.name,
                    "schema": request.output_schema.schema,
                }
            }
        body.update(
            request.renamed({
                "max_tokens": "max_output_tokens",
                "temperature": "temperature",
                "top_p": "top_p",
                "parallel_tool_calls": "parallel_tool_calls",
            })
        )
        reasoning: dict[str, Any] = {}
        if (effort := request.options.get("reasoning_effort")) is not None:
            reasoning["effort"] = effort
        if request.options.get("include_reasoning"):
            reasoning["summary"] = "auto"
        if reasoning:
            body["reasoning"] = reasoning
        if stream:
            body["stream"] = True
        return HttpRequest("/responses", request.body(body))

    def parse_chat(self, data: Mapping[str, Any]) -> Iterable[Delta]:
        _raise_for_failure(data)
        for item in data.get("output") or ():
            yield from _item_deltas(item, streamed=False)
        yield from _response_deltas(data)

    def stream_parser(self) -> StreamParser:
        return _ResponsesStreamParser()

    def _input_items(self, message: Message) -> list[Mapping[str, Any]]:
        match message.role:
            case "system":
                return [{"role": "system", "content": message.text}]
            case "user":
                return [{"role": "user", "content": [_user_part(part) for part in message.parts]}]
            case "assistant":
                items: list[Mapping[str, Any]] = provider_payloads(message, self.name)
                if message.text:
                    items.append({"role": "assistant", "content": message.text})
                items.extend(
                    {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                )
                return items
            case "tool":
                return [
                    {"type": "function_call_output", "call_id": result.call.id, "output": result.output}
                    for result in message.tool_results
                ]


class _ResponsesStreamParser(StreamParser):
    def __init__(self) -> None:
        self._streamed_calls: set[str] = set()

    def feed(self, event: str, data: str) -> Iterable[Delta]:
        payload = json.loads(data)
        match payload.get("type"):
            case "response.output_text.delta":
                yield TextDelta(payload["delta"])
            case "response.refusal.delta":
                yield RefusalDelta(payload["delta"])
            case "response.reasoning_summary_text.delta" | "response.reasoning_text.delta":
                yield ReasoningDelta(payload["delta"])
            case "response.reasoning_summary_part.added" if payload.get("summary_index", 0) > 0:
                yield ReasoningDelta("\n\n")
            case "response.output_item.added" if payload["item"].get("type") == "function_call":
                item = payload["item"]
                self._streamed_calls.add(_call_key(item))
                yield ToolCallFragment(_call_key(item), id=item["call_id"], name=item["name"])
            case "response.function_call_arguments.delta":
                yield ToolCallFragment(payload["item_id"], arguments=payload["delta"])
            case "response.output_item.done" if _call_key_or_none(payload["item"]) in self._streamed_calls:
                yield ToolCallFragment(_call_key(payload["item"]), done=True)
            case "response.output_item.done":
                yield from _item_deltas(payload["item"], streamed=True)
            case _:
                yield from _lifecycle_deltas(payload)


def _lifecycle_deltas(payload: Mapping[str, Any]) -> Iterable[Delta]:
    match payload.get("type"):
        case "response.created":
            yield _info(payload["response"])
        case "response.completed" | "response.incomplete":
            yield from _response_deltas(payload["response"])
        case "response.failed":
            _raise_for_failure(payload["response"])
        case "error":
            raise APIResponseError(payload.get("message") or json.dumps(payload))


def _user_part(part: object) -> dict[str, Any]:
    match part:
        case Text(text=text):
            return {"type": "input_text", "text": text}
        case Image():
            return {"type": "input_image", "image_url": part.data_url}
        case Video():
            raise unsupported_media(ResponsesFormat.name, "video")
    raise TypeError(f"Unexpected user content: {part!r}")


def _item_deltas(item: Mapping[str, Any], *, streamed: bool) -> Iterable[Delta]:
    """Deltas for a complete output item; ``streamed`` skips what earlier stream events delivered."""
    match item.get("type"):
        case "message" if not streamed:
            for content in item.get("content") or ():
                if content.get("type") == "output_text":
                    yield TextDelta(content["text"])
                elif content.get("type") == "refusal":
                    yield RefusalDelta(content["refusal"])
        case "function_call":
            yield ToolCallFragment(
                _call_key(item), id=item["call_id"], name=item["name"], arguments=item["arguments"], done=True
            )
        case "image_generation_call" if item.get("result"):
            media_type = f"image/{item.get('output_format', 'png')}"
            yield ImageReady(Image(media_type, data=base64.b64decode(item["result"])))
        case "reasoning":
            if not streamed and (reasoning := _reasoning_text(item)):
                yield ReasoningDelta(reasoning)
            yield ProviderData(ResponsesFormat.name, item)


def _call_key(item: Mapping[str, Any]) -> str:
    """Argument deltas refer to the output item id, not the call id."""
    return item.get("id") or item["call_id"]


def _call_key_or_none(item: Mapping[str, Any]) -> str | None:
    return _call_key(item) if item.get("type") == "function_call" else None


def _response_deltas(response: Mapping[str, Any]) -> Iterable[Delta]:
    yield _info(response)
    if usage := response.get("usage"):
        yield _usage(usage)


def _info(response: Mapping[str, Any]) -> ResponseInfo:
    finish_reason: FinishReason | None = None
    if response.get("status") == "completed":
        finish_reason = "stop"
    elif response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        finish_reason = _INCOMPLETE_REASONS.get(reason, "other")
    return ResponseInfo(id=response.get("id"), model=response.get("model"), finish_reason=finish_reason)


def _reasoning_text(item: Mapping[str, Any]) -> str:
    """Summaries when requested with ``reasoning.summary``, or raw text from open-weight models."""
    parts = [*(item.get("summary") or ()), *(item.get("content") or ())]
    return "\n\n".join(part["text"] for part in parts if part.get("type") in _REASONING_TEXT_TYPES)


def _usage(usage: Mapping[str, Any]) -> UsageReport:
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return UsageReport(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reasoning_tokens=output_details.get("reasoning_tokens"),
        cached_tokens=input_details.get("cached_tokens"),
    )


def _raise_for_failure(response: Mapping[str, Any]) -> None:
    if error := response.get("error"):
        raise APIResponseError(json.dumps(error))
