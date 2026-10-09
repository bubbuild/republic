"""The OpenAI Responses format."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Mapping
from typing import Any

from republic._content import Image, Message, ProviderData, Text, Tool, Video
from republic._errors import APIResponseError
from republic._options import ReasoningEffort
from republic._response import BuiltinToolCall, Citation, FinishReason
from republic.events import BuiltinToolCallReady, CitationAdded, ImageReady, ReasoningDelta, RefusalDelta, TextDelta
from republic.tools import BuiltinTool, CodeExecution, ImageGeneration, NativeTool, WebSearch

from ._base import (
    ChatApiFormat,
    ChatRequest,
    Delta,
    HttpRequest,
    ResponseInfo,
    StreamParser,
    ToolCallFragment,
    UsageReport,
    approximate_location,
    deep_merge,
    provider_payloads,
    strict_schema,
    unsupported_media,
    unsupported_tool,
)

_REASONING_TEXT_TYPES = frozenset({"summary_text", "reasoning_text"})
_BUILTIN_NAMES = {
    "web_search_call": "web_search",
    "code_interpreter_call": "code_execution",
    "image_generation_call": "image_generation",
}
_BUILTIN_OUTPUT_KEYS = ("output", "outputs", "results", "result")
_BUILTIN_RESERVED_KEYS = frozenset({"id", "type", "status", *_BUILTIN_OUTPUT_KEYS})
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
        builtin_tools = request.builtin_tools(self.name)
        if tools := [_function_tool(tool) for tool in request.tools] + [_builtin_tool(t) for t in builtin_tools]:
            body["tools"] = tools
        if any(isinstance(tool, WebSearch) for tool in builtin_tools):
            body["include"] = ["web_search_call.action.sources"]
        if (tool_choice := request.options.get("tool_choice")) is not None:
            body["tool_choice"] = (
                {"type": "function", "name": tool_choice.name} if isinstance(tool_choice, Tool) else tool_choice
            )
        if request.output_schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema.name,
                    "schema": strict_schema(request.output_schema.schema, require_all=True),
                    "strict": True,
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
        effort = request.options.get("reasoning_effort")
        body = deep_merge(
            body, self.reasoning_fields(effort, include_reasoning=request.options.get("include_reasoning", False))
        )
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

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        reasoning: dict[str, Any] = {}
        if effort is not None:
            reasoning["effort"] = effort
        if include_reasoning:
            reasoning["summary"] = "auto"
        return {"reasoning": reasoning} if reasoning else {}

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
        if data == "[DONE]":
            self.completed = True
            return
        payload = json.loads(data)
        if payload.get("type") in {"response.completed", "response.incomplete"}:
            self.completed = True
        yield from self._feed_payload(payload)

    def _feed_payload(self, payload: Mapping[str, Any]) -> Iterable[Delta]:
        match payload.get("type"):
            case "response.output_text.delta":
                yield TextDelta(payload["delta"])
            case "response.output_text.annotation.added":
                yield from _citation_deltas([payload["annotation"]])
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


def _function_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        # The Responses API defaults to strict, which rejects most ordinary schemas.
        "strict": tool.strict,
    }


def _builtin_tool(tool: BuiltinTool) -> Mapping[str, Any]:
    match tool:
        case NativeTool(definition=definition):
            return definition
        case WebSearch(max_uses=int()):
            raise unsupported_tool(ResponsesFormat.name, tool, "max_uses")
        case WebSearch(blocked_domains=[_, *_]):
            raise unsupported_tool(ResponsesFormat.name, tool, "blocked_domains")
        case WebSearch():
            definition: dict[str, Any] = {"type": "web_search"}
            if tool.allowed_domains:
                definition["filters"] = {"allowed_domains": list(tool.allowed_domains)}
            if tool.user_location is not None:
                definition["user_location"] = approximate_location(tool.user_location)
            return definition
        case CodeExecution():
            return {"type": "code_interpreter", "container": {"type": "auto"}}
        case ImageGeneration():
            return {"type": "image_generation"}
    raise unsupported_tool(ResponsesFormat.name, tool)


def _item_deltas(item: Mapping[str, Any], *, streamed: bool) -> Iterable[Delta]:
    """Deltas for a complete output item; ``streamed`` skips what earlier stream events delivered."""
    match item.get("type"):
        case "message" if streamed:
            pass
        case "message":
            for content in item.get("content") or ():
                if content.get("type") == "output_text":
                    yield TextDelta(content["text"])
                    yield from _citation_deltas(content.get("annotations") or ())
                elif content.get("type") == "refusal":
                    yield RefusalDelta(content["refusal"])
        case "function_call":
            yield ToolCallFragment(
                _call_key(item), id=item["call_id"], name=item["name"], arguments=item["arguments"], done=True
            )
        case "reasoning":
            if not streamed and (reasoning := _reasoning_text(item)):
                yield ReasoningDelta(reasoning)
            yield ProviderData(ResponsesFormat.name, item)
        case str(item_type):
            yield from _builtin_item_deltas(item_type, item)


def _builtin_item_deltas(item_type: str, item: Mapping[str, Any]) -> Iterable[Delta]:
    """Server-side tool items. All are sent back, since reasoning items require what followed them."""
    if item_type == "image_generation_call" and item.get("result"):
        media_type = f"image/{item.get('output_format', 'png')}"
        yield ImageReady(Image(media_type, data=base64.b64decode(item["result"])))
    if item_type.endswith("_call"):
        output = next((item[key] for key in _BUILTIN_OUTPUT_KEYS if key in item and key != "result"), None)
        yield BuiltinToolCallReady(
            BuiltinToolCall(
                name=_BUILTIN_NAMES.get(item_type, item_type.removesuffix("_call")),
                input={key: value for key, value in item.items() if key not in _BUILTIN_RESERVED_KEYS},
                output=output,
                id=item.get("id"),
            )
        )
    yield ProviderData(ResponsesFormat.name, item)


def _citation_deltas(annotations: Iterable[Mapping[str, Any]]) -> Iterable[Delta]:
    for annotation in annotations:
        if annotation.get("type") == "url_citation":
            yield CitationAdded(Citation(annotation["url"], title=annotation.get("title")))


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
