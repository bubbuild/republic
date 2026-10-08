"""The OpenAI Chat Completions format, widely adopted by compatible gateways."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from republic._content import Image, Message, Text, Tool, Video, media_from_data_url
from republic._errors import APIResponseError
from republic._options import ReasoningEffort
from republic._response import Citation, FinishReason
from republic.events import CitationAdded, ImageReady, ReasoningDelta, RefusalDelta, TextDelta
from republic.tools import NativeTool, WebSearch

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
    strict_schema,
    unsupported_tool,
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


class ChatFormat(ChatApiFormat):
    name = "chat"

    def chat_request(self, request: ChatRequest, *, stream: bool) -> HttpRequest:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [entry for message in request.messages for entry in _message_entries(message)],
        }
        tools: list[Mapping[str, Any]] = [_tool(tool) for tool in request.tools]
        for builtin_tool in request.builtin_tools(self.name):
            match builtin_tool:
                case NativeTool(definition=definition):
                    tools.append(definition)
                case WebSearch():
                    body["web_search_options"] = _web_search_options(builtin_tool)
                case _:
                    raise unsupported_tool(self.name, builtin_tool)
        if tools:
            body["tools"] = tools
        if (tool_choice := request.options.get("tool_choice")) is not None:
            body["tool_choice"] = (
                {"type": "function", "function": {"name": tool_choice.name}}
                if isinstance(tool_choice, Tool)
                else tool_choice
            )
        if request.output_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.name,
                    "schema": strict_schema(request.output_schema.schema, require_all=True),
                    "strict": True,
                },
            }
        if (max_tokens := request.options.get("max_tokens")) is not None:
            body.update(self.max_tokens_fields(max_tokens))
        body.update(
            request.renamed({
                "temperature": "temperature",
                "top_p": "top_p",
                # Not part of the OpenAI API, but accepted by most compatible servers.
                "top_k": "top_k",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
                "seed": "seed",
                "parallel_tool_calls": "parallel_tool_calls",
            })
        )
        effort = request.options.get("reasoning_effort")
        body = deep_merge(
            body, self.reasoning_fields(effort, include_reasoning=request.options.get("include_reasoning", False))
        )
        if (stop := request.options.get("stop")) is not None:
            body["stop"] = list(stop)
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return HttpRequest("/chat/completions", request.body(body))

    def parse_chat(self, data: Mapping[str, Any]) -> Iterable[Delta]:
        _raise_for_error(data)
        choice = data["choices"][0]
        message = choice["message"]
        if reasoning := self.reasoning_text(message):
            yield ReasoningDelta(reasoning)
        if content := message.get("content"):
            yield TextDelta(content)
        yield from _citation_deltas(message)
        if refusal := message.get("refusal"):
            yield RefusalDelta(refusal)
        yield from _image_deltas(message)
        for call in message.get("tool_calls") or ():
            function = call["function"]
            yield ToolCallFragment(
                call["id"], id=call["id"], name=function["name"], arguments=function["arguments"], done=True
            )
        yield _info(data, choice)
        if usage := data.get("usage"):
            yield _usage(usage)

    def stream_parser(self) -> StreamParser:
        return _ChatStreamParser(self)

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        # Compatible servers that return reasoning do so by default, so include_reasoning needs no field.
        return {} if effort is None else {"reasoning_effort": effort}

    def max_tokens_fields(self, max_tokens: int) -> dict[str, Any]:
        """OpenAI deprecated ``max_tokens``; override for servers that only accept it."""
        return {"max_completion_tokens": max_tokens}

    def reasoning_text(self, message: Mapping[str, Any]) -> str | None:
        """Reasoning in a message or delta, as exposed by vLLM, DeepSeek (``reasoning_content``) or OpenRouter."""
        return message.get("reasoning_content") or message.get("reasoning")


class _ChatStreamParser(StreamParser):
    def __init__(self, api_format: ChatFormat) -> None:
        self._api_format = api_format

    def feed(self, event: str, data: str) -> Iterable[Delta]:
        chunk = json.loads(data)
        _raise_for_error(chunk)
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            if reasoning := self._api_format.reasoning_text(delta):
                yield ReasoningDelta(reasoning)
            if content := delta.get("content"):
                yield TextDelta(content)
            yield from _citation_deltas(delta)
            if refusal := delta.get("refusal"):
                yield RefusalDelta(refusal)
            yield from _image_deltas(delta)
            for call in delta.get("tool_calls") or ():
                function = call.get("function") or {}
                # Calls complete only when the response ends; the builder releases them then.
                yield ToolCallFragment(
                    call.get("index", 0),
                    id=call.get("id"),
                    name=function.get("name"),
                    arguments=function.get("arguments") or "",
                )
            yield _info(chunk, choice)
        if usage := chunk.get("usage"):
            yield _usage(usage)


def _message_entries(message: Message) -> list[dict[str, Any]]:
    match message.role:
        case "system":
            return [{"role": "system", "content": message.text}]
        case "user":
            return [{"role": "user", "content": _user_content(message)}]
        case "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                    for call in message.tool_calls
                ]
            return [entry]
        case "tool":
            return [
                {"role": "tool", "tool_call_id": result.call.id, "content": result.output}
                for result in message.tool_results
            ]


def _user_content(message: Message) -> str | list[dict[str, Any]]:
    if all(isinstance(part, Text) for part in message.parts):
        return message.text
    content: list[dict[str, Any]] = []
    for part in message.parts:
        match part:
            case Text(text=text):
                content.append({"type": "text", "text": text})
            case Image():
                content.append({"type": "image_url", "image_url": {"url": part.data_url}})
            case Video():
                content.append({"type": "video_url", "video_url": {"url": part.data_url}})
    return content


def _tool(tool: Tool) -> dict[str, Any]:
    function = {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
    if tool.strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _web_search_options(tool: WebSearch) -> dict[str, Any]:
    """Search options for search-enabled chat models, as accepted by OpenAI and OpenRouter."""
    for setting in ("max_uses", "allowed_domains", "blocked_domains"):
        if getattr(tool, setting):
            raise unsupported_tool(ChatFormat.name, tool, setting)
    options: dict[str, Any] = {}
    if tool.user_location is not None:
        location = approximate_location(tool.user_location)
        options["user_location"] = {"type": location.pop("type"), "approximate": location}
    return options


def _citation_deltas(message: Mapping[str, Any]) -> Iterable[Delta]:
    for annotation in message.get("annotations") or ():
        if annotation.get("type") == "url_citation":
            citation = annotation["url_citation"]
            yield CitationAdded(Citation(citation["url"], title=citation.get("title")))


def _info(data: Mapping[str, Any], choice: Mapping[str, Any]) -> ResponseInfo:
    finish_reason = choice.get("finish_reason")
    return ResponseInfo(
        id=data.get("id"),
        model=data.get("model"),
        finish_reason=None if finish_reason is None else _FINISH_REASONS.get(finish_reason, "other"),
    )


def _image_deltas(message: Mapping[str, Any]) -> Iterable[Delta]:
    """Images returned by gateways such as OpenRouter."""
    for item in message.get("images") or ():
        url = item["image_url"]["url"]
        image = media_from_data_url(Image, url) if url.startswith("data:") else Image("image/png", url=url)
        yield ImageReady(image)


def _usage(usage: Mapping[str, Any]) -> UsageReport:
    input_details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    return UsageReport(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        reasoning_tokens=output_details.get("reasoning_tokens"),
        cached_tokens=input_details.get("cached_tokens"),
    )


def _raise_for_error(data: Mapping[str, Any]) -> None:
    if error := data.get("error"):
        raise APIResponseError(json.dumps(error))
