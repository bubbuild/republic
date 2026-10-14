"""The Anthropic Messages format."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from republic._content import Audio, Image, Message, ProviderData, Text, Tool, ToolCall, Video
from republic._options import ReasoningEffort, ToolChoice
from republic._response import BuiltinToolCall, Citation, FinishReason
from republic.errors import APIResponseError
from republic.events import BuiltinToolCallReady, CitationAdded, ReasoningDelta, RefusalDelta, TextDelta
from republic.tools import BuiltinTool, CodeExecution, NativeTool, WebFetch, WebSearch

from ._base import (
    ChatApiFormat,
    ChatRequest,
    Delta,
    HttpRequest,
    ResponseInfo,
    StreamParser,
    ToolCallFragment,
    UsageReport,
    answered_call,
    approximate_location,
    deep_merge,
    merge_same_role,
    provider_payloads,
    strict_schema,
    unsupported_media,
    unsupported_tool,
)

DEFAULT_MAX_TOKENS = 16000
"""``max_tokens`` is required by the Messages API; this leaves room for long answers."""

_BUILTIN_USE_BLOCKS = frozenset({"server_tool_use", "mcp_tool_use"})
_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "refusal",
    "pause_turn": "pause",
}


class MessagesFormat(ChatApiFormat):
    name = "messages"
    headers: ClassVar[Mapping[str, str]] = {"anthropic-version": "2023-06-01"}

    def chat_request(self, request: ChatRequest, *, stream: bool) -> HttpRequest:
        request.reject(self.name, "seed", "presence_penalty", "frequency_penalty")
        options = request.options
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": options.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": merge_same_role(
                [self._entry(message) for message in request.messages if message.role != "system"],
                content_key="content",
            ),
        }
        if system := "\n\n".join(message.text for message in request.messages if message.role == "system"):
            body["system"] = system
        builtin_tools = request.builtin_tools(self.name)
        if tools := [_tool(tool) for tool in request.tools] + [_builtin_tool(tool) for tool in builtin_tools]:
            body["tools"] = tools
        if tool_choice := _tool_choice(options.get("tool_choice"), options.get("parallel_tool_calls")):
            body["tool_choice"] = tool_choice
        if output_config := _output_config(request):
            body["output_config"] = output_config
        effort = options.get("reasoning_effort")
        body = deep_merge(
            body, self.reasoning_fields(effort, include_reasoning=options.get("include_reasoning", False))
        )
        body.update(request.renamed({"temperature": "temperature", "top_p": "top_p", "top_k": "top_k"}))
        if (stop := options.get("stop")) is not None:
            body["stop_sequences"] = list(stop)
        if stream:
            body["stream"] = True
        return HttpRequest("/messages", request.body(body))

    def parse_chat(self, data: Mapping[str, Any]) -> Iterable[Delta]:
        builtin_calls = _BuiltinCalls()
        order = _BlockOrder()
        for block in data["content"]:
            match block["type"]:
                case "text":
                    order.text(block["text"])
                    yield TextDelta(block["text"])
                    yield from _citation_deltas(block.get("citations") or ())
                case "tool_use":
                    order.tool_use(block["id"])
                    yield ToolCallFragment(
                        block["id"],
                        id=block["id"],
                        name=block["name"],
                        arguments=json.dumps(block["input"]),
                        done=True,
                    )
                case _:
                    yield from order.flush()
                    yield from _round_trip_deltas(block, builtin_calls)
        yield from _stop_deltas(data)
        yield ResponseInfo(id=data.get("id"), model=data.get("model"))
        yield _usage(data["usage"])

    def stream_parser(self) -> StreamParser:
        return _MessagesStreamParser()

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        """Thinking must be enabled explicitly on some models; ``display`` makes it readable."""
        if effort == "none":
            return {"thinking": {"type": "disabled"}}
        if effort is None and not include_reasoning:
            return {}
        fields: dict[str, Any] = {"thinking": {"type": "adaptive"}}
        if include_reasoning:
            fields["thinking"]["display"] = "summarized"
        if effort is not None:
            fields["output_config"] = {"effort": effort}
        return fields

    def _entry(self, message: Message) -> dict[str, Any]:
        if message.role == "tool":
            content = (
                message.text
                if all(isinstance(part, Text) for part in message.parts)
                else [_user_block(part) for part in message.parts]
            )
            result = {
                "type": "tool_result",
                "tool_use_id": answered_call(message).id,
                "content": content,
                "is_error": message.is_error,
            }
            return {"role": "user", "content": [result]}
        if message.role == "user":
            return {"role": "user", "content": [_user_block(part) for part in message.parts]}
        return {"role": "assistant", "content": _assistant_content(provider_payloads(message, self.name), message)}


class _MessagesStreamParser(StreamParser):
    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._block_inputs: dict[int, list[str]] = {}
        self._tool_indexes: set[int] = set()
        self._builtin_calls = _BuiltinCalls()
        self._order = _BlockOrder()
        self._texts: dict[int, dict[str, Any]] = {}

    def feed(self, event: str, data: str) -> Iterable[Delta]:
        payload = json.loads(data)
        match payload["type"]:
            case "message_stop":
                self.completed = True
            case "message_start":
                message = payload["message"]
                yield ResponseInfo(id=message.get("id"), model=message.get("model"))
                yield _usage(message["usage"])
            case "content_block_start":
                yield from self._start(payload["index"], payload["content_block"])
            case "content_block_delta":
                yield from self._delta(payload["index"], payload["delta"])
            case "content_block_stop":
                yield from self._stop(payload["index"])
            case "message_delta":
                yield from _stop_deltas(payload["delta"])
                yield _usage(payload["usage"])
            case "error":
                raise APIResponseError(json.dumps(payload["error"]))

    def _start(self, index: int, block: Mapping[str, Any]) -> Iterable[Delta]:
        match block["type"]:
            case "text":
                self._texts[index] = self._order.text(block["text"])
                if block["text"]:
                    yield TextDelta(block["text"])
            case "tool_use":
                self._order.tool_use(block["id"])
                self._tool_indexes.add(index)
                yield ToolCallFragment(index, id=block["id"], name=block["name"])
            case _:
                self._blocks[index] = dict(block)

    def _delta(self, index: int, delta: Mapping[str, Any]) -> Iterable[Delta]:
        match delta["type"]:
            case "text_delta":
                if (marker := self._texts.get(index)) is not None:
                    marker["text"] += delta["text"]
                yield TextDelta(delta["text"])
            case "citations_delta":
                yield from _citation_deltas([delta["citation"]])
            case "input_json_delta" if index in self._tool_indexes:
                yield ToolCallFragment(index, arguments=delta["partial_json"])
            case "input_json_delta":
                # Server tool input, such as a search query, arrives in fragments too.
                self._block_inputs.setdefault(index, []).append(delta["partial_json"])
            case "thinking_delta":
                block = self._blocks[index]
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
                yield ReasoningDelta(delta["thinking"])
            case "signature_delta":
                block = self._blocks[index]
                block["signature"] = block.get("signature", "") + delta["signature"]

    def _stop(self, index: int) -> Iterable[Delta]:
        if index in self._tool_indexes:
            yield ToolCallFragment(index, done=True)
        if (block := self._blocks.pop(index, None)) is None:
            return
        if (fragments := self._block_inputs.pop(index, None)) is not None:
            block["input"] = json.loads("".join(fragments) or "{}")
        yield from self._order.flush()
        # Streamed thinking was already yielded as reasoning deltas.
        yield from _round_trip_deltas(block, self._builtin_calls, include_reasoning=False)


_ORDER_MARKERS = frozenset({"text", "tool_use"})


class _BlockOrder:
    """Records text and tool use blocks that come before a round-tripped block.

    A message keeps its text and tool calls apart from provider data, while some
    servers validate thinking against the original block order, for example when a
    tool use precedes a built-in tool result. Those blocks are marked in place as
    provider data so :func:`_assistant_content` can restore the order. Blocks after
    the last round-tripped block need no marker.
    """

    def __init__(self) -> None:
        self._pending: list[dict[str, Any]] = []

    def text(self, text: str) -> dict[str, Any]:
        marker = {"type": "text", "text": text}
        self._pending.append(marker)
        return marker

    def tool_use(self, call_id: str) -> None:
        self._pending.append({"type": "tool_use", "id": call_id})

    def flush(self) -> Iterable[Delta]:
        for marker in self._pending:
            if marker["type"] != "text" or marker["text"]:
                yield ProviderData(MessagesFormat.name, marker)
        self._pending = []


def _assistant_content(payloads: list[Mapping[str, Any]], message: Message) -> list[Mapping[str, Any]]:
    """Assistant blocks in their original order, with text and tool uses placed by their markers.

    A message whose text or tool calls no longer match its markers was changed after it
    was received; its round-tripped blocks then come first, as for unmarked messages.
    """
    calls = {call.id: call for call in message.tool_calls}
    marked_text = "".join(payload["text"] for payload in payloads if payload.get("type") == "text")
    marked_calls = {payload["id"] for payload in payloads if payload.get("type") == "tool_use"}
    if not message.text.startswith(marked_text) or not marked_calls <= calls.keys():
        payloads = [payload for payload in payloads if payload.get("type") not in _ORDER_MARKERS]
        marked_text, marked_calls = "", set()
    content: list[Mapping[str, Any]] = []
    for payload in payloads:
        match payload.get("type"):
            case "text":
                content.append({"type": "text", "text": payload["text"]})
            case "tool_use":
                content.append(_tool_use_block(calls[payload["id"]]))
            case _:
                content.append(payload)
    if rest := message.text[len(marked_text) :]:
        content.append({"type": "text", "text": rest})
    content.extend(_tool_use_block(call) for call in message.tool_calls if call.id not in marked_calls)
    return content


def _tool_use_block(call: ToolCall) -> dict[str, Any]:
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}


class _BuiltinCalls:
    """Pairs server tool uses with their results, which arrive as separate blocks."""

    def __init__(self) -> None:
        self._uses: dict[str, Mapping[str, Any]] = {}

    def add(self, block: Mapping[str, Any]) -> Iterable[Delta]:
        if block["type"] in _BUILTIN_USE_BLOCKS:
            self._uses[block["id"]] = block
        elif (use := self._uses.pop(block.get("tool_use_id", ""), None)) is not None:
            name = use["name"]
            yield BuiltinToolCallReady(
                BuiltinToolCall(
                    name="code_execution" if name.endswith("code_execution") else name,
                    input=use.get("input") or {},
                    output=block.get("content"),
                    id=use["id"],
                )
            )


def _round_trip_deltas(
    block: Mapping[str, Any], builtin_calls: _BuiltinCalls, *, include_reasoning: bool = True
) -> Iterable[Delta]:
    """Blocks other than text and tool use go back verbatim; built-in tool results are also surfaced."""
    if include_reasoning and (thinking := block.get("thinking")):
        yield ReasoningDelta(thinking)
    yield from builtin_calls.add(block)
    yield ProviderData(MessagesFormat.name, block)


def _citation_deltas(citations: Iterable[Mapping[str, Any]]) -> Iterable[Delta]:
    for citation in citations:
        if url := citation.get("url"):
            yield CitationAdded(Citation(url, title=citation.get("title"), cited_text=citation.get("cited_text")))


def _tool(tool: Tool) -> dict[str, Any]:
    definition = {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
    if tool.strict:
        definition["strict"] = True
    return definition


def _builtin_tool(tool: BuiltinTool) -> Mapping[str, Any]:
    match tool:
        case NativeTool(definition=definition):
            return definition
        case WebSearch():
            definition = {"type": "web_search_20260209", "name": "web_search", **_domain_settings(tool)}
            if tool.user_location is not None:
                definition["user_location"] = approximate_location(tool.user_location)
            return definition
        case WebFetch():
            return {"type": "web_fetch_20260209", "name": "web_fetch", **_domain_settings(tool)}
        case CodeExecution():
            return {"type": "code_execution_20260521", "name": "code_execution"}
    raise unsupported_tool(MessagesFormat.name, tool)


def _domain_settings(tool: WebSearch | WebFetch) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if tool.max_uses is not None:
        settings["max_uses"] = tool.max_uses
    if tool.allowed_domains:
        settings["allowed_domains"] = list(tool.allowed_domains)
    if tool.blocked_domains:
        settings["blocked_domains"] = list(tool.blocked_domains)
    return settings


def _output_config(request: ChatRequest) -> dict[str, Any]:
    output_config: dict[str, Any] = {}
    if request.output_schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": strict_schema(request.output_schema.schema, require_all=False),
        }
    return output_config


def _stop_deltas(message: Mapping[str, Any]) -> Iterable[Delta]:
    stop_reason = message.get("stop_reason")
    if stop_reason is None:
        return
    if stop_reason == "refusal" and (explanation := (message.get("stop_details") or {}).get("explanation")):
        yield RefusalDelta(explanation)
    yield ResponseInfo(finish_reason=_STOP_REASONS.get(stop_reason, "other"))


def _tool_choice(tool_choice: ToolChoice | None, parallel_tool_calls: bool | None) -> dict[str, Any] | None:
    match tool_choice:
        case None if parallel_tool_calls is None:
            return None
        case None | "auto":
            choice: dict[str, Any] = {"type": "auto"}
        case "none":
            return {"type": "none"}
        case "required":
            choice = {"type": "any"}
        case Tool(name=name):
            choice = {"type": "tool", "name": name}
    if parallel_tool_calls is not None:
        choice["disable_parallel_tool_use"] = not parallel_tool_calls
    return choice


def _user_block(part: object) -> dict[str, Any]:
    match part:
        case Text(text=text):
            return {"type": "text", "text": text}
        case Image(url=str() as url):
            return {"type": "image", "source": {"type": "url", "url": url}}
        case Image():
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": part.media_type, "data": part.base64_data},
            }
        case Audio():
            raise unsupported_media(MessagesFormat.name, "audio")
        case Video():
            raise unsupported_media(MessagesFormat.name, "video")
    raise TypeError(f"Unexpected user content: {part!r}")


def _usage(usage: Mapping[str, Any]) -> UsageReport:
    cached_tokens = usage.get("cache_read_input_tokens")
    cache_write_tokens = usage.get("cache_creation_input_tokens")
    input_tokens = usage.get("input_tokens")
    if input_tokens is not None:
        # Anthropic reports cache reads and writes apart from the uncached input.
        input_tokens += (cached_tokens or 0) + (cache_write_tokens or 0)
    return UsageReport(
        input_tokens=input_tokens,
        output_tokens=usage.get("output_tokens"),
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
