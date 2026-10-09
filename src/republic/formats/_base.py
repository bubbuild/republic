from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from republic._content import Message, Part, ProviderData, Reasoning, Text, Tool, ToolCall
from republic._errors import UnsupportedFeatureError
from republic._options import ChatOptions, ReasoningEffort
from republic._response import EmbeddingResponse, FinishReason, Response, TokenUsage
from republic.decisions import DecisionResponse, JSONValue, Question
from republic.events import (
    BuiltinToolCallReady,
    CitationAdded,
    Event,
    ImageReady,
    ReasoningDelta,
    RefusalDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallReady,
    UsageDelta,
)
from republic.tools import BuiltinTool, NativeTool, UserLocation


@dataclass(frozen=True)
class OutputSchema:
    name: str
    schema: Mapping[str, Any]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``, combining nested mappings instead of replacing them."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def strict_schema(schema: Mapping[str, Any], *, require_all: bool) -> dict[str, Any]:
    """Adapt a JSON schema to the strict structured-output rules of provider APIs.

    Every object schema gets ``additionalProperties: false``. With
    ``require_all``, as OpenAI's strict mode demands, every property is also
    required, ``default: null`` is dropped, and ``$ref`` siblings are inlined.
    """
    root = deepcopy(dict(schema))
    return _strict_node(root, root, require_all=require_all)


def _strict_node(node: dict[str, Any], root: Mapping[str, Any], *, require_all: bool) -> dict[str, Any]:
    def visit(child: Any) -> Any:
        return _strict_node(child, root, require_all=require_all) if isinstance(child, dict) else child

    for key in ("$defs", "definitions", "properties"):
        if isinstance(node.get(key), dict):
            node[key] = {name: visit(child) for name, child in node[key].items()}
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(node.get(key), list):
            node[key] = [visit(child) for child in node[key]]
    for key in ("items", "additionalProperties", "not"):
        if key in node:
            node[key] = visit(node[key])
    if node.get("type") == "object" and "additionalProperties" not in node:
        node["additionalProperties"] = False
    if require_all:
        _require_all(node, root, visit)
    return node


def _require_all(node: dict[str, Any], root: Mapping[str, Any], visit: Callable[[Any], Any]) -> None:
    """The extra rules of OpenAI's strict mode, applied to one schema node."""
    if isinstance(properties := node.get("properties"), dict):
        node["required"] = list(properties)
    if "default" in node and node["default"] is None:
        del node["default"]
    if isinstance(all_of := node.get("allOf"), list) and len(all_of) == 1:
        node.update(node.pop("allOf")[0])
    if isinstance(ref := node.get("$ref"), str) and len(node) > 1:
        # OpenAI rejects keywords next to $ref, so inline the referenced schema.
        del node["$ref"]
        node.update({**visit(deepcopy(_resolve_ref(root, ref))), **node})


def _resolve_ref(root: Mapping[str, Any], ref: str) -> dict[str, Any]:
    target: Any = root
    for segment in ref.removeprefix("#/").split("/"):
        target = target[segment]
    return dict(target)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: Sequence[Message]
    """Normalized messages; see :func:`normalize`."""
    options: ChatOptions
    output_schema: OutputSchema | None = None

    @property
    def tools(self) -> list[Tool]:
        """Function tools, executed by the caller."""
        return [tool for tool in self.options.get("tools", ()) if isinstance(tool, Tool)]

    def builtin_tools(self, api_format: str) -> list[BuiltinTool]:
        """Built-in tools, run by the provider. Native tools must target ``api_format``."""
        builtin = [tool for tool in self.options.get("tools", ()) if not isinstance(tool, Tool)]
        for tool in builtin:
            if isinstance(tool, NativeTool) and tool.api_format != api_format:
                raise UnsupportedFeatureError(
                    f"A native tool for {tool.api_format!r} cannot be sent through the {api_format!r} API format"
                )
        return builtin

    def reject(self, api_format: str, *unsupported: str) -> None:
        """Raise if any option this API format cannot express was set."""
        if rejected := [name for name in unsupported if name in self.options]:
            raise UnsupportedFeatureError(f"The {api_format!r} API format does not support {', '.join(rejected)}")

    def renamed(self, wire_names: Mapping[str, str]) -> dict[str, Any]:
        """Options that map one-to-one onto wire fields, keyed by their wire names."""
        return {wire_names[name]: value for name, value in self.options.items() if name in wire_names}

    def body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Merge ``extra_body`` over the body built by the API format."""
        return deep_merge(body, self.options.get("extra_body", {}))


@dataclass(frozen=True)
class HttpRequest:
    path: str
    body: Mapping[str, Any]
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallFragment:
    """Part of a tool call. Fragments sharing a key are concatenated.

    ``done`` marks the call complete; calls never marked complete are
    released when the response ends.
    """

    key: Hashable
    id: str | None = None
    name: str | None = None
    arguments: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    done: bool = False


@dataclass(frozen=True)
class UsageReport:
    """Cumulative usage reported so far. ``None`` keeps the previously reported value."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True)
class ResponseInfo:
    """Response metadata. ``None`` keeps the previously reported value."""

    id: str | None = None
    model: str | None = None
    finish_reason: FinishReason | None = None


Delta = (
    TextDelta
    | ReasoningDelta
    | RefusalDelta
    | ImageReady
    | CitationAdded
    | BuiltinToolCallReady
    | ToolCallFragment
    | UsageReport
    | ResponseInfo
    | ProviderData
)


class StreamParser(ABC):
    """Turns server-sent events of one response into deltas."""

    @abstractmethod
    def feed(self, event: str, data: str) -> Iterable[Delta]: ...


class ApiFormat(ABC):
    """One wire protocol. Each kind of model picks among the formats of its own kind."""

    name: ClassVar[str]
    kind: ClassVar[str]
    headers: ClassVar[Mapping[str, str]] = {}


class ChatApiFormat(ApiFormat):
    """A chat wire protocol.

    Option mappings that differ between services speaking the same protocol
    are hook methods, so a provider can return a subclass from
    :meth:`~republic.providers.Provider.select_api_format`.
    """

    kind = "chat"

    @abstractmethod
    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        """The request body fragment for reasoning options, deep-merged into the body."""

    @abstractmethod
    def chat_request(self, request: ChatRequest, *, stream: bool) -> HttpRequest: ...

    @abstractmethod
    def parse_chat(self, data: Mapping[str, Any]) -> Iterable[Delta]: ...

    @abstractmethod
    def stream_parser(self) -> StreamParser: ...


class EmbeddingApiFormat(ApiFormat):
    kind = "embedding"

    @abstractmethod
    def embedding_request(self, model: str, texts: Sequence[str], *, dimensions: int | None) -> HttpRequest: ...

    @abstractmethod
    def parse_embedding(self, data: Mapping[str, Any]) -> EmbeddingResponse: ...


class DecisionApiFormat(ApiFormat):
    kind = "decision"

    @abstractmethod
    def decision_request(self, model: str, state: JSONValue, questions: Mapping[str, Question]) -> HttpRequest: ...

    @abstractmethod
    def parse_decision(self, data: Mapping[str, Any]) -> DecisionResponse: ...


@dataclass
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    released: bool = False

    def build(self) -> ToolCall:
        return ToolCall(self.id, self.name, "".join(self.arguments) or "{}", metadata=self.metadata)


ContentEvent = TextDelta | ReasoningDelta | RefusalDelta | ImageReady | CitationAdded | BuiltinToolCallReady


class ResponseBuilder:
    """Accumulates deltas into the final response and yields the public events they produce."""

    def __init__(self) -> None:
        self._content: list[ContentEvent] = []
        self._provider_data: list[ProviderData] = []
        self._calls: dict[Hashable, _PartialCall] = {}
        self._usage = TokenUsage()
        self._info = ResponseInfo()

    def add(self, delta: Delta) -> list[Event]:
        """Record a delta and return the public events it produces."""
        match delta:
            case ProviderData():
                self._provider_data.append(delta)
                return []
            case ToolCallFragment():
                return self._add_fragment(delta)
            case UsageReport():
                previous, self._usage = self._usage, replace(self._usage, **_reported(delta))
                return [UsageDelta(_subtract(self._usage, previous))] if self._usage != previous else []
            case ResponseInfo():
                self._info = replace(self._info, **_reported(delta))
                return []
            case _:
                self._content.append(delta)
                return [delta]

    def release_tool_calls(self) -> list[Event]:
        """Release calls whose completion the API format does not signal."""
        return [event for key in self._calls for event in self._add_fragment(ToolCallFragment(key, done=True))]

    def response(self, output: Any = None) -> Response[Any]:
        return Response(
            self.message,
            self._usage,
            output,
            finish_reason=self.finish_reason,
            refusal=self._joined(RefusalDelta) or None,
            citations=tuple(dict.fromkeys(e.citation for e in self._content if isinstance(e, CitationAdded))),
            builtin_tool_calls=tuple(e.call for e in self._content if isinstance(e, BuiltinToolCallReady)),
            id=self._info.id,
            model=self._info.model,
        )

    @property
    def message(self) -> Message:
        parts: list[Part] = [*self._provider_data]
        if reasoning := self._joined(ReasoningDelta):
            parts.append(Reasoning(reasoning))
        if text := self._joined(TextDelta):
            parts.append(Text(text))
        parts.extend(event.image for event in self._content if isinstance(event, ImageReady))
        calls = tuple(call.build() for call in self._calls.values())
        return Message("assistant", tuple(parts), tool_calls=calls)

    @property
    def finish_reason(self) -> FinishReason | None:
        reason = self._info.finish_reason
        if reason not in (None, "stop"):
            return reason
        if self._joined(RefusalDelta):
            return "refusal"
        if self._calls:
            return "tool_calls"
        return reason

    def _joined(self, chunk_type: type[TextDelta | ReasoningDelta | RefusalDelta]) -> str:
        return "".join(event.chunk for event in self._content if isinstance(event, chunk_type))

    def _add_fragment(self, fragment: ToolCallFragment) -> list[Event]:
        call = self._calls.setdefault(fragment.key, _PartialCall())
        call.id = fragment.id or call.id
        call.name = fragment.name or call.name
        call.arguments.append(fragment.arguments)
        call.metadata.update(fragment.metadata)
        events: list[Event] = []
        if fragment.arguments:
            events.append(ToolCallDelta(call.id, call.name, fragment.arguments))
        if fragment.done and not call.released:
            call.released = True
            events.append(ToolCallReady(call.build()))
        return events


def _reported(report: UsageReport | ResponseInfo) -> dict[str, Any]:
    return {name: value for name, value in vars(report).items() if value is not None}


def _subtract(current: TokenUsage, previous: TokenUsage) -> TokenUsage:
    return TokenUsage(**{name: value - getattr(previous, name) for name, value in vars(current).items()})


def normalize(messages: Iterable[Message]) -> list[Message]:
    """Split tool results into ``tool`` messages that follow their calls.

    An assistant message carrying tool results also announces their calls,
    unless an earlier assistant message already did.
    """
    normalized: list[Message] = []
    announced: set[str] = set()
    for message in messages:
        if message.role != "assistant":
            normalized.append(message)
            continue
        result_calls = [result.call for result in message.tool_results]
        calls = [call for call in (*message.tool_calls, *result_calls) if call.id not in announced]
        announced.update(call.id for call in calls)
        if message.parts or calls:
            normalized.append(Message("assistant", message.parts, tool_calls=tuple(dict.fromkeys(calls))))
        if message.tool_results:
            normalized.append(Message("tool", tool_results=message.tool_results))
    return normalized


def merge_same_role(entries: list[dict[str, Any]], *, content_key: str) -> list[dict[str, Any]]:
    """Merge consecutive entries of the same role by concatenating their content lists."""
    merged: list[dict[str, Any]] = []
    for entry in entries:
        if merged and merged[-1]["role"] == entry["role"]:
            merged[-1] = {**merged[-1], content_key: [*merged[-1][content_key], *entry[content_key]]}
        else:
            merged.append(entry)
    return merged


def provider_payloads(message: Message, api_format: str) -> list[Mapping[str, Any]]:
    """Opaque items this API format produced earlier in the conversation."""
    return [part.payload for part in message.parts if isinstance(part, ProviderData) and part.api_format == api_format]


def approximate_location(location: UserLocation) -> dict[str, Any]:
    """The ``approximate`` user location shape shared by OpenAI and Anthropic."""
    return {"type": "approximate", **{key: value for key, value in vars(location).items() if value is not None}}


def unsupported_tool(api_format: str, tool: object, setting: str | None = None) -> UnsupportedFeatureError:
    subject = type(tool).__name__ if setting is None else f"{type(tool).__name__}.{setting}"
    return UnsupportedFeatureError(f"The {api_format!r} API format does not support {subject}")


def unsupported_media(api_format: str, kind: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(f"The {api_format!r} API format does not accept {kind} input")
