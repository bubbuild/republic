"""The Google Gemini ``generateContent`` format."""

from __future__ import annotations

import base64
import itertools
import json
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from republic._content import Image, Message, ProviderData, Text, Tool, ToolResult, Video, _Media
from republic._errors import APIResponseError, UnsupportedFeatureError
from republic._options import ReasoningEffort, ToolChoice
from republic._response import BuiltinToolCall, Citation, FinishReason
from republic.events import BuiltinToolCallReady, CitationAdded, ImageReady, ReasoningDelta, TextDelta
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
    deep_merge,
    merge_same_role,
    provider_payloads,
    unsupported_tool,
)

# Gemini only sends call ids for some models. Keep the original so a generated
# placeholder id is never sent back.
_CALL_ID = "gemini_call_id"
_THOUGHT_SIGNATURE = "gemini_thought_signature"
_BLOCKED_REASONS = frozenset({
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
})


class GeminiFormat(ChatApiFormat):
    name = "gemini"

    def chat_request(self, request: ChatRequest, *, stream: bool) -> HttpRequest:
        request.reject(self.name, "parallel_tool_calls")
        if any(tool.strict for tool in request.tools):
            raise UnsupportedFeatureError(f"The {self.name!r} API format has no strict tool mode")
        options = request.options
        body: dict[str, Any] = {
            "contents": merge_same_role(
                [_content(message) for message in request.messages if message.role != "system"],
                content_key="parts",
            ),
        }
        if system := "\n\n".join(message.text for message in request.messages if message.role == "system"):
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools := _tools(request, self.name):
            body["tools"] = tools
        if (tool_choice := options.get("tool_choice")) is not None:
            body["toolConfig"] = {"functionCallingConfig": _function_calling_config(tool_choice)}
        if config := _generation_config(request):
            body["generationConfig"] = config
        effort = options.get("reasoning_effort")
        body = deep_merge(
            body, self.reasoning_fields(effort, include_reasoning=options.get("include_reasoning", False))
        )
        if stream:
            return HttpRequest(
                f"/models/{request.model}:streamGenerateContent", request.body(body), params={"alt": "sse"}
            )
        return HttpRequest(f"/models/{request.model}:generateContent", request.body(body))

    def parse_chat(self, data: Mapping[str, Any]) -> Iterable[Delta]:
        return _GeminiStreamParser().chunk_deltas(data)

    def stream_parser(self) -> StreamParser:
        return _GeminiStreamParser()

    def reasoning_fields(self, effort: ReasoningEffort | None, *, include_reasoning: bool) -> dict[str, Any]:
        """Gemini 3 levels; override for Gemini 2.5, which takes ``thinkingBudget`` instead."""
        thinking: dict[str, Any] = {}
        if effort is not None:
            thinking["thinkingLevel"] = effort
        if include_reasoning:
            thinking["includeThoughts"] = True
        return {"generationConfig": {"thinkingConfig": thinking}} if thinking else {}


class _GeminiStreamParser(StreamParser):
    """Parses response chunks; a full response is a single chunk."""

    def __init__(self) -> None:
        self._call_keys = itertools.count()
        self._pending_code: Mapping[str, Any] | None = None

    def feed(self, event: str, data: str) -> Iterable[Delta]:
        return self.chunk_deltas(json.loads(data))

    def chunk_deltas(self, data: Mapping[str, Any]) -> Iterable[Delta]:
        if error := data.get("error"):
            raise APIResponseError(json.dumps(error))
        candidates = data.get("candidates") or ()
        if not candidates and (feedback := data.get("promptFeedback", {}).get("blockReason")):
            raise APIResponseError(f"Prompt blocked: {feedback}")
        finish_reason: FinishReason | None = None
        for candidate in candidates[:1]:
            for part in candidate.get("content", {}).get("parts") or ():
                yield from self._part_deltas(part)
            yield from _grounding_deltas(candidate)
            if reason := candidate.get("finishReason"):
                finish_reason = _finish_reason(reason)
        yield ResponseInfo(id=data.get("responseId"), model=data.get("modelVersion"), finish_reason=finish_reason)
        if usage := data.get("usageMetadata"):
            yield _usage(usage)

    def _part_deltas(self, part: Mapping[str, Any]) -> Iterable[Delta]:
        if part.get("thought"):
            # Thought summaries, returned when thinkingConfig.includeThoughts is set.
            if text := part.get("text"):
                yield ReasoningDelta(text)
        elif "text" in part:
            yield TextDelta(part["text"])
        elif call := part.get("functionCall"):
            yield _function_call_fragment(part, call, next(self._call_keys))
        elif (inline := part.get("inlineData")) and inline["mimeType"].startswith("image/"):
            yield ImageReady(Image(inline["mimeType"], data=base64.b64decode(inline["data"])))
        elif "executableCode" in part:
            self._pending_code = part["executableCode"]
            yield ProviderData(GeminiFormat.name, part)
        elif "codeExecutionResult" in part:
            code, self._pending_code = self._pending_code or {}, None
            yield BuiltinToolCallReady(
                BuiltinToolCall("code_execution", input=code, output=part["codeExecutionResult"])
            )
            yield ProviderData(GeminiFormat.name, part)


def _function_call_fragment(part: Mapping[str, Any], call: Mapping[str, Any], key: int) -> ToolCallFragment:
    metadata = {}
    if call_id := call.get("id"):
        metadata[_CALL_ID] = call_id
    if signature := part.get("thoughtSignature"):
        metadata[_THOUGHT_SIGNATURE] = signature
    return ToolCallFragment(
        key,
        id=call_id or f"call_{uuid.uuid4().hex}",
        name=call["name"],
        arguments=json.dumps(call.get("args") or {}),
        metadata=metadata,
        done=True,
    )


def _grounding_deltas(candidate: Mapping[str, Any]) -> Iterable[Delta]:
    """Search and URL context report what they did as metadata, not as parts."""
    grounding = candidate.get("groundingMetadata") or {}
    if queries := grounding.get("webSearchQueries"):
        yield BuiltinToolCallReady(BuiltinToolCall("web_search", input={"queries": queries}))
    for chunk in grounding.get("groundingChunks") or ():
        if web := chunk.get("web"):
            yield CitationAdded(Citation(web["uri"], title=web.get("title")))
    if url_metadata := (candidate.get("urlContextMetadata") or {}).get("urlMetadata"):
        urls = [entry.get("retrievedUrl") for entry in url_metadata]
        yield BuiltinToolCallReady(BuiltinToolCall("web_fetch", input={"urls": urls}, output=url_metadata))


def _usage(usage: Mapping[str, Any]) -> UsageReport:
    thoughts_tokens = usage.get("thoughtsTokenCount", 0)
    return UsageReport(
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount", 0) + thoughts_tokens,
        reasoning_tokens=thoughts_tokens,
        cached_tokens=usage.get("cachedContentTokenCount"),
    )


def _tools(request: ChatRequest, api_format: str) -> list[Mapping[str, Any]]:
    tools: list[Mapping[str, Any]] = []
    if request.tools:
        declarations = [
            {"name": tool.name, "description": tool.description, "parametersJsonSchema": tool.parameters}
            for tool in request.tools
        ]
        tools.append({"functionDeclarations": declarations})
    tools.extend(_builtin_tool(tool, api_format) for tool in request.builtin_tools(api_format))
    return tools


def _builtin_tool(tool: BuiltinTool, api_format: str) -> Mapping[str, Any]:
    match tool:
        case NativeTool(definition=definition):
            return definition
        case WebSearch() | WebFetch() if any(vars(tool).values()):
            # Gemini's search and URL context tools take no settings.
            raise unsupported_tool(api_format, tool, "settings")
        case WebSearch():
            return {"googleSearch": {}}
        case WebFetch():
            return {"urlContext": {}}
        case CodeExecution():
            return {"codeExecution": {}}
    raise unsupported_tool(api_format, tool)


def _generation_config(request: ChatRequest) -> dict[str, Any]:
    options = request.options
    config = request.renamed({
        "max_tokens": "maxOutputTokens",
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "presence_penalty": "presencePenalty",
        "frequency_penalty": "frequencyPenalty",
        "seed": "seed",
    })
    if request.output_schema is not None:
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = request.output_schema.schema
    if (stop := options.get("stop")) is not None:
        config["stopSequences"] = list(stop)
    return config


def _finish_reason(reason: str) -> FinishReason:
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _BLOCKED_REASONS:
        return "content_filter"
    return "other"


def _function_calling_config(tool_choice: ToolChoice) -> dict[str, Any]:
    match tool_choice:
        case Tool(name=name):
            return {"mode": "ANY", "allowedFunctionNames": [name]}
        case "required":
            return {"mode": "ANY"}
        case _:
            return {"mode": tool_choice.upper()}


def _content(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "user", "parts": [_function_response(result) for result in message.tool_results]}
    parts: list[Mapping[str, Any]] = provider_payloads(message, GeminiFormat.name)
    parts.extend(_part(part) for part in message.parts if isinstance(part, Text | Image | Video))
    for call in message.tool_calls:
        function_call: dict[str, Any] = {"name": call.name, "args": call.args}
        if call_id := call.metadata.get(_CALL_ID):
            function_call["id"] = call_id
        part: dict[str, Any] = {"functionCall": function_call}
        if signature := call.metadata.get(_THOUGHT_SIGNATURE):
            part["thoughtSignature"] = signature
        parts.append(part)
    return {"role": "model" if message.role == "assistant" else "user", "parts": parts}


def _function_response(result: ToolResult) -> dict[str, Any]:
    response: dict[str, Any] = {
        "name": result.call.name,
        "response": {"error" if result.is_error else "output": result.output},
    }
    if call_id := result.call.metadata.get(_CALL_ID):
        response["id"] = call_id
    return {"functionResponse": response}


def _part(part: Text | _Media) -> dict[str, Any]:
    if isinstance(part, Text):
        return {"text": part.text}
    if part.url is not None:
        return {"fileData": {"mimeType": part.media_type, "fileUri": part.url}}
    return {"inlineData": {"mimeType": part.media_type, "data": part.base64_data}}
