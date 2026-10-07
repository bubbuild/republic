"""The Google Gemini ``generateContent`` format."""

from __future__ import annotations

import base64
import itertools
import json
import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from republic._content import Image, Message, Text, Tool, ToolResult, Video, _Media
from republic._errors import APIResponseError, UnsupportedFeatureError
from republic._options import ToolChoice
from republic._response import FinishReason
from republic.events import ImageReady, ReasoningDelta, TextDelta

from .base import (
    ChatApiFormat,
    ChatRequest,
    Delta,
    HttpRequest,
    ResponseInfo,
    StreamParser,
    ToolCallFragment,
    UsageReport,
    merge_same_role,
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
        if request.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": tool.name, "description": tool.description, "parametersJsonSchema": tool.parameters}
                        for tool in request.tools
                    ]
                }
            ]
        if (tool_choice := options.get("tool_choice")) is not None:
            body["toolConfig"] = {"functionCallingConfig": _function_calling_config(tool_choice)}
        if config := _generation_config(request):
            body["generationConfig"] = config
        if stream:
            return HttpRequest(
                f"/models/{request.model}:streamGenerateContent", request.body(body), params={"alt": "sse"}
            )
        return HttpRequest(f"/models/{request.model}:generateContent", request.body(body))

    def parse_chat(self, data: Mapping[str, Any]) -> Iterable[Delta]:
        return _chunk_deltas(data, itertools.count())

    def stream_parser(self) -> StreamParser:
        return _GeminiStreamParser()


class _GeminiStreamParser(StreamParser):
    def __init__(self) -> None:
        self._call_keys = itertools.count()

    def feed(self, event: str, data: str) -> Iterable[Delta]:
        return _chunk_deltas(json.loads(data), self._call_keys)


def _chunk_deltas(data: Mapping[str, Any], call_keys: Iterator[int]) -> Iterable[Delta]:
    if error := data.get("error"):
        raise APIResponseError(json.dumps(error))
    candidates = data.get("candidates") or ()
    if not candidates and (feedback := data.get("promptFeedback", {}).get("blockReason")):
        raise APIResponseError(f"Prompt blocked: {feedback}")
    finish_reason: FinishReason | None = None
    for candidate in candidates[:1]:
        for part in candidate.get("content", {}).get("parts") or ():
            yield from _part_deltas(part, call_keys)
        if reason := candidate.get("finishReason"):
            finish_reason = _finish_reason(reason)
    yield ResponseInfo(id=data.get("responseId"), model=data.get("modelVersion"), finish_reason=finish_reason)
    if usage := data.get("usageMetadata"):
        thoughts_tokens = usage.get("thoughtsTokenCount", 0)
        yield UsageReport(
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount", 0) + thoughts_tokens,
            reasoning_tokens=thoughts_tokens,
            cached_tokens=usage.get("cachedContentTokenCount"),
        )


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
    thinking: dict[str, Any] = {}
    if (effort := options.get("reasoning_effort")) is not None:
        thinking["thinkingLevel"] = effort
    if options.get("include_reasoning"):
        thinking["includeThoughts"] = True
    if thinking:
        config["thinkingConfig"] = thinking
    return config


def _finish_reason(reason: str) -> FinishReason:
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _BLOCKED_REASONS:
        return "content_filter"
    return "other"


def _part_deltas(part: Mapping[str, Any], call_keys: Iterator[int]) -> Iterable[Delta]:
    if part.get("thought"):
        # Thought summaries, returned when thinkingConfig.includeThoughts is set.
        if text := part.get("text"):
            yield ReasoningDelta(text)
        return
    if "text" in part:
        yield TextDelta(part["text"])
    elif call := part.get("functionCall"):
        metadata = {}
        if call_id := call.get("id"):
            metadata[_CALL_ID] = call_id
        if signature := part.get("thoughtSignature"):
            metadata[_THOUGHT_SIGNATURE] = signature
        yield ToolCallFragment(
            next(call_keys),
            id=call_id or f"call_{uuid.uuid4().hex}",
            name=call["name"],
            arguments=json.dumps(call.get("args") or {}),
            metadata=metadata,
            done=True,
        )
    elif (inline := part.get("inlineData")) and inline["mimeType"].startswith("image/"):
        yield ImageReady(Image(inline["mimeType"], data=base64.b64decode(inline["data"])))


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
    parts: list[dict[str, Any]] = [_part(part) for part in message.parts if isinstance(part, Text | Image | Video)]
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
