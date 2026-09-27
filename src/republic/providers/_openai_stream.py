# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: delayed tool headers, strict termination and tail usage.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""State for one Chat Completions response (never a tool executor)."""

from dataclasses import dataclass, field
from typing import Any, cast

from republic import events
from republic.errors import IncompleteStreamError
from republic.providers._openai_chat import (
    check_message_fields,
    finish_reason,
    invalid_response,
    response_metadata,
    usage,
)


@dataclass
class _Tool:
    call_id: str = ""
    name: str = ""
    pending: list[str] = field(default_factory=list)
    started: bool = False

    def update(self, data: dict[str, Any]) -> None:
        if data.get("type", "function") != "function":
            raise invalid_response(detail="only function tool calls are supported")
        function = data.get("function") or {}
        self.call_id = self._header(self.call_id, data.get("id"))
        self.name = self._header(self.name, function.get("name"))
        if (arguments := function.get("arguments")) is not None:
            if not isinstance(arguments, str):
                raise invalid_response(detail="tool arguments must be a string")
            self.pending.append(arguments)

    @staticmethod
    def _header(previous: str, value: Any) -> str:
        if value is None or value == "":
            return previous
        if not isinstance(value, str) or (previous and previous != value):
            raise invalid_response(detail="tool ID/name changed; fragmented headers are unsupported")
        return value


class ChatStream:
    def __init__(self) -> None:
        self.reason: str | None = None
        self.identity: dict[str, Any] = {}
        self.raw_usage: dict[str, Any] | None = None
        self.tools: dict[int, _Tool] = {}
        self.blocks: set[str] = set()
        self.refusal: str | None = None
        self.logprobs: dict[str, list[Any]] = {}
        self._call_ids: set[str] = set()

    def feed(self, chunk: dict[str, Any]) -> list[events.Event]:
        self._record(chunk)
        choices = chunk.get("choices") or []
        if not choices:
            return []
        if len(choices) != 1 or choices[0].get("index") != 0:
            raise invalid_response(detail="expected exactly one choice at index 0")
        if self.reason is not None:
            raise invalid_response(detail="choice after finish reason")
        choice = choices[0]
        delta = choice.get("delta") or {}
        check_message_fields(delta)
        result = self._content(delta)
        result.extend(self._tools(delta.get("tool_calls") or []))
        for key, values in (choice.get("logprobs") or {}).items():
            if values is not None:
                if not isinstance(values, list):
                    raise invalid_response(detail="logprobs entries must be arrays")
                self.logprobs.setdefault(key, []).extend(values)
        if (reason := choice.get("finish_reason")) is not None:
            if not isinstance(reason, str) or not reason:
                raise invalid_response(detail="invalid finish reason")
            self.reason = reason
            result.extend(self._end_blocks())
        return result

    def _record(self, chunk: dict[str, Any]) -> None:
        for key in ("id", "model", "system_fingerprint", "service_tier"):
            if (value := chunk.get(key)) is not None:
                if key in {"id", "model"} and self.identity.get(key, value) != value:
                    raise invalid_response(detail=f"response {key} changed within one stream")
                self.identity[key] = value
        if chunk.get("usage") is not None:
            self.raw_usage = chunk["usage"]

    def _content(self, delta: dict[str, Any]) -> list[events.Event]:
        result: list[events.Event] = []
        for key in ("reasoning", "reasoning_content", "content"):
            if (text := delta.get(key)) is None:
                continue
            if not isinstance(text, str):
                raise invalid_response(detail=f"{key} must be a string")
            if key not in self.blocks:
                self.blocks.add(key)
                if key == "content":
                    result.append(events.TextStart(block_id=key))
                else:
                    result.append(events.ReasoningStart(block_id=key, provider_metadata={"openai": {"field": key}}))
            if key == "content":
                result.append(events.TextDelta(block_id=key, chunk=text))
            else:
                result.append(events.ReasoningDelta(block_id=key, chunk=text))
        if (refusal := delta.get("refusal")) is not None:
            if not isinstance(refusal, str):
                raise invalid_response(detail="refusal must be a string")
            self.refusal = (self.refusal or "") + refusal
        return result

    def _tools(self, calls: list[dict[str, Any]]) -> list[events.Event]:
        for call in calls:
            index = call.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise invalid_response(detail="tool delta needs a nonnegative index")
            self.tools.setdefault(index, _Tool()).update(call)
        result: list[events.Event] = []
        # Keep first-observed tool order, even if an earlier header is delayed.
        for tool in self.tools.values():
            if not tool.started:
                if not tool.call_id or not tool.name:
                    break
                if tool.call_id in self._call_ids:
                    raise invalid_response(detail="duplicate tool call ID across indices")
                self._call_ids.add(tool.call_id)
                tool.started = True
                result.append(events.ToolStart(tool_call_id=tool.call_id, tool_name=tool.name))
            result.extend(events.ToolDelta(tool_call_id=tool.call_id, chunk=chunk) for chunk in tool.pending)
            tool.pending.clear()
        return result

    def _end_blocks(self) -> list[events.Event]:
        result: list[events.Event] = []
        for key in sorted(self.blocks):
            if key == "content":
                result.append(events.TextEnd(block_id=key))
            else:
                result.append(events.ReasoningEnd(block_id=key))
        for tool in self.tools.values():
            if not tool.started:
                raise invalid_response(detail="finished tool call without ID or name")
            result.append(events.ToolEnd(tool_call_id=tool.call_id))
        return result

    def end(self) -> events.StreamEnd:
        if self.reason is None:
            raise IncompleteStreamError
        metadata = response_metadata(self.identity, self.reason) or {"openai": {}}
        details = cast(dict[str, Any], metadata["openai"])
        if self.refusal is not None:
            details["refusal"] = self.refusal
        if self.logprobs:
            details["logprobs"] = self.logprobs
        return events.StreamEnd(
            finish_reason=finish_reason(self.reason),
            response_id=self.identity.get("id"),
            response_model=self.identity.get("model"),
            usage=usage(self.raw_usage),
            provider_metadata=metadata if details else None,
        )
