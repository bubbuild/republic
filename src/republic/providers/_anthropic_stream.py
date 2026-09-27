# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: raw SDK events, signature aggregation, strict terminal.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""One Messages SSE sequence; no SDK tool runner or JSON repair."""

from dataclasses import dataclass
from typing import Any

from republic import events
from republic.errors import IncompleteStreamError
from republic.providers import _anthropic_messages as wire
from republic.types import ReasoningPart, TextPart, ToolCallPart


@dataclass
class _Block:
    part: TextPart | ReasoningPart | ToolCallPart
    kind: str
    signature: str = ""
    fragments: bool = False
    closed: bool = False

    def start(self, index: int) -> list[events.Event]:
        part = self.part
        if isinstance(part, ToolCallPart):
            return [
                events.ToolStart(
                    tool_call_id=part.tool_call_id, tool_name=part.tool_name, provider_metadata=part.provider_metadata
                )
            ]
        start_type, delta_type = (
            (events.TextStart, events.TextDelta)
            if isinstance(part, TextPart)
            else (events.ReasoningStart, events.ReasoningDelta)
        )
        result: list[events.Event] = [start_type(block_id=str(index), provider_metadata=part.provider_metadata)]
        if part.text:
            result.append(delta_type(block_id=str(index), chunk=part.text))
        return result

    def delta(self, index: int, raw: dict[str, Any]) -> list[events.Event]:
        kind = raw.get("type")
        if kind == "text_delta" and self.kind == "text":
            return [events.TextDelta(block_id=str(index), chunk=wire.string(raw, "text"))]
        if kind == "thinking_delta" and self.kind == "thinking":
            return [events.ReasoningDelta(block_id=str(index), chunk=wire.string(raw, "thinking"))]
        if kind == "signature_delta" and self.kind == "thinking":
            self.signature += wire.string(raw, "signature")
            return []
        if kind == "input_json_delta" and isinstance(self.part, ToolCallPart):
            chunk = wire.string(raw, "partial_json")
            if self.part.tool_args != "{}" and chunk:
                raise wire.invalid_response(detail="JSON deltas conflict with nonempty initial tool input")
            self.fragments = self.fragments or bool(chunk)
            return [events.ToolDelta(tool_call_id=self.part.tool_call_id, chunk=chunk)]
        raise wire.invalid_response(detail=f"unsupported delta {kind!r} for {self.kind}")

    def stop(self, index: int) -> list[events.Event]:
        self.closed = True
        part = self.part
        if isinstance(part, TextPart):
            return [events.TextEnd(block_id=str(index))]
        if isinstance(part, ReasoningPart):
            info = {"anthropic": {"signature": self.signature}} if self.kind == "thinking" else part.provider_metadata
            return [events.ReasoningEnd(block_id=str(index), provider_metadata=info)]
        result: list[events.Event] = []
        if not self.fragments:
            result.append(events.ToolDelta(tool_call_id=part.tool_call_id, chunk=part.tool_args))
        result.append(events.ToolEnd(tool_call_id=part.tool_call_id))
        return result


class MessagesStream:
    def __init__(self) -> None:
        self.message: dict[str, Any] | None = None
        self.blocks: list[_Block] = []
        self.call_ids: set[str] = set()
        self.terminal: events.StreamEnd | None = None

    def _start(self, raw: dict[str, Any]) -> None:
        if self.message is not None:
            raise wire.invalid_response(detail="duplicate message_start")
        wire.check_message(raw)
        if raw.get("content") != [] or raw.get("stop_reason") is not None:
            raise wire.invalid_response(detail="message_start must have empty content and no stop reason")
        self.message = raw

    def _index(self, raw: dict[str, Any]) -> int:
        index = raw.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise wire.invalid_response(detail="expected nonnegative block index")
        return index

    def _block_start(self, raw: dict[str, Any]) -> list[events.Event]:
        index = self._index(raw)
        if index != len(self.blocks):
            raise wire.invalid_response(detail="block indices must start once in order")
        content = raw["content_block"]
        part = wire.block_part(content)
        if isinstance(part, ToolCallPart):
            if part.tool_call_id in self.call_ids:
                raise wire.invalid_response(detail="duplicate tool call ID")
            self.call_ids.add(part.tool_call_id)
        block = _Block(part, content["type"], signature=content.get("signature", ""))
        self.blocks.append(block)
        return block.start(index)

    def _block_event(self, raw: dict[str, Any]) -> list[events.Event]:
        index = self._index(raw)
        if index >= len(self.blocks) or self.blocks[index].closed:
            raise wire.invalid_response(detail="event has no open content block")
        block = self.blocks[index]
        if raw["type"] == "content_block_stop":
            return block.stop(index)
        return block.delta(index, raw["delta"])

    def _message_delta(self, raw: dict[str, Any], message: dict[str, Any]) -> None:
        delta = raw["delta"]
        if delta.keys() - {"stop_reason", "stop_sequence", "stop_details"}:
            raise wire.invalid_response(detail="unsupported message_delta fields")
        if message.get("stop_reason") and delta.get("stop_reason") not in (None, message["stop_reason"]):
            raise wire.invalid_response(detail="stop reason changed")
        message.update(delta)
        if raw.get("usage") is not None:
            # Cumulative patches replace counters, never add them to start usage.
            message["usage"] = {**(message.get("usage") or {}), **raw["usage"]}

    def feed(self, raw: dict[str, Any]) -> list[events.Event]:
        kind = raw.get("type")
        if kind == "ping":
            return []  # The official SDK normally filters pings itself.
        if kind == "message_start":
            self._start(raw["message"])
            return []
        message = self.message
        if message is None:
            raise wire.invalid_response(detail="event before message_start")
        if kind in ("content_block_start", "content_block_delta", "content_block_stop"):
            if message.get("stop_reason"):
                raise wire.invalid_response(detail="content event after stop reason")
            return self._block_start(raw) if kind == "content_block_start" else self._block_event(raw)
        if kind == "message_delta":
            self._message_delta(raw, message)
        elif kind == "message_stop":
            if any(not block.closed for block in self.blocks):
                raise wire.invalid_response(detail="message_stop with open content blocks")
            self.terminal = events.StreamEnd(
                usage=wire.usage(message.get("usage")),
                finish_reason=wire.finish_reason(message),
                response_id=message["id"],
                response_model=message["model"],
                provider_metadata=wire.message_metadata(message),
            )
        else:
            raise wire.invalid_response(detail=f"unsupported event {kind!r}")
        return []

    def end(self) -> events.StreamEnd:
        if self.terminal is None:
            raise IncompleteStreamError
        return self.terminal
