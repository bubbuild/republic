# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: reconcile snapshots, preserve native items and terminals.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""Responses SSE state, independent of transport and the public accumulator."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from republic import events
from republic.errors import IncompleteStreamError, ProviderError
from republic.providers import _openai_responses as wire

_TEXT_FIELDS = {"text", "refusal", "arguments"}
_TERMINALS = {"response.completed", "response.incomplete", "response.failed"}
_LIFECYCLE = {"response.created", "response.in_progress", "response.queued"}
# Native reasoning text is retained separately from the display summary.
_TEXT_EVENTS = {
    "output_text": ("content", "content_index", "output_text", "text"),
    "refusal": ("content", "content_index", "refusal", "refusal"),
    "reasoning_summary_text": ("summary", "summary_index", "summary_text", "text"),
    "reasoning_text": ("content", "content_index", "reasoning_text", "text"),
}


def _merge(old: Any, new: Any, *, key: str = "", closed: bool = False) -> Any:
    """Snapshots may fill a prefix, but may never rewrite received data."""
    if old is None:
        return deepcopy(new)
    if isinstance(old, dict) and isinstance(new, dict):
        result = deepcopy(old)
        for name, value in new.items():
            result[name] = _merge(old.get(name), value, key=name, closed=closed)
        return result
    if isinstance(old, list) and isinstance(new, list) and len(new) >= len(old):
        if closed and key in ("content", "summary") and len(new) != len(old):
            raise wire.invalid_response(detail="content added after item done")
        return [_merge(old[i], v, closed=closed) if i < len(old) else deepcopy(v) for i, v in enumerate(new)]
    if key in _TEXT_FIELDS and isinstance(old, str) and isinstance(new, str) and not closed and new.startswith(old):
        return new
    if key == "status" and old in ("in_progress", "queued") and new in ("in_progress", "completed", "incomplete"):
        return new
    if old != new:
        raise wire.invalid_response(detail=f"conflicting snapshot field {key!r}")
    return deepcopy(new)


def _index(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise wire.invalid_response(detail=f"expected nonnegative {key}")
    return value


@dataclass
class _Item:
    raw: dict[str, Any]
    emitted: str = ""
    started: bool = False
    closed: bool = False
    sealed: set[tuple[str, int]] = field(default_factory=set)

    def snapshot(self, raw: dict[str, Any], *, done: bool) -> None:
        merged = _merge(self.raw, raw, closed=self.closed)
        for name, index in self.sealed:
            old = self.raw[name][index] if name != "arguments" else self.raw[name]
            new = merged[name][index] if name != "arguments" else merged[name]
            _merge(old, new, key=name, closed=True)
        self.raw = merged
        self.closed = self.closed or done

    def slot(self, name: str, index: int, kind: str) -> dict[str, Any]:
        parts = self.raw.setdefault(name, [])
        if index > len(parts):
            raise wire.invalid_response(detail="content indices must start in order")
        if index == len(parts):
            parts.append({"type": kind})
        if parts[index]["type"] != kind:
            raise wire.invalid_response(detail="content type changed")
        return parts[index]

    def text_event(self, data: dict[str, Any], name: str, *, done: bool) -> None:
        group, index_key, kind, text_key = _TEXT_EVENTS[name]
        expected_item = "reasoning" if name.startswith("reasoning_") else "message"
        if self.raw["type"] != expected_item:
            raise wire.invalid_response(detail="text event has incompatible item type")
        index = _index(data, index_key)
        part = self.slot(group, index, kind)
        if not done and (self.closed or (group, index) in self.sealed):
            raise wire.invalid_response(detail="delta after content done")
        text = wire.required_string(data, text_key if done else "delta")
        part[text_key] = (
            _merge(part.get(text_key), text, key=text_key, closed=self.closed or (group, index) in self.sealed)
            if done
            else part.get(text_key, "") + text
        )
        if data.get("logprobs"):
            part["logprobs"] = (
                _merge(part.get("logprobs"), data["logprobs"]) if done else part.get("logprobs", []) + data["logprobs"]
            )
        if done:
            self.sealed.add((group, index))

    def part_event(self, data: dict[str, Any], *, summary: bool, done: bool) -> None:
        group, index_key = ("summary", "summary_index") if summary else ("content", "content_index")
        index = _index(data, index_key)
        raw = data["part"]
        allowed = ("summary_text",) if summary else ("output_text", "refusal")
        if self.raw["type"] != ("reasoning" if summary else "message") or raw.get("type") not in allowed:
            raise wire.invalid_response(detail="unsupported content part")
        slot = self.slot(group, index, raw["type"])
        self.raw[group][index] = _merge(slot, raw, closed=self.closed or (group, index) in self.sealed)
        if done:
            self.sealed.add((group, index))

    def arguments(self, data: dict[str, Any], *, done: bool) -> None:
        if self.raw["type"] != "function_call":
            raise wire.invalid_response(detail="arguments for a non-function item")
        if not done and (self.closed or ("arguments", 0) in self.sealed):
            raise wire.invalid_response(detail="arguments delta after done")
        value = wire.required_string(data, "arguments" if done else "delta")
        old = self.raw.get("arguments", "")
        self.raw["arguments"] = (
            _merge(old, value, key="arguments", closed=self.closed or ("arguments", 0) in self.sealed)
            if done
            else old + value
        )
        if data.get("name") is not None:
            self.raw["name"] = _merge(self.raw.get("name"), data["name"])
        if done:
            self.sealed.add(("arguments", 0))

    def visible(self) -> str:
        if self.raw["type"] == "function_call":
            return self.raw.get("arguments", "")
        group = "summary" if self.raw["type"] == "reasoning" else "content"
        text = ""
        for index, part in enumerate(self.raw.get(group, [])):
            if part["type"] in ("summary_text", "output_text"):
                text += part.get("text", "")
            if not self.closed and (group, index) not in self.sealed:
                break
        return text

    def publish(self, *, end: bool = False) -> list[events.Event]:
        kind = self.raw["type"]
        info = wire.item_metadata(self.raw)
        emitted: list[events.Event] = []
        if kind == "function_call":
            # Wait for actual call identity; never substitute the output item ID.
            if not self.raw.get("call_id") or not self.raw.get("name"):
                if end:
                    raise wire.invalid_response(detail="missing function call ID/name")
                return []
            start = events.ToolStart(
                tool_call_id=self.raw["call_id"], tool_name=self.raw["name"], provider_metadata=info
            )
            delta = events.ToolDelta(tool_call_id=self.raw["call_id"], chunk="")
            stop = events.ToolEnd(tool_call_id=self.raw["call_id"], provider_metadata=info)
        else:
            block = self.raw["id"]
            start_type, delta_type, end_type = (
                (events.ReasoningStart, events.ReasoningDelta, events.ReasoningEnd)
                if kind == "reasoning"
                else (events.TextStart, events.TextDelta, events.TextEnd)
            )
            start = start_type(block_id=block, provider_metadata=info)
            delta = delta_type(block_id=block, chunk="")
            stop = end_type(block_id=block, provider_metadata=info)
        if not self.started:
            emitted.append(start)
            self.started = True
        text = self.visible()
        if not text.startswith(self.emitted):
            raise wire.invalid_response(detail="snapshot rewrites emitted content")
        if text != self.emitted:
            delta.chunk = text[len(self.emitted) :]
            emitted.append(delta)
            self.emitted = text
        if end:
            emitted.append(stop)
        return emitted


class ResponsesStream:
    def __init__(self) -> None:
        self.items: list[_Item] = []
        self.identity: dict[str, Any] = {}
        self.terminal: events.StreamEnd | None = None

    def _identity(self, raw: dict[str, Any]) -> None:
        self.identity = _merge(self.identity, {k: raw[k] for k in ("id", "model") if k in raw})

    def _item(self, index: int, raw: dict[str, Any], *, done: bool) -> _Item:
        if index > len(self.items):
            raise wire.invalid_response(detail="output indices must start in order")
        if index == len(self.items):
            wire.required_string(raw, "id", nonempty=True)
            if raw.get("type") not in ("message", "reasoning", "function_call"):
                raise wire.invalid_response(detail=f"unsupported output item {raw.get('type')!r}")
            if any(item.raw["id"] == raw["id"] for item in self.items):
                raise wire.invalid_response(detail="duplicate output item ID")
            self.items.append(_Item(raw={}))
        item = self.items[index]
        item.snapshot(raw, done=done)
        return item

    def _lookup(self, data: dict[str, Any]) -> _Item:
        index = _index(data, "output_index")
        if index >= len(self.items) or data.get("item_id") != self.items[index].raw["id"]:
            raise wire.invalid_response(detail="event has no matching output item")
        return self.items[index]

    def _finish(self, data: dict[str, Any]) -> list[events.Event]:
        raw = data["response"]
        if data["type"] != f"response.{raw.get('status')}":
            raise wire.invalid_response(detail="terminal event and response status differ")
        self._identity(raw)
        if len(raw["output"]) < len(self.items):
            raise wire.invalid_response(detail="terminal snapshot omits received items")
        for index, item in enumerate(raw["output"]):
            self._item(index, item, done=True)
        # Include data from earlier events when the final snapshot omits optional
        # fields (e.g. annotations), after checking all overlapping fields.
        result = wire.response({**raw, "output": [item.raw for item in self.items]})
        emitted = [event for item in self.items for event in item.publish(end=True)]
        self.terminal = events.StreamEnd(
            usage=result.usage,
            finish_reason=result.finish_reason,
            response_id=result.response_id,
            response_model=result.response_model,
            provider_metadata=result.message.provider_metadata,
        )
        return emitted

    def feed(self, data: dict[str, Any]) -> list[events.Event]:
        kind = data.get("type", "")
        if kind in _LIFECYCLE:
            self._identity(data["response"])
            return []
        if kind in _TERMINALS:
            return self._finish(data)
        if kind == "error":
            raise ProviderError(str(data.get("message", data)), provider="openai", code=data.get("code"))
        if kind in ("response.output_item.added", "response.output_item.done"):
            item = self._item(_index(data, "output_index"), data["item"], done=kind.endswith(".done"))
        else:
            item = self._lookup(data)
            self._content_event(item, data)
        # Headers establish output order. If a call header is incomplete, hold
        # later items until its real call ID/name arrives.
        emitted = []
        for state in self.items:
            emitted.extend(state.publish())
            if not state.started:
                break
        return emitted

    def _content_event(self, item: _Item, data: dict[str, Any]) -> None:
        kind = data["type"].removeprefix("response.")
        name, _, suffix = kind.rpartition(".")
        done = suffix == "done"
        if name in _TEXT_EVENTS and suffix in ("delta", "done"):
            item.text_event(data, name, done=done)
        elif name in ("content_part", "reasoning_summary_part") and suffix in ("added", "done"):
            item.part_event(data, summary=name == "reasoning_summary_part", done=done)
        elif name == "function_call_arguments" and suffix in ("delta", "done"):
            item.arguments(data, done=done)
        elif kind == "output_text.annotation.added":
            slot = item.slot("content", _index(data, "content_index"), "output_text")
            annotations = slot.setdefault("annotations", [])
            index = _index(data, "annotation_index")
            if index != len(annotations):
                raise wire.invalid_response(detail="annotation indices must start in order")
            annotations.append(deepcopy(data["annotation"]))
        else:
            raise wire.invalid_response(detail=f"unsupported Responses event {data['type']!r}")

    def end(self) -> events.StreamEnd:
        if self.terminal is None:
            raise IncompleteStreamError
        return self.terminal
