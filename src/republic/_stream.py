# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: one response, strict ordering and explicit cleanup.
# See NOTICE for the upstream revision and the scope of the extraction.
"""Aggregation and lifetime of a single provider stream."""

import asyncio
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Literal

from republic import events
from republic.errors import IncompleteStreamError, StreamProtocolError
from republic.types import Message, ProviderMetadata, ReasoningPart, Response, TextPart, ToolCallPart

_BlockEvent = (
    events.TextStart
    | events.TextDelta
    | events.TextEnd
    | events.ReasoningStart
    | events.ReasoningDelta
    | events.ReasoningEnd
    | events.ToolStart
    | events.ToolDelta
    | events.ToolEnd
)
_Block = TextPart | ReasoningPart | ToolCallPart


def _merge_metadata(old: ProviderMetadata | None, new: ProviderMetadata | None) -> ProviderMetadata | None:
    """Merge objects recursively; leaves are snapshots, never string fragments."""
    if new is None:
        return deepcopy(old)
    merged = deepcopy(old) if old is not None else {}
    for key, value in new.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _merge_metadata(previous, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _block_key(event: _BlockEvent) -> tuple[str, str]:
    if isinstance(event, events.ToolStart | events.ToolDelta | events.ToolEnd):
        return "tool", event.tool_call_id
    if isinstance(event, events.TextStart | events.TextDelta | events.TextEnd):
        return "text", event.block_id
    return "reasoning", event.block_id


class _Accumulator:
    def __init__(self) -> None:
        self.message = Message(role="assistant", parts=[])
        self.response: Response | None = None
        self._parts: dict[tuple[str, str], _Block] = {}
        self._active: set[tuple[str, str]] = set()

    def feed(self, event: events.Event) -> None:
        if isinstance(event, events.StreamEnd):
            if self._active:
                raise StreamProtocolError("unfinished_blocks", sorted(self._active))
            self.message.provider_metadata = deepcopy(event.provider_metadata)
            self.response = Response(
                message=self.message.model_copy(deep=True),
                usage=event.usage.model_copy(deep=True) if event.usage is not None else None,
                finish_reason=event.finish_reason,
                response_id=event.response_id,
                response_model=event.response_model,
            )
        elif isinstance(event, events.FileEvent):
            self.message.parts.append(event.part.model_copy(deep=True))
        elif isinstance(event, events.TextStart | events.ReasoningStart | events.ToolStart):
            self._start(event)
        else:
            self._update(event)

    def _start(self, event: events.TextStart | events.ReasoningStart | events.ToolStart) -> None:
        key = _block_key(event)
        if not key[1] or key in self._parts:
            raise StreamProtocolError("empty_or_reused_id", [key])
        part: _Block
        if isinstance(event, events.ToolStart):
            part = ToolCallPart(tool_call_id=event.tool_call_id, tool_name=event.tool_name, tool_args="")
        elif isinstance(event, events.TextStart):
            part = TextPart(text="")
        else:
            part = ReasoningPart(text="")
        part.provider_metadata = deepcopy(event.provider_metadata)
        self._parts[key] = part
        self._active.add(key)
        self.message.parts.append(part)

    def _update(self, event: _BlockEvent) -> None:
        key = _block_key(event)
        if key not in self._active:
            raise StreamProtocolError("unknown_or_closed_block", [key])
        part = self._parts[key]
        if isinstance(event, events.TextDelta | events.ReasoningDelta) and isinstance(part, TextPart | ReasoningPart):
            part.text += event.chunk
        elif isinstance(event, events.ToolDelta) and isinstance(part, ToolCallPart):
            part.tool_args += event.chunk
        else:
            self._active.remove(key)
        part.provider_metadata = _merge_metadata(part.provider_metadata, event.provider_metadata)


class Stream:
    """Single-consumer event iterator, used inside ``republic.stream``.

    Message and response properties return detached snapshots. Early close
    preserves partial output without draining the provider or inventing a result.
    Cancel a task consuming the stream before closing it from another task.
    """

    def __init__(self, source: AsyncGenerator[events.Event, None]) -> None:
        self._source = source
        self._accumulator = _Accumulator()
        self._close_task: asyncio.Task[None] | None = None
        self._status: Literal["open", "completed", "incomplete", "closed", "cancelled", "failed"] = "open"

    @property
    def status(self) -> str:
        """Lifecycle outcome; 'completed' means a valid terminal event arrived."""
        return self._status

    @property
    def message(self) -> Message:
        """Snapshot of the output received so far, including partial tool JSON."""
        return self._accumulator.message.model_copy(deep=True)

    @property
    def response(self) -> Response | None:
        """Terminal response, or None on early close, failure or incomplete input."""
        response = self._accumulator.response
        return response.model_copy(deep=True) if response is not None else None

    def __aiter__(self) -> "Stream":
        return self

    async def __anext__(self) -> events.Event:
        if self._close_task is not None:
            raise StopAsyncIteration
        try:
            event = await anext(self._source)
            self._accumulator.feed(event)
        except StopAsyncIteration:
            self._status = "incomplete"
            await self.aclose()
            raise IncompleteStreamError from None
        except IncompleteStreamError:
            self._status = "incomplete"
            await self.aclose()
            raise
        except asyncio.CancelledError:
            self._cancel()
            await self.aclose()
            raise
        except Exception:
            self._status = "failed"
            await self.aclose()
            raise
        if isinstance(event, events.StreamEnd):
            self._status = "completed"
            await self.aclose()
        return event

    def _cancel(self) -> None:
        if self._status in {"open", "closed"}:
            self._status = "cancelled"

    async def aclose(self) -> None:
        """Close once, without draining; finish cleanup before propagating cancellation."""
        if self._status == "open":
            self._status = "closed"
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._source.aclose())
        cancellation = None
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError as exc:
                if self._close_task.cancelled():
                    raise
                self._cancel()
                cancellation = exc
        if cancellation is not None:
            raise cancellation
