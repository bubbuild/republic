# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: model events only; see NOTICE for source and changes.
"""Provider events for exactly one response; all block IDs are explicit."""

from typing import Annotated, Literal

from pydantic import Field

from republic.types import FilePart, FinishReason, ProviderMetadata, Usage, _Data


class _Event(_Data):
    provider_metadata: ProviderMetadata | None = None


class TextStart(_Event):
    kind: Literal["text_start"] = "text_start"
    block_id: str


class TextDelta(_Event):
    kind: Literal["text_delta"] = "text_delta"
    block_id: str
    chunk: str


class TextEnd(_Event):
    kind: Literal["text_end"] = "text_end"
    block_id: str


class ReasoningStart(_Event):
    kind: Literal["reasoning_start"] = "reasoning_start"
    block_id: str


class ReasoningDelta(_Event):
    kind: Literal["reasoning_delta"] = "reasoning_delta"
    block_id: str
    chunk: str


class ReasoningEnd(_Event):
    kind: Literal["reasoning_end"] = "reasoning_end"
    block_id: str


class ToolStart(_Event):
    kind: Literal["tool_start"] = "tool_start"
    tool_call_id: str
    tool_name: str


class ToolDelta(_Event):
    kind: Literal["tool_delta"] = "tool_delta"
    tool_call_id: str
    chunk: str


class ToolEnd(_Event):
    kind: Literal["tool_end"] = "tool_end"
    tool_call_id: str


class FileEvent(_Data):
    kind: Literal["file"] = "file"
    part: FilePart


class StreamEnd(_Event):
    """Final event, emitted only after the provider confirms termination.

    Adapters must consume late usage chunks before emitting this event. Unknown
    native finish reasons map to 'other', with the raw reason in metadata.
    """

    kind: Literal["stream_end"] = "stream_end"
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    response_id: str | None = None
    response_model: str | None = None


Event = Annotated[
    TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolStart
    | ToolDelta
    | ToolEnd
    | FileEvent
    | StreamEnd,
    Field(discriminator="kind"),
]
