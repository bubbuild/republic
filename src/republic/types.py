# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: single-call data only; see NOTICE for source and changes.
"""JSON data shared by callers and provider adapters (Python 3.11+)."""

import base64
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue

ProviderMetadata = dict[str, JsonValue]


class _Data(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class TextPart(_Data):
    kind: Literal["text"] = "text"
    text: str
    provider_metadata: ProviderMetadata | None = None


class FilePart(_Data):
    """A URL or standard base64 data; never a file handle or fetched resource."""

    kind: Literal["file"] = "file"
    data: str
    media_type: str
    encoding: Literal["url", "base64"] = "url"
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None

    @classmethod
    def from_bytes(cls, data: bytes, *, media_type: str, filename: str | None = None) -> Self:
        """Normalize bytes to standard base64 before persistence."""
        return cls(
            data=base64.b64encode(data).decode("ascii"),
            encoding="base64",
            media_type=media_type,
            filename=filename,
        )


class ReasoningPart(_Data):
    """Reasoning text and opaque provider data, including signatures."""

    kind: Literal["reasoning"] = "reasoning"
    text: str
    provider_metadata: ProviderMetadata | None = None


class ToolCallPart(_Data):
    """A model's request to the caller; arguments stay verbatim JSON text."""

    kind: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    tool_args: str
    provider_metadata: ProviderMetadata | None = None


class ToolResultPart(_Data):
    """Caller-supplied tool output. Republic never produces it by execution."""

    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    result: JsonValue
    is_error: bool = False
    provider_metadata: ProviderMetadata | None = None


Part = Annotated[TextPart | FilePart | ReasoningPart | ToolCallPart | ToolResultPart, Field(discriminator="kind")]


class Message(_Data):
    """Persistable history; role/part compatibility is checked by each adapter."""

    role: Literal["system", "user", "assistant", "tool"]
    parts: list[Part]
    provider_metadata: ProviderMetadata | None = None

    @property
    def text(self) -> str:
        """Concatenate text parts, excluding reasoning and tool data."""
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Return tool-call data in model output order."""
        return [part for part in self.parts if isinstance(part, ToolCallPart)]


class Tool(_Data):
    """A name, description and JSON Schema; there is no callable field."""

    name: str
    description: str | None = None
    parameters: dict[str, JsonValue]
    provider_metadata: ProviderMetadata | None = None


class ToolChoice(_Data):
    """Select a named tool, including tools named 'auto' or 'none'."""

    name: str


class RequestOptions(_Data):
    """Common options. None means omitted; adapters must reject unsupported values."""

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    stop: list[str] | None = None
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    provider_options: dict[str, JsonValue] = Field(default_factory=dict)


class Request(_Data):
    """One model request, separate from runtime provider/client configuration."""

    model: str
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    options: RequestOptions = Field(default_factory=RequestOptions)


class Usage(_Data):
    """One response's usage snapshot. Missing counts remain unknown, not zero.

    input_tokens includes cached input; cache counts are a breakdown, not extra
    tokens to add again. An adapter reports an unknown total if native fields
    needed to compute that inclusive total are missing.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    raw: dict[str, JsonValue] | None = None

    @property
    def total_tokens(self) -> int | None:
        """Known input plus output, or None if either count is unknown."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


FinishReason = Literal["stop", "length", "content_filter", "tool_call", "error", "other"]


class Response(_Data):
    """One terminal model response. 'length' still describes truncated output."""

    message: Message
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    response_id: str | None = None
    response_model: str | None = None
