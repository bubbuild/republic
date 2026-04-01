"""Typed conversation model used across core and provider codecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ContentPart:
    content: Any


@dataclass(frozen=True)
class ToolCallPart:
    name: str
    arguments: Any
    call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResultPart:
    call_id: str
    output: Any


ConversationPart: TypeAlias = ContentPart | ToolCallPart | ToolResultPart


@dataclass(frozen=True)
class ConversationMessage:
    role: MessageRole
    parts: tuple[ConversationPart, ...]


@dataclass(frozen=True)
class Conversation:
    messages: tuple[ConversationMessage, ...]


class UnsupportedMessageRoleError(ValueError):
    @classmethod
    def at_index(cls, role: Any, *, index: int) -> UnsupportedMessageRoleError:
        return cls(f"Unsupported message role at index {index}: {role!r}")


def conversation_from_messages(messages: list[dict[str, Any]]) -> Conversation:
    return Conversation(tuple(_message_from_payload(message, index) for index, message in enumerate(messages)))


def _message_from_payload(message: dict[str, Any], index: int) -> ConversationMessage:
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise UnsupportedMessageRoleError.at_index(role, index=index)

    parts: list[ConversationPart] = []
    content = message.get("content")
    if role == "tool":
        call_id = message.get("tool_call_id") or message.get("call_id")
        if call_id:
            parts.append(ToolResultPart(call_id=call_id, output=content))
        return ConversationMessage(role=role, parts=tuple(parts))

    if content not in (None, ""):
        parts.append(ContentPart(content))

    if role == "assistant":
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            metadata = {
                key: value for key, value in tool_call.items() if key not in {"id", "call_id", "type", "function"}
            }
            parts.append(
                ToolCallPart(
                    name=name,
                    arguments=function.get("arguments", ""),
                    call_id=tool_call.get("id") or tool_call.get("call_id"),
                    metadata=metadata,
                )
            )

    return ConversationMessage(role=role, parts=tuple(parts))
