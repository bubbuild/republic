"""Provider-facing codecs for the internal conversation model."""

from __future__ import annotations

import json
from typing import Any

from republic.conversation import ContentPart, Conversation, ConversationMessage, ToolCallPart, ToolResultPart


def conversation_to_completion_messages(conversation: Conversation) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in conversation.messages:
        if (entry := _completion_message(message)) is not None:
            payload.append(entry)
    return payload


def conversation_to_openai_responses_input(conversation: Conversation) -> tuple[str | None, list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for message in conversation.messages:
        if message.role in {"system", "developer"}:
            instructions_parts.extend(_string_parts(message))
            continue

        input_items.extend(_responses_input_items(message))

    instructions = "\n\n".join(part for part in instructions_parts if part.strip()) or None
    return instructions, input_items


def conversation_to_anthropic_messages(conversation: Conversation) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in conversation.messages:
        if message.role in {"system", "developer"}:
            system_parts.extend(_string_parts(message))
            continue

        if message.role == "user":
            converted.append({"role": "user", "content": _anthropic_message_content(message)})
            continue

        if message.role == "assistant":
            converted.append({"role": "assistant", "content": _anthropic_assistant_content(message)})
            continue

        if message.role != "tool":
            continue
        tool_results = _anthropic_tool_results(message)
        if not tool_results:
            continue
        if (
            converted
            and converted[-1]["role"] == "user"
            and isinstance(previous_content := converted[-1]["content"], list)
            and previous_content
            and previous_content[0].get("type") == "tool_result"
        ):
            previous_content.extend(tool_results)
            continue
        converted.append({"role": "user", "content": tool_results})

    system = "\n\n".join(part for part in system_parts if part.strip()) or None
    return system, converted


def _completion_message(message: ConversationMessage) -> dict[str, Any] | None:
    if message.role == "tool":
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                return {
                    "role": "tool",
                    "tool_call_id": part.call_id,
                    "content": _tool_output_content(part.output),
                }
        return None

    content = _completion_content(message)
    payload: dict[str, Any] = {"role": message.role}
    if content not in (None, ""):
        payload["content"] = content

    if message.role == "assistant":
        tool_calls = [_completion_tool_call(part) for part in message.parts if isinstance(part, ToolCallPart)]
        if tool_calls:
            payload["tool_calls"] = tool_calls
    return payload


def _completion_content(message: ConversationMessage) -> Any:
    contents = [part.content for part in message.parts if isinstance(part, ContentPart)]
    if not contents:
        return ""
    if len(contents) == 1:
        return contents[0]
    return contents


def _response_message_content(message: ConversationMessage) -> Any:
    contents = [part.content for part in message.parts if isinstance(part, ContentPart)]
    if not contents:
        return ""
    if len(contents) == 1:
        return contents[0]
    return [_responses_message_content_item(content) for content in contents]


def _responses_input_items(message: ConversationMessage) -> list[dict[str, Any]]:
    if message.role == "tool":
        return _responses_tool_outputs(message)

    items: list[dict[str, Any]] = []
    if message.role in {"user", "assistant"}:
        content = _response_message_content(message)
        if content not in (None, "", []):
            items.append({"role": message.role, "content": content, "type": "message"})
    if message.role == "assistant":
        items.extend(_responses_tool_calls(message))
    return items


def _responses_tool_calls(message: ConversationMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, part in enumerate(message.parts):
        if not isinstance(part, ToolCallPart):
            continue
        items.append({
            "type": "function_call",
            "name": part.name,
            "arguments": _json_string(part.arguments),
            "call_id": part.call_id or f"call_{index}",
        })
    return items


def _responses_tool_outputs(message: ConversationMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in message.parts:
        if not isinstance(part, ToolResultPart):
            continue
        items.append({
            "type": "function_call_output",
            "call_id": part.call_id,
            "output": _responses_tool_output(part.output),
        })
    return items


def _responses_message_content_item(content: Any) -> dict[str, Any] | Any:
    if isinstance(content, str):
        return {"type": "input_text", "text": content}
    if isinstance(content, dict) and isinstance(content.get("type"), str):
        return content
    return {"type": "input_text", "text": _json_string(content)}


def _responses_tool_output(output: Any) -> Any:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        typed_items = [item for item in output if isinstance(item, dict) and isinstance(item.get("type"), str)]
        if len(typed_items) == len(output):
            return typed_items
    if output is None:
        return ""
    return _json_string(output)


def _anthropic_message_content(message: ConversationMessage) -> Any:
    contents = [part.content for part in message.parts if isinstance(part, ContentPart)]
    if not contents:
        return ""
    if len(contents) == 1:
        return _anthropic_content(contents[0])
    return [_anthropic_content(content) for content in contents]


def _anthropic_assistant_content(message: ConversationMessage) -> Any:
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, ContentPart):
            blocks.extend(_anthropic_text_blocks(part.content))
            continue
        if isinstance(part, ToolCallPart):
            blocks.append({
                "type": "tool_use",
                "id": part.call_id or part.name,
                "name": part.name,
                "input": _json_value(part.arguments),
            })
    return blocks or ""


def _anthropic_tool_results(message: ConversationMessage) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for part in message.parts:
        if not isinstance(part, ToolResultPart):
            continue
        results.append({
            "type": "tool_result",
            "tool_use_id": part.call_id,
            "content": _tool_output_content(part.output),
        })
    return results


def _anthropic_text_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                blocks.append(item)
            else:
                blocks.append({"type": "text", "text": _json_string(item)})
        return blocks
    if isinstance(content, dict) and isinstance(content.get("type"), str):
        return [content]
    return [{"type": "text", "text": _json_string(content)}]


def _anthropic_content(content: Any) -> Any:
    if isinstance(content, (str, list)):
        return content
    if isinstance(content, dict) and isinstance(content.get("type"), str):
        return [content]
    return _json_string(content)


def _string_parts(message: ConversationMessage) -> list[str]:
    return [_json_string(part.content) for part in message.parts if isinstance(part, ContentPart)]


def _tool_output_content(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return _json_string(output)


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return value


def _completion_tool_call(part: ToolCallPart) -> dict[str, Any]:
    payload = {
        "id": part.call_id,
        "type": "function",
        "function": {
            "name": part.name,
            "arguments": _json_string(part.arguments),
        },
    }
    payload.update(part.metadata)
    return payload
