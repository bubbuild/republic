"""Request-shape adapters for different upstream APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from republic.core import provider_policies

DEFAULT_MESSAGES_MAX_TOKENS = 8192

__all__ = [
    "TransportInvocation",
    "build_completion_invocation",
    "build_messages_invocation",
    "build_responses_invocation",
    "normalize_responses_kwargs",
    "split_messages_for_responses",
]

_REASONING_EFFORT_TO_THINKING = {
    "none": {"type": "disabled"},
    "minimal": {"type": "enabled", "budget_tokens": 1024},
    "low": {"type": "enabled", "budget_tokens": 2048},
    "medium": {"type": "enabled", "budget_tokens": 8192},
    "high": {"type": "enabled", "budget_tokens": 24576},
    "xhigh": {"type": "enabled", "budget_tokens": 32768},
    "auto": {"type": "enabled", "budget_tokens": 8192},
}


@dataclass(frozen=True)
class TransportInvocation:
    method_name: Literal["completion", "responses", "messages"]
    kwargs: dict[str, Any]


def build_completion_invocation(
    *,
    provider_name: str,
    model_id: str,
    messages_payload: list[dict[str, Any]],
    tools_payload: list[dict[str, Any]] | None,
    max_tokens: int | None,
    stream: bool,
    reasoning_effort: Any | None,
    kwargs: dict[str, Any],
) -> TransportInvocation:
    completion_kwargs = _build_completion_kwargs(
        provider_name=provider_name,
        max_tokens=max_tokens,
        stream=stream,
        kwargs=kwargs,
    )
    return TransportInvocation(
        method_name="completion",
        kwargs={
            "model": model_id,
            "messages": messages_payload,
            "tools": tools_payload,
            "stream": stream,
            "reasoning_effort": reasoning_effort,
            **completion_kwargs,
        },
    )


def build_responses_invocation(
    *,
    model_id: str,
    messages_payload: list[dict[str, Any]],
    tools_payload: list[dict[str, Any]] | None,
    max_tokens: int | None,
    stream: bool,
    reasoning_effort: Any | None,
    kwargs: dict[str, Any],
    drop_extra_headers: bool,
) -> TransportInvocation:
    instructions, input_items = split_messages_for_responses(messages_payload)
    responses_kwargs = _build_responses_kwargs(
        kwargs,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        drop_extra_headers=drop_extra_headers,
    )
    return TransportInvocation(
        method_name="responses",
        kwargs={
            "model": model_id,
            "input_data": input_items,
            "tools": _convert_tools_for_responses(tools_payload),
            "stream": stream,
            "instructions": instructions,
            **responses_kwargs,
        },
    )


def build_messages_invocation(
    *,
    model_id: str,
    messages_payload: list[dict[str, Any]],
    tools_payload: list[dict[str, Any]] | None,
    max_tokens: int | None,
    stream: bool,
    reasoning_effort: Any | None,
    kwargs: dict[str, Any],
) -> TransportInvocation:
    system_prompt, native_messages = _build_messages_payload(messages_payload)
    messages_kwargs = _build_messages_kwargs(kwargs, reasoning_effort=reasoning_effort)
    resolved_max_tokens, request_kwargs = _resolve_messages_max_tokens(max_tokens, messages_kwargs)
    explicit_system = request_kwargs.pop("system", None)
    return TransportInvocation(
        method_name="messages",
        kwargs={
            "model": model_id,
            "messages": native_messages,
            "max_tokens": resolved_max_tokens,
            "stream": stream,
            "system": _merge_system_prompts(system_prompt, explicit_system),
            "tools": _convert_tools_for_messages(tools_payload),
            **request_kwargs,
        },
    )


def split_messages_for_responses(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    filtered_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            if content := _stringify_content(message.get("content")):
                instructions_parts.append(content)
            continue
        filtered_messages.append(message)

    instructions = "\n\n".join(part for part in instructions_parts if part.strip())
    if not instructions:
        instructions = None
    return instructions, _convert_messages_to_responses_input(filtered_messages)


def normalize_responses_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize completion-style kwargs into responses-compatible shapes."""
    tool_choice = kwargs.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return kwargs

    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return kwargs

    function_name = function.get("name")
    if not isinstance(function_name, str) or not function_name:
        return kwargs

    normalized_tool_choice = dict(tool_choice)
    normalized_tool_choice.pop("function", None)
    normalized_tool_choice["type"] = normalized_tool_choice.get("type", "function")
    normalized_tool_choice["name"] = function_name
    return {**kwargs, "tool_choice": normalized_tool_choice}


def _build_completion_kwargs(
    *,
    provider_name: str,
    max_tokens: int | None,
    stream: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    completion_kwargs = dict(kwargs)
    max_tokens_arg = provider_policies.completion_max_tokens_arg(provider_name)
    if max_tokens_arg not in completion_kwargs:
        completion_kwargs[max_tokens_arg] = max_tokens

    if (
        stream
        and provider_policies.should_include_completion_stream_usage(provider_name)
        and "stream_options" not in completion_kwargs
    ):
        completion_kwargs["stream_options"] = {"include_usage": True}
    return completion_kwargs


def _build_responses_kwargs(
    kwargs: dict[str, Any],
    *,
    max_tokens: int | None,
    reasoning_effort: Any | None,
    drop_extra_headers: bool,
) -> dict[str, Any]:
    responses_kwargs = dict(kwargs)
    if drop_extra_headers:
        # any-llm responses params currently reject extra_headers, so drop it on the wrapped path only.
        responses_kwargs.pop("extra_headers", None)
    responses_kwargs = normalize_responses_kwargs(responses_kwargs)

    if reasoning_effort is not None and "reasoning" not in responses_kwargs:
        responses_kwargs["reasoning"] = {"effort": reasoning_effort}
    if "max_output_tokens" not in responses_kwargs and max_tokens is not None:
        responses_kwargs["max_output_tokens"] = max_tokens
    return responses_kwargs


def _convert_tools_for_responses(
    tools_payload: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools_payload:
        return tools_payload

    converted_tools: list[dict[str, Any]] = []
    for tool in tools_payload:
        function = tool.get("function")
        if isinstance(function, dict):
            converted: dict[str, Any] = {
                "type": tool.get("type", "function"),
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
            if "strict" in function:
                converted["strict"] = function["strict"]
            converted_tools.append(converted)
            continue
        converted_tools.append(dict(tool))
    return converted_tools


def _convert_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content not in (None, ""):
            input_items.append({"role": role, "content": content, "type": "message"})

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            for index, tool_call in enumerate(tool_calls):
                func = tool_call.get("function") or {}
                name = func.get("name")
                if not name:
                    continue
                call_id = tool_call.get("id") or tool_call.get("call_id") or f"call_{index}"
                input_items.append({
                    "type": "function_call",
                    "name": name,
                    "arguments": func.get("arguments", ""),
                    "call_id": call_id,
                })

        if role == "tool":
            call_id = message.get("tool_call_id") or message.get("call_id")
            if not call_id:
                continue
            input_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": message.get("content", ""),
            })
    return input_items


def _merge_system_prompts(
    message_system: str | None,
    explicit_system: str | None,
) -> str | None:
    if message_system and explicit_system:
        return f"{message_system}\n\n{explicit_system}"
    return explicit_system or message_system


def _build_messages_payload(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    native_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _stringify_content(message.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "assistant" and message.get("tool_calls"):
            native_messages.append(_convert_assistant_tool_call_message(message))
            continue
        if role == "tool":
            _append_tool_result_message(native_messages, message)
            continue
        native_messages.append(_convert_plain_message(message))

    system = "\n\n".join(part for part in system_parts if part.strip())
    return system or None, native_messages


def _append_tool_result_message(
    native_messages: list[dict[str, Any]],
    message: dict[str, Any],
) -> None:
    tool_result_message = _convert_tool_result_message(message)
    if tool_result_message is None:
        return
    if _is_tool_result_user_message(native_messages[-1] if native_messages else None):
        native_messages[-1]["content"].extend(tool_result_message["content"])
        return
    native_messages.append(tool_result_message)


def _convert_tools_for_messages(
    tools_payload: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools_payload:
        return tools_payload

    converted_tools: list[dict[str, Any]] = []
    for tool in tools_payload:
        if "input_schema" in tool and "name" in tool:
            converted_tools.append(dict(tool))
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            converted_tools.append(dict(tool))
            continue

        converted_tools.append({
            "name": function.get("name"),
            "description": function.get("description", ""),
            "input_schema": function.get("parameters", {}),
        })
    return converted_tools


def _resolve_messages_max_tokens(
    max_tokens: int | None,
    kwargs: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    clean_kwargs = dict(kwargs)
    explicit_max_tokens = clean_kwargs.pop("max_tokens", None)
    resolved_max_tokens = explicit_max_tokens if explicit_max_tokens is not None else max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = DEFAULT_MESSAGES_MAX_TOKENS
    return resolved_max_tokens, clean_kwargs


def _build_messages_kwargs(
    kwargs: dict[str, Any],
    *,
    reasoning_effort: Any | None,
) -> dict[str, Any]:
    clean_kwargs = dict(kwargs)

    stop = clean_kwargs.pop("stop", None)
    if stop is not None and "stop_sequences" not in clean_kwargs:
        clean_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    parallel_tool_calls = clean_kwargs.pop("parallel_tool_calls", None)
    normalized_tool_choice = _normalize_messages_tool_choice(
        clean_kwargs.get("tool_choice"),
        parallel_tool_calls=parallel_tool_calls,
    )
    if normalized_tool_choice is not None:
        clean_kwargs["tool_choice"] = normalized_tool_choice

    if reasoning_effort is not None and "thinking" not in clean_kwargs:
        thinking = _REASONING_EFFORT_TO_THINKING.get(str(reasoning_effort))
        if thinking is not None:
            clean_kwargs["thinking"] = thinking

    return clean_kwargs


def _convert_assistant_tool_call_message(message: dict[str, Any]) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str):
        if content:
            content_blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        converted_blocks = _convert_content_blocks(content)
        if isinstance(converted_blocks, list):
            content_blocks.extend(converted_blocks)

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        content_blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id") or tool_call.get("call_id"),
            "name": name,
            "input": _parse_tool_arguments(function.get("arguments")),
        })

    return {"role": "assistant", "content": content_blocks or ""}


def _convert_plain_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": message.get("role"),
        "content": _convert_content_blocks(message.get("content")),
    }


def _convert_tool_result_message(message: dict[str, Any]) -> dict[str, Any] | None:
    call_id = message.get("tool_call_id") or message.get("call_id")
    if not call_id:
        return None

    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"
        )

    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": str(content),
            }
        ],
    }


def _is_tool_result_user_message(message: dict[str, Any] | None) -> bool:
    if not message or message.get("role") != "user":
        return False
    content = message.get("content")
    return bool(
        isinstance(content, list)
        and content
        and isinstance(content[0], dict)
        and content[0].get("type") == "tool_result"
    )


def _convert_content_blocks(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    converted: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "image_url":
            converted.append(_convert_image_block(block))
            continue
        if block_type == "file":
            converted.append(_convert_file_block(block))
            continue
        converted.append(dict(block))
    return converted


def _convert_image_block(block: dict[str, Any]) -> dict[str, Any]:
    url = block.get("image_url", {}).get("url", "")
    if isinstance(url, str) and url.startswith("data:") and "base64," in url:
        mime_part = url[5:].split(";", 1)[0]
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_part or "image/png",
                "data": url.split("base64,", 1)[1],
            },
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_file_block(block: dict[str, Any]) -> dict[str, Any]:
    file_data = block.get("file", {}).get("file_data", "")
    if isinstance(file_data, str) and file_data.startswith("data:") and "base64," in file_data:
        mime_part = file_data[5:].split(";", 1)[0]
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": mime_part or "application/octet-stream",
                "data": file_data.split("base64,", 1)[1],
            },
        }
    return {"type": "document", "source": {"type": "url", "url": file_data}}


def _normalize_messages_tool_choice(
    tool_choice: Any,
    *,
    parallel_tool_calls: Any,
) -> dict[str, Any] | None:
    if isinstance(tool_choice, str):
        normalized: dict[str, Any] | None = {"type": {"required": "any"}.get(tool_choice, tool_choice)}
    elif isinstance(tool_choice, dict):
        normalized = _normalize_messages_tool_choice_dict(tool_choice)
    else:
        normalized = None

    if normalized is None:
        if parallel_tool_calls is None:
            return None
        normalized = {"type": "auto"}

    if parallel_tool_calls is not None and normalized.get("type") in {"auto", "any"}:
        normalized["disable_parallel_tool_use"] = not bool(parallel_tool_calls)
    return normalized


def _normalize_messages_tool_choice_dict(tool_choice: dict[str, Any]) -> dict[str, Any] | None:
    tool_choice_type = tool_choice.get("type")
    if tool_choice_type in {"auto", "any", "none"}:
        return {"type": tool_choice_type}
    if tool_choice_type == "tool":
        return {"type": "tool", "name": tool_choice.get("name")}

    nested_name = _nested_tool_choice_name(tool_choice, tool_choice_type)
    if nested_name:
        return {"type": "tool", "name": nested_name}
    if tool_choice.get("name") and tool_choice_type:
        return {"type": tool_choice_type, "name": tool_choice["name"]}
    return None


def _nested_tool_choice_name(tool_choice: dict[str, Any], tool_choice_type: Any) -> str | None:
    if tool_choice_type in {"function", "custom"}:
        nested = tool_choice.get(tool_choice_type)
        if isinstance(nested, dict) and nested.get("name"):
            return nested["name"]

    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return function["name"]
    return None


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""
