"""Native Responses wire fixtures shared by API-key and Codex tests."""

from typing import Any


def message(text: str = "Hello", **extra: Any) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        **extra,
    }


def reasoning(text: str = "Summary", **extra: Any) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": text}],
        "encrypted_content": "opaque",
        **extra,
    }


def call(index: int = 1, args: str = '{"city":"SF"}', **extra: Any) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "weather",
        "arguments": args,
        "status": "completed",
        **extra,
    }


def response(output: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "object": "response",
        "id": "resp_1",
        "model": "resolved-model",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [message()] if output is None else output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        **extra,
    }


def event(kind: str, **data: Any) -> dict[str, Any]:
    return {"type": f"response.{kind}", "sequence_number": 1, **data}


def added(item: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return event("output_item.added", output_index=index, item=item)


def delta(
    text: str, *, item_id: str = "msg_1", index: int = 0, kind: str = "output_text", **extra: Any
) -> dict[str, Any]:
    return event(
        f"{kind}.delta", **{"output_index": index, "item_id": item_id, "content_index": 0, "delta": text, **extra}
    )


def terminal(output: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    raw = response(output, **extra)
    return event(raw["status"], response=raw)
