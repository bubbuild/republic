"""Helpers for OpenAI Codex OAuth-backed sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.responses import Response

from republic.auth.openai_codex import extract_openai_codex_account_id

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
DEFAULT_CODEX_ORIGINATOR = "republic"
DEFAULT_CODEX_INCLUDE = ("reasoning.encrypted_content",)
DEFAULT_CODEX_INSTRUCTIONS = "You are Codex."
DEFAULT_CODEX_TEXT_CONFIG = {"verbosity": "medium"}


class OpenAICodexTransportError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OpenAICodexProvider(BaseOpenAIProvider):
    SUPPORTS_RESPONSES = True
    PREFERRED_TRANSPORT = "responses"
    PRESERVE_EXTRA_HEADERS_IN_RESPONSES = True

    def __init__(
        self,
        default_instructions: str = DEFAULT_CODEX_INSTRUCTIONS,
        default_include: tuple[str, ...] = DEFAULT_CODEX_INCLUDE,
        default_text: dict[str, Any] | None = None,
        store: bool = False,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._default_instructions = default_instructions
        self._default_include = list(default_include)
        self._default_text = dict(default_text or DEFAULT_CODEX_TEXT_CONFIG)
        self._store = store
        default_headers = dict(kwargs.pop("default_headers", {}))
        default_headers.update(build_openai_codex_default_headers(api_key or ""))
        super().__init__(
            api_key=api_key,
            api_base=resolve_openai_codex_api_base(api_base),
            default_headers=default_headers,
            **kwargs,
        )

    async def aresponses(self, **kwargs: Any) -> Any:
        payload = self._build_payload(kwargs)
        response = await self.client.responses.create(**payload)
        if kwargs.get("stream"):
            return response
        return self._collect_responses_events(await self._collect_events(response))

    def _build_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload = dict(kwargs)
        input_data = payload.pop("input_data", None)
        payload["input"] = input_data
        payload["stream"] = True
        # ChatGPT Codex backend currently rejects response token limit parameters on this path.
        payload.pop("max_output_tokens", None)
        payload.pop("max_tokens", None)
        payload["instructions"] = payload.get("instructions") or self._default_instructions
        payload["store"] = payload.get("store", self._store)
        payload["include"] = payload.get("include", list(self._default_include))

        text = payload.get("text")
        if isinstance(text, dict):
            payload["text"] = {**self._default_text, **text}
        elif text is None:
            payload["text"] = dict(self._default_text)

        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    async def _collect_events(response: AsyncIterator[Any]) -> list[Any]:
        events: list[Any] = []
        async for event in response:
            events.append(event)
        return events

    @staticmethod
    def _collect_responses_events(events: list[Any]) -> Response:
        text_parts: list[str] = []
        tool_calls: dict[str, dict[str, Any]] = {}
        usage: dict[str, Any] | Any | None = None
        completed_response: Any | None = None

        for event in events:
            completed_response, usage = OpenAICodexProvider._handle_response_event(
                event=event,
                text_parts=text_parts,
                tool_calls=tool_calls,
                completed_response=completed_response,
                usage=usage,
            )

        return OpenAICodexProvider._build_response(
            completed_response=completed_response,
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
        )

    @staticmethod
    def _build_response(
        *,
        completed_response: Any | None,
        text: str,
        tool_calls: dict[str, dict[str, Any]],
        usage: dict[str, Any] | Any | None,
    ) -> Response:
        # The Codex backend sometimes returns a completed SDK Response with output=[],
        # even though earlier stream events (output_text.delta etc.) carried the full text.
        # Only fall through to reconstruction when ALL of these hold:
        #   1. status == "completed" (non-completed responses are returned as-is to preserve error info)
        #   2. output is empty
        #   3. the stream actually collected text or tool_calls
        if isinstance(completed_response, Response):
            if completed_response.output:
                return completed_response
            if completed_response.status != "completed":
                return completed_response
            if not text and not tool_calls:
                return completed_response

        payload: dict[str, Any] = {
            "id": getattr(completed_response, "id", None) or "resp_codex",
            "created_at": getattr(completed_response, "created_at", None) or 0,
            "model": getattr(completed_response, "model", None) or "gpt-5-codex",
            "object": getattr(completed_response, "object", None) or "response",
            "output": OpenAICodexProvider._build_response_output(
                completed_response=completed_response,
                text=text,
                tool_calls=tool_calls,
            ),
            "parallel_tool_calls": getattr(completed_response, "parallel_tool_calls", None) or False,
            "tool_choice": getattr(completed_response, "tool_choice", None) or "auto",
            "tools": getattr(completed_response, "tools", None) or [],
            "usage": OpenAICodexProvider._normalize_response_usage(
                getattr(completed_response, "usage", None) or usage,
            ),
        }
        return Response.model_validate(payload)

    @staticmethod
    def _build_response_output(
        *,
        completed_response: Any | None,
        text: str,
        tool_calls: dict[str, dict[str, Any]],
    ) -> list[Any]:
        existing_output = getattr(completed_response, "output", None)
        if isinstance(existing_output, list) and existing_output:
            return existing_output

        output: list[dict[str, Any]] = []
        if text:
            output.append({
                "id": "msg_codex",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            })
        output.extend(
            {
                "type": "function_call",
                "call_id": call["call_id"],
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("arguments", ""),
                "status": "completed",
            }
            for call in tool_calls.values()
        )
        return output

    @staticmethod
    def _normalize_response_usage(usage: dict[str, Any] | Any | None) -> dict[str, Any]:
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            usage = model_dump()

        payload = dict(usage) if isinstance(usage, dict) else {}
        input_tokens = payload.get("input_tokens")
        output_tokens = payload.get("output_tokens")
        total_tokens = payload.get("total_tokens")

        if not isinstance(input_tokens, int):
            input_tokens = 0
        if not isinstance(output_tokens, int):
            output_tokens = 0
        if not isinstance(total_tokens, int):
            total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "input_tokens_details": payload.get("input_tokens_details") or {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": payload.get("output_tokens_details") or {"reasoning_tokens": 0},
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _handle_response_event(
        *,
        event: Any,
        text_parts: list[str],
        tool_calls: dict[str, dict[str, Any]],
        completed_response: Any | None,
        usage: dict[str, Any] | Any | None,
    ) -> tuple[Any | None, dict[str, Any] | Any | None]:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            OpenAICodexProvider._append_text_delta(event, text_parts)
            return completed_response, usage
        if event_type == "response.output_item.done":
            OpenAICodexProvider._record_tool_call(
                tool_calls,
                OpenAICodexProvider._function_call_from_output_item(event),
            )
            return completed_response, usage
        if event_type == "response.function_call_arguments.done":
            OpenAICodexProvider._record_tool_call(
                tool_calls,
                OpenAICodexProvider._function_call_from_arguments_done(event),
            )
            return completed_response, usage
        if event_type == "response.completed":
            completed_response = getattr(event, "response", None)
            usage = getattr(completed_response, "usage", None) or usage
            return completed_response, usage
        return completed_response, getattr(event, "usage", None) or usage

    @staticmethod
    def _append_text_delta(event: Any, text_parts: list[str]) -> None:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            text_parts.append(delta)

    @staticmethod
    def _record_tool_call(
        tool_calls: dict[str, dict[str, Any]],
        tool_call: dict[str, Any] | None,
    ) -> None:
        if tool_call is None:
            return
        tool_calls[tool_call["call_id"]] = tool_call

    @staticmethod
    def _function_call_from_output_item(event: Any) -> dict[str, Any] | None:
        item = getattr(event, "item", None)
        if getattr(item, "type", None) != "function_call":
            return None
        call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
        if not isinstance(call_id, str) or not call_id:
            return None
        return {
            "type": "function_call",
            "call_id": call_id,
            "id": getattr(item, "id", None),
            "name": getattr(item, "name", None),
            "arguments": getattr(item, "arguments", "") or "",
        }

    @staticmethod
    def _function_call_from_arguments_done(event: Any) -> dict[str, Any] | None:
        call_id = getattr(event, "call_id", None) or getattr(event, "item_id", None)
        if not isinstance(call_id, str) or not call_id:
            return None
        return {
            "type": "function_call",
            "call_id": call_id,
            "id": getattr(event, "item_id", None),
            "name": getattr(event, "name", None),
            "arguments": getattr(event, "arguments", "") or "",
        }


def should_use_openai_codex_backend(provider: str, api_key: str | None) -> bool:
    if provider != "openai" or not api_key:
        return False
    return extract_openai_codex_account_id(api_key) is not None


def resolve_openai_codex_api_base(api_base: str | None) -> str:
    raw = (api_base or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if raw.endswith("/responses"):
        raw = raw[: -len("/responses")]
    if raw.endswith("/codex"):
        return raw
    return f"{raw}/codex"


def build_openai_codex_default_headers(
    api_key: str,
    *,
    originator: str = DEFAULT_CODEX_ORIGINATOR,
) -> dict[str, str]:
    account_id = extract_openai_codex_account_id(api_key)
    if account_id is None:
        raise OpenAICodexTransportError(None, "OpenAI Codex OAuth token is missing chatgpt_account_id")
    return {
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
    }


__all__ = [
    "DEFAULT_CODEX_BASE_URL",
    "DEFAULT_CODEX_INCLUDE",
    "DEFAULT_CODEX_INSTRUCTIONS",
    "DEFAULT_CODEX_ORIGINATOR",
    "DEFAULT_CODEX_TEXT_CONFIG",
    "OpenAICodexProvider",
    "OpenAICodexTransportError",
    "build_openai_codex_default_headers",
    "resolve_openai_codex_api_base",
    "should_use_openai_codex_backend",
]
