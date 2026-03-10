"""Helpers for OpenAI Codex OAuth-backed sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.utils.aio import run_async_in_sync

from republic.auth.openai_codex import extract_openai_codex_account_id
from republic.clients._async_bridge import threaded_async_call_to_sync_iter

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


class OpenAICodexResponsesClient:
    SUPPORTS_RESPONSES = True
    PREFERRED_TRANSPORT = "responses"
    PRESERVE_EXTRA_HEADERS_IN_RESPONSES = True

    def __init__(
        self,
        base_client: Any,
        *,
        default_instructions: str = DEFAULT_CODEX_INSTRUCTIONS,
        default_include: tuple[str, ...] = DEFAULT_CODEX_INCLUDE,
        default_text: dict[str, Any] | None = None,
        store: bool = False,
    ) -> None:
        self._base_client = base_client
        self._default_instructions = default_instructions
        self._default_include = list(default_include)
        self._default_text = dict(default_text or DEFAULT_CODEX_TEXT_CONFIG)
        self._store = store
        self.client = getattr(base_client, "client", None)
        responses_api = getattr(self.client, "responses", None)
        create = getattr(responses_api, "create", None)
        if not callable(create):
            raise OpenAICodexTransportError(None, "OpenAI-compatible any-llm client is missing client.responses.create")
        self._responses_create = create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_client, name)

    def responses(
        self,
        *,
        allow_running_loop: bool = INSIDE_NOTEBOOK,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("stream"):
            return threaded_async_call_to_sync_iter(self.aresponses(**kwargs))

        response = run_async_in_sync(
            self.aresponses(**kwargs),
            allow_running_loop=allow_running_loop,
        )
        return response

    async def aresponses(self, **kwargs: Any) -> Any:
        payload = self._build_payload(kwargs)
        response = await self._responses_create(**payload)
        if kwargs.get("stream"):
            return response
        if hasattr(response, "__aiter__"):
            return self._collect_responses_events(await self._collect_events(response))
        return response

    @staticmethod
    async def _collect_events(response: AsyncIterator[Any]) -> list[Any]:
        events: list[Any] = []
        async for event in response:
            events.append(event)
        return events

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
    def _collect_responses_events(events: list[Any]) -> Any:
        text_parts: list[str] = []
        tool_calls: dict[str, dict[str, Any]] = {}
        usage: dict[str, Any] | Any | None = None
        completed_response: Any | None = None

        for event in events:
            completed_response, usage = OpenAICodexResponsesClient._handle_response_event(
                event=event,
                text_parts=text_parts,
                tool_calls=tool_calls,
                completed_response=completed_response,
                usage=usage,
            )

        if completed_response is not None and (
            getattr(completed_response, "output", None) is not None
            or getattr(completed_response, "output_text", None) is not None
        ):
            return completed_response

        model_dump = getattr(usage, "model_dump", None)
        if usage is not None and callable(model_dump):
            usage = model_dump()

        return SimpleNamespace(
            output_text="".join(text_parts),
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id=call["call_id"],
                    id=call.get("id"),
                    name=call.get("name"),
                    arguments=call.get("arguments", ""),
                )
                for call in tool_calls.values()
            ],
            usage=usage,
        )

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
            OpenAICodexResponsesClient._append_text_delta(event, text_parts)
            return completed_response, usage
        if event_type == "response.output_item.done":
            OpenAICodexResponsesClient._record_tool_call(
                tool_calls,
                OpenAICodexResponsesClient._function_call_from_output_item(event),
            )
            return completed_response, usage
        if event_type == "response.function_call_arguments.done":
            OpenAICodexResponsesClient._record_tool_call(
                tool_calls,
                OpenAICodexResponsesClient._function_call_from_arguments_done(event),
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
    "OpenAICodexResponsesClient",
    "OpenAICodexTransportError",
    "build_openai_codex_default_headers",
    "resolve_openai_codex_api_base",
    "should_use_openai_codex_backend",
]
