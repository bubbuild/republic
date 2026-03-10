"""Helpers for OpenAI Codex OAuth-backed sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.utils.aio import async_iter_to_sync_iter, run_async_in_sync

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
        response = run_async_in_sync(
            self.aresponses(**kwargs),
            allow_running_loop=allow_running_loop,
        )
        if kwargs.get("stream") and hasattr(response, "__aiter__"):
            return async_iter_to_sync_iter(response)
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
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str):
                    text_parts.append(delta)
                continue
            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                    if isinstance(call_id, str) and call_id:
                        tool_calls[call_id] = {
                            "type": "function_call",
                            "call_id": call_id,
                            "id": getattr(item, "id", None),
                            "name": getattr(item, "name", None),
                            "arguments": getattr(item, "arguments", "") or "",
                        }
                continue
            if event_type == "response.function_call_arguments.done":
                call_id = getattr(event, "call_id", None) or getattr(event, "item_id", None)
                if isinstance(call_id, str) and call_id:
                    tool_calls[call_id] = {
                        "type": "function_call",
                        "call_id": call_id,
                        "id": getattr(event, "item_id", None),
                        "name": getattr(event, "name", None),
                        "arguments": getattr(event, "arguments", "") or "",
                    }
                continue
            if event_type == "response.completed":
                completed_response = getattr(event, "response", None)
                maybe_usage = getattr(completed_response, "usage", None)
                if maybe_usage is not None:
                    usage = maybe_usage
                continue
            maybe_usage = getattr(event, "usage", None)
            if maybe_usage is not None:
                usage = maybe_usage

        if completed_response is not None and (
            getattr(completed_response, "output", None) is not None
            or getattr(completed_response, "output_text", None) is not None
        ):
            return completed_response

        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump()

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
    "build_openai_codex_default_headers",
    "DEFAULT_CODEX_BASE_URL",
    "DEFAULT_CODEX_INCLUDE",
    "DEFAULT_CODEX_INSTRUCTIONS",
    "DEFAULT_CODEX_ORIGINATOR",
    "DEFAULT_CODEX_TEXT_CONFIG",
    "OpenAICodexResponsesClient",
    "OpenAICodexTransportError",
    "resolve_openai_codex_api_base",
    "should_use_openai_codex_backend",
]
