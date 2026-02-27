"""Minimal ChatGPT Codex backend for OAuth-backed OpenAI sessions."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from republic.auth.openai_codex import extract_openai_codex_account_id

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"


class OpenAICodexTransportError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class OpenAICodexBackendConfig:
    api_key: str
    api_base: str | None = None
    originator: str = "republic"
    timeout_seconds: float = 60.0


def should_use_openai_codex_backend(provider: str, api_key: str | None) -> bool:
    if provider != "openai" or not api_key:
        return False
    return extract_openai_codex_account_id(api_key) is not None


class OpenAICodexClient:
    def __init__(self, config: OpenAICodexBackendConfig) -> None:
        self._config = config
        account_id = extract_openai_codex_account_id(config.api_key)
        if account_id is None:
            raise OpenAICodexTransportError(None, "OpenAI Codex OAuth token is missing chatgpt_account_id")
        self._account_id = account_id

    def completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> Any:
        payload = self._build_payload(
            model=model,
            messages=messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            max_tokens=max_completion_tokens if max_completion_tokens is not None else max_tokens,
        )
        return self._perform_request(payload, stream=stream)

    async def acompletion(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.completion, **kwargs)

    def embedding(self, **_: Any) -> Any:
        raise OpenAICodexTransportError(None, "OpenAI Codex backend does not support embeddings")

    async def aembedding(self, **_: Any) -> Any:
        raise OpenAICodexTransportError(None, "OpenAI Codex backend does not support embeddings")

    def _perform_request(self, payload: dict[str, Any], *, stream: bool) -> Any:
        request = urllib.request.Request(  # noqa: S310
            self._resolve_url(self._config.api_base),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._build_headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenAICodexTransportError(exc.code, self._format_http_error(exc.code, body), body) from exc
        except urllib.error.URLError as exc:
            raise OpenAICodexTransportError(None, str(exc.reason)) from exc

        parsed = self._parse_sse(body)
        response = self._build_response(parsed)
        if stream:
            return response
        return response

    def _build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        reasoning_effort: str | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        instructions, input_items = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": instructions or "You are Codex.",
            "input": input_items,
            "include": ["reasoning.encrypted_content"],
            "text": {"verbosity": "medium"},
        }
        responses_tools = self._convert_tools(tools)
        if responses_tools:
            payload["tools"] = responses_tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        return payload

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "chatgpt-account-id": self._account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": self._config.originator,
            "accept": "text/event-stream",
            "content-type": "application/json",
            "user-agent": "republic-openai-codex/0",
        }

    @staticmethod
    def _resolve_url(api_base: str | None) -> str:
        raw = (api_base or DEFAULT_CODEX_BASE_URL).rstrip("/")
        if raw.endswith("/codex/responses"):
            return raw
        if raw.endswith("/codex"):
            return f"{raw}/responses"
        return f"{raw}/codex/responses"

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                parameters = function.get("parameters")
                if isinstance(name, str) and isinstance(parameters, dict):
                    converted.append(
                        {
                            "type": "function",
                            "name": name,
                            "description": function.get("description", "") or "",
                            "parameters": parameters,
                        }
                    )
                    continue
            if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                converted.append(dict(tool))
        return converted or None

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                return text
        return str(content)

    @classmethod
    def _convert_messages(cls, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = cls._stringify_content(message.get("content", ""))
            if role == "system":
                if content:
                    instructions.append(content)
                continue
            if role == "assistant":
                if content:
                    items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                continue
            if role == "tool":
                call_id = message.get("tool_call_id")
                if isinstance(call_id, str) and call_id and content:
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": content,
                        }
                    )
                continue
            if content:
                items.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
        return ("\n\n".join(instructions) or None), items

    @staticmethod
    def _extract_fallback_text(item: Any) -> str | None:
        if not isinstance(item, dict) or item.get("type") != "message":
            return None
        content = item.get("content")
        if not isinstance(content, list):
            return None
        collected: list[str] = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if isinstance(text, str) and text:
                collected.append(text)
        return "".join(collected) or None

    @staticmethod
    def _extract_tool_call(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return None
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            return None
        call: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id:
            call["id"] = call_id
        return call

    @staticmethod
    def _update_usage_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
        response = event.get("response")
        if not isinstance(response, dict):
            return None
        raw_usage = response.get("usage")
        if isinstance(raw_usage, dict):
            return raw_usage
        return None

    @staticmethod
    def _raise_event_error(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "error":
            message = event.get("message")
            if isinstance(message, str) and message:
                raise OpenAICodexTransportError(502, message)
            return
        if event_type != "response.failed":
            return
        response = event.get("response")
        if not isinstance(response, dict):
            return
        error = response.get("error")
        if not isinstance(error, dict):
            return
        message = error.get("message")
        if isinstance(message, str) and message:
            raise OpenAICodexTransportError(502, message)

    @classmethod
    def _parse_sse(cls, raw: str) -> dict[str, Any]:
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        fallback_text: str | None = None
        tool_calls: list[dict[str, Any]] = []
        for chunk in raw.split("\n\n"):
            lines = [line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")]
            if not lines:
                continue
            data = "\n".join(lines).strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            cls._raise_event_error(event)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
                continue
            if event_type == "response.output_item.done":
                item = event.get("item")
                tool_call = cls._extract_tool_call(item)
                if tool_call is not None:
                    tool_calls.append(tool_call)
                    continue
                fallback_text = cls._extract_fallback_text(item) or fallback_text
                continue
            if event_type in {"response.completed", "response.done"}:
                usage = cls._update_usage_from_event(event) or usage
        return {
            "text": "".join(parts) or fallback_text or "",
            "usage": usage,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _build_response(parsed: dict[str, Any]) -> Any:
        tool_calls_raw = parsed.get("tool_calls") or []
        tool_calls = [
            SimpleNamespace(
                id=item.get("id"),
                type=item.get("type"),
                function=SimpleNamespace(
                    name=item.get("function", {}).get("name"),
                    arguments=item.get("function", {}).get("arguments"),
                ),
            )
            for item in tool_calls_raw
            if isinstance(item, dict)
        ]
        message = SimpleNamespace(content=parsed.get("text", ""), tool_calls=tool_calls)
        choice = SimpleNamespace(message=message)
        usage = parsed.get("usage")
        return SimpleNamespace(choices=[choice], usage=usage)

    @staticmethod
    def _format_http_error(status_code: int, body: str) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail:
                return f"Error code: {status_code} - {detail}"
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return f"Error code: {status_code} - {payload}"
        return f"Error code: {status_code} - {body}"


__all__ = [
    "DEFAULT_CODEX_BASE_URL",
    "OpenAICodexBackendConfig",
    "OpenAICodexClient",
    "OpenAICodexTransportError",
    "should_use_openai_codex_backend",
]
