"""GitHub Copilot backend implemented via github-copilot-sdk."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS = 180.0


class GitHubCopilotTransportError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class GitHubCopilotBackendConfig:
    api_key: str
    api_base: str | None = None
    cli_path: str | None = None
    cli_url: str | None = None
    use_stdio: bool | None = None
    log_level: str = "info"
    timeout_seconds: float = DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS
    config_dir: str | None = None


@dataclass(frozen=True)
class CopilotSdkBindings:
    client_type: Any
    permission_result_type: Any
    event_type: Any
    tool_type: Any


@dataclass
class CollectedCopilotResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    chunks: list[Any] = field(default_factory=list)


def should_use_github_copilot_backend(provider: str) -> bool:
    return provider == "github-copilot"


def _load_copilot_sdk() -> CopilotSdkBindings:
    try:
        from copilot import CopilotClient, PermissionRequestResult, Tool  # type: ignore
        from copilot.generated.session_events import SessionEventType  # type: ignore
    except Exception as exc:
        raise GitHubCopilotTransportError(
            None,
            "github-copilot-sdk is required for provider `github-copilot`. "
            "Install Republic with that dependency available.",
        ) from exc
    return CopilotSdkBindings(
        client_type=CopilotClient,
        permission_result_type=PermissionRequestResult,
        event_type=SessionEventType,
        tool_type=Tool,
    )


class GitHubCopilotClient:
    def __init__(self, config: GitHubCopilotBackendConfig) -> None:
        self._config = config

    def completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        self._ensure_api_key()

        if stream:
            collected = _run_sync(self._collect_response(model=model, messages=messages, tools=tools, stream=True))
            return iter(collected["chunks"])
        return _run_sync(self._collect_response(model=model, messages=messages, tools=tools, stream=False))["response"]

    async def acompletion(self, **kwargs: Any) -> Any:
        self._ensure_api_key()

        if kwargs.get("stream"):
            collected = await self._collect_response(
                model=str(kwargs["model"]),
                messages=list(kwargs["messages"]),
                tools=kwargs.get("tools"),
                stream=True,
            )

            async def _iterator() -> Any:
                for chunk in collected["chunks"]:
                    yield chunk

            return _iterator()

        return (await self._collect_response(
            model=str(kwargs["model"]),
            messages=list(kwargs["messages"]),
            tools=kwargs.get("tools"),
            stream=False,
        ))["response"]

    def embedding(self, **_: Any) -> Any:
        raise GitHubCopilotTransportError(None, "GitHub Copilot backend does not support embeddings")

    async def aembedding(self, **_: Any) -> Any:
        raise GitHubCopilotTransportError(None, "GitHub Copilot backend does not support embeddings")

    def _ensure_api_key(self) -> None:
        if not self._config.api_key.strip():
            raise GitHubCopilotTransportError(401, "GitHub Copilot OAuth token is missing")

    async def _collect_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        sdk = _load_copilot_sdk()

        prompt, system_message = self._build_prompt(messages)
        if not prompt.strip():
            prompt = "Continue."

        client = sdk.client_type(self._build_client_options())
        session = None
        parts: list[str] = []
        final_text: str | None = None
        usage: dict[str, Any] | None = None
        chunks: list[Any] = []
        session_error: str | None = None
        tool_calls: list[dict[str, Any]] = []
        finished = asyncio.Event()
        saw_idle = False

        def _handle_event(event: Any) -> None:
            nonlocal final_text, usage, session_error, saw_idle
            event_type = getattr(event, "type", None)
            data = getattr(event, "data", None)
            if event_type == sdk.event_type.ASSISTANT_MESSAGE_DELTA:
                delta = self._coerce_text(getattr(data, "delta_content", None))
                if delta:
                    parts.append(delta)
                    chunks.append(self._make_text_chunk(delta))
                return
            if event_type == sdk.event_type.ASSISTANT_MESSAGE:
                final_text = self._coerce_text(getattr(data, "content", None))
                return
            if event_type == sdk.event_type.ASSISTANT_USAGE:
                usage = self._extract_usage(data)
                if usage is not None:
                    chunks.append(self._make_usage_chunk(usage))
                return
            if event_type == sdk.event_type.EXTERNAL_TOOL_REQUESTED:
                tool_calls.extend(self._extract_tool_calls(data))
                finished.set()
                return
            if event_type == sdk.event_type.SESSION_ERROR:
                session_error = self._coerce_text(getattr(data, "message", None)) or "GitHub Copilot session error"
                finished.set()
                return
            if event_type == sdk.event_type.SESSION_IDLE:
                saw_idle = True
                finished.set()

        try:
            await client.start()
            session_config: dict[str, Any] = {
                "model": model,
                "streaming": True,
                "on_permission_request": lambda request, invocation: self._handle_permission_request(
                    request=request,
                    invocation=invocation,
                    permission_result_type=sdk.permission_result_type,
                ),
            }
            if system_message:
                session_config["system_message"] = {"mode": "append", "content": system_message}
            sdk_tools = self._build_tools(tools, sdk.tool_type)
            if sdk_tools:
                session_config["tools"] = sdk_tools
            session = await client.create_session(session_config)
            unsubscribe = session.on(_handle_event)
            try:
                await session.send({"prompt": prompt})
                await asyncio.wait_for(finished.wait(), timeout=self._config.timeout_seconds)
                if tool_calls:
                    await session.abort()
                event = None
            finally:
                unsubscribe()
                if saw_idle:
                    event = None
        except GitHubCopilotTransportError:
            raise
        except TimeoutError as exc:
            raise GitHubCopilotTransportError(None, f"Timeout after {self._config.timeout_seconds}s waiting for Copilot") from exc
        except Exception as exc:
            raise GitHubCopilotTransportError(None, str(exc)) from exc
        finally:
            if session is not None:
                try:
                    await session.destroy()
                except Exception:
                    pass
            try:
                await client.stop()
            except Exception:
                pass

        if event is not None and final_text is None:
            final_text = self._coerce_text(getattr(getattr(event, "data", None), "content", None))
        if session_error is not None:
            raise GitHubCopilotTransportError(None, session_error)

        collected = CollectedCopilotResponse(
            text=final_text if final_text is not None else "".join(parts),
            tool_calls=tool_calls,
            usage=usage,
            chunks=chunks,
        )
        if collected.tool_calls:
            for call in collected.tool_calls:
                collected.chunks.append(self._make_tool_chunk(call))
        if stream and not collected.chunks and collected.text:
            collected.chunks.append(self._make_text_chunk(collected.text))
        return {
            "response": self._make_response(
                text=collected.text,
                tool_calls=collected.tool_calls,
                usage=collected.usage,
            ),
            "chunks": collected.chunks,
        }

    def _build_client_options(self) -> dict[str, Any]:
        env = os.environ.copy()
        env["GH_TOKEN"] = self._config.api_key
        env["GITHUB_TOKEN"] = self._config.api_key
        domain = self._resolve_domain(self._config.api_base)
        if domain is not None:
            env["COPILOT_DOMAIN"] = domain

        options: dict[str, Any] = {
            "env": env,
            "log_level": self._config.log_level,
        }
        if self._config.cli_url:
            options["cli_url"] = self._config.cli_url
        if self._config.cli_path:
            options["cli_path"] = str(Path(self._config.cli_path).expanduser())
        if self._config.use_stdio is not None:
            options["use_stdio"] = self._config.use_stdio
        if self._config.config_dir:
            options["config_dir"] = str(Path(self._config.config_dir).expanduser())
        return options

    @staticmethod
    def _resolve_domain(api_base: str | None) -> str | None:
        if not api_base:
            return None
        parsed = urlsplit(api_base)
        if parsed.scheme and parsed.netloc:
            return parsed.netloc
        normalized = api_base.strip().strip("/")
        return normalized or None

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _build_prompt(cls, messages: list[dict[str, Any]]) -> tuple[str, str | None]:
        system_parts: list[str] = []
        conversation_parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = cls._coerce_text(message.get("content"))
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
            elif role == "tool":
                conversation_parts.append(f"Tool: {content}")
            else:
                conversation_parts.append(f"User: {content}")
        return "\n\n".join(conversation_parts), ("\n\n".join(system_parts) or None)

    @staticmethod
    def _extract_usage(data: Any) -> dict[str, Any] | None:
        usage: dict[str, Any] = {}
        for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            value = getattr(data, field, None)
            if isinstance(value, (int, float)):
                usage[field] = int(value)
        if usage:
            usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return usage or None

    @staticmethod
    def _stringify_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        if arguments is None:
            return "{}"
        try:
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"

    @classmethod
    def _extract_tool_calls(cls, data: Any) -> list[dict[str, Any]]:
        raw_requests = getattr(data, "tool_requests", None)
        if isinstance(raw_requests, list) and raw_requests:
            extracted = [cls._tool_call_from_request(item) for item in raw_requests]
            return [item for item in extracted if item is not None]
        single = cls._tool_call_from_request(data)
        return [single] if single is not None else []

    @classmethod
    def _tool_call_from_request(cls, request: Any) -> dict[str, Any] | None:
        name = getattr(request, "name", None) or getattr(request, "tool_name", None)
        tool_call_id = getattr(request, "tool_call_id", None)
        arguments = getattr(request, "arguments", None)
        if not isinstance(name, str) or not name:
            return None
        call: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": cls._stringify_arguments(arguments),
            },
        }
        if isinstance(tool_call_id, str) and tool_call_id:
            call["id"] = tool_call_id
        return call

    @staticmethod
    def _build_tools(tools: list[dict[str, Any]] | None, sdk_tool_type: Any) -> list[Any] | None:
        if not tools:
            return None
        built: list[Any] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            built.append(
                sdk_tool_type(
                    name=name,
                    description=str(function.get("description", "") or ""),
                    handler=_placeholder_tool_handler,
                    parameters=function.get("parameters"),
                )
            )
        return built or None

    @staticmethod
    def _handle_permission_request(*, request: Any, invocation: dict[str, Any], permission_result_type: Any) -> Any:
        del invocation
        request_kind = getattr(getattr(request, "kind", None), "value", getattr(request, "kind", None))
        if request_kind == "custom-tool":
            return permission_result_type(kind="approved")
        return permission_result_type()

    @staticmethod
    def _make_response(*, text: str, tool_calls: list[dict[str, Any]], usage: dict[str, Any] | None) -> Any:
        normalized_calls = [
            SimpleNamespace(
                id=call.get("id"),
                type=call.get("type"),
                function=SimpleNamespace(
                    name=call.get("function", {}).get("name"),
                    arguments=call.get("function", {}).get("arguments"),
                ),
            )
            for call in tool_calls
            if isinstance(call, dict)
        ]
        message = SimpleNamespace(content=text, tool_calls=normalized_calls)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=usage)

    @staticmethod
    def _make_text_chunk(text: str) -> Any:
        delta = SimpleNamespace(content=text, tool_calls=[])
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice], usage=None)

    @staticmethod
    def _make_tool_chunk(tool_call: dict[str, Any]) -> Any:
        function = tool_call.get("function", {})
        delta_tool_call = SimpleNamespace(
            id=tool_call.get("id"),
            type=tool_call.get("type"),
            function=SimpleNamespace(
                name=function.get("name"),
                arguments=function.get("arguments"),
            ),
        )
        delta = SimpleNamespace(content="", tool_calls=[delta_tool_call])
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice], usage=None)

    @staticmethod
    def _make_usage_chunk(usage: dict[str, Any]) -> Any:
        delta = SimpleNamespace(content="", tool_calls=[])
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice], usage=usage)


def _run_sync(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        raise GitHubCopilotTransportError(None, "Synchronous GitHub Copilot calls cannot run inside an active event loop")
    return asyncio.run(awaitable)


async def _placeholder_tool_handler(invocation: Any) -> Any:
    # Keep the SDK-side tool pending long enough for Republic to intercept the
    # external tool request and route it through its own tool execution flow.
    await asyncio.sleep(3600)
    return {
        "text_result_for_llm": "",
        "result_type": "success",
        "tool_telemetry": {},
    }


__all__ = [
    "DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS",
    "GitHubCopilotBackendConfig",
    "GitHubCopilotClient",
    "GitHubCopilotTransportError",
    "should_use_github_copilot_backend",
]
