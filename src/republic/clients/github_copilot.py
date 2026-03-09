"""GitHub Copilot backend implemented via github-copilot-sdk."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
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


@dataclass
class _SessionState:
    parts: list[str] = field(default_factory=list)
    final_text: str | None = None
    usage: dict[str, Any] | None = None
    chunks: list[Any] = field(default_factory=list)
    session_error: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    saw_idle: bool = False


def should_use_github_copilot_backend(provider: str) -> bool:
    return provider == "github-copilot"


def _load_copilot_sdk() -> CopilotSdkBindings:
    try:
        from copilot import CopilotClient, PermissionRequestResult, Tool
        from copilot.generated.session_events import SessionEventType
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

        return (
            await self._collect_response(
                model=str(kwargs["model"]),
                messages=list(kwargs["messages"]),
                tools=kwargs.get("tools"),
                stream=False,
            )
        )["response"]

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
        state = _SessionState()
        _, event = await self._run_session(
            client=client,
            model=model,
            prompt=prompt,
            system_message=system_message,
            tools=tools,
            sdk=sdk,
            state=state,
        )
        self._raise_for_session_error(state)
        self._populate_final_text_from_event(state, event)
        collected = self._build_collected_response(state)
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

    async def _run_session(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        system_message: str | None,
        tools: list[dict[str, Any]] | None,
        sdk: CopilotSdkBindings,
        state: _SessionState,
    ) -> tuple[Any, Any]:
        session = None
        event = None
        finished = asyncio.Event()
        handler = self._build_event_handler(sdk=sdk, state=state, finished=finished)

        try:
            await client.start()
            session_config = self._build_session_config(
                model=model,
                system_message=system_message,
                tools=tools,
                sdk=sdk,
            )
            session = await client.create_session(session_config)
            unsubscribe = session.on(handler)
            try:
                await session.send({"prompt": prompt})
                await asyncio.wait_for(finished.wait(), timeout=self._config.timeout_seconds)
                if state.tool_calls:
                    await session.abort()
            finally:
                unsubscribe()
                if state.saw_idle:
                    event = None
        except GitHubCopilotTransportError:
            raise
        except TimeoutError as exc:
            raise GitHubCopilotTransportError(
                None, f"Timeout after {self._config.timeout_seconds}s waiting for Copilot"
            ) from exc
        except Exception as exc:
            raise GitHubCopilotTransportError(None, str(exc)) from exc
        finally:
            if session is not None:
                with suppress(Exception):
                    await session.destroy()
            with suppress(Exception):
                await client.stop()
        return session, event

    def _build_event_handler(
        self,
        *,
        sdk: CopilotSdkBindings,
        state: _SessionState,
        finished: asyncio.Event,
    ) -> Any:
        def _handle_event(event: Any) -> None:
            event_type = getattr(event, "type", None)
            data = getattr(event, "data", None)
            if event_type == sdk.event_type.ASSISTANT_MESSAGE_DELTA:
                delta = self._coerce_text(getattr(data, "delta_content", None))
                if delta:
                    state.parts.append(delta)
                    state.chunks.append(self._make_text_chunk(delta))
                return
            if event_type == sdk.event_type.ASSISTANT_MESSAGE:
                state.final_text = self._coerce_text(getattr(data, "content", None))
                return
            if event_type == sdk.event_type.ASSISTANT_USAGE:
                state.usage = self._extract_usage(data)
                if state.usage is not None:
                    state.chunks.append(self._make_usage_chunk(state.usage))
                return
            if event_type == sdk.event_type.EXTERNAL_TOOL_REQUESTED:
                state.tool_calls.extend(self._extract_tool_calls(data))
                finished.set()
                return
            if event_type == sdk.event_type.SESSION_ERROR:
                state.session_error = (
                    self._coerce_text(getattr(data, "message", None)) or "GitHub Copilot session error"
                )
                finished.set()
                return
            if event_type == sdk.event_type.SESSION_IDLE:
                state.saw_idle = True
                finished.set()

        return _handle_event

    def _build_session_config(
        self,
        *,
        model: str,
        system_message: str | None,
        tools: list[dict[str, Any]] | None,
        sdk: CopilotSdkBindings,
    ) -> dict[str, Any]:
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
        return session_config

    @staticmethod
    def _raise_for_session_error(state: _SessionState) -> None:
        if state.session_error is not None:
            raise GitHubCopilotTransportError(None, state.session_error)

    def _populate_final_text_from_event(self, state: _SessionState, event: Any) -> None:
        if event is not None and state.final_text is None:
            state.final_text = self._coerce_text(getattr(getattr(event, "data", None), "content", None))

    def _build_collected_response(self, state: _SessionState) -> CollectedCopilotResponse:
        collected = CollectedCopilotResponse(
            text=state.final_text if state.final_text is not None else "".join(state.parts),
            tool_calls=list(state.tool_calls),
            usage=state.usage,
            chunks=list(state.chunks),
        )
        if collected.tool_calls:
            for call in collected.tool_calls:
                collected.chunks.append(self._make_tool_chunk(call))
        return collected

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
        for usage_field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            value = getattr(data, usage_field, None)
            if isinstance(value, (int, float)):
                usage[usage_field] = int(value)
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
        raise GitHubCopilotTransportError(
            None, "Synchronous GitHub Copilot calls cannot run inside an active event loop"
        )
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
