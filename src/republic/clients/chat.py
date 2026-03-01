"""Chat helpers for Republic."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

from republic.core.errors import ErrorKind, RepublicError
from republic.core.execution import LLMCore
from republic.core.results import (
    AsyncStreamEvents,
    AsyncTextStream,
    ErrorPayload,
    StreamEvent,
    StreamEvents,
    StreamState,
    TextStream,
    ToolAutoResult,
)
from republic.tape.context import TapeContext
from republic.tape.manager import AsyncTapeManager, TapeManager
from republic.tools.context import ToolContext
from republic.tools.executor import ToolExecutor
from republic.tools.schema import ToolInput, ToolSet, normalize_tools

MessageInput = dict[str, Any]


def _field(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


@dataclass(frozen=True)
class PreparedChat:
    payload: list[dict[str, Any]]
    new_messages: list[dict[str, Any]]
    toolset: ToolSet
    tape: str | None
    should_update: bool
    context_error: ErrorPayload | None
    run_id: str
    system_prompt: str | None
    context: TapeContext | None


class ToolCallAssembler:
    def __init__(self) -> None:
        self._calls: dict[object, dict[str, Any]] = {}
        self._order: list[object] = []
        self._index_to_key: dict[Any, object] = {}

    def _replace_key(self, old_key: object, new_key: object) -> None:
        entry = self._calls.pop(old_key)
        self._calls[new_key] = entry
        self._order[self._order.index(old_key)] = new_key
        for index, key in list(self._index_to_key.items()):
            if key == old_key:
                self._index_to_key[index] = new_key

    def _key_at_position(self, position: int) -> object | None:
        if position < len(self._order):
            return self._order[position]
        return None

    def _resolve_key_by_id(self, call_id: str, index: Any, position: int) -> object:
        id_key: object = ("id", call_id)
        if id_key in self._calls:
            if index is not None:
                self._index_to_key[index] = id_key
            return id_key

        mapped_key = self._index_to_key.get(index) if index is not None else None
        if mapped_key is not None and mapped_key in self._calls and mapped_key != id_key:
            self._replace_key(mapped_key, id_key)
            self._index_to_key[index] = id_key
            return id_key

        if index is not None:
            index_key: object = ("index", index)
            if index_key in self._calls:
                self._replace_key(index_key, id_key)
                self._index_to_key[index] = id_key
                return id_key

        position_key = self._key_at_position(position)
        if position_key is not None and position_key in self._calls:
            self._replace_key(position_key, id_key)
            if index is not None:
                self._index_to_key[index] = id_key
            return id_key
        if index is not None:
            self._index_to_key[index] = id_key
        return id_key

    def _resolve_key_by_index(self, tool_call: Any, index: Any, position: int) -> object:
        mapped_key = self._index_to_key.get(index)
        if mapped_key is not None and mapped_key in self._calls:
            return mapped_key

        index_key: object = ("index", index)
        if index_key in self._calls:
            self._index_to_key[index] = index_key
            return index_key

        position_key = self._key_at_position(position)
        func = _field(tool_call, "function")
        tool_name = _field(func, "name") if func is not None else None

        if (tool_name is None or tool_name == "") and position_key is not None and position_key in self._calls:
            self._index_to_key[index] = position_key
            return position_key

        if (
            position_key is not None
            and position_key in self._calls
            and isinstance(position_key, tuple)
            and position_key[0] == "position"
        ):
            self._replace_key(position_key, index_key)
            self._index_to_key[index] = index_key
            return index_key
        self._index_to_key[index] = index_key
        return index_key

    def _resolve_key(self, tool_call: Any, position: int) -> object:
        call_id = _field(tool_call, "id")
        index = _field(tool_call, "index")

        if call_id is not None:
            return self._resolve_key_by_id(call_id, index, position)

        if index is not None:
            return self._resolve_key_by_index(tool_call, index, position)

        # Some providers omit id/index for follow-up tool-call deltas.
        # Merge by positional order so argument fragments are not split into fake calls.
        position_key = self._key_at_position(position)
        if position_key is not None:
            return position_key
        return ("position", position)

    @staticmethod
    def _merge_arguments(
        entry: dict[str, Any],
        *,
        arguments: Any,
        arguments_complete: bool,
    ) -> None:
        if arguments is None:
            return
        if not isinstance(arguments, str):
            entry["function"]["arguments"] = arguments
            return
        if arguments_complete:
            entry["function"]["arguments"] = arguments
            return

        existing = entry["function"].get("arguments", "")
        if not isinstance(existing, str):
            existing = ""
        entry["function"]["arguments"] = existing + arguments

    def add_deltas(self, tool_calls: list[Any]) -> None:
        for position, tool_call in enumerate(tool_calls):
            key = self._resolve_key(tool_call, position)
            if key not in self._calls:
                self._order.append(key)
                self._calls[key] = {"function": {"name": "", "arguments": ""}}
            entry = self._calls[key]
            call_id = _field(tool_call, "id")
            if call_id:
                entry["id"] = call_id
            call_type = _field(tool_call, "type")
            if call_type:
                entry["type"] = call_type
            func = _field(tool_call, "function")
            if func is None:
                continue
            name = _field(func, "name")
            if name:
                entry["function"]["name"] = name
            arguments = _field(func, "arguments")
            self._merge_arguments(
                entry,
                arguments=arguments,
                arguments_complete=bool(_field(tool_call, "arguments_complete", False)),
            )

    def finalize(self) -> list[dict[str, Any]]:
        return [self._calls[key] for key in self._order]


class ChatClient:
    """Chat operations with structured outputs."""

    def __init__(
        self,
        core: LLMCore,
        tool_executor: ToolExecutor,
        tape: TapeManager,
        async_tape: AsyncTapeManager,
    ) -> None:
        self._core = core
        self._tool_executor = tool_executor
        self._tape = tape
        self._async_tape = async_tape

    @property
    def default_context(self) -> TapeContext:
        return self._tape.default_context

    @staticmethod
    def _is_non_stream_response(response: Any) -> bool:
        return (
            isinstance(response, str)
            or _field(response, "choices") is not None
            or _field(response, "output") is not None
            or _field(response, "output_text") is not None
        )

    def _validate_chat_input(
        self,
        *,
        prompt: str | None,
        messages: list[MessageInput] | None,
        system_prompt: str | None,
        tape: str | None,
    ) -> None:
        if prompt is not None and messages is not None:
            raise ErrorPayload(ErrorKind.INVALID_INPUT, "Provide either prompt or messages, not both.")
        if prompt is None and messages is None:
            raise ErrorPayload(ErrorKind.INVALID_INPUT, "Either prompt or messages is required.")
        if messages is not None and (system_prompt is not None or tape is not None):
            raise ErrorPayload(
                ErrorKind.INVALID_INPUT,
                "system_prompt and tape are not supported with messages input.",
            )

    def _prepare_messages(
        self,
        prompt: str | None,
        system_prompt: str | None,
        tape: str | None,
        messages: list[MessageInput] | None,
        context: TapeContext | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._validate_chat_input(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tape=tape,
        )

        if messages is not None:
            payload = [dict(message) for message in messages]
            return payload, []

        if prompt is None:
            raise ErrorPayload(ErrorKind.INVALID_INPUT, "prompt is required when messages is not provided")

        user_message = {"role": "user", "content": prompt}

        if tape is None:
            payload: list[dict[str, Any]] = []
            if system_prompt:
                payload.append({"role": "system", "content": system_prompt})
            payload.append(user_message)
            return payload, []

        history = self._tape.read_messages(tape, context=context)
        payload = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(history)
        payload.append(user_message)
        return payload, [user_message]

    async def _prepare_messages_async(
        self,
        prompt: str | None,
        system_prompt: str | None,
        tape: str | None,
        messages: list[MessageInput] | None,
        context: TapeContext | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._validate_chat_input(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tape=tape,
        )

        if messages is not None:
            payload = [dict(message) for message in messages]
            return payload, []

        if prompt is None:
            raise ErrorPayload(ErrorKind.INVALID_INPUT, "prompt is required when messages is not provided")

        user_message = {"role": "user", "content": prompt}

        if tape is None:
            payload: list[dict[str, Any]] = []
            if system_prompt:
                payload.append({"role": "system", "content": system_prompt})
            payload.append(user_message)
            return payload, []

        history = await self._async_tape.read_messages(tape, context=context)
        payload = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(history)
        payload.append(user_message)
        return payload, [user_message]

    def _prepare_request(
        self,
        *,
        prompt: str | None,
        system_prompt: str | None,
        messages: list[MessageInput] | None,
        tape: str | None,
        context: TapeContext | None,
        tools: ToolInput,
        require_tools: bool = False,
        require_runnable: bool = False,
    ) -> PreparedChat:
        context_error: ErrorPayload | None = None
        payload: list[dict[str, Any]] = []
        new_messages: list[dict[str, Any]] = []
        try:
            payload, new_messages = self._prepare_messages(
                prompt,
                system_prompt,
                tape,
                messages,
                context=context,
            )
            if require_tools and not tools:
                raise ErrorPayload(ErrorKind.INVALID_INPUT, "tools are required for this operation.")  # noqa: TRY301
            toolset = self._normalize_tools(tools)
        except ErrorPayload as exc:
            context_error = exc
            toolset = ToolSet([], [])
        else:
            if require_runnable:
                try:
                    toolset.require_runnable()
                except ValueError as exc:
                    context_error = ErrorPayload(ErrorKind.INVALID_INPUT, str(exc))

        should_update = tape is not None and messages is None
        run_id = uuid.uuid4().hex
        return PreparedChat(
            payload=payload,
            new_messages=new_messages,
            toolset=toolset,
            tape=tape,
            should_update=should_update,
            context_error=context_error,
            run_id=run_id,
            system_prompt=system_prompt,
            context=context,
        )

    async def _prepare_request_async(
        self,
        *,
        prompt: str | None,
        system_prompt: str | None,
        messages: list[MessageInput] | None,
        tape: str | None,
        context: TapeContext | None,
        tools: ToolInput,
        require_tools: bool = False,
        require_runnable: bool = False,
    ) -> PreparedChat:
        context_error: ErrorPayload | None = None
        payload: list[dict[str, Any]] = []
        new_messages: list[dict[str, Any]] = []
        try:
            payload, new_messages = await self._prepare_messages_async(
                prompt,
                system_prompt,
                tape,
                messages,
                context=context,
            )
            if require_tools and not tools:
                raise ErrorPayload(ErrorKind.INVALID_INPUT, "tools are required for this operation.")  # noqa: TRY301
            toolset = self._normalize_tools(tools)
        except ErrorPayload as exc:
            context_error = exc
            toolset = ToolSet([], [])
        else:
            if require_runnable:
                try:
                    toolset.require_runnable()
                except ValueError as exc:
                    context_error = ErrorPayload(ErrorKind.INVALID_INPUT, str(exc))

        should_update = tape is not None and messages is None
        run_id = uuid.uuid4().hex
        return PreparedChat(
            payload=payload,
            new_messages=new_messages,
            toolset=toolset,
            tape=tape,
            should_update=should_update,
            context_error=context_error,
            run_id=run_id,
            system_prompt=system_prompt,
            context=context,
        )

    def _execute_sync(
        self,
        prepared: PreparedChat,
        *,
        tools_payload: list[dict[str, Any]] | None,
        model: str | None,
        provider: str | None,
        max_tokens: int | None,
        stream: bool,
        kwargs: dict[str, Any],
        on_response: Callable[[Any, str, str, int], Any],
    ) -> Any:
        if prepared.context_error is not None:
            raise prepared.context_error
        try:
            return self._core.run_chat_sync(
                messages_payload=prepared.payload,
                tools_payload=tools_payload,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=stream,
                reasoning_effort=None,
                kwargs=kwargs,
                on_response=on_response,
            )
        except RepublicError as exc:
            raise ErrorPayload(exc.kind, exc.message) from exc

    async def _execute_async(
        self,
        prepared: PreparedChat,
        *,
        tools_payload: list[dict[str, Any]] | None,
        model: str | None,
        provider: str | None,
        max_tokens: int | None,
        stream: bool,
        kwargs: dict[str, Any],
        on_response: Callable[[Any, str, str, int], Any],
    ) -> Any:
        if prepared.context_error is not None:
            raise prepared.context_error
        try:
            return await self._core.run_chat_async(
                messages_payload=prepared.payload,
                tools_payload=tools_payload,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=stream,
                reasoning_effort=None,
                kwargs=kwargs,
                on_response=on_response,
            )
        except RepublicError as exc:
            raise ErrorPayload(exc.kind, exc.message) from exc

    def _normalize_tools(self, tools: ToolInput) -> ToolSet:
        try:
            return normalize_tools(tools)
        except (ValueError, TypeError) as exc:
            raise ErrorPayload(ErrorKind.INVALID_INPUT, str(exc)) from exc

    def _update_tape(
        self,
        prepared: PreparedChat,
        response_text: str | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        error: ErrorPayload | None = None,
        response: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if not prepared.should_update:
            return
        if prepared.tape is None:
            return
        self._tape.record_chat(
            tape=prepared.tape,
            run_id=prepared.run_id,
            system_prompt=prepared.system_prompt,
            context_error=prepared.context_error,
            new_messages=prepared.new_messages,
            response_text=response_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            error=error,
            response=response,
            provider=provider,
            model=model,
            usage=usage,
        )

    async def _update_tape_async(
        self,
        prepared: PreparedChat,
        response_text: str | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        error: ErrorPayload | None = None,
        response: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if not prepared.should_update:
            return
        if prepared.tape is None:
            return
        await self._async_tape.record_chat(
            tape=prepared.tape,
            run_id=prepared.run_id,
            system_prompt=prepared.system_prompt,
            context_error=prepared.context_error,
            new_messages=prepared.new_messages,
            response_text=response_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            error=error,
            response=response,
            provider=provider,
            model=model,
            usage=usage,
        )

    @staticmethod
    def _empty_iterator() -> Iterator[str]:
        return iter(())

    @staticmethod
    async def _empty_async_iterator() -> AsyncIterator[str]:
        if False:
            yield ""

    def _stream_error_result(self, prepared: PreparedChat, error: ErrorPayload) -> TextStream:
        self._update_tape(prepared, None, error=error)
        return TextStream(self._empty_iterator(), state=StreamState(error=error))

    async def _stream_async_error_result(self, prepared: PreparedChat, error: ErrorPayload) -> AsyncTextStream:
        await self._update_tape_async(prepared, None, error=error)
        return AsyncTextStream(self._empty_async_iterator(), state=StreamState(error=error))

    def _event_error_result(self, prepared: PreparedChat, error: ErrorPayload) -> StreamEvents:
        self._update_tape(prepared, None, error=error)
        state = StreamState(error=error)
        events = [
            StreamEvent("error", error.as_dict()),
            StreamEvent(
                "final",
                self._final_event_data(
                    text=None,
                    tool_calls=[],
                    tool_results=[],
                    usage=None,
                    error=error,
                ),
            ),
        ]
        return StreamEvents(iter(events), state=state)

    async def _event_async_error_result(self, prepared: PreparedChat, error: ErrorPayload) -> AsyncStreamEvents:
        await self._update_tape_async(prepared, None, error=error)
        state = StreamState(error=error)

        async def _iterator() -> AsyncIterator[StreamEvent]:
            yield StreamEvent("error", error.as_dict())
            yield StreamEvent(
                "final",
                self._final_event_data(
                    text=None,
                    tool_calls=[],
                    tool_results=[],
                    usage=None,
                    error=error,
                ),
            )

        return AsyncStreamEvents(_iterator(), state=state)

    @staticmethod
    def _final_event_data(
        *,
        text: str | None,
        tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        usage: dict[str, Any] | None,
        error: ErrorPayload | None,
    ) -> dict[str, Any]:
        return {
            "text": text,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "usage": usage,
            "ok": error is None,
        }

    def _finalize_text_stream(
        self,
        prepared: PreparedChat,
        *,
        text: str | None,
        tool_calls: list[dict[str, Any]],
        state: StreamState,
        provider_name: str,
        model_id: str,
        attempt: int,
        usage: dict[str, Any] | None = None,
        response: Any | None = None,
        log_empty: bool = False,
    ) -> None:
        if not text and not tool_calls and state.error is None:
            empty_error = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
            if log_empty:
                self._core.log_error(empty_error, provider_name, model_id, attempt)
            state.error = ErrorPayload(empty_error.kind, empty_error.message)
        state.usage = usage
        self._update_tape(
            prepared,
            text or None,
            tool_calls=tool_calls or None,
            tool_results=None,
            error=state.error,
            response=response,
            provider=provider_name,
            model=model_id,
            usage=usage,
        )

    async def _finalize_text_stream_async(
        self,
        prepared: PreparedChat,
        *,
        text: str | None,
        tool_calls: list[dict[str, Any]],
        state: StreamState,
        provider_name: str,
        model_id: str,
        attempt: int,
        usage: dict[str, Any] | None = None,
        response: Any | None = None,
        log_empty: bool = False,
    ) -> None:
        if not text and not tool_calls and state.error is None:
            empty_error = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
            if log_empty:
                self._core.log_error(empty_error, provider_name, model_id, attempt)
            state.error = ErrorPayload(empty_error.kind, empty_error.message)
        state.usage = usage
        await self._update_tape_async(
            prepared,
            text or None,
            tool_calls=tool_calls or None,
            tool_results=None,
            error=state.error,
            response=response,
            provider=provider_name,
            model=model_id,
            usage=usage,
        )

    def _finalize_event_stream(
        self,
        prepared: PreparedChat,
        *,
        parts: list[str],
        tool_calls: list[dict[str, Any]],
        state: StreamState,
        provider_name: str,
        model_id: str,
        attempt: int,
        usage: dict[str, Any] | None,
    ) -> tuple[list[StreamEvent], list[Any]]:
        events: list[StreamEvent] = []
        for idx, call in enumerate(tool_calls):
            events.append(StreamEvent("tool_call", {"index": idx, "call": call}))

        tool_results: list[Any] = []
        try:
            tool_results = self._execute_tool_calls(
                prepared,
                tool_calls,
                provider_name,
                model_id,
            )
        except ErrorPayload as exc:
            state.error = exc
            events.append(StreamEvent("error", exc.as_dict()))
        if tool_results:
            for idx, result in enumerate(tool_results):
                events.append(StreamEvent("tool_result", {"index": idx, "result": result}))

        if not parts and not tool_calls and state.error is None:
            empty = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
            self._core.log_error(empty, provider_name, model_id, attempt)
            empty_error = ErrorPayload(empty.kind, empty.message)
            state.error = empty_error
            events.append(StreamEvent("error", empty_error.as_dict()))

        if usage is not None:
            events.append(StreamEvent("usage", usage))

        events.append(
            StreamEvent(
                "final",
                self._final_event_data(
                    text="".join(parts) if parts else None,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=state.error,
                ),
            )
        )
        return events, tool_results

    async def _finalize_event_stream_async(
        self,
        prepared: PreparedChat,
        *,
        parts: list[str],
        tool_calls: list[dict[str, Any]],
        state: StreamState,
        provider_name: str,
        model_id: str,
        attempt: int,
        usage: dict[str, Any] | None,
    ) -> tuple[list[StreamEvent], list[Any]]:
        events: list[StreamEvent] = []
        for idx, call in enumerate(tool_calls):
            events.append(StreamEvent("tool_call", {"index": idx, "call": call}))

        tool_results: list[Any] = []
        try:
            tool_results = await self._execute_tool_calls_async(
                prepared,
                tool_calls,
                provider_name,
                model_id,
            )
        except ErrorPayload as exc:
            state.error = exc
            events.append(StreamEvent("error", exc.as_dict()))
        if tool_results:
            for idx, result in enumerate(tool_results):
                events.append(StreamEvent("tool_result", {"index": idx, "result": result}))

        if not parts and not tool_calls and state.error is None:
            empty = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
            self._core.log_error(empty, provider_name, model_id, attempt)
            empty_error = ErrorPayload(empty.kind, empty.message)
            state.error = empty_error
            events.append(StreamEvent("error", empty_error.as_dict()))

        if usage is not None:
            events.append(StreamEvent("usage", usage))

        events.append(
            StreamEvent(
                "final",
                self._final_event_data(
                    text="".join(parts) if parts else None,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=state.error,
                ),
            )
        )
        return events, tool_results

    def _finalize_event_stream_state(
        self,
        prepared: PreparedChat,
        *,
        parts: list[str],
        tool_calls: list[dict[str, Any]] | None,
        tool_results: list[Any],
        state: StreamState,
        provider_name: str,
        model_id: str,
        usage: dict[str, Any] | None,
        assembler: ToolCallAssembler,
    ) -> list[dict[str, Any]]:
        state.usage = usage
        final_calls = tool_calls or assembler.finalize()
        self._update_tape(
            prepared,
            "".join(parts) if parts else None,
            tool_calls=final_calls or None,
            tool_results=tool_results or None,
            error=state.error,
            provider=provider_name,
            model=model_id,
            usage=usage,
        )
        return final_calls

    async def _finalize_event_stream_state_async(
        self,
        prepared: PreparedChat,
        *,
        parts: list[str],
        tool_calls: list[dict[str, Any]] | None,
        tool_results: list[Any],
        state: StreamState,
        provider_name: str,
        model_id: str,
        usage: dict[str, Any] | None,
        assembler: ToolCallAssembler,
    ) -> list[dict[str, Any]]:
        state.usage = usage
        final_calls = tool_calls or assembler.finalize()
        await self._update_tape_async(
            prepared,
            "".join(parts) if parts else None,
            tool_calls=final_calls or None,
            tool_results=tool_results or None,
            error=state.error,
            provider=provider_name,
            model=model_id,
            usage=usage,
        )
        return final_calls

    def _error_event_sequence(
        self,
        *,
        parts: list[str],
        tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        usage: dict[str, Any] | None,
        error: ErrorPayload,
    ) -> list[StreamEvent]:
        return [
            StreamEvent("error", error.as_dict()),
            StreamEvent(
                "final",
                self._final_event_data(
                    text="".join(parts) if parts else None,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=error,
                ),
            ),
        ]

    def _handle_create_response(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> str | object:
        text = self._extract_text(response)
        if text:
            self._update_tape(
                prepared,
                text,
                response=response,
                provider=provider_name,
                model=model_id,
            )
            return text
        empty_error = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
        self._core.log_error(empty_error, provider_name, model_id, attempt)
        return self._core.RETRY

    async def _handle_create_response_async(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> str | object:
        text = self._extract_text(response)
        if text:
            await self._update_tape_async(
                prepared,
                text,
                response=response,
                provider=provider_name,
                model=model_id,
            )
            return text
        empty_error = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
        self._core.log_error(empty_error, provider_name, model_id, attempt)
        return self._core.RETRY

    def _handle_tool_calls_response(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        calls = self._extract_tool_calls(response)
        self._update_tape(
            prepared,
            None,
            tool_calls=calls,
            tool_results=[],
            response=response,
            provider=provider_name,
            model=model_id,
        )
        return calls

    async def _handle_tool_calls_response_async(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        calls = self._extract_tool_calls(response)
        await self._update_tape_async(
            prepared,
            None,
            tool_calls=calls,
            tool_results=[],
            response=response,
            provider=provider_name,
            model=model_id,
        )
        return calls

    def _handle_tools_auto_response(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> ToolAutoResult | object:
        tool_calls = self._extract_tool_calls(response)
        if tool_calls:
            execution = self._tool_executor.execute(
                tool_calls,
                tools=prepared.toolset.runnable,
                context=self._make_tool_context(prepared, provider_name, model_id),
            )
            self._update_tape(
                prepared,
                None,
                tool_calls=execution.tool_calls,
                tool_results=execution.tool_results,
                response=response,
                provider=provider_name,
                model=model_id,
            )
            return ToolAutoResult.tools_result(execution.tool_calls, execution.tool_results)

        text = self._extract_text(response)
        if text:
            self._update_tape(
                prepared,
                text,
                response=response,
                provider=provider_name,
                model=model_id,
            )
            return ToolAutoResult.text_result(text)

        empty_error = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
        self._core.log_error(empty_error, provider_name, model_id, attempt)
        return self._core.RETRY

    async def _handle_tools_auto_response_async(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> ToolAutoResult | object:
        tool_calls = self._extract_tool_calls(response)
        if tool_calls:
            execution = await self._tool_executor.execute_async(
                tool_calls,
                tools=prepared.toolset.runnable,
                context=self._make_tool_context(prepared, provider_name, model_id),
            )
            await self._update_tape_async(
                prepared,
                None,
                tool_calls=execution.tool_calls,
                tool_results=execution.tool_results,
                response=response,
                provider=provider_name,
                model=model_id,
            )
            return ToolAutoResult.tools_result(execution.tool_calls, execution.tool_results)

        text = self._extract_text(response)
        if text:
            await self._update_tape_async(
                prepared,
                text,
                response=response,
                provider=provider_name,
                model=model_id,
            )
            return ToolAutoResult.text_result(text)

        empty_error = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
        self._core.log_error(empty_error, provider_name, model_id, attempt)
        return self._core.RETRY

    def chat(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> str:
        prepared = self._prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=None,
        )
        try:
            return self._execute_sync(
                prepared,
                tools_payload=None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=False,
                kwargs=kwargs,
                on_response=partial(self._handle_create_response, prepared),
            )
        except ErrorPayload as error:
            self._update_tape(prepared, None, error=error)
            raise

    def tool_calls(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        prepared = self._prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=tools,
            require_tools=True,
        )
        try:
            return self._execute_sync(
                prepared,
                tools_payload=prepared.toolset.payload,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=False,
                kwargs=kwargs,
                on_response=partial(self._handle_tool_calls_response, prepared),
            )
        except ErrorPayload as error:
            self._update_tape(prepared, None, error=error)
            raise

    def run_tools(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> ToolAutoResult:
        prepared = self._prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=tools,
            require_tools=True,
            require_runnable=True,
        )
        try:
            return self._execute_sync(
                prepared,
                tools_payload=prepared.toolset.payload,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=False,
                kwargs=kwargs,
                on_response=partial(self._handle_tools_auto_response, prepared),
            )
        except ErrorPayload as error:
            self._update_tape(prepared, None, error=error)
            return ToolAutoResult.error_result(error)

    async def chat_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> str:
        prepared = await self._prepare_request_async(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=None,
        )
        try:
            return await self._execute_async(
                prepared,
                tools_payload=None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=False,
                kwargs=kwargs,
                on_response=partial(self._handle_create_response_async, prepared),
            )
        except ErrorPayload as error:
            await self._update_tape_async(prepared, None, error=error)
            raise

    async def tool_calls_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        prepared = await self._prepare_request_async(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=tools,
            require_tools=True,
        )
        try:
            return await self._execute_async(
                prepared,
                tools_payload=prepared.toolset.payload,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=False,
                kwargs=kwargs,
                on_response=partial(self._handle_tool_calls_response_async, prepared),
            )
        except ErrorPayload as error:
            await self._update_tape_async(prepared, None, error=error)
            raise

    async def run_tools_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> ToolAutoResult:
        prepared = await self._prepare_request_async(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=tools,
            require_tools=True,
            require_runnable=True,
        )
        try:
            return await self._execute_async(
                prepared,
                tools_payload=prepared.toolset.payload,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=False,
                kwargs=kwargs,
                on_response=partial(self._handle_tools_auto_response_async, prepared),
            )
        except ErrorPayload as error:
            await self._update_tape_async(prepared, None, error=error)
            return ToolAutoResult.error_result(error)

    def stream(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> TextStream:
        prepared = self._prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=None,
        )
        try:
            return self._execute_sync(
                prepared,
                tools_payload=None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=True,
                kwargs=kwargs,
                on_response=partial(self._build_text_stream, prepared),
            )
        except ErrorPayload as error:
            return self._stream_error_result(prepared, error)

    async def stream_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> AsyncTextStream:
        prepared = await self._prepare_request_async(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=None,
        )
        try:
            return await self._execute_async(
                prepared,
                tools_payload=None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=True,
                kwargs=kwargs,
                on_response=partial(self._build_async_text_stream, prepared),
            )
        except ErrorPayload as error:
            return await self._stream_async_error_result(prepared, error)

    def stream_events(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> StreamEvents:
        prepared = self._prepare_request(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=tools,
        )
        try:
            return self._execute_sync(
                prepared,
                tools_payload=prepared.toolset.payload or None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=True,
                kwargs=kwargs,
                on_response=partial(self._build_event_stream, prepared),
            )
        except ErrorPayload as error:
            return self._event_error_result(prepared, error)

    async def stream_events_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[MessageInput] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> AsyncStreamEvents:
        prepared = await self._prepare_request_async(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tape=tape,
            context=context,
            tools=tools,
        )
        try:
            return await self._execute_async(
                prepared,
                tools_payload=prepared.toolset.payload or None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                stream=True,
                kwargs=kwargs,
                on_response=partial(self._build_async_event_stream, prepared),
            )
        except ErrorPayload as error:
            return await self._event_async_error_result(prepared, error)

    def _build_text_stream(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> TextStream:
        if self._is_non_stream_response(response):
            text = self._extract_text(response)
            tool_calls = self._extract_tool_calls(response)
            state = StreamState()
            self._finalize_text_stream(
                prepared,
                text=text or None,
                tool_calls=tool_calls,
                state=state,
                provider_name=provider_name,
                model_id=model_id,
                attempt=attempt,
                usage=self._extract_usage(response),
                response=response,
                log_empty=False,
            )
            return TextStream(iter([text]) if text else self._empty_iterator(), state=state)

        state = StreamState()
        parts: list[str] = []
        assembler = ToolCallAssembler()

        usage: dict[str, Any] | None = None

        def _iterator() -> Iterator[str]:
            nonlocal usage
            try:
                for chunk in response:
                    deltas = self._extract_chunk_tool_call_deltas(chunk)
                    if deltas:
                        assembler.add_deltas(deltas)
                    text = self._extract_chunk_text(chunk)
                    if text:
                        parts.append(text)
                        yield text
                    usage = self._extract_usage(chunk) or usage
            except Exception as exc:
                kind = self._core.classify_exception(exc)
                wrapped = self._core.wrap_error(exc, kind, provider_name, model_id)
                state.error = ErrorPayload(wrapped.kind, wrapped.message)
            finally:
                tool_calls = assembler.finalize()
                self._finalize_text_stream(
                    prepared,
                    text="".join(parts) if parts else None,
                    tool_calls=tool_calls,
                    state=state,
                    provider_name=provider_name,
                    model_id=model_id,
                    attempt=attempt,
                    usage=usage,
                    log_empty=True,
                )

        return TextStream(_iterator(), state=state)

    async def _build_async_text_stream(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> AsyncTextStream:
        if self._is_non_stream_response(response):
            text = self._extract_text(response)
            tool_calls = self._extract_tool_calls(response)
            state = StreamState()
            await self._finalize_text_stream_async(
                prepared,
                text=text or None,
                tool_calls=tool_calls,
                state=state,
                provider_name=provider_name,
                model_id=model_id,
                attempt=attempt,
                usage=self._extract_usage(response),
                response=response,
                log_empty=False,
            )

            async def _single() -> AsyncIterator[str]:
                if text:
                    yield text

            return AsyncTextStream(_single(), state=state)

        state = StreamState()
        parts: list[str] = []
        usage: dict[str, Any] | None = None
        assembler = ToolCallAssembler()

        async def _iterator() -> AsyncIterator[str]:
            nonlocal usage
            try:
                async for chunk in response:
                    deltas = self._extract_chunk_tool_call_deltas(chunk)
                    if deltas:
                        assembler.add_deltas(deltas)
                    text = self._extract_chunk_text(chunk)
                    if text:
                        parts.append(text)
                        yield text
                    usage = self._extract_usage(chunk) or usage
            except Exception as exc:
                kind = self._core.classify_exception(exc)
                wrapped = self._core.wrap_error(exc, kind, provider_name, model_id)
                state.error = ErrorPayload(wrapped.kind, wrapped.message)
            finally:
                tool_calls = assembler.finalize()
                await self._finalize_text_stream_async(
                    prepared,
                    text="".join(parts) if parts else None,
                    tool_calls=tool_calls,
                    state=state,
                    provider_name=provider_name,
                    model_id=model_id,
                    attempt=attempt,
                    usage=usage,
                    log_empty=True,
                )

        return AsyncTextStream(_iterator(), state=state)

    @staticmethod
    def _chunk_has_tool_calls(chunk: Any) -> bool:
        return bool(ChatClient._extract_chunk_tool_call_deltas(chunk))

    @staticmethod
    def _responses_tool_delta_from_args_event(chunk: Any, event_type: str) -> list[Any]:
        item_id = _field(chunk, "item_id")
        if not item_id:
            return []
        arguments = _field(chunk, "delta")
        if event_type == "response.function_call_arguments.done":
            arguments = _field(chunk, "arguments")
        if not isinstance(arguments, str):
            return []
        call_id = _field(chunk, "call_id")
        payload: dict[str, Any] = {
            "index": item_id,
            "type": "function",
            "function": SimpleNamespace(name=_field(chunk, "name") or "", arguments=arguments),
            "arguments_complete": event_type == "response.function_call_arguments.done",
        }
        if call_id:
            payload["id"] = call_id
        return [SimpleNamespace(**payload)]

    @staticmethod
    def _responses_tool_delta_from_output_item_event(chunk: Any, event_type: str) -> list[Any]:
        item = _field(chunk, "item")
        if _field(item, "type") != "function_call":
            return []
        item_id = _field(item, "id")
        call_id = _field(item, "call_id") or item_id
        if not call_id:
            return []
        arguments = _field(item, "arguments")
        if not isinstance(arguments, str):
            arguments = ""
        return [
            SimpleNamespace(
                id=call_id,
                index=item_id or call_id,
                type="function",
                function=SimpleNamespace(name=_field(item, "name") or "", arguments=arguments),
                arguments_complete=event_type == "response.output_item.done",
            )
        ]

    @staticmethod
    def _extract_chunk_tool_call_deltas(chunk: Any) -> list[Any]:
        event_type = _field(chunk, "type")
        if event_type in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
            return ChatClient._responses_tool_delta_from_args_event(chunk, event_type)
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            return ChatClient._responses_tool_delta_from_output_item_event(chunk, event_type)

        choices = _field(chunk, "choices")
        if not choices:
            return []
        delta = _field(choices[0], "delta")
        if delta is None:
            return []
        return _field(delta, "tool_calls") or []

    @staticmethod
    def _extract_chunk_text(chunk: Any) -> str:
        event_type = _field(chunk, "type")
        if event_type == "response.output_text.delta":
            delta = _field(chunk, "delta")
            if isinstance(delta, str):
                return delta
            return ""

        choices = _field(chunk, "choices")
        if not choices:
            return ""
        delta = _field(choices[0], "delta")
        if delta is None:
            return ""
        return _field(delta, "content", "") or ""

    def _build_event_stream(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> StreamEvents:
        if self._is_non_stream_response(response):
            return self._build_event_stream_from_response(
                prepared,
                response,
                provider_name,
                model_id,
            )

        state = StreamState()
        usage: dict[str, Any] | None = None
        parts: list[str] = []
        tool_calls: list[dict[str, Any]] | None = None
        tool_results: list[Any] = []
        assembler = ToolCallAssembler()

        def _iterator() -> Iterator[StreamEvent]:
            nonlocal usage, tool_calls, tool_results
            try:
                for chunk in response:
                    usage = self._extract_usage(chunk) or usage
                    assembler.add_deltas(self._extract_chunk_tool_call_deltas(chunk))
                    text = self._extract_chunk_text(chunk)
                    if text:
                        parts.append(text)
                        yield StreamEvent("text", {"delta": text})

                tool_calls = assembler.finalize()
                events, tool_results = self._finalize_event_stream(
                    prepared,
                    parts=parts,
                    tool_calls=tool_calls,
                    state=state,
                    provider_name=provider_name,
                    model_id=model_id,
                    attempt=attempt,
                    usage=usage,
                )
                yield from events
            except Exception as exc:
                kind = self._core.classify_exception(exc)
                wrapped = self._core.wrap_error(exc, kind, provider_name, model_id)
                state.error = ErrorPayload(wrapped.kind, wrapped.message)
                final_calls = tool_calls or assembler.finalize()
                yield from self._error_event_sequence(
                    parts=parts,
                    tool_calls=final_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=state.error,
                )
            finally:
                tool_calls = self._finalize_event_stream_state(
                    prepared,
                    parts=parts,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    state=state,
                    provider_name=provider_name,
                    model_id=model_id,
                    usage=usage,
                    assembler=assembler,
                )

        return StreamEvents(_iterator(), state=state)

    def _build_async_event_stream(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
        attempt: int,
    ) -> AsyncStreamEvents:
        if self._is_non_stream_response(response):
            return self._build_async_event_stream_from_response(
                prepared,
                response,
                provider_name,
                model_id,
            )

        state = StreamState()
        usage: dict[str, Any] | None = None
        parts: list[str] = []
        tool_calls: list[dict[str, Any]] | None = None
        tool_results: list[Any] = []
        assembler = ToolCallAssembler()

        async def _iterator() -> AsyncIterator[StreamEvent]:
            nonlocal usage, tool_calls, tool_results
            try:
                async for chunk in response:
                    usage = self._extract_usage(chunk) or usage
                    assembler.add_deltas(self._extract_chunk_tool_call_deltas(chunk))
                    text = self._extract_chunk_text(chunk)
                    if text:
                        parts.append(text)
                        yield StreamEvent("text", {"delta": text})

                tool_calls = assembler.finalize()
                events, tool_results = await self._finalize_event_stream_async(
                    prepared,
                    parts=parts,
                    tool_calls=tool_calls,
                    state=state,
                    provider_name=provider_name,
                    model_id=model_id,
                    attempt=attempt,
                    usage=usage,
                )
                for event in events:
                    yield event
            except Exception as exc:
                kind = self._core.classify_exception(exc)
                wrapped = self._core.wrap_error(exc, kind, provider_name, model_id)
                state.error = ErrorPayload(wrapped.kind, wrapped.message)
                final_calls = tool_calls or assembler.finalize()
                for event in self._error_event_sequence(
                    parts=parts,
                    tool_calls=final_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=state.error,
                ):
                    yield event
            finally:
                tool_calls = await self._finalize_event_stream_state_async(
                    prepared,
                    parts=parts,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    state=state,
                    provider_name=provider_name,
                    model_id=model_id,
                    usage=usage,
                    assembler=assembler,
                )

        return AsyncStreamEvents(_iterator(), state=state)

    def _build_event_stream_from_response(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
    ) -> StreamEvents:
        text = self._extract_text(response)
        tool_calls = self._extract_tool_calls(response)
        usage = self._extract_usage(response)
        state = StreamState(usage=usage)
        tool_results: list[Any] = []
        try:
            tool_results = self._execute_tool_calls(prepared, tool_calls, provider_name, model_id)
        except ErrorPayload as exc:
            state.error = exc
        if not text and not tool_calls and state.error is None:
            empty = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
            self._core.log_error(empty, provider_name, model_id, 0)
            state.error = ErrorPayload(empty.kind, empty.message)
        events: list[StreamEvent] = []
        if text:
            events.append(StreamEvent("text", {"delta": text}))
        for idx, call in enumerate(tool_calls):
            events.append(StreamEvent("tool_call", {"index": idx, "call": call}))
        for idx, result in enumerate(tool_results):
            events.append(StreamEvent("tool_result", {"index": idx, "result": result}))
        if state.error is not None:
            events.append(StreamEvent("error", state.error.as_dict()))
        if usage is not None:
            events.append(StreamEvent("usage", usage))
        events.append(
            StreamEvent(
                "final",
                self._final_event_data(
                    text=text or None,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=state.error,
                ),
            )
        )
        self._update_tape(
            prepared,
            text or None,
            tool_calls=tool_calls or None,
            tool_results=tool_results or None,
            error=state.error,
            response=response,
            provider=provider_name,
            model=model_id,
            usage=usage,
        )
        return StreamEvents(iter(events), state=state)

    def _build_async_event_stream_from_response(
        self,
        prepared: PreparedChat,
        response: Any,
        provider_name: str,
        model_id: str,
    ) -> AsyncStreamEvents:
        text = self._extract_text(response)
        tool_calls = self._extract_tool_calls(response)
        usage = self._extract_usage(response)
        state = StreamState(usage=usage)
        tool_results: list[Any] = []

        async def _iterator() -> AsyncIterator[StreamEvent]:
            nonlocal tool_results
            try:
                tool_results = await self._execute_tool_calls_async(prepared, tool_calls, provider_name, model_id)
            except ErrorPayload as exc:
                state.error = exc
            if not text and not tool_calls and state.error is None:
                empty = RepublicError(ErrorKind.TEMPORARY, f"{provider_name}:{model_id}: empty response")
                self._core.log_error(empty, provider_name, model_id, 0)
                state.error = ErrorPayload(empty.kind, empty.message)

            if text:
                yield StreamEvent("text", {"delta": text})
            for idx, call in enumerate(tool_calls):
                yield StreamEvent("tool_call", {"index": idx, "call": call})
            for idx, result in enumerate(tool_results):
                yield StreamEvent("tool_result", {"index": idx, "result": result})
            if state.error is not None:
                yield StreamEvent("error", state.error.as_dict())
            if usage is not None:
                yield StreamEvent("usage", usage)
            yield StreamEvent(
                "final",
                self._final_event_data(
                    text=text or None,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    usage=usage,
                    error=state.error,
                ),
            )

            await self._update_tape_async(
                prepared,
                text or None,
                tool_calls=tool_calls or None,
                tool_results=tool_results or None,
                error=state.error,
                response=response,
                provider=provider_name,
                model=model_id,
                usage=usage,
            )

        return AsyncStreamEvents(_iterator(), state=state)

    def _execute_tool_calls(
        self,
        prepared: PreparedChat,
        tool_calls: list[dict[str, Any]],
        provider_name: str,
        model_id: str,
    ) -> list[Any]:
        if not tool_calls:
            return []
        if not prepared.toolset.runnable:
            raise ErrorPayload(ErrorKind.TOOL, "No runnable tools are available.")
        execution = self._tool_executor.execute(
            tool_calls,
            tools=prepared.toolset.runnable,
            context=self._make_tool_context(prepared, provider_name, model_id),
        )
        return execution.tool_results

    async def _execute_tool_calls_async(
        self,
        prepared: PreparedChat,
        tool_calls: list[dict[str, Any]],
        provider_name: str,
        model_id: str,
    ) -> list[Any]:
        if not tool_calls:
            return []
        if not prepared.toolset.runnable:
            raise ErrorPayload(ErrorKind.TOOL, "No runnable tools are available.")
        execution = await self._tool_executor.execute_async(
            tool_calls,
            tools=prepared.toolset.runnable,
            context=self._make_tool_context(prepared, provider_name, model_id),
        )
        return execution.tool_results

    @staticmethod
    def _make_tool_context(prepared: PreparedChat, provider_name: str, model_id: str) -> ToolContext:
        return ToolContext(
            tape=prepared.tape,
            run_id=prepared.run_id,
            meta={"provider": provider_name, "model": model_id},
            state={} if prepared.context is None else prepared.context.state,
        )

    @staticmethod
    def _extract_text_from_responses_output(output: Any) -> str:
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            if _field(item, "type") != "message":
                continue
            content = _field(item, "content") or []
            for entry in content:
                if _field(entry, "type") == "output_text":
                    text = _field(entry, "text")
                    if text:
                        parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        output_text = _field(response, "output_text")
        if isinstance(output_text, str):
            return output_text
        output = _field(response, "output")
        text_from_output = ChatClient._extract_text_from_responses_output(output)
        if text_from_output:
            return text_from_output
        choices = _field(response, "choices")
        if not choices:
            return ""
        message = _field(choices[0], "message")
        if message is None:
            return ""
        return _field(message, "content", "") or ""

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        output = _field(response, "output")
        if output is not None:
            return ChatClient._extract_responses_tool_calls(output)
        return ChatClient._extract_completion_tool_calls(response)

    @staticmethod
    def _extract_responses_tool_calls(output: Any) -> list[dict[str, Any]]:
        if not isinstance(output, list):
            return []
        calls: list[dict[str, Any]] = []
        for item in output:
            if _field(item, "type") != "function_call":
                continue
            name = _field(item, "name")
            arguments = _field(item, "arguments")
            if not name:
                continue
            entry: dict[str, Any] = {"function": {"name": name, "arguments": arguments or ""}}
            call_id = _field(item, "call_id") or _field(item, "id")
            if call_id:
                entry["id"] = call_id
            entry["type"] = "function"
            calls.append(entry)
        return calls

    @staticmethod
    def _extract_completion_tool_calls(response: Any) -> list[dict[str, Any]]:
        choices = _field(response, "choices")
        if not choices:
            return []
        message = _field(choices[0], "message")
        if message is None:
            return []
        tool_calls = _field(message, "tool_calls") or []
        calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = _field(tool_call, "function")
            if function is None:
                continue
            entry: dict[str, Any] = {
                "function": {
                    "name": _field(function, "name"),
                    "arguments": _field(function, "arguments"),
                }
            }
            call_id = _field(tool_call, "id")
            if call_id:
                entry["id"] = call_id
            call_type = _field(tool_call, "type")
            if call_type:
                entry["type"] = call_type
            calls.append(entry)
        return calls

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any] | None:
        event_type = _field(response, "type")
        if event_type in {"response.completed", "response.in_progress", "response.failed", "response.incomplete"}:
            usage = _field(_field(response, "response"), "usage")
        else:
            usage = _field(response, "usage")
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return dict(usage)
        data: dict[str, Any] = {}
        for field in ("input_tokens", "output_tokens", "total_tokens", "requests"):
            value = _field(usage, field)
            if value is not None:
                data[field] = value
        return data or None
