"""Tape session view helpers for Republic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from republic.core.results import (
    AsyncStreamEvents,
    AsyncTextStream,
    StreamEvents,
    TextStream,
    ToolAutoResult,
)
from republic.tape.context import TapeContext
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery
from republic.tape.store import AsyncTapeStore, TapeStore
from republic.tools.schema import ToolInput

if TYPE_CHECKING:
    from republic.clients.chat import ChatClient


class _TapeBase:
    def __init__(
        self,
        name: str,
        *,
        chat_client: ChatClient,
        context: TapeContext | None = None,
    ) -> None:
        self._name = name
        self._client = chat_client
        self._local_context = context

    def __repr__(self) -> str:
        return f"<Tape name={self._name}>"

    @property
    def name(self) -> str:
        return self._name

    @property
    def context(self) -> TapeContext:
        return self._local_context or self._client.default_context

    @context.setter
    def context(self, value: TapeContext | None) -> None:
        self._local_context = value


class Tape(_TapeBase):
    """A scoped LLM session that interacts with a specific tape."""

    def read_messages(self, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = context or self.context
        return self._client._tape.read_messages(self._name, context=active_context)

    @property
    def query(self) -> TapeQuery[TapeStore]:
        return self._client._tape.query_tape(self._name)

    def append(self, entry: TapeEntry) -> TapeEntry:
        return self._client._tape.append_entry(self._name, entry)

    def handoff(self, name: str, payload: Any | None = None, **meta: Any) -> TapeEntry:
        return self._client._tape.handoff(self._name, name, payload, **meta)

    def chat(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        return self._client.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            **kwargs,
        )

    def tool_calls(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return self._client.tool_calls(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            tools=tools,
            **kwargs,
        )

    def run_tools(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> ToolAutoResult:
        return self._client.run_tools(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            tools=tools,
            **kwargs,
        )

    def stream(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> TextStream:
        return self._client.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            **kwargs,
        )

    def stream_events(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> StreamEvents:
        return self._client.stream_events(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            tools=tools,
            **kwargs,
        )

    @property
    def query_async(self) -> TapeQuery[AsyncTapeStore]:
        return self._client._async_tape.query_tape(self._name)

    async def read_messages_async(self, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = context or self.context
        return await self._client._async_tape.read_messages(self._name, context=active_context)

    async def handoff_async(self, name: str, payload: Any | None = None, **meta: Any) -> TapeEntry:
        return await self._client._async_tape.handoff(self._name, name, payload, **meta)

    async def append_async(self, entry: TapeEntry) -> TapeEntry:
        return await self._client._async_tape.append_entry(self._name, entry)

    async def chat_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        return await self._client.chat_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            **kwargs,
        )

    async def tool_calls_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return await self._client.tool_calls_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            tools=tools,
            **kwargs,
        )

    async def run_tools_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> ToolAutoResult:
        return await self._client.run_tools_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            tools=tools,
            **kwargs,
        )

    async def stream_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncTextStream:
        return await self._client.stream_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            **kwargs,
        )

    async def stream_events_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> AsyncStreamEvents:
        return await self._client.stream_events_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=self._name,
            context=self.context,
            tools=tools,
            **kwargs,
        )
