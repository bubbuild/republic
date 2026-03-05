"""Republic LLM facade."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

from republic.__about__ import DEFAULT_MODEL
from republic.clients._internal import InternalOps
from republic.clients.chat import ChatClient
from republic.clients.embedding import EmbeddingClient
from republic.clients.text import TextClient
from republic.core.errors import ErrorKind, RepublicError
from republic.core.execution import LLMCore
from republic.core.results import (
    AsyncStreamEvents,
    AsyncTextStream,
    StreamEvents,
    TextStream,
    ToolAutoResult,
)
from republic.tape import (
    AsyncTapeManager,
    AsyncTapeStore,
    AsyncTapeStoreAdapter,
    InMemoryTapeStore,
    Tape,
    TapeContext,
    TapeManager,
    TapeStore,
)
from republic.tape.store import UnavailableTapeStore, is_async_tape_store
from republic.tools.executor import ToolExecutor
from republic.tools.schema import ToolInput


class LLM:
    """Developer-first LLM client powered by any-llm."""

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        api_key: str | dict[str, str] | None = None,
        api_base: str | dict[str, str] | None = None,
        client_args: dict[str, Any] | None = None,
        api_format: Literal["completion", "responses", "messages"] = "completion",
        verbose: int = 0,
        tape_store: TapeStore | AsyncTapeStore | None = None,
        context: TapeContext | None = None,
        error_classifier: Callable[[Exception], ErrorKind | None] | None = None,
    ) -> None:
        if verbose not in (0, 1, 2):
            raise RepublicError(ErrorKind.INVALID_INPUT, "verbose must be 0, 1, or 2")
        if max_retries < 0:
            raise RepublicError(ErrorKind.INVALID_INPUT, "max_retries must be >= 0")
        if api_format not in {"completion", "responses", "messages"}:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                "api_format must be 'completion', 'responses', or 'messages'",
            )

        if not model:
            model = DEFAULT_MODEL
            warnings.warn(f"No model was provided, defaulting to {model}", UserWarning, stacklevel=2)

        resolved_provider, resolved_model = LLMCore.resolve_model_provider(model, provider)

        self._core = LLMCore(
            provider=resolved_provider,
            model=resolved_model,
            fallback_models=fallback_models or [],
            max_retries=max_retries,
            api_key=api_key,
            api_base=api_base,
            client_args=client_args or {},
            api_format=api_format,
            verbose=verbose,
            error_classifier=error_classifier,
        )
        tool_executor = ToolExecutor()
        if tape_store is None:
            shared_tape_store = InMemoryTapeStore()
            sync_tape_store = shared_tape_store
            async_tape_store = AsyncTapeStoreAdapter(shared_tape_store)
        elif is_async_tape_store(tape_store):
            sync_tape_store = UnavailableTapeStore(
                "Sync tape APIs are unavailable when tape_store is AsyncTapeStore; use async chat/tool APIs.",
            )
            async_tape_store = tape_store
        else:
            tape_store = cast(TapeStore, tape_store)
            sync_tape_store = tape_store
            async_tape_store = AsyncTapeStoreAdapter(tape_store)

        self._tape = TapeManager(store=sync_tape_store, default_context=context)
        self._async_tape = AsyncTapeManager(store=async_tape_store, default_context=context)
        self._chat_client = ChatClient(
            self._core,
            tool_executor,
            tape=self._tape,
            async_tape=self._async_tape,
        )
        self._text_client = TextClient(self._chat_client)
        self.embeddings = EmbeddingClient(self._core)
        self.tools = tool_executor
        self._internal = InternalOps(self._core)

    @property
    def model(self) -> str:
        return self._core.model

    @property
    def provider(self) -> str:
        return self._core.provider

    @property
    def fallback_models(self) -> list[str]:
        return self._core.fallback_models

    @property
    def context(self) -> TapeContext:
        return self._async_tape.default_context

    @context.setter
    def context(self, value: TapeContext) -> None:
        self._tape.default_context = value
        self._async_tape.default_context = value

    def tape(self, name: str, *, context: TapeContext | None = None) -> Tape:
        return Tape(name, chat_client=self._chat_client, context=context)

    def chat(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> str:
        return self._chat_client.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
            **kwargs,
        )

    async def chat_async(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> str:
        return await self._chat_client.chat_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return self._chat_client.tool_calls(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
            tools=tools,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return await self._chat_client.tool_calls_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> ToolAutoResult:
        return self._chat_client.run_tools(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> ToolAutoResult:
        return await self._chat_client.run_tools_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
            tools=tools,
            **kwargs,
        )

    def if_(
        self,
        input_text: str,
        question: str,
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> bool:
        return self._text_client.if_(input_text, question, tape=tape, context=context)

    async def if_async(
        self,
        input_text: str,
        question: str,
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> bool:
        return await self._text_client.if_async(input_text, question, tape=tape, context=context)

    def classify(
        self,
        input_text: str,
        choices: list[str],
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> str:
        return self._text_client.classify(input_text, choices, tape=tape, context=context)

    async def classify_async(
        self,
        input_text: str,
        choices: list[str],
        *,
        tape: str | None = None,
        context: TapeContext | None = None,
    ) -> str:
        return await self._text_client.classify_async(input_text, choices, tape=tape, context=context)

    def embed(
        self,
        inputs: str | list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ):
        return self.embeddings.embed(inputs, model=model, provider=provider, **kwargs)

    async def embed_async(
        self,
        inputs: str | list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ):
        return await self.embeddings.embed_async(inputs, model=model, provider=provider, **kwargs)

    def stream(
        self,
        prompt: str | None = None,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> TextStream:
        return self._chat_client.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        **kwargs: Any,
    ) -> AsyncTextStream:
        return await self._chat_client.stream_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> StreamEvents:
        return self._chat_client.stream_events(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
            tools=tools,
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
        tape: str | None = None,
        context: TapeContext | None = None,
        tools: ToolInput = None,
        **kwargs: Any,
    ) -> AsyncStreamEvents:
        return await self._chat_client.stream_events_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            max_tokens=max_tokens,
            tape=tape,
            context=context,
            tools=tools,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"<LLM provider={self._core.provider} model={self._core.model} "
            f"fallback_models={self._core.fallback_models} max_retries={self._core.max_retries}>"
        )
