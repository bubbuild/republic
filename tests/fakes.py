from __future__ import annotations

import asyncio
import queue
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

from republic.core.errors import ErrorKind
from republic.core.execution import TransportResponse
from republic.core.results import RepublicError
from republic.providers.codecs import conversation_to_completion_messages, conversation_to_openai_responses_input
from republic.providers.types import (
    ANTHROPIC_COMPAT_TRANSPORTS,
    COMPLETION_TRANSPORTS,
    OPENAI_TRANSPORTS,
    OPENROUTER_TRANSPORTS,
    ProviderBackend,
    ProviderCapabilities,
    ProviderContext,
    ProviderHook,
)


class FakeQueueEmptyError(AssertionError):
    def __init__(self, provider: str, kind: str) -> None:
        super().__init__(f"No queued {kind} value for provider={provider}")


class FakeProviderBackend(ProviderBackend):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.capabilities = _capabilities_for_provider(provider)
        self.calls: list[dict[str, Any]] = []
        self.completion_queue: deque[Any] = deque()
        self.acompletion_queue: deque[Any] = deque()
        self.responses_queue: deque[Any] = deque()
        self.aresponses_queue: deque[Any] = deque()
        self.embedding_queue: deque[Any] = deque()
        self.aembedding_queue: deque[Any] = deque()

    def queue_completion(self, *items: Any) -> None:
        self.completion_queue.extend(items)

    def queue_acompletion(self, *items: Any) -> None:
        self.acompletion_queue.extend(items)

    def queue_responses(self, *items: Any) -> None:
        self.responses_queue.extend(items)

    def queue_aresponses(self, *items: Any) -> None:
        self.aresponses_queue.extend(items)

    def queue_embedding(self, *items: Any) -> None:
        self.embedding_queue.extend(items)

    def queue_aembedding(self, *items: Any) -> None:
        self.aembedding_queue.extend(items)

    def _next(self, items: deque[Any], kind: str) -> Any:
        if not items:
            raise FakeQueueEmptyError(self.provider, kind)
        item = items.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def chat(self, request) -> TransportResponse:
        if request.transport == "responses":
            instructions, input_items = conversation_to_openai_responses_input(request.conversation)
            call = {
                "responses": True,
                "model": request.model,
                "input_data": input_items,
                "instructions": instructions,
                "tools": request.tools,
                "stream": request.stream,
                **dict(request.kwargs),
            }
            self.calls.append(call)
            if request.stream and not self.responses_queue and self.aresponses_queue:
                response = self._next(self.aresponses_queue, "aresponses")
                if hasattr(response, "__aiter__"):
                    response = self._async_iter_to_sync_iter(response)
            else:
                response = self._next(self.responses_queue, "responses")
            return TransportResponse(transport="responses", payload=response)

        messages = conversation_to_completion_messages(request.conversation)
        call = {
            "model": request.model,
            "messages": messages,
            "tools": request.tools,
            "stream": request.stream,
            "reasoning_effort": request.reasoning_effort,
            **dict(request.kwargs),
        }
        self.calls.append(call)
        response = self._next(self.completion_queue, "completion")
        return TransportResponse(transport=request.transport, payload=response)

    async def achat(self, request) -> TransportResponse:
        if request.transport == "responses":
            instructions, input_items = conversation_to_openai_responses_input(request.conversation)
            call = {
                "responses": True,
                "model": request.model,
                "input_data": input_items,
                "instructions": instructions,
                "tools": request.tools,
                "stream": request.stream,
                **dict(request.kwargs),
            }
            self.calls.append(call)
            queue_items = self.aresponses_queue if self.aresponses_queue else self.responses_queue
            response = self._next(queue_items, "aresponses")
            return TransportResponse(transport="responses", payload=response)

        messages = conversation_to_completion_messages(request.conversation)
        call = {
            "model": request.model,
            "messages": messages,
            "tools": request.tools,
            "stream": request.stream,
            "reasoning_effort": request.reasoning_effort,
            **dict(request.kwargs),
        }
        self.calls.append(call)
        queue_items = self.acompletion_queue if self.acompletion_queue else self.completion_queue
        response = self._next(queue_items, "acompletion")
        return TransportResponse(transport=request.transport, payload=response)

    def validate_chat_request(self, request) -> None:
        normalized_provider = self.provider.strip().lower()
        normalized_model = request.model.strip().lower()
        if (
            request.transport == "responses"
            and normalized_provider == "openrouter"
            and normalized_model.startswith("anthropic/")
            and request.tools
        ):
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                f"{self.provider}:{request.model}: responses format is not supported for this model when tools are enabled",
            )

    @staticmethod
    def _async_iter_to_sync_iter(async_iter: AsyncIterator[Any]) -> Iterator[Any]:
        item_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _runner() -> None:
            async def _consume() -> None:
                try:
                    async for item in async_iter:
                        item_queue.put(("item", item))
                except Exception as exc:
                    item_queue.put(("error", exc))
                finally:
                    item_queue.put(("done", None))

            asyncio.run(_consume())

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        try:
            while True:
                kind, value = item_queue.get()
                if kind == "item":
                    yield value
                    continue
                if kind == "error":
                    raise value
                break
        finally:
            thread.join()

    def embed(self, request) -> Any:
        self.calls.append({"embedding": True, "model": request.model, "inputs": request.inputs, **dict(request.kwargs)})
        return self._next(self.embedding_queue, "embedding")

    async def aembed(self, request) -> Any:
        self.calls.append({
            "aembedding": True,
            "model": request.model,
            "inputs": request.inputs,
            **dict(request.kwargs),
        })
        queue_items = self.aembedding_queue if self.aembedding_queue else self.embedding_queue
        return self._next(queue_items, "aembedding")

    def responses(self, request) -> Any:
        self.calls.append({
            "responses": True,
            "model": request.model,
            "input_data": request.input_data,
            **dict(request.kwargs),
        })
        return self._next(self.responses_queue, "responses")

    async def aresponses(self, request) -> Any:
        self.calls.append({
            "responses": True,
            "model": request.model,
            "input_data": request.input_data,
            **dict(request.kwargs),
        })
        queue_items = self.aresponses_queue if self.aresponses_queue else self.responses_queue
        return self._next(queue_items, "aresponses")


class FakeProviderFactory:
    def __init__(self) -> None:
        self.clients: dict[str, FakeProviderBackend] = {}

    def ensure(self, provider: str) -> FakeProviderBackend:
        if provider not in self.clients:
            self.clients[provider] = FakeProviderBackend(provider)
        return self.clients[provider]

    def create(self, provider: str, **_: Any) -> FakeProviderBackend:
        return self.ensure(provider)


def _capabilities_for_provider(provider: str) -> ProviderCapabilities:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return ProviderCapabilities(
            transports=OPENAI_TRANSPORTS,
            supports_embeddings=True,
            preserves_responses_extra_headers=True,
            default_completion_stream_usage=True,
            completion_max_tokens_arg="max_completion_tokens",
        )
    if normalized == "anthropic":
        return ProviderCapabilities(
            transports=ANTHROPIC_COMPAT_TRANSPORTS,
            completion_max_tokens_arg="max_tokens",
        )
    if normalized == "openrouter":
        return ProviderCapabilities(
            transports=OPENROUTER_TRANSPORTS,
            completion_max_tokens_arg="max_tokens",
        )
    if normalized == "github-copilot":
        return ProviderCapabilities(
            transports=COMPLETION_TRANSPORTS,
            default_completion_stream_usage=True,
            completion_max_tokens_arg="max_tokens",
        )
    return ProviderCapabilities(
        transports=COMPLETION_TRANSPORTS,
        completion_max_tokens_arg="max_tokens",
    )


class FakeProviderHook(ProviderHook):
    name = "fake"

    def __init__(
        self, factory: FakeProviderFactory, created: list[tuple[str, dict[str, object]]] | None = None
    ) -> None:
        self._factory = factory
        self._created = created

    def matches(self, provider: str) -> bool:
        return True

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        if self._created is not None:
            self._created.append((
                context.provider,
                {
                    "api_key": context.api_key,
                    "api_base": context.api_base,
                    **dict(context.client_args),
                },
            ))
        return self._factory.ensure(context.provider)


class FakeProviderRuntime:
    def __init__(self, hook: FakeProviderHook) -> None:
        self._hook = hook

    def provider_for(self, provider_name: str) -> FakeProviderHook:
        return self._hook


def install_fake_provider_runtime(
    monkeypatch, created: list[tuple[str, dict[str, object]]] | None = None
) -> FakeProviderFactory:
    import republic.core.execution as execution

    factory = FakeProviderFactory()
    runtime = FakeProviderRuntime(FakeProviderHook(factory, created))
    monkeypatch.setattr(execution, "default_provider_runtime", lambda: runtime)
    return factory


def make_tool_call(
    name: str,
    arguments: dict[str, Any] | str,
    *,
    call_id: str | None = "call_1",
    call_type: str | None = "function",
) -> Any:
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response(
    *,
    text: str = "",
    tool_calls: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> Any:
    message = SimpleNamespace(content=text, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_chunk(
    *,
    text: str = "",
    tool_calls: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> Any:
    delta = SimpleNamespace(content=text, tool_calls=tool_calls or [])
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_responses_output_text(text: str) -> Any:
    return SimpleNamespace(type="output_text", text=text)


def make_responses_message(text: str) -> Any:
    return SimpleNamespace(type="message", content=[make_responses_output_text(text)])


def make_responses_function_call(name: str, arguments: str, call_id: str = "call_1") -> Any:
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def make_responses_response(
    *,
    text: str = "",
    tool_calls: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> Any:
    output: list[Any] = []
    if text:
        output.append({
            "id": "msg_1",
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
    if tool_calls:
        output.extend(
            {
                "type": "function_call",
                "call_id": getattr(call, "call_id", None) or getattr(call, "id", None) or "call_1",
                "id": getattr(call, "id", None),
                "name": getattr(call, "name", None),
                "arguments": getattr(call, "arguments", "") or "",
                "status": "completed",
            }
            for call in tool_calls
        )
    usage_payload = dict(usage or {})
    input_tokens = usage_payload.get("input_tokens", 0)
    output_tokens = usage_payload.get("output_tokens", 0)
    total_tokens = usage_payload.get("total_tokens", input_tokens + output_tokens)
    return {
        "id": "resp_1",
        "created_at": 1,
        "model": "gpt-5-codex",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": total_tokens,
        },
    }


def make_responses_reasoning_response(
    *,
    usage: dict[str, Any] | None = None,
    status: str = "completed",
    incomplete_details: Any = None,
) -> Any:
    return SimpleNamespace(
        id="resp_reasoning_1",
        object="response",
        status=status,
        incomplete_details=incomplete_details,
        model="gpt-5.4-pro",
        output_text=None,
        output=[SimpleNamespace(type="reasoning", id="rs_1", summary=[], content=None, status=None)],
        usage=usage or {"input_tokens": 1, "output_tokens": 128, "total_tokens": 129},
    )


def make_responses_text_delta(delta: str) -> Any:
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def make_responses_function_delta(delta: str, *, item_id: str = "call_1") -> Any:
    return SimpleNamespace(type="response.function_call_arguments.delta", delta=delta, item_id=item_id, output_index=0)


def make_responses_function_done(name: str, arguments: str, *, item_id: str = "call_1") -> Any:
    return SimpleNamespace(
        type="response.function_call_arguments.done",
        name=name,
        arguments=arguments,
        item_id=item_id,
        output_index=0,
    )


def make_responses_completed(usage: dict[str, Any] | None = None) -> Any:
    response = SimpleNamespace(usage=usage)
    return SimpleNamespace(type="response.completed", response=response)


def make_responses_output_item_added(
    *,
    item_id: str = "fc_1",
    call_id: str = "call_1",
    name: str = "echo",
    arguments: str = "",
) -> Any:
    item = SimpleNamespace(type="function_call", id=item_id, call_id=call_id, name=name, arguments=arguments)
    return SimpleNamespace(type="response.output_item.added", item=item)


def make_responses_reasoning_item_added(*, item_id: str = "rs_1") -> Any:
    item = SimpleNamespace(type="reasoning", id=item_id, summary=[], content=None, status=None)
    return SimpleNamespace(type="response.output_item.added", item=item)


def make_responses_output_item_done(
    *,
    item_id: str = "fc_1",
    call_id: str = "call_1",
    name: str = "echo",
    arguments: str = '{"text":"tokyo"}',
) -> Any:
    item = SimpleNamespace(type="function_call", id=item_id, call_id=call_id, name=name, arguments=arguments)
    return SimpleNamespace(type="response.output_item.done", item=item)


def make_responses_reasoning_item_done(*, item_id: str = "rs_1") -> Any:
    item = SimpleNamespace(type="reasoning", id=item_id, summary=[], content=None, status=None)
    return SimpleNamespace(type="response.output_item.done", item=item)
