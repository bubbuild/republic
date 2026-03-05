from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any


class FakeQueueEmptyError(AssertionError):
    def __init__(self, provider: str, kind: str) -> None:
        super().__init__(f"No queued {kind} value for provider={provider}")


class FakeAnyLLMClient:
    SUPPORTS_RESPONSES = True

    def __init__(self, provider: str) -> None:
        self.provider = provider
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

    def _next(self, queue: deque[Any], kind: str) -> Any:
        if not queue:
            raise FakeQueueEmptyError(self.provider, kind)
        item = queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._next(self.completion_queue, "completion")

    async def acompletion(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        queue = self.acompletion_queue if self.acompletion_queue else self.completion_queue
        return self._next(queue, "acompletion")

    def responses(self, **kwargs: Any) -> Any:
        self.calls.append({"responses": True, **dict(kwargs)})
        return self._next(self.responses_queue, "responses")

    async def aresponses(self, **kwargs: Any) -> Any:
        self.calls.append({"responses": True, **dict(kwargs)})
        queue = self.aresponses_queue if self.aresponses_queue else self.responses_queue
        return self._next(queue, "aresponses")

    def _embedding(self, **kwargs: Any) -> Any:
        self.calls.append({"embedding": True, **dict(kwargs)})
        return self._next(self.embedding_queue, "embedding")

    async def aembedding(self, **kwargs: Any) -> Any:
        self.calls.append({"aembedding": True, **dict(kwargs)})
        queue = self.aembedding_queue if self.aembedding_queue else self.embedding_queue
        return self._next(queue, "aembedding")


class FakeAnyLLMFactory:
    def __init__(self) -> None:
        self.clients: dict[str, FakeAnyLLMClient] = {}

    def ensure(self, provider: str) -> FakeAnyLLMClient:
        if provider not in self.clients:
            self.clients[provider] = FakeAnyLLMClient(provider)
        return self.clients[provider]

    def create(self, provider: str, **_: Any) -> FakeAnyLLMClient:
        return self.ensure(provider)


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
        output.append(make_responses_message(text))
    if tool_calls:
        output.extend(tool_calls)
    return SimpleNamespace(output=output, usage=usage)


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


def make_responses_output_item_done(
    *,
    item_id: str = "fc_1",
    call_id: str = "call_1",
    name: str = "echo",
    arguments: str = '{"text":"tokyo"}',
) -> Any:
    item = SimpleNamespace(type="function_call", id=item_id, call_id=call_id, name=name, arguments=arguments)
    return SimpleNamespace(type="response.output_item.done", item=item)
