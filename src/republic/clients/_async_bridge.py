"""Utilities for exposing async iterators through sync interfaces."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import suppress
from typing import Any, Generic, TypeVar

T = TypeVar("T")
_QUEUE_TIMEOUT_SECONDS = 0.1
_THREAD_JOIN_TIMEOUT_SECONDS = 1.0


class _StreamItem(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value


class _StreamError:
    def __init__(self, error: Exception) -> None:
        self.error = error


class _StreamDone:
    pass


StreamMessage = _StreamItem[T] | _StreamError | _StreamDone


def _pending_tasks() -> list[asyncio.Task[object]]:
    current_task = asyncio.current_task()
    return [task for task in asyncio.all_tasks() if task is not current_task and not task.done()]


async def _cleanup_stream(stream_source: object | None, *, stop_requested: threading.Event) -> None:
    if not stop_requested.is_set() or stream_source is None:
        return

    aclose = getattr(stream_source, "aclose", None)
    close = getattr(stream_source, "close", None)
    if callable(aclose):
        with suppress(Exception):
            await aclose()
        return
    if callable(close):
        with suppress(Exception):
            close()


async def _gather_pending_tasks() -> None:
    if pending_tasks := _pending_tasks():
        await asyncio.gather(*pending_tasks, return_exceptions=True)


async def _emit_async_items(
    stream_source: AsyncIterator[T],
    *,
    messages: queue.Queue[StreamMessage[T]],
    stop_requested: threading.Event,
) -> None:
    async for item in stream_source:
        if stop_requested.is_set():
            break
        messages.put(_StreamItem(item))


def _emit_sync_items(
    stream_source: Iterator[T],
    *,
    messages: queue.Queue[StreamMessage[T]],
    stop_requested: threading.Event,
) -> None:
    for item in stream_source:
        if stop_requested.is_set():
            break
        messages.put(_StreamItem(item))


def _yield_messages(
    *,
    messages: queue.Queue[StreamMessage[T]],
    worker: threading.Thread,
    stop_requested: threading.Event,
) -> Iterator[T]:
    try:
        while True:
            try:
                message = messages.get(timeout=_QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                if worker.is_alive():
                    continue
                break

            if isinstance(message, _StreamItem):
                yield message.value
                continue
            if isinstance(message, _StreamError):
                raise message.error
            break
    finally:
        stop_requested.set()
        worker.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)


def _start_worker(*, runner: Coroutine[Any, Any, None], thread_name: str) -> threading.Thread:
    worker = threading.Thread(target=lambda: asyncio.run(runner), name=thread_name, daemon=True)
    worker.start()
    return worker


def threaded_async_call_to_sync_iter(awaitable: Coroutine[Any, Any, AsyncIterator[T] | Iterator[T]]) -> Iterator[T]:
    """Run an async call and consume its resulting stream on the same event loop thread."""

    messages: queue.Queue[StreamMessage[T]] = queue.Queue()
    stop_requested = threading.Event()

    async def _drain_stream() -> None:
        stream_source: AsyncIterator[T] | Iterator[T] | None = None
        try:
            stream_source = await awaitable
            if isinstance(stream_source, AsyncIterator):
                await _emit_async_items(
                    stream_source,
                    messages=messages,
                    stop_requested=stop_requested,
                )
            else:
                _emit_sync_items(
                    stream_source,
                    messages=messages,
                    stop_requested=stop_requested,
                )
        except Exception as exc:
            messages.put(_StreamError(exc))
        finally:
            await _cleanup_stream(stream_source, stop_requested=stop_requested)
            await _gather_pending_tasks()
            messages.put(_StreamDone())

    worker = _start_worker(runner=_drain_stream(), thread_name="republic-async-call-stream")
    yield from _yield_messages(messages=messages, worker=worker, stop_requested=stop_requested)
