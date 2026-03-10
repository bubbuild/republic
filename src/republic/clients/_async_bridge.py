"""Utilities for exposing async iterators through sync interfaces."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator
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


def threaded_async_iter_to_sync_iter(async_iter: AsyncIterator[T]) -> Iterator[T]:
    """Bridge an async iterator into a sync iterator on a dedicated event loop thread."""

    messages: queue.Queue[StreamMessage[T]] = queue.Queue()
    stop_requested = threading.Event()

    async def _drain_async_iterator() -> None:
        try:
            async for item in async_iter:
                if stop_requested.is_set():
                    break
                messages.put(_StreamItem(item))
        except Exception as exc:
            messages.put(_StreamError(exc))
        finally:
            aclose = getattr(async_iter, "aclose", None)
            if stop_requested.is_set() and callable(aclose):
                try:
                    await aclose()
                except Exception:
                    pass
            pending_tasks = _pending_tasks()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            messages.put(_StreamDone())

    def _run_worker() -> None:
        asyncio.run(_drain_async_iterator())

    worker = threading.Thread(target=_run_worker, name="republic-async-stream", daemon=True)
    worker.start()

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


def threaded_async_call_to_sync_iter(awaitable: Coroutine[Any, Any, AsyncIterator[T] | Iterator[T]]) -> Iterator[T]:
    """Run an async call and consume its resulting stream on the same event loop thread."""

    messages: queue.Queue[StreamMessage[T]] = queue.Queue()
    stop_requested = threading.Event()

    async def _drain_stream() -> None:
        stream_source: AsyncIterator[T] | Iterator[T] | None = None
        try:
            stream_source = await awaitable
            if isinstance(stream_source, AsyncIterator):
                async for item in stream_source:
                    if stop_requested.is_set():
                        break
                    messages.put(_StreamItem(item))
            else:
                for item in stream_source:
                    if stop_requested.is_set():
                        break
                    messages.put(_StreamItem(item))
        except Exception as exc:
            messages.put(_StreamError(exc))
        finally:
            if stop_requested.is_set() and stream_source is not None:
                aclose = getattr(stream_source, "aclose", None)
                close = getattr(stream_source, "close", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass
                elif callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            pending_tasks = _pending_tasks()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            messages.put(_StreamDone())

    def _run_worker() -> None:
        asyncio.run(_drain_stream())

    worker = threading.Thread(target=_run_worker, name="republic-async-call-stream", daemon=True)
    worker.start()

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
