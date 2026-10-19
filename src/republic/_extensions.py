"""Private call ownership and dispatch for synchronous extension observers."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from republic.extensions import Extension, Observer, Request, ResponseValue

if TYPE_CHECKING:
    from republic.providers import Provider

logger = logging.getLogger(__name__)
_current: ContextVar[tuple[Provider, _Observation] | None] = ContextVar("republic_observation", default=None)


def _invoke(receiver: object, method: str, *args: Any) -> Any:
    try:
        callback: Callable[..., Any] = getattr(receiver, method)
        result = callback(*args)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            logger.warning("Extension callback %s must be synchronous", method)
            return None
    except (Exception, asyncio.CancelledError):
        logger.warning("Extension callback %s failed", method, exc_info=True)
        return None
    return result


class _Observation:
    def __init__(self, extensions: Sequence[Extension], request: Request) -> None:
        self.response: ResponseValue | None = None
        self._done = False
        self._observers: list[Observer] = []
        for extension in extensions:
            observer = _invoke(extension, "on_start", request)
            if observer is not None:
                self._observers.append(observer)

    def retry(self, error: Exception, attempt: int, delay: float) -> None:
        for observer in self._observers:
            _invoke(observer, "on_retry", error, attempt, delay)

    def finish(self, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        for observer in self._observers:
            _invoke(observer, "on_finish", self.response, error)


def _active(provider: Provider) -> _Observation | None:
    current = _current.get()
    return current[1] if current is not None and current[0] is provider else None


@contextmanager
def _bind(provider: Provider, observation: _Observation) -> Iterator[None]:
    token = _current.set((provider, observation))
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def _observe(provider: Provider, operation: str, model: str | None = None) -> Iterator[_Observation]:
    observation = _Observation(provider.extensions, Request(provider.name, operation, model))
    with _bind(provider, observation):
        try:
            yield observation
        except BaseException as error:
            observation.finish(error)
            raise
        else:
            observation.finish()
