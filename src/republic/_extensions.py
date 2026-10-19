"""Private call ownership and synchronous extension dispatch."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from republic.extensions import Extension, ResponseValue

if TYPE_CHECKING:
    from republic.providers import Provider

logger = logging.getLogger(__name__)
_current: ContextVar[tuple[Provider, _Observation] | None] = ContextVar("republic_observation", default=None)


def _invoke(extension: Extension, method: str, *args: Any) -> Any:
    try:
        result = getattr(extension, method)(*args)
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
    def __init__(self, extensions: Sequence[Extension], provider: str, operation: str, model: str | None) -> None:
        self.response: ResponseValue | None = None
        self._done = False
        self._contexts = [
            (extension, _invoke(extension, "on_start", provider, operation, model)) for extension in extensions
        ]

    def retry(self, error: Exception, attempt: int, delay: float) -> None:
        for extension, context in self._contexts:
            _invoke(extension, "on_retry", context, error, attempt, delay)

    def finish(self, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        for extension, context in self._contexts:
            _invoke(extension, "on_finish", context, self.response, error)


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
    observation = _Observation(provider.extensions, provider.name, operation, model)
    with _bind(provider, observation):
        try:
            yield observation
        except BaseException as error:
            observation.finish(error)
            raise
        else:
            observation.finish()
