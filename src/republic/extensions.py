"""Optional synchronous observers for public Republic calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from republic._response import EmbeddingResponse, ImageResponse, ModelInfo, Response
from republic.decisions import DecisionResponse

__all__ = ["Extension", "Observer", "Request", "ResponseValue"]

ResponseValue = Response[Any] | EmbeddingResponse | ImageResponse | DecisionResponse | list[ModelInfo]


@dataclass(frozen=True)
class Request:
    """Read-only call metadata, without prompts or mutable request options.

    ``operation`` is ``chat``, ``stream``, ``embed`` (including batches),
    ``generate``, ``edit``, ``decide``, or ``list_models``.
    """

    provider: str
    operation: str
    model: str | None = None


class Observer:
    """A receiver for one call, returned by :meth:`Extension.on_start`.

    Methods are synchronous and called in extension order. Their return values
    are ignored and failures are logged without changing the call's outcome.
    Borrowed responses must not be mutated.
    """

    def on_retry(self, error: Exception, attempt: int, delay: float) -> None:
        """Observe a scheduled retry after response cleanup, before waiting.

        ``attempt`` starts at one for each HTTP retry loop, including each
        page of model listing. ``delay`` is the selected wait in seconds.
        """

    def on_finish(self, response: ResponseValue | None, error: BaseException | None) -> None:
        """Observe one terminal outcome, with an available response and original error.

        Both may be present if validation or history fails after a response
        was prepared. Both are absent for an unfinished stream. Successful
        streams notify before ``Completed``, independently of context exit.
        """


class Extension:
    """Configure an observer factory and supply it through ``extensions=``."""

    def on_start(self, request: Request) -> Observer | None:
        """Create a receiver when a call starts; streams start on context entry.

        Keep per-call state in the returned observer so this extension can be
        reused across concurrent calls. Return ``None`` to skip a call.
        """
