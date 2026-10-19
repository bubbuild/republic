"""Optional synchronous extensions for public Republic calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from republic._response import EmbeddingResponse, ImageResponse, ModelInfo, Response
from republic.decisions import DecisionResponse

__all__ = ["Extension", "Request", "ResponseValue"]

ResponseValue = Response[Any] | EmbeddingResponse | ImageResponse | DecisionResponse | list[ModelInfo]


@dataclass(frozen=True, eq=False)
class Request:
    """Read-only call metadata and identity, shared by all three callbacks.

    Distinct calls have distinct, hashable request objects, even when their
    metadata is identical. Use them as keys when keeping per-call state.
    """

    provider: str
    operation: str
    model: str | None = None


class Extension:
    """Configure an extension and pass it through ``extensions=``.

    Methods are synchronous, run in extension order, and default to no-ops.
    Return values are ignored; callback failures are logged and isolated.
    Borrowed responses must not be mutated.
    """

    def on_start(self, request: Request) -> None:
        """Observe a call starting; streams start on context entry."""

    def on_retry(self, request: Request, error: Exception, attempt: int, delay: float) -> None:
        """Observe a retry after response cleanup, before the selected wait.

        ``attempt`` starts at one for each HTTP retry loop, including each
        page of model listing. ``delay`` is the wait in seconds.
        """

    def on_finish(self, request: Request, response: ResponseValue | None, error: BaseException | None) -> None:
        """Observe exactly one terminal outcome, with the original error.

        Both response and error may be present after validation/history failure.
        Both are absent for an unfinished stream. Successful streams notify
        before ``Completed``, independently of context exit.
        """
