"""Optional synchronous extensions for public Republic calls."""

from __future__ import annotations

from typing import Any

from republic._response import EmbeddingResponse, ImageResponse, ModelInfo, Response
from republic.decisions import DecisionResponse

__all__ = ["Extension", "ResponseValue"]

ResponseValue = Response[Any] | EmbeddingResponse | ImageResponse | DecisionResponse | list[ModelInfo]


class Extension:
    """Configure an extension and pass it through ``extensions=``.

    Methods are synchronous, run in extension order, and default to no-ops.
    Callback failures are logged and isolated. Retry/finish return values are ignored.
    Borrowed responses must not be mutated.
    """

    def on_start(self, provider: str, operation: str, model: str | None) -> Any:
        """Return this extension's context for one call; the default is ``None``.

        Streams start on context entry. Retry and finish receive the returned
        object unchanged, allowing extensions to keep their own per-call state.
        If this method fails, subsequent methods receive ``None``.
        """

    def on_retry(self, context: Any, error: Exception, attempt: int, delay: float) -> None:
        """Observe a retry after response cleanup, before the selected wait.

        ``attempt`` starts at one for each HTTP retry loop, including each
        page of model listing. ``delay`` is the wait in seconds.
        """

    def on_finish(self, context: Any, response: ResponseValue | None, error: BaseException | None) -> None:
        """Observe exactly one terminal outcome, with the original error.

        Both response and error may be present after validation/history failure.
        Both are absent for an unfinished stream. Successful streams notify
        before ``Completed``, independently of context exit.
        """
