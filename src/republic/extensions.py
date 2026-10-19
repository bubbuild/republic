"""Optional callbacks for provider calls; HTTP hooks still belong to the HTTP client."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from republic._response import EmbeddingResponse, ImageResponse, ModelInfo, Response
from republic.decisions import DecisionResponse
from republic.errors import request_id

__all__ = ["Call", "Extension"]


@dataclass(eq=False)
class Call:
    """One provider call, shared by its callbacks and never reused for another call.

    Republic maintains the fields; extensions may keep their own per-call data
    in ``state``. No prompt, credentials or HTTP request body is included.
    """

    provider: str
    model: str | None
    operation: str
    api_format: str
    streaming: bool = False
    attempts: int = 0
    elapsed: float = 0
    """Seconds since the call started, updated before callbacks."""
    status: Literal["pending", "success", "error", "cancelled", "closed"] = "pending"
    response: Response[Any] | EmbeddingResponse | ImageResponse | DecisionResponse | list[ModelInfo] | None = None
    error: BaseException | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    """Headers from the most recent HTTP response, when available."""
    state: dict[Any, Any] = field(default_factory=dict, repr=False)
    _started: float = field(default_factory=time.monotonic, init=False, repr=False)

    @property
    def request_id(self) -> str | None:
        return request_id(self.headers)


@dataclass(frozen=True)
class Extension:
    """A bundle of optional callbacks, passed through ``extensions=``.

    Callbacks may be regular or async functions and run in extension order.
    Observer exceptions are logged and skipped; cancellation propagates.
    A ``retry_delay`` error propagates, since it controls execution.
    """

    on_start: Callable[[Call], Awaitable[None] | None] | None = None
    """Observe a call before request preparation."""
    on_finish: Callable[[Call], Awaitable[None] | None] | None = None
    """Observe its final outcome and public response, including parsing failures.

    A fully parsed chat response is available even if output validation fails.
    Streams finish when their context exits. An unconsumed or partially
    consumed stream has status ``closed`` unless it failed or was cancelled.
    """
    on_retry: Callable[[Call, Exception, float], Awaitable[None] | None] | None = None
    """Observe a scheduled retry after closing the failed response."""
    retry_delay: Callable[[Call, Exception, float | None], Awaitable[float | None] | float | None] | None = None
    """Adjust the proposed wait in seconds, or return None to stop retrying.

    The proposed wait is None for a nonretryable failure. Each callback sees
    the preceding callback's result. Retry count and maximum wait still apply.
    Only HTTP status and transport failures reach this callback; opened streams,
    payload errors and cancellation are never replayed. Errors and invalid
    return values stop the call, closing any failed HTTP response first.
    """
