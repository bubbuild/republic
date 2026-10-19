"""Optional provider callbacks, grouped into one extension object."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from republic._response import EmbeddingResponse, ImageResponse, ModelInfo, Response
from republic.decisions import DecisionResponse

__all__ = ["Extension"]

_Response = Response[Any] | EmbeddingResponse | ImageResponse | DecisionResponse | list[ModelInfo]


@dataclass(frozen=True)
class Extension:
    """Optional synchronous callbacks, called in extension order.

    Pass instances through ``extensions=``. No registration or subclass is
    needed. Observer failures are logged; retry-policy failures propagate.
    """

    on_response: Callable[[_Response], None] | None = None
    """Observe a completed public response, after parsing, validation and history.

    For streams this runs at completion, before the ``Completed`` event;
    closing an unfinished stream produces no response notification.
    """
    on_retry: Callable[[Exception, int, float], None] | None = None
    """Observe a retry's error, failed attempt number (starting at 1), and wait.

    Called after releasing the failed HTTP response and before waiting.
    """
    retry_delay: Callable[[Exception, int, float], float] | None = None
    """Adjust an already planned retry's wait in seconds.

    Receives the error, failed attempt number, and proposed wait. The result
    must be finite and nonnegative, and is capped by ``max_retry_delay``.
    This does not change which failures are retried or ``max_retries``.
    """
