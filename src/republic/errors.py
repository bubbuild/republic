"""Errors at the single-call boundary. Cancellation remains CancelledError."""


class RepublicError(Exception):
    """Base class for SDK errors."""


class ProviderError(RepublicError):
    """An adapter's mapped request/transport failure, with its cause preserved."""

    def __init__(self, message: str, *, provider: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class IncompleteStreamError(RepublicError):
    """No terminal response is available; partial output is on Stream.message."""

    def __init__(self) -> None:
        super().__init__("Provider stream exhausted without StreamEnd")


class StreamProtocolError(RepublicError):
    """Invalid event order or block identity from an adapter."""

    def __init__(self, reason: str, blocks: list[tuple[str, str]]) -> None:
        self.reason = reason
        self.blocks = blocks
        super().__init__(f"Invalid provider event ({reason}): {blocks}")
