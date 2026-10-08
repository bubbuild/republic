from __future__ import annotations


class RepublicError(Exception):
    """Base class for errors raised by Republic."""


class AuthenticationError(RepublicError):
    """Raised when local credentials cannot be loaded or refreshed."""


class ProviderNotFoundError(RepublicError, LookupError):
    """Raised when no provider is registered under the requested name."""


class UnsupportedApiFormatError(RepublicError, ValueError):
    """Raised when a provider does not support the requested API format."""


class UnsupportedFeatureError(RepublicError):
    """Raised when the selected API format cannot express a requested feature."""


class APIStatusError(RepublicError):
    """Raised when the provider answers with an HTTP error status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Provider returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class APIResponseError(RepublicError):
    """Raised when the provider reports a failure inside a successful HTTP response."""


class StreamNotFinishedError(RepublicError):
    """Raised when reading the final result of a stream before it completes."""
