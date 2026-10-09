from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "APIConnectionError",
    "APIResponseError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "ProviderNotFoundError",
    "RepublicError",
    "StreamIncompleteError",
    "StreamNotFinishedError",
    "UnsupportedApiFormatError",
    "UnsupportedFeatureError",
]


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


def request_id(headers: Mapping[str, str]) -> str | None:
    """The HTTP request ID, distinct from a generated message or response ID."""
    return headers.get("x-request-id") or headers.get("request-id") or headers.get("x-goog-request-id")


class _APIError(RepublicError):
    def __init__(self, message: str, *, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self._set_headers(headers or {})

    def _set_headers(self, headers: Mapping[str, str]) -> None:
        self.headers = {key.lower(): value for key, value in headers.items()}
        self.request_id = request_id(self.headers)


class APIStatusError(_APIError):
    """Raised for an HTTP error, preserving status, body and response headers."""

    def __init__(self, status_code: int, body: str, *, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(f"Provider returned HTTP {status_code}: {body}", headers=headers)
        self.status_code = status_code
        self.body = body


class APIConnectionError(_APIError):
    """Raised for a transport failure; the original HTTP exception is its cause."""


class APITimeoutError(APIConnectionError):
    """Raised when a provider request or response read times out."""


class APIResponseError(_APIError):
    """Raised when the provider reports a failure inside a successful HTTP response."""


class StreamIncompleteError(APIResponseError):
    """Raised when a stream ends without its protocol's completion signal."""


class StreamNotFinishedError(RepublicError):
    """Raised when reading the final result of a stream before it completes."""
