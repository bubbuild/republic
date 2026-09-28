"""Validate supplied media data without fetching URLs or opening files."""

import base64
from urllib.parse import urlsplit

from republic.errors import UnsupportedRequestError
from republic.types import FilePart


def base64_data(part: FilePart) -> str:
    data = part.data
    if part.encoding == "url" and data.startswith(f"data:{part.media_type};base64,"):
        data = data.split(",", 1)[1]
    elif part.encoding != "base64":
        raise UnsupportedRequestError("file.encoding", "expected explicit base64 data")
    try:
        decoded = base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise UnsupportedRequestError("file.data", "expected nonempty standard base64") from exc
    if not decoded:
        raise UnsupportedRequestError("file.data", "expected nonempty standard base64")
    return data


def file_url(part: FilePart) -> str:
    if part.encoding == "base64" or (part.encoding == "url" and part.data.startswith("data:")):
        return f"data:{part.media_type};base64,{base64_data(part)}"
    if part.encoding != "url":
        raise UnsupportedRequestError("file.encoding", "this wire field needs a URL or base64 data")
    url = urlsplit(part.data)
    if url.scheme not in {"http", "https"} or not url.netloc:
        raise UnsupportedRequestError("file.URL", "expected an HTTP(S) or matching base64 data URL")
    return part.data


def file_id(part: FilePart) -> str:
    if not part.data or part.encoding != "file_id":
        raise UnsupportedRequestError("file_id", "expected a nonempty provider file reference")
    return part.data


def detail_option(options: dict) -> None:
    if "detail" in options and options["detail"] not in ("auto", "low", "high", "original"):
        raise UnsupportedRequestError("file.detail", "expected auto, low, high or original")


def text_data(part: FilePart) -> str:
    """Upstream text-file conversion: inline UTF-8 or the supplied URL as text."""
    if part.encoding == "url" and not part.data.startswith("data:"):
        return file_url(part)
    try:
        return base64.b64decode(base64_data(part)).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedRequestError("file.data", "text input needs UTF-8") from exc
