from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

Role = Literal["system", "user", "assistant", "tool"]

_REMOTE_PREFIXES = ("http://", "https://", "gs://")


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Reasoning:
    """Readable reasoning returned by the model.

    It is only sent back to chat servers that require it, such as DeepSeek during tool use.
    """

    text: str


@dataclass(frozen=True)
class _Media:
    """Inline bytes or a remote URL with a media type."""

    kind: ClassVar[str]

    media_type: str
    data: bytes | None = None
    url: str | None = None

    @property
    def base64_data(self) -> str:
        if self.data is None:
            raise ValueError(f"This {self.kind} has no inline data")
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        """The URL form accepted by most APIs: the remote URL or an inline data URL."""
        if self.url is not None:
            return self.url
        return f"data:{self.media_type};base64,{self.base64_data}"


@dataclass(frozen=True)
class Image(_Media):
    kind: ClassVar[str] = "image"

    def to_pil(self) -> PILImage:
        """Decode inline image data with Pillow (install ``republic[image]``)."""
        from io import BytesIO

        from PIL import Image as PILImageModule

        if self.data is None:
            raise ValueError(f"Image at {self.url} has no inline data to decode")
        return PILImageModule.open(BytesIO(self.data))


@dataclass(frozen=True)
class Video(_Media):
    kind: ClassVar[str] = "video"


@dataclass(frozen=True)
class ProviderData:
    """An opaque item that must be sent back verbatim to the API format that produced it.

    Reasoning items and thinking blocks are kept this way so multi-turn tool use
    stays valid. Other API formats skip it.
    """

    api_format: str
    payload: Mapping[str, Any]


Part = Text | Reasoning | Image | Video | ProviderData

_MediaT = TypeVar("_MediaT", bound=_Media)


@dataclass(frozen=True)
class Tool:
    """A tool schema offered to the model. Republic never executes tools."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False
    """Ask the provider to guarantee arguments match ``parameters``; the schema must meet its strict-mode rules."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    """The raw JSON arguments produced by the model."""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    """Provider details needed to send the call back, such as Gemini thought signatures."""

    @property
    def args(self) -> Any:
        """The decoded JSON arguments."""
        return json.loads(self.arguments or "{}")


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    role: Role
    parts: tuple[Part, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, Text))

    @property
    def reasoning(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, Reasoning))


UserContent = str | Image | Video


def _to_part(content: UserContent) -> Part:
    return Text(content) if isinstance(content, str) else content


def system(text: str) -> Message:
    """Build a system message."""
    return Message("system", (Text(text),))


def user(*content: UserContent) -> Message:
    """Build a user message from text, images, and videos."""
    return Message("user", tuple(_to_part(item) for item in content))


def assistant(
    *content: UserContent,
    tool_calls: Iterable[ToolCall] = (),
    tool_results: Iterable[ToolResult] = (),
) -> Message:
    """Build an assistant turn, optionally carrying tool calls and their results.

    A tool result already holds its call, so ``assistant(tool_results=[...])`` is
    enough to continue after executing the calls from a response. Calls that
    already appear earlier in the conversation are not sent twice.
    """
    return Message(
        "assistant",
        tuple(_to_part(item) for item in content),
        tool_calls=tuple(tool_calls),
        tool_results=tuple(tool_results),
    )


def tool_result(call: ToolCall, output: str, *, is_error: bool = False) -> ToolResult:
    """Pair a tool call with the output produced by executing it."""
    return ToolResult(call, output, is_error=is_error)


def image(source: str | os.PathLike[str] | bytes, *, media_type: str | None = None) -> Image:
    """Load an image from a path, a URL, a data URL, or raw bytes."""
    return _load_media(Image, source, media_type)


def video(source: str | os.PathLike[str] | bytes, *, media_type: str | None = None) -> Video:
    """Load a video from a path, a URL, a data URL, or raw bytes."""
    return _load_media(Video, source, media_type)


def _load_media(media_class: type[_MediaT], source: str | os.PathLike[str] | bytes, media_type: str | None) -> _MediaT:
    if isinstance(source, bytes):
        if media_type is None:
            raise ValueError(f"media_type is required for raw {media_class.kind} bytes")
        return media_class(media_type, data=source)
    if isinstance(source, str) and source.startswith("data:"):
        return media_from_data_url(media_class, source)
    if isinstance(source, str) and source.startswith(_REMOTE_PREFIXES):
        return media_class(media_type or _guess_media_type(media_class, source), url=source)
    path = Path(source)
    return media_class(media_type or _guess_media_type(media_class, path.name), data=path.read_bytes())


def media_from_data_url(media_class: type[_MediaT], url: str) -> _MediaT:
    header, _, payload = url.partition(",")
    media_type = header.removeprefix("data:").removesuffix(";base64")
    try:
        data = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 {media_class.kind} data URL") from exc
    return media_class(media_type, data=data)


def _guess_media_type(media_class: type[_Media], name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    if guessed is None:
        raise ValueError(f"Cannot guess the media type of {name!r}; pass media_type explicitly")
    return guessed


Input = str | Message | Iterable[str | Message]


def to_messages(value: Input) -> list[Message]:
    """Turn chat input into messages. Plain strings become user messages."""
    if isinstance(value, str | Message):
        value = [value]
    return [user(item) if isinstance(item, str) else item for item in value]
