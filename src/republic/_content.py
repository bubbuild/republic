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
from urllib.parse import urlsplit

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
class Audio(_Media):
    """Audio input, represented by MIME type and inline bytes or a remote URL."""

    kind: ClassVar[str] = "audio"


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


Part = Text | Reasoning | Image | Audio | Video | ProviderData

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
class Message:
    role: Role
    parts: tuple[Part, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call: ToolCall | None = None
    """The call a ``tool`` message answers; its parts hold the output."""
    is_error: bool = False
    """Whether a ``tool`` message reports a failed call."""

    def __post_init__(self) -> None:
        if (self.role == "tool") != (self.tool_call is not None):
            raise ValueError("tool_call is required for tool messages and only allowed on them")

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, Text))

    @property
    def reasoning(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, Reasoning))

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a JSON-compatible dict.

        A message holding a single text part stores it as a plain ``content`` string;
        any other parts are stored as a list of typed items.
        """
        data: dict[str, Any] = {"role": self.role}
        match self.parts:
            case ():
                pass
            case (Text(text),):
                data["content"] = text
            case parts:
                data["content"] = [_part_to_dict(part) for part in parts]
        if self.tool_calls:
            data["tool_calls"] = [_tool_call_to_dict(call) for call in self.tool_calls]
        if self.tool_call is not None:
            data["tool_call"] = _tool_call_to_dict(self.tool_call)
        if self.is_error:
            data["is_error"] = True
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        """Build a message from a dict produced by :meth:`to_dict`."""
        content = data.get("content")
        if content is None:
            parts: tuple[Part, ...] = ()
        elif isinstance(content, str):
            parts = (Text(content),)
        else:
            parts = tuple(_part_from_dict(item) for item in content)
        return cls(
            data["role"],
            parts,
            tool_calls=tuple(_tool_call_from_dict(call) for call in data.get("tool_calls", ())),
            tool_call=None if (call := data.get("tool_call")) is None else _tool_call_from_dict(call),
            is_error=data.get("is_error", False),
        )


def _part_to_dict(part: Part) -> dict[str, Any]:
    match part:
        case Text(text):
            return {"type": "text", "text": text}
        case Reasoning(text):
            return {"type": "reasoning", "text": text}
        case Image() | Audio() | Video():
            data: dict[str, Any] = {"type": part.kind, "media_type": part.media_type}
            if part.url is not None:
                data["url"] = part.url
            if part.data is not None:
                data["data"] = part.base64_data
            return data
        case ProviderData(api_format, payload):
            return {"type": "provider_data", "api_format": api_format, "payload": dict(payload)}


def _part_from_dict(data: Mapping[str, Any]) -> Part:
    match data["type"]:
        case "text":
            return Text(data["text"])
        case "reasoning":
            return Reasoning(data["text"])
        case "image" | "audio" | "video" as kind:
            media_class = {"image": Image, "audio": Audio, "video": Video}[kind]
            inline = data.get("data")
            return media_class(
                data["media_type"],
                data=None if inline is None else base64.b64decode(inline, validate=True),
                url=data.get("url"),
            )
        case "provider_data":
            return ProviderData(data["api_format"], data["payload"])
        case other:
            raise ValueError(f"Unknown message part type: {other!r}")


def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    data: dict[str, Any] = {"id": call.id, "name": call.name, "arguments": call.arguments}
    if call.metadata:
        data["metadata"] = dict(call.metadata)
    return data


def _tool_call_from_dict(data: Mapping[str, Any]) -> ToolCall:
    return ToolCall(data["id"], data["name"], data["arguments"], metadata=dict(data.get("metadata", {})))


UserContent = str | Image | Audio | Video


def _to_part(content: UserContent) -> Part:
    return Text(content) if isinstance(content, str) else content


def system(text: str) -> Message:
    """Build a system message."""
    return Message("system", (Text(text),))


def user(*content: UserContent) -> Message:
    """Build a user message from text, images, audio, and videos."""
    return Message("user", tuple(_to_part(item) for item in content))


def assistant(*content: UserContent, tool_calls: Iterable[ToolCall] = ()) -> Message:
    """Build an assistant turn, optionally carrying tool calls."""
    return Message("assistant", tuple(_to_part(item) for item in content), tool_calls=tuple(tool_calls))


def tool(call: ToolCall, *content: UserContent, is_error: bool = False) -> Message:
    """Build a tool message holding the output of executing ``call``.

    The output may mix text, images, audio, and videos. A call that does not appear
    earlier in the conversation is announced automatically before its result.
    """
    return Message("tool", tuple(_to_part(item) for item in content), tool_call=call, is_error=is_error)


def image(source: str | os.PathLike[str] | bytes, *, media_type: str | None = None) -> Image:
    """Load an image from a path, a URL, a data URL, or raw bytes."""
    return _load_media(Image, source, media_type)


def audio(source: str | os.PathLike[str] | bytes, *, media_type: str | None = None) -> Audio:
    """Load audio from a path, a URL, a data URL, or raw bytes."""
    return _load_media(Audio, source, media_type)


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
    path = urlsplit(name).path if name.startswith(_REMOTE_PREFIXES) else name
    guessed, _ = mimetypes.guess_type(path)
    if guessed is None:
        raise ValueError(f"Cannot guess the media type of {name!r}; pass media_type explicitly")
    return guessed


Input = str | Message | Iterable[str | Message]


def to_messages(value: Input) -> list[Message]:
    """Turn chat input into messages. Plain strings become user messages."""
    if isinstance(value, str | Message):
        value = [value]
    return [user(item) if isinstance(item, str) else item for item in value]
