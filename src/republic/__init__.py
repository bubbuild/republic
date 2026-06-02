"""Republic public API."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

from republic.auth import (
    github_copilot_oauth_resolver,
    load_openai_codex_oauth_tokens,
    login_github_copilot_oauth,
    login_openai_codex_oauth,
    multi_api_key_resolver,
    openai_codex_oauth_resolver,
)
from republic.core.results import (
    AsyncStreamEvents,
    AsyncTextStream,
    RepublicError,
    StreamEvent,
    StreamEvents,
    StreamState,
    TextStream,
    ToolAutoResult,
)
from republic.llm import LLM
from republic.tape import (
    TAPE_ANCHOR_KIND,
    TAPE_ANCHOR_NAME_KEY,
    TAPE_CLOSE_ANCHOR,
    TAPE_NOW,
    TAPE_START,
    AsyncTapeManager,
    AsyncTapeStore,
    AsyncTapeStream,
    Tape,
    TapeContext,
    TapeEntry,
    TapeManager,
    TapeQuery,
    TapeStream,
    TapeStreamInfo,
    TapeStreamQuery,
    TapeStreamView,
    entry_offset,
)
from republic.tools import Tool, ToolContext, ToolSet, schema_from_model, tool, tool_from_model

__all__ = [
    "LLM",
    "TAPE_ANCHOR_KIND",
    "TAPE_ANCHOR_NAME_KEY",
    "TAPE_CLOSE_ANCHOR",
    "TAPE_NOW",
    "TAPE_START",
    "AsyncStreamEvents",
    "AsyncTapeManager",
    "AsyncTapeStore",
    "AsyncTapeStream",
    "AsyncTextStream",
    "RepublicError",
    "StreamEvent",
    "StreamEvents",
    "StreamState",
    "Tape",
    "TapeContext",
    "TapeEntry",
    "TapeManager",
    "TapeQuery",
    "TapeStream",
    "TapeStreamInfo",
    "TapeStreamQuery",
    "TapeStreamView",
    "TextStream",
    "Tool",
    "ToolAutoResult",
    "ToolContext",
    "ToolSet",
    "entry_offset",
    "github_copilot_oauth_resolver",
    "load_openai_codex_oauth_tokens",
    "login_github_copilot_oauth",
    "login_openai_codex_oauth",
    "multi_api_key_resolver",
    "openai_codex_oauth_resolver",
    "schema_from_model",
    "tool",
    "tool_from_model",
]

try:
    __version__ = import_module("republic._version").version
except ModuleNotFoundError:
    try:
        __version__ = metadata_version("republic")
    except PackageNotFoundError:
        __version__ = "0.0.0"
