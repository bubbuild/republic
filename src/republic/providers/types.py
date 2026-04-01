"""Provider hook contracts and request types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from republic.conversation import Conversation
from republic.core.errors import ErrorKind

TransportName = Literal["completion", "responses", "messages"]


def transport_set(*names: TransportName) -> frozenset[TransportName]:
    return frozenset(names)


COMPLETION_TRANSPORTS = transport_set("completion")
RESPONSES_TRANSPORTS = transport_set("responses")
MESSAGES_TRANSPORTS = transport_set("messages")
OPENAI_TRANSPORTS = transport_set("completion", "responses")
OPENROUTER_TRANSPORTS = transport_set("completion", "responses", "messages")
ANTHROPIC_COMPAT_TRANSPORTS = transport_set("completion", "messages")


@dataclass(frozen=True)
class ProviderContext:
    provider: str
    api_key: str | None
    api_base: str | None
    client_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    transports: frozenset[TransportName]
    preferred_transport: TransportName | None = None
    supports_embeddings: bool = False
    preserves_responses_extra_headers: bool = False
    default_completion_stream_usage: bool = False
    completion_max_tokens_arg: str = "max_tokens"


@dataclass(frozen=True)
class ChatRequest:
    transport: TransportName
    model: str
    conversation: Conversation
    stream: bool
    reasoning_effort: Any | None
    kwargs: dict[str, Any]
    tools: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class EmbedRequest:
    model: str
    inputs: str | list[str]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class ResponsesRequest:
    model: str
    input_data: Any
    kwargs: dict[str, Any]


class ProviderBackend:
    """Base backend for one resolved provider configuration."""

    capabilities = ProviderCapabilities(transports=COMPLETION_TRANSPORTS)

    def chat(self, request: ChatRequest):
        raise NotImplementedError

    async def achat(self, request: ChatRequest):
        raise NotImplementedError

    def validate_chat_request(
        self,
        request: ChatRequest,
    ) -> None:
        return None

    def embed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError

    async def aembed(self, request: EmbedRequest) -> Any:
        raise NotImplementedError

    def responses(self, request: ResponsesRequest) -> Any:
        raise NotImplementedError

    async def aresponses(self, request: ResponsesRequest) -> Any:
        raise NotImplementedError

    def list_models(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def alist_models(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def create_batch(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acreate_batch(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def list_batches(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def alist_batches(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class ProviderHook:
    """Base provider hook."""

    name = ""

    def matches(self, provider: str) -> bool:
        return provider == self.name

    def resolve_api_key(self, provider: str) -> str | None:
        return None

    def login(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def create_backend(self, context: ProviderContext) -> ProviderBackend:
        raise NotImplementedError

    def classify_error(self, exc: Exception) -> ErrorKind | None:
        return None
