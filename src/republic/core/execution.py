"""Core execution utilities for Republic."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, NoReturn

from republic.conversation import conversation_from_messages
from republic.core.errors import ErrorKind, RepublicError
from republic.core.request_adapters import normalize_responses_kwargs
from republic.providers.runtime import default_provider_runtime
from republic.providers.types import MESSAGES_TRANSPORTS, ChatRequest, ProviderBackend, ProviderContext

logger = logging.getLogger(__name__)


class AttemptDecision(Enum):
    """What to do after one failed attempt."""

    RETRY_SAME_MODEL = auto()
    TRY_NEXT_MODEL = auto()


@dataclass(frozen=True)
class AttemptOutcome:
    """Result of classifying and deciding how to handle one exception."""

    error: RepublicError
    decision: AttemptDecision


@dataclass(frozen=True)
class TransportResponse:
    transport: Literal["completion", "responses", "messages"]
    payload: Any


@dataclass(frozen=True)
class TransportCallRequest:
    backend: ProviderBackend
    provider_name: str
    model_id: str
    messages_payload: list[dict[str, Any]]
    tools_payload: list[dict[str, Any]] | None
    max_tokens: int | None
    stream: bool
    reasoning_effort: Any | None
    kwargs: dict[str, Any]


class LLMCore:
    """Shared LLM execution utilities (provider resolution, retries, backend cache)."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        fallback_models: list[str],
        max_retries: int,
        api_key: str | dict[str, str] | None,
        api_key_resolver: Callable[[str], str | None] | None,
        api_base: str | dict[str, str] | None,
        client_args: dict[str, Any],
        api_format: Literal["completion", "responses", "messages"],
        verbose: int,
        error_classifier: Callable[[Exception], ErrorKind | None] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._fallback_models = fallback_models
        self._max_retries = max_retries
        self._api_key = api_key
        self._api_key_resolver = api_key_resolver
        self._api_base = api_base
        self._client_args = client_args
        self._api_format = api_format
        self._verbose = verbose
        self._error_classifier = error_classifier
        self._client_cache: dict[str, ProviderBackend] = {}
        self._provider_runtime = default_provider_runtime()

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def fallback_models(self) -> list[str]:
        return self._fallback_models

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def max_attempts(self) -> int:
        return max(1, 1 + self._max_retries)

    @staticmethod
    def resolve_model_provider(model: str, provider: str | None) -> tuple[str, str]:
        if provider:
            if ":" in model:
                raise RepublicError(
                    ErrorKind.INVALID_INPUT,
                    "When provider is specified, model must not include a provider prefix.",
                )
            return provider, model

        if ":" not in model:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Model must be in 'provider:model' format.")

        provider_name, model_id = model.split(":", 1)
        if not provider_name or not model_id:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Model must be in 'provider:model' format.")
        return provider_name, model_id

    def resolve_fallback(self, model: str) -> tuple[str, str]:
        if ":" in model:
            provider_name, model_id = model.split(":", 1)
            if not provider_name or not model_id:
                raise RepublicError(ErrorKind.INVALID_INPUT, "Fallback models must be in 'provider:model' format.")
            return provider_name, model_id
        if self._provider:
            return self._provider, model
        raise RepublicError(
            ErrorKind.INVALID_INPUT,
            "Fallback models must include provider or LLM must be initialized with a provider.",
        )

    def model_candidates(self, override_model: str | None, override_provider: str | None) -> list[tuple[str, str]]:
        if override_model:
            provider, model = self.resolve_model_provider(override_model, override_provider)
            return [(provider, model)]

        candidates = [(self._provider, self._model)]
        for model in self._fallback_models:
            candidates.append(self.resolve_fallback(model))
        return candidates

    def iter_clients(self, override_model: str | None, override_provider: str | None):
        for provider_name, model_id in self.model_candidates(override_model, override_provider):
            yield provider_name, model_id, self.get_backend(provider_name)

    def _resolve_api_key(self, provider: str) -> str | None:
        if isinstance(self._api_key, dict):
            if (key := self._api_key.get(provider)) is not None:
                return key
            if self._api_key_resolver is not None:
                return self._api_key_resolver(provider)
            if hook := self._provider_runtime.provider_for(provider):
                return hook.resolve_api_key(provider)
            return None
        if self._api_key is not None:
            return self._api_key
        if self._api_key_resolver is not None:
            return self._api_key_resolver(provider)
        if hook := self._provider_runtime.provider_for(provider):
            return hook.resolve_api_key(provider)
        return None

    def _resolve_api_base(self, provider: str) -> str | None:
        if isinstance(self._api_base, dict):
            return self._api_base.get(provider)
        return self._api_base

    def _freeze_cache_key(self, provider: str, api_key: str | None, api_base: str | None) -> str:
        def _freeze(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (tuple, list)):
                return [_freeze(item) for item in value]
            if isinstance(value, dict):
                return {str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
            return repr(value)

        payload = {
            "provider": provider,
            "api_key": api_key,
            "api_base": api_base,
            "client_args": _freeze(self._client_args),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def get_backend(self, provider: str) -> ProviderBackend:
        api_key = self._resolve_api_key(provider)
        api_base = self._resolve_api_base(provider)
        cache_key = self._freeze_cache_key(provider, api_key, api_base)
        if cache_key not in self._client_cache:
            hook = self._provider_runtime.provider_for(provider)
            if hook is None:
                raise RepublicError(ErrorKind.INVALID_INPUT, f"Unsupported provider: {provider}")
            backend = hook.create_backend(
                ProviderContext(
                    provider=provider,
                    api_key=api_key,
                    api_base=api_base,
                    client_args=dict(self._client_args),
                )
            )
            self._client_cache[cache_key] = backend
        return self._client_cache[cache_key]

    def log_error(self, error: RepublicError, provider: str, model: str, attempt: int) -> None:
        if self._verbose == 0:
            return
        prefix = f"[{provider}:{model}] attempt {attempt + 1}/{self.max_attempts()}"
        logger.warning("%s failed: %s", prefix, error)

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    @staticmethod
    def _text_matches(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def _classify_provider_exception(self, provider: str, exc: Exception) -> ErrorKind | None:
        hook = self._provider_runtime.provider_for(provider)
        if hook is None:
            return None
        return hook.classify_error(exc)

    @staticmethod
    def _classify_known_exception_names(exc: Exception) -> ErrorKind | None:
        error_name = type(exc).__name__
        if error_name in {"AuthenticationError", "MissingApiKeyError", "PermissionDeniedError"}:
            return ErrorKind.CONFIG
        if error_name in {
            "UnsupportedProviderError",
            "UnsupportedParameterError",
            "InvalidRequestError",
            "ModelNotFoundError",
            "ContextLengthExceededError",
            "NotFoundError",
            "BadRequestError",
            "UnprocessableEntityError",
        }:
            return ErrorKind.INVALID_INPUT
        if error_name in {"RateLimitError", "ConflictError"}:
            return ErrorKind.TEMPORARY
        if error_name in {"APIError", "ProviderError", "InternalServerError"}:
            return ErrorKind.PROVIDER
        return None

    def _classify_by_http_status(self, exc: Exception) -> ErrorKind | None:
        status = self._extract_status_code(exc)
        if status in {401, 403}:
            return ErrorKind.CONFIG
        if status in {400, 404, 413, 422}:
            return ErrorKind.INVALID_INPUT
        if status in {408, 409, 425, 429}:
            return ErrorKind.TEMPORARY
        if status is not None and 500 <= status < 600:
            return ErrorKind.PROVIDER
        return None

    def _classify_by_text_signature(self, exc: Exception) -> ErrorKind | None:
        name = type(exc).__name__.lower()
        text = f"{name} {exc!s}".lower()

        if self._text_matches(
            text,
            (
                r"auth|authentication|unauthorized|forbidden|permission denied|access denied",
                r"invalid[_\s-]?api[_\s-]?key|incorrect api key|api key.*not valid",
            ),
        ):
            return ErrorKind.CONFIG

        if self._text_matches(
            text,
            (
                r"ratelimit|rate[_\s-]?limit|too many requests|quota exceeded",
                r"\b429\b",
            ),
        ):
            return ErrorKind.TEMPORARY

        if self._text_matches(
            text,
            (
                r"invalid request|bad request|validation|unprocessable",
                r"model.*not.*found|does not exist",
                r"context.*length|maximum.*context|token limit",
                r"unsupported parameter",
            ),
        ):
            return ErrorKind.INVALID_INPUT

        if self._text_matches(
            text,
            (
                r"timeout|timed out|connection error|network error",
                r"internal server|service unavailable|gateway timeout",
            ),
        ):
            return ErrorKind.PROVIDER
        return None

    def classify_exception(self, exc: Exception, *, provider: str | None = None) -> ErrorKind:
        if isinstance(exc, RepublicError):
            return exc.kind
        if (custom_kind := self._classify_with_custom_classifier(exc)) is not None:
            return custom_kind
        if self._is_pydantic_validation_error(exc):
            return ErrorKind.INVALID_INPUT

        for classifier in self._exception_classifiers(provider):
            mapped = classifier(exc)
            if mapped is not None:
                return mapped

        return ErrorKind.UNKNOWN

    def _classify_with_custom_classifier(self, exc: Exception) -> ErrorKind | None:
        if self._error_classifier is None:
            return None
        try:
            kind = self._error_classifier(exc)
        except Exception as classifier_exc:
            logger.warning("error_classifier failed: %r", classifier_exc)
            return None
        if isinstance(kind, ErrorKind):
            return kind
        return None

    @staticmethod
    def _pydantic_validation_error_type() -> type[Exception] | None:
        try:
            from pydantic import ValidationError as PydanticValidationError
        except ImportError:
            return None
        return PydanticValidationError

    def _is_pydantic_validation_error(self, exc: Exception) -> bool:
        validation_error_type = self._pydantic_validation_error_type()
        return validation_error_type is not None and isinstance(exc, validation_error_type)

    def _exception_classifiers(
        self,
        provider: str | None,
    ) -> list[Callable[[Exception], ErrorKind | None]]:
        classifiers: list[Callable[[Exception], ErrorKind | None]] = [
            self._classify_known_exception_names,
            self._classify_by_http_status,
            self._classify_by_text_signature,
        ]
        if provider is None:
            return classifiers
        return [lambda error: self._classify_provider_exception(provider, error), *classifiers]

    def should_retry(self, kind: ErrorKind) -> bool:
        return kind in {ErrorKind.TEMPORARY, ErrorKind.PROVIDER}

    def wrap_error(self, exc: Exception, provider: str, model: str) -> RepublicError:
        kind = self.classify_exception(exc, provider=provider)
        message = f"{provider}:{model}: {exc}"
        return RepublicError(kind, message)

    def raise_wrapped(self, exc: Exception, provider: str, model: str) -> NoReturn:
        raise self.wrap_error(exc, provider, model) from exc

    def _handle_attempt_error(self, exc: Exception, provider_name: str, model_id: str, attempt: int) -> AttemptOutcome:
        wrapped = exc if isinstance(exc, RepublicError) else self.wrap_error(exc, provider_name, model_id)
        kind = wrapped.kind
        self.log_error(wrapped, provider_name, model_id, attempt)
        can_retry_same_model = self.should_retry(kind) and attempt + 1 < self.max_attempts()
        if can_retry_same_model:
            return AttemptOutcome(error=wrapped, decision=AttemptDecision.RETRY_SAME_MODEL)
        return AttemptOutcome(error=wrapped, decision=AttemptDecision.TRY_NEXT_MODEL)

    def _decide_kwargs_for_backend(
        self,
        backend: ProviderBackend,
        max_tokens: int | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        clean_kwargs = dict(kwargs)
        max_tokens_arg = backend.capabilities.completion_max_tokens_arg
        if max_tokens_arg in clean_kwargs:
            return clean_kwargs
        return {**clean_kwargs, max_tokens_arg: max_tokens}

    def _decide_responses_kwargs(
        self,
        max_tokens: int | None,
        kwargs: dict[str, Any],
        *,
        drop_extra_headers: bool = True,
    ) -> dict[str, Any]:
        clean_kwargs = dict(kwargs)
        if drop_extra_headers:
            clean_kwargs.pop("extra_headers", None)
        clean_kwargs = normalize_responses_kwargs(clean_kwargs)
        if "max_output_tokens" in clean_kwargs or max_tokens is None:
            return clean_kwargs
        return {**clean_kwargs, "max_output_tokens": max_tokens}

    def _with_default_completion_stream_options(
        self,
        backend: ProviderBackend,
        stream: bool,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if not stream:
            return kwargs
        if not backend.capabilities.default_completion_stream_usage:
            return kwargs
        if "stream_options" in kwargs:
            return kwargs
        return {**kwargs, "stream_options": {"include_usage": True}}

    @staticmethod
    def _with_responses_reasoning(
        kwargs: dict[str, Any],
        reasoning_effort: Any | None,
    ) -> dict[str, Any]:
        if reasoning_effort is None:
            return kwargs
        if "reasoning" in kwargs:
            return kwargs
        return {**kwargs, "reasoning": {"effort": reasoning_effort}}

    @staticmethod
    def _convert_tools_for_responses(tools_payload: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools_payload:
            return tools_payload

        converted_tools: list[dict[str, Any]] = []
        for tool in tools_payload:
            function = tool.get("function")
            if isinstance(function, dict):
                converted: dict[str, Any] = {
                    "type": tool.get("type", "function"),
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
                if "strict" in function:
                    converted["strict"] = function["strict"]
                converted_tools.append(converted)
                continue
            converted_tools.append(dict(tool))
        return converted_tools

    def _selected_transport(
        self,
        backend: ProviderBackend,
        *,
        provider_name: str,
        model_id: str,
    ) -> Literal["completion", "responses", "messages"]:
        capabilities = backend.capabilities
        if capabilities.preferred_transport is not None:
            return capabilities.preferred_transport
        if self._api_format in capabilities.transports:
            return self._api_format
        if self._api_format == "completion" and capabilities.transports == MESSAGES_TRANSPORTS:
            return "messages"
        available = ", ".join(sorted(capabilities.transports))
        raise RepublicError(
            ErrorKind.INVALID_INPUT,
            f"{provider_name}:{model_id}: requested transport '{self._api_format}' is not supported; "
            f"available transports: {available}",
        )

    def _build_chat_request(
        self,
        request: TransportCallRequest,
        *,
        transport: Literal["completion", "responses", "messages"],
    ) -> ChatRequest:
        conversation = conversation_from_messages(request.messages_payload)
        if transport == "responses":
            request_kwargs = self._with_responses_reasoning(request.kwargs, request.reasoning_effort)
            request_kwargs = self._decide_responses_kwargs(
                request.max_tokens,
                request_kwargs,
                drop_extra_headers=not request.backend.capabilities.preserves_responses_extra_headers,
            )
            tools = self._convert_tools_for_responses(request.tools_payload)
        else:
            request_kwargs = self._decide_kwargs_for_backend(request.backend, request.max_tokens, request.kwargs)
            request_kwargs = self._with_default_completion_stream_options(
                request.backend,
                request.stream,
                request_kwargs,
            )
            tools = request.tools_payload

        return ChatRequest(
            transport=transport,
            model=request.model_id,
            conversation=conversation,
            stream=request.stream,
            reasoning_effort=request.reasoning_effort,
            kwargs=request_kwargs,
            tools=tools,
        )

    def _call_client_sync(
        self,
        *,
        client: ProviderBackend,
        provider_name: str,
        model_id: str,
        messages_payload: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        max_tokens: int | None,
        stream: bool,
        reasoning_effort: Any | None,
        kwargs: dict[str, Any],
    ) -> Any:
        request = TransportCallRequest(
            backend=client,
            provider_name=provider_name,
            model_id=model_id,
            messages_payload=messages_payload,
            tools_payload=tools_payload,
            max_tokens=max_tokens,
            stream=stream,
            reasoning_effort=reasoning_effort,
            kwargs=kwargs,
        )
        transport = self._selected_transport(
            client,
            provider_name=provider_name,
            model_id=model_id,
        )
        chat_request = self._build_chat_request(request, transport=transport)
        client.validate_chat_request(chat_request)
        return client.chat(chat_request)

    async def _call_client_async(
        self,
        *,
        client: ProviderBackend,
        provider_name: str,
        model_id: str,
        messages_payload: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        max_tokens: int | None,
        stream: bool,
        reasoning_effort: Any | None,
        kwargs: dict[str, Any],
    ) -> Any:
        request = TransportCallRequest(
            backend=client,
            provider_name=provider_name,
            model_id=model_id,
            messages_payload=messages_payload,
            tools_payload=tools_payload,
            max_tokens=max_tokens,
            stream=stream,
            reasoning_effort=reasoning_effort,
            kwargs=kwargs,
        )
        transport = self._selected_transport(
            client,
            provider_name=provider_name,
            model_id=model_id,
        )
        chat_request = self._build_chat_request(request, transport=transport)
        client.validate_chat_request(chat_request)
        return await client.achat(chat_request)

    def run_chat_sync(
        self,
        *,
        messages_payload: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        model: str | None,
        provider: str | None,
        max_tokens: int | None,
        stream: bool,
        reasoning_effort: Any | None,
        kwargs: dict[str, Any],
        on_response: Callable[[Any, str, str, int], Any],
    ) -> Any:
        last_provider: str | None = None
        last_model: str | None = None
        last_error: RepublicError | None = None
        for provider_name, model_id, client in self.iter_clients(model, provider):
            last_provider, last_model = provider_name, model_id
            for attempt in range(self.max_attempts()):
                try:
                    response = self._call_client_sync(
                        client=client,
                        provider_name=provider_name,
                        model_id=model_id,
                        messages_payload=messages_payload,
                        tools_payload=tools_payload,
                        max_tokens=max_tokens,
                        stream=stream,
                        reasoning_effort=reasoning_effort,
                        kwargs=kwargs,
                    )
                except Exception as exc:
                    outcome = self._handle_attempt_error(exc, provider_name, model_id, attempt)
                    last_error = outcome.error
                    if outcome.decision is AttemptDecision.RETRY_SAME_MODEL:
                        continue
                    break
                else:
                    try:
                        result = on_response(response, provider_name, model_id, attempt)
                    except RepublicError as exc:
                        self.log_error(exc, provider_name, model_id, attempt)
                        if exc.kind == ErrorKind.TEMPORARY:
                            continue
                        raise
                    return result

        if last_error is not None:
            raise last_error
        if last_provider and last_model:
            raise RepublicError(
                ErrorKind.TEMPORARY,
                f"{last_provider}:{last_model}: LLM call failed after retries",
            )
        raise RepublicError(ErrorKind.TEMPORARY, "LLM call failed after retries")

    async def run_chat_async(  # noqa: C901
        self,
        *,
        messages_payload: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        model: str | None,
        provider: str | None,
        max_tokens: int | None,
        stream: bool,
        reasoning_effort: Any | None,
        kwargs: dict[str, Any],
        on_response: Callable[[Any, str, str, int], Any],
    ) -> Any:
        last_provider: str | None = None
        last_model: str | None = None
        last_error: RepublicError | None = None
        for provider_name, model_id, client in self.iter_clients(model, provider):
            last_provider, last_model = provider_name, model_id
            for attempt in range(self.max_attempts()):
                try:
                    response = await self._call_client_async(
                        client=client,
                        provider_name=provider_name,
                        model_id=model_id,
                        messages_payload=messages_payload,
                        tools_payload=tools_payload,
                        max_tokens=max_tokens,
                        stream=stream,
                        reasoning_effort=reasoning_effort,
                        kwargs=kwargs,
                    )
                except Exception as exc:
                    outcome = self._handle_attempt_error(exc, provider_name, model_id, attempt)
                    last_error = outcome.error
                    if outcome.decision is AttemptDecision.RETRY_SAME_MODEL:
                        continue
                    break
                else:
                    try:
                        result = on_response(response, provider_name, model_id, attempt)
                        if inspect.isawaitable(result):
                            result = await result
                    except RepublicError as exc:
                        self.log_error(exc, provider_name, model_id, attempt)
                        if exc.kind == ErrorKind.TEMPORARY:
                            continue
                        raise
                    return result

        if last_error is not None:
            raise last_error
        if last_provider and last_model:
            raise RepublicError(
                ErrorKind.TEMPORARY,
                f"{last_provider}:{last_model}: LLM call failed after retries",
            )
        raise RepublicError(ErrorKind.TEMPORARY, "LLM call failed after retries")
