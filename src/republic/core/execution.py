"""Core execution utilities for Republic."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, NoReturn

from any_llm import AnyLLM
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
    UnsupportedProviderError,
)

from republic.core.errors import ErrorKind, RepublicError

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


class LLMCore:
    """Shared LLM execution utilities (provider resolution, retries, client cache)."""

    RETRY = object()

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        fallback_models: list[str],
        max_retries: int,
        api_key: str | dict[str, str] | None,
        api_base: str | dict[str, str] | None,
        client_args: dict[str, Any],
        use_responses: bool,
        verbose: int,
        error_classifier: Callable[[Exception], ErrorKind | None] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._fallback_models = fallback_models
        self._max_retries = max_retries
        self._api_key = api_key
        self._api_base = api_base
        self._client_args = client_args
        self._use_responses = use_responses
        self._verbose = verbose
        self._error_classifier = error_classifier
        self._client_cache: dict[str, AnyLLM] = {}

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
            yield provider_name, model_id, self.get_client(provider_name)

    def _resolve_api_key(self, provider: str) -> str | None:
        if isinstance(self._api_key, dict):
            return self._api_key.get(provider)
        return self._api_key

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

    def get_client(self, provider: str) -> AnyLLM:
        api_key = self._resolve_api_key(provider)
        api_base = self._resolve_api_base(provider)
        cache_key = self._freeze_cache_key(provider, api_key, api_base)
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = AnyLLM.create(
                provider,
                api_key=api_key,
                api_base=api_base,
                **self._client_args,
            )
        return self._client_cache[cache_key]

    def log_error(self, error: RepublicError, provider: str, model: str, attempt: int) -> None:
        if self._verbose == 0:
            return

        prefix = f"[{provider}:{model}] attempt {attempt + 1}/{self.max_attempts()}"
        if error.cause:
            logger.warning("%s failed: %s (cause=%r)", prefix, error, error.cause)
        else:
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

    def _classify_anyllm_exception(self, exc: Exception) -> ErrorKind | None:
        error_map = [
            ((MissingApiKeyError, AuthenticationError), ErrorKind.CONFIG),
            (
                (
                    UnsupportedProviderError,
                    UnsupportedParameterError,
                    InvalidRequestError,
                    ModelNotFoundError,
                    ContextLengthExceededError,
                ),
                ErrorKind.INVALID_INPUT,
            ),
            ((RateLimitError, ContentFilterError), ErrorKind.TEMPORARY),
            ((ProviderError, AnyLLMError), ErrorKind.PROVIDER),
        ]
        for types, kind in error_map:
            if isinstance(exc, types):
                return kind
        return None

    def _classify_by_http_status(self, exc: Exception) -> ErrorKind | None:
        # SDK-native HTTP status handling (any-llm may pass these through unless unified mode is enabled).
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

    def classify_exception(self, exc: Exception) -> ErrorKind:
        if isinstance(exc, RepublicError):
            return exc.kind
        if self._error_classifier is not None:
            try:
                kind = self._error_classifier(exc)
            except Exception as classifier_exc:
                logger.warning("error_classifier failed: %r", classifier_exc)
            else:
                if isinstance(kind, ErrorKind):
                    return kind
        try:
            from pydantic import ValidationError as PydanticValidationError

            validation_error_type: type[Exception] | None = PydanticValidationError
        except ImportError:
            validation_error_type = None
        if validation_error_type is not None and isinstance(exc, validation_error_type):
            return ErrorKind.INVALID_INPUT

        for classifier in (
            self._classify_anyllm_exception,
            self._classify_by_http_status,
            self._classify_by_text_signature,
        ):
            mapped = classifier(exc)
            if mapped is not None:
                return mapped

        return ErrorKind.UNKNOWN

    def should_retry(self, kind: ErrorKind) -> bool:
        return kind in {ErrorKind.TEMPORARY, ErrorKind.PROVIDER}

    def wrap_error(self, exc: Exception, kind: ErrorKind, provider: str, model: str) -> RepublicError:
        message = f"{provider}:{model}: {exc}"
        return RepublicError(kind, message, cause=exc)

    def raise_wrapped(self, exc: Exception, provider: str, model: str) -> NoReturn:
        kind = self.classify_exception(exc)
        raise self.wrap_error(exc, kind, provider, model) from exc

    def _handle_attempt_error(self, exc: Exception, provider_name: str, model_id: str, attempt: int) -> AttemptOutcome:
        kind = self.classify_exception(exc)
        wrapped = self.wrap_error(exc, kind, provider_name, model_id)
        self.log_error(wrapped, provider_name, model_id, attempt)
        can_retry_same_model = self.should_retry(kind) and attempt + 1 < self.max_attempts()
        if can_retry_same_model:
            return AttemptOutcome(error=wrapped, decision=AttemptDecision.RETRY_SAME_MODEL)
        return AttemptOutcome(error=wrapped, decision=AttemptDecision.TRY_NEXT_MODEL)

    def _decide_kwargs_for_provider(
        self, provider: str, max_tokens: int | None, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        use_completion_tokens = "openai" in provider.lower()
        if not use_completion_tokens:
            return {**kwargs, "max_tokens": max_tokens}
        if "max_completion_tokens" in kwargs:
            return kwargs
        return {**kwargs, "max_completion_tokens": max_tokens}

    def _decide_responses_kwargs(self, max_tokens: int | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "extra_headers"}
        if "max_output_tokens" in clean_kwargs:
            return clean_kwargs
        return {**clean_kwargs, "max_output_tokens": max_tokens}

    def _should_use_responses(self, client: AnyLLM, *, stream: bool) -> bool:
        return not stream and self._use_responses and bool(getattr(client, "SUPPORTS_RESPONSES", False))

    def _call_client_sync(
        self,
        *,
        client: AnyLLM,
        provider_name: str,
        model_id: str,
        messages_payload: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        max_tokens: int | None,
        stream: bool,
        reasoning_effort: Any | None,
        kwargs: dict[str, Any],
    ) -> Any:
        if self._should_use_responses(client, stream=stream):
            instructions, input_items = self._split_messages_for_responses(messages_payload)
            return client.responses(
                model=model_id,
                input_data=input_items,
                tools=tools_payload,
                stream=stream,
                instructions=instructions,
                **self._decide_responses_kwargs(max_tokens, kwargs),
            )
        return client.completion(
            model=model_id,
            messages=messages_payload,
            tools=tools_payload,
            stream=stream,
            reasoning_effort=reasoning_effort,
            **self._decide_kwargs_for_provider(provider_name, max_tokens, kwargs),
        )

    async def _call_client_async(
        self,
        *,
        client: AnyLLM,
        provider_name: str,
        model_id: str,
        messages_payload: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        max_tokens: int | None,
        stream: bool,
        reasoning_effort: Any | None,
        kwargs: dict[str, Any],
    ) -> Any:
        if self._should_use_responses(client, stream=stream):
            instructions, input_items = self._split_messages_for_responses(messages_payload)
            return await client.aresponses(
                model=model_id,
                input_data=input_items,
                tools=tools_payload,
                stream=stream,
                instructions=instructions,
                **self._decide_responses_kwargs(max_tokens, kwargs),
            )
        return await client.acompletion(
            model=model_id,
            messages=messages_payload,
            tools=tools_payload,
            stream=stream,
            reasoning_effort=reasoning_effort,
            **self._decide_kwargs_for_provider(provider_name, max_tokens, kwargs),
        )

    @staticmethod
    def _split_messages_for_responses(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions_parts: list[str] = []
        filtered_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role in {"system", "developer"}:
                content = message.get("content")
                if content not in (None, ""):
                    instructions_parts.append(str(content))
                continue
            filtered_messages.append(message)

        instructions = "\n\n".join(part for part in instructions_parts if part.strip())
        if not instructions:
            instructions = None
        return instructions, LLMCore._convert_messages_to_responses_input(filtered_messages)

    @staticmethod
    def _convert_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content not in (None, ""):
                input_items.append({"role": role, "content": content, "type": "message"})

            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                for index, tool_call in enumerate(tool_calls):
                    func = tool_call.get("function") or {}
                    name = func.get("name")
                    if not name:
                        continue
                    call_id = tool_call.get("id") or tool_call.get("call_id") or f"call_{index}"
                    input_items.append({
                        "type": "function_call",
                        "name": name,
                        "arguments": func.get("arguments", ""),
                        "call_id": call_id,
                    })

            if role == "tool":
                call_id = message.get("tool_call_id") or message.get("call_id")
                if not call_id:
                    continue
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.get("content", ""),
                })
        return input_items

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
                    result = on_response(response, provider_name, model_id, attempt)
                    if result is self.RETRY:
                        continue
                    return result

        if last_error is not None:
            raise last_error
        if last_provider and last_model:
            raise RepublicError(
                ErrorKind.TEMPORARY,
                f"{last_provider}:{last_model}: LLM call failed after retries",
            )
        raise RepublicError(ErrorKind.TEMPORARY, "LLM call failed after retries")

    async def run_chat_async(
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
                    result = on_response(response, provider_name, model_id, attempt)
                    if inspect.isawaitable(result):
                        result = await result
                    if result is self.RETRY:
                        continue
                    return result

        if last_error is not None:
            raise last_error
        if last_provider and last_model:
            raise RepublicError(
                ErrorKind.TEMPORARY,
                f"{last_provider}:{last_model}: LLM call failed after retries",
            )
        raise RepublicError(ErrorKind.TEMPORARY, "LLM call failed after retries")
