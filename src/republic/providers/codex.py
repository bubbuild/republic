# Adapted from Bub 357901db and OpenAI Codex 21eb3551 (Apache-2.0); see NOTICE.
"""ChatGPT-backed Codex Responses, with explicit tokens and one SSE request."""

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import openai

from republic import events
from republic.api import stream
from republic.auth.codex import CodexAuthError, CodexTokens
from republic.errors import IncompleteStreamError, ProviderError, UnsupportedRequestError
from republic.providers import _openai_responses as responses
from republic.providers._openai_client import OpenAIClient
from republic.providers._openai_responses_stream import ResponsesStream
from republic.types import Request, Response

_BASE_URL = "https://chatgpt.com/backend-api/codex"
_OPTIONS = {"text", "reasoning", "store", "include", "service_tier", "prompt_cache_key", "timeout"}


def _payload(request: Request) -> dict[str, Any]:
    for field in ("temperature", "top_p", "max_output_tokens", "stop"):
        if getattr(request.options, field) is not None:
            raise UnsupportedRequestError(field, "not supported by the Codex adapter")
    if request.options.provider_options.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", "unsupported or managed Codex option")
    in_history = False
    for message in request.messages:
        if message.role == "system" and in_history:
            raise UnsupportedRequestError("messages.system", "only leading system messages can become instructions")
        in_history = in_history or message.role != "system"
    # Reuse native Responses conversion/validation, including retained raw items.
    payload = responses.request_payload(request)
    payload.pop("truncation")  # Shared converter default, never caller input.
    if payload["include"] != ["reasoning.encrypted_content"]:
        raise UnsupportedRequestError("include", "Codex requires encrypted reasoning replay")
    instructions = []
    history = []
    for item in payload["input"]:
        if item.get("role") == "system":
            instructions.extend(part["text"] for part in item["content"])
        else:
            history.append(item)
    payload.update(instructions="\n\n".join(instructions), input=history)
    payload.setdefault("tools", [])
    payload.setdefault("tool_choice", "auto")
    payload.setdefault("parallel_tool_calls", True)
    return payload


def _failure(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    codes = {401: "unauthorized", 403: "forbidden", 429: "rate_limit"}
    code = codes.get(status, "request_failed")
    if isinstance(exc, ProviderError):
        code = "invalid_response" if exc.code == "invalid_response" else "stream_error"
    return ProviderError(f"Codex request: {code}", provider="codex", status_code=status, code=code)


class OpenAICodex(OpenAIClient):
    """Codex-only bearer tokens; generate aggregates one streaming wire request.

    A borrowed AsyncOpenAI contributes its HTTP transport/timeout, not its API
    key, endpoint, organization or custom headers/query. Its settings and lifetime
    remain untouched. After explicit refresh, construct a new provider with the
    new immutable tokens (the same borrowed client may be reused).
    """

    def __init__(self, tokens: CodexTokens, *, client: openai.AsyncOpenAI | None = None) -> None:
        if tokens.account_id is None:
            raise CodexAuthError("missing_account")
        if client is not None and client._client.follow_redirects:
            raise UnsupportedRequestError(
                "client.follow_redirects", "disable redirects for a single Codex HTTP attempt"
            )
        self._tokens = tokens
        headers = {"chatgpt-account-id": tokens.account_id, "originator": "republic"}
        if client is None:
            owned = openai.AsyncOpenAI(
                api_key=tokens.access_token,
                base_url=_BASE_URL,
                max_retries=0,
                default_headers=headers,
                http_client=openai.DefaultAsyncHttpxClient(follow_redirects=False),
            )
            super().__init__(client=owned)
            self._owns_client = True
        else:
            super().__init__(
                client=client.with_options(
                    api_key=tokens.access_token,
                    base_url=_BASE_URL,
                    set_default_headers=headers,
                    set_default_query={},
                    max_retries=0,
                )
            )
        # SDK copy() inherits these even when None is passed explicitly. Only
        # our private copy is changed; API account routing does not apply here.
        self._client.organization = None
        self._client.project = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderError("closed", provider="codex", code="closed")

    async def generate(self, request: Request) -> Response:
        """Collect a single SSE request; no non-streaming attempt or fallback."""
        async with stream(self, request) as output:
            async for _ in output:
                pass
        result = output.response
        if result is None:
            raise IncompleteStreamError
        return result

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """Use native Responses events and release the wire before StreamEnd."""
        self._ensure_open()
        if self._tokens.is_expired():
            raise CodexAuthError("expired")
        payload = _payload(request)
        state = ResponsesStream()
        try:
            source = await self._client.responses.create(**payload, stream=True)
            async with source:
                async for chunk in source:
                    raw = chunk.model_dump(mode="json", exclude_none=True)
                    if isinstance(raw.get("response"), dict) and raw["response"].get("error") is not None:
                        # Server diagnostic text can echo credentials. Keep the
                        # failed status/partial output, not untrusted error text.
                        raw["response"]["error"] = {"code": "response_failed", "message": "Codex response failed"}
                    for event in state.feed(raw):
                        yield event
                    if state.terminal is not None:
                        break
                terminal = state.end()
        except (openai.OpenAIError, httpx.HTTPError, ProviderError) as exc:
            failure = _failure(exc)
        except (ValueError, TypeError, KeyError, AttributeError):
            failure = ProviderError("invalid_response", provider="codex", code="invalid_response")
        else:
            yield terminal
            return
        # Unlike API-key adapters, do not retain OAuth-bearing SDK error objects.
        raise failure
