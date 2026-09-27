# Copyright 2023-2026 SpaceXAI. Licensed under Apache-2.0.
# Protocol subset adapted from grok-build f0e3be11; see NOTICE.
"""Grok Build OAuth proxy, using native Responses and one SSE request."""

import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import openai

from republic import events
from republic.api import stream
from republic.auth.grok import GrokAuthError, GrokTokens, _version
from republic.errors import IncompleteStreamError, ProviderError, UnsupportedRequestError
from republic.providers import _openai_responses as responses
from republic.providers._openai_client import OpenAIClient, oauth_client, oauth_error
from republic.providers._openai_responses_stream import ResponsesStream
from republic.types import Request, Response

_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
_OPTIONS = {"text", "reasoning", "store", "include", "prompt_cache_key", "timeout"}


def _payload(request: Request) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", request.model):
        raise UnsupportedRequestError("model", "expected a nonempty ASCII model identifier for proxy routing")
    if request.options.parallel_tool_calls is not None:
        raise UnsupportedRequestError("parallel_tool_calls", "not established for this proxy adapter")
    if request.options.provider_options.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", "unsupported or managed Grok option")
    payload = responses.request_payload(request)
    payload.pop("truncation")  # Shared default; not accepted as caller input.
    if payload["include"] != ["reasoning.encrypted_content"]:
        raise UnsupportedRequestError("include", "retain encrypted reasoning for self-contained history")
    return payload


class GrokOAuth(OpenAIClient):
    """Explicit Grok OAuth tokens, not an api.x.ai API key.

    client_version identifies the caller-selected Grok Build wire version (source
    reference: 1.0.41). This does not establish third-party client entitlement.
    Borrowed SDK settings remain untouched; only its transport/timeout is reused.
    After explicit refresh, construct a new provider with the returned tokens.
    """

    def __init__(self, tokens: GrokTokens, *, client_version: str, client: openai.AsyncOpenAI | None = None) -> None:
        _version(client_version)
        self._tokens = tokens
        headers = {
            "X-XAI-Token-Auth": "xai-grok-cli",
            "x-authenticateresponse": "authenticate-response",
            "x-grok-client-version": client_version,
            "x-grok-client-identifier": "republic",
            "x-grok-client-mode": "headless",
            "User-Agent": "republic",
            "Accept": "text/event-stream",
        }
        self._client, self._owns_client = oauth_client(tokens.access_token, _BASE_URL, headers, client)
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderError("closed", provider="grok", code="closed")

    async def generate(self, request: Request) -> Response:
        """Aggregate one SSE request; no JSON-first attempt, retry or fallback."""
        async with stream(self, request) as output:
            async for _ in output:
                pass
        result = output.response
        if result is None:
            raise IncompleteStreamError
        return result

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """Reuse native Responses parsing, preserving item identity and raw reasoning."""
        self._ensure_open()
        if self._tokens.is_expired():
            raise GrokAuthError("expired")
        payload = _payload(request)
        state = ResponsesStream()
        try:
            source = await self._client.responses.create(
                **payload, stream=True, extra_headers={"x-grok-model-override": request.model}
            )
            async with source:
                async for chunk in source:
                    raw = chunk.model_dump(mode="json", exclude_none=True)
                    if isinstance(raw.get("response"), dict) and raw["response"].get("error") is not None:
                        raw["response"]["error"] = {"code": "response_failed", "message": "Grok response failed"}
                    for event in state.feed(raw):
                        yield event
                    if state.terminal is not None:
                        break
                terminal = state.end()
        except (openai.OpenAIError, httpx.HTTPError, ProviderError) as exc:
            failure = oauth_error(exc, provider="grok")
        except (ValueError, TypeError, KeyError, AttributeError):
            failure = ProviderError("invalid_response", provider="grok", code="invalid_response")
        else:
            yield terminal
            return
        raise failure
