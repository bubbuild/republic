# Protocol references: Microsoft Copilot Chat (MIT); see NOTICE.
"""Explicit Copilot Chat Completions using an already exchanged inference token."""

import re
from collections.abc import AsyncGenerator
from importlib.metadata import version
from typing import Any

import httpx
import openai

from republic import events
from republic.auth.github_copilot import CopilotAuthError, CopilotToken
from republic.errors import ProviderError, UnsupportedRequestError
from republic.providers import _openai_chat as chat
from republic.providers._openai_client import OpenAIClient, oauth_client, oauth_error
from republic.providers._openai_stream import ChatStream
from republic.types import FilePart, ReasoningPart, Request, Response


def _payload(request: Request) -> dict[str, Any]:
    if request.options.tool_choice == "required":
        raise UnsupportedRequestError(
            "tool_choice", "Copilot Chat does not support required in the referenced protocol"
        )
    if any(isinstance(part, FilePart | ReasoningPart) for message in request.messages for part in message.parts):
        raise UnsupportedRequestError("message.part", "Copilot media/native reasoning is not implemented")
    payload = chat.request_payload(request)
    if "max_completion_tokens" in payload:
        payload["max_tokens"] = payload.pop("max_completion_tokens")
    return payload


def _check_output(raw: dict[str, Any]) -> dict[str, Any]:
    # CAPI reasoning is NOT OpenAI reasoning_content. Fail instead of discarding
    # opaque history required by some models; those models are outside this subset.
    unsupported = {
        "reasoning_text",
        "reasoning_opaque",
        "cot_id",
        "cot_summary",
        "thinking",
        "signature",
        "reasoning",
        "reasoning_content",
        "copilot_references",
        "copilot_confirmations",
    }
    for choice in raw.get("choices", []):
        data = choice.get("message", choice.get("delta")) or {}
        if any(data.get(key) is not None for key in unsupported):
            raise chat.invalid_response(detail="unsupported Copilot reasoning/reference output")
    return raw


class GitHubCopilot(OpenAIClient):
    """One Chat Completions request. Authentication/renewal stays with the caller.

    integration_id must be the caller's service-recognized integration. Republic
    does not impersonate a first-party editor or promise third-party entitlement.
    Borrowed client policy is retained; owned clients default to zero retries.
    """

    def __init__(
        self,
        token: CopilotToken | str,
        *,
        integration_id: str,
        client: openai.AsyncOpenAI | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        if isinstance(token, str):
            token = CopilotToken(token)
        if not isinstance(token, CopilotToken):
            raise CopilotAuthError("inference_token_required")
        if not isinstance(integration_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", integration_id):
            raise UnsupportedRequestError("integration_id", "expected a nonempty service integration identifier")
        self._token = token
        identity = f"republic/{version('republic')}"
        defaults = {
            "User-Agent": "republic",
            "Editor-Version": identity,
            "Editor-Plugin-Version": identity,
            "Copilot-Integration-Id": integration_id,
            "X-GitHub-Api-Version": "2025-10-01",
            "OpenAI-Intent": "conversation-panel",
        }
        self._client, self._owns_client = oauth_client(
            token.token,
            token.api_endpoint,
            defaults,
            client,
            endpoint=base_url,
            extra_headers=headers,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderError("closed", provider="github-copilot", code="closed")

    def _request(self, request: Request) -> dict[str, Any]:
        self._ensure_open()
        return _payload(request)

    async def generate(self, request: Request) -> Response:
        """One non-streaming POST /chat/completions; never exchange/refresh tokens."""
        payload = self._request(request)
        self._request_headers(payload)
        try:
            result = await self._client.chat.completions.create(**payload, stream=False)
            return chat.response(_check_output(result.model_dump(mode="json", exclude_none=True)))
        except (openai.OpenAIError, httpx.HTTPError, ProviderError) as exc:
            failure = oauth_error(exc, provider="github-copilot")
        except (ValueError, TypeError, KeyError, AttributeError):
            failure = ProviderError("invalid_response", provider="github-copilot", code="invalid_response")
        raise failure

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """One SSE request; consume tail usage and close the response before end."""
        payload = self._request(request)
        state = ChatStream()
        self._request_headers(payload)
        try:
            source = await self._client.chat.completions.create(
                **payload,
                stream=True,
                stream_options={"include_usage": True},
            )
            async with source:
                async for chunk in source:
                    for event in state.feed(_check_output(chunk.model_dump(mode="json", exclude_none=True))):
                        yield event
                terminal = state.end()
        except (openai.OpenAIError, httpx.HTTPError, ProviderError) as exc:
            failure = oauth_error(exc, provider="github-copilot")
        except (ValueError, TypeError, KeyError, AttributeError):
            failure = ProviderError("invalid_response", provider="github-copilot", code="invalid_response")
        else:
            yield terminal
            return
        raise failure
