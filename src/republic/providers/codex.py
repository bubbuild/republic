# Adapted from Bub 357901db and OpenAI Codex 21eb3551 (Apache-2.0); see NOTICE.
"""ChatGPT-backed Codex Responses, with explicit tokens and one SSE request."""

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import openai

from republic import events
from republic.api import stream
from republic.auth.codex import CodexTokens
from republic.errors import IncompleteStreamError, ProviderError, UnsupportedRequestError
from republic.providers import _openai_responses as responses
from republic.providers._files import file_url
from republic.providers._openai_client import OpenAIClient, oauth_client, oauth_error
from republic.providers._openai_responses_stream import ResponsesStream
from republic.types import FilePart, Request, Response

_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _file(part: FilePart) -> dict[str, Any]:
    if part.media_type.startswith("image/"):
        return responses._file(part)
    if part.media_type.startswith("audio/"):
        responses.metadata(part.provider_metadata, set(), "file.metadata")
        if part.filename is not None:
            raise UnsupportedRequestError("file.filename", "Codex input_audio has no filename field")
        # Codex ContentItem has audio_url, unlike Chat's input_audio object.
        # Keep this sourced wire extension out of standard OpenAI Responses.
        return {"type": "input_audio", "audio_url": file_url(part)}
    raise UnsupportedRequestError("file", "the Codex adapter supports user images/audio, not documents/video")


def _payload(request: Request) -> dict[str, Any]:
    for field in ("temperature", "top_p", "max_output_tokens", "stop"):
        if getattr(request.options, field) is not None:
            raise UnsupportedRequestError(field, "not supported by the Codex adapter")
    in_history = False
    for message in request.messages:
        if message.role == "system" and in_history:
            raise UnsupportedRequestError("messages.system", "only leading system messages can become instructions")
        in_history = in_history or message.role != "system"
    # Reuse native Responses conversion/validation, including retained raw items.
    payload = responses.request_payload(request, file_part=_file)
    if "truncation" not in request.options.provider_options:
        payload.pop("truncation")
    instructions = []
    history = []
    for item in payload["input"]:
        if item.get("role") == "system":
            instructions.extend(part["text"] for part in item["content"])
        else:
            history.append(item)
    if instructions and "instructions" in payload:
        raise UnsupportedRequestError("instructions", "conflicts with leading system messages")
    payload.setdefault("instructions", "\n\n".join(instructions))
    payload["input"] = history
    payload.setdefault("tools", [])
    payload.setdefault("tool_choice", "auto")
    payload.setdefault("parallel_tool_calls", True)
    return payload


class OpenAICodex(OpenAIClient):
    """Codex-only bearer tokens; generate aggregates one streaming wire request.

    Accept existing access tokens or optional lifecycle data. Borrowed client
    settings are retained unless explicitly overridden. Refresh, storage and
    expiry policy belong to the caller; providers never initiate authentication.
    """

    def __init__(
        self,
        tokens: CodexTokens | str,
        *,
        account_id: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        credentials = CodexTokens(tokens, account_id=account_id) if isinstance(tokens, str) else tokens
        defaults = {"originator": "republic"}
        if (account := account_id or credentials.account_id) is not None:
            defaults["chatgpt-account-id"] = account
        self._client, self._owns_client = oauth_client(
            credentials.access_token,
            _BASE_URL,
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
        payload = _payload(request)
        state = ResponsesStream()
        self._request_headers(payload)
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
            failure = oauth_error(exc, provider="codex")
        except (ValueError, TypeError, KeyError, AttributeError):
            failure = ProviderError("invalid_response", provider="codex", code="invalid_response")
        else:
            yield terminal
            return
        # Unlike API-key adapters, do not retain OAuth-bearing SDK error objects.
        raise failure
