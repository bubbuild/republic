# Copyright 2026 Vercel, Inc. Licensed under the Apache License, Version 2.0.
# Modified for Republic: native generate, explicit ownership and caller-configurable retries.
# Source: ai-python c788059dd1db2d93ae1c3da6daffb660eca07dbb; see NOTICE.
"""Explicit Chat Completions and Responses adapters using the official async client."""

from collections.abc import AsyncGenerator

import httpx
import openai

from republic import events
from republic.providers import _openai_chat as chat
from republic.providers import _openai_responses as responses
from republic.providers._openai_client import OpenAIClient, provider_error
from republic.providers._openai_responses_stream import ResponsesStream
from republic.providers._openai_stream import ChatStream
from republic.types import Request, Response


class OpenAIChatCompletions(OpenAIClient):
    """One request per operation, with owned or borrowed AsyncOpenAI clients.

    Without client, Republic creates and owns an AsyncOpenAI(max_retries=0).
    With client, it preserves configuration through with_options(),
    leaving the caller's settings and lifetime unchanged. Custom transports and
    endpoints must not independently retry if one HTTP attempt is required.
    """

    async def generate(self, request: Request) -> Response:
        """Perform one non-streaming POST /chat/completions."""
        self._ensure_open()
        payload = chat.request_payload(request)
        self._request_headers(payload)
        try:
            result = await self._client.chat.completions.create(**payload, stream=False)
            return chat.response(result.model_dump(mode="json", exclude_none=True))
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise chat.invalid_response(detail=str(exc)) from exc

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """Stream one request; read trailing usage before emitting StreamEnd."""
        self._ensure_open()
        payload = chat.request_payload(request)
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
                    for event in state.feed(chunk.model_dump(mode="json", exclude_none=True)):
                        yield event
                terminal = state.end()
            # Close the wire response before handing completion to the caller.
            yield terminal
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise chat.invalid_response(detail=str(exc)) from exc


class OpenAIResponses(OpenAIClient):
    """One Responses call with self-contained history and store=False defaults.

    Uses the same owned/borrowed client configuration as Chat Completions.
    Reasoning and output item metadata survive serialization and history replay.
    """

    async def generate(self, request: Request) -> Response:
        """Perform one non-streaming POST /responses; no parsing/repair turns."""
        self._ensure_open()
        payload = responses.request_payload(request)
        self._request_headers(payload)
        try:
            result = await self._client.responses.create(**payload, stream=False)
            return responses.response(result.model_dump(mode="json", exclude_none=True))
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise responses.invalid_response(detail=str(exc)) from exc

    async def stream(self, request: Request) -> AsyncGenerator[events.Event, None]:
        """Reconcile native deltas/snapshots and close before StreamEnd."""
        self._ensure_open()
        payload = responses.request_payload(request)
        state = ResponsesStream()
        self._request_headers(payload)
        try:
            source = await self._client.responses.create(**payload, stream=True)
            async with source:
                async for chunk in source:
                    for event in state.feed(chunk.model_dump(mode="json", exclude_none=True)):
                        yield event
                    if state.terminal is not None:
                        break
                terminal = state.end()
            yield terminal
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise provider_error(exc) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise responses.invalid_response(detail=str(exc)) from exc
