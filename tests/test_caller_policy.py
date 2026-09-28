"""Caller policy stays outside SDK mechanisms; real HTTPX/OpenAI/Anthropic wires."""

import json
from pathlib import Path

import anthropic
import httpx
import openai
import pytest

from republic import ProviderError, generate
from republic.auth import codex
from republic.auth.github_copilot import CopilotToken
from republic.auth.grok import GrokTokens
from republic.providers.anthropic import AnthropicMessages
from republic.providers.codex import OpenAICodex
from republic.providers.github_copilot import GitHubCopilot
from republic.providers.grok import GrokOAuth
from republic.providers.openai import OpenAIChatCompletions, OpenAIResponses
from tests.anthropic_fixtures import message as anthropic_message
from tests.http_fixtures import Bytes, Transport, streaming
from tests.openai_fixtures import completion, request, sse
from tests.responses_fixtures import response, terminal


def reply(protocol: str) -> httpx.Response:
    if protocol in {"codex", "grok"}:
        return streaming(Bytes([sse(terminal())]))
    payload = (
        anthropic_message() if protocol == "anthropic" else response() if protocol == "responses" else completion()
    )
    return httpx.Response(200, json=payload)


def adapter(protocol: str, client, **options):
    if protocol == "codex":
        return OpenAICodex("fixture-access", client=client, **options)
    if protocol == "grok":
        return GrokOAuth("fixture-access", client_version="1.0.41", client=client, **options)
    if protocol == "copilot":
        return GitHubCopilot("fixture-inference", integration_id="fixture", client=client, **options)
    cls = {"chat": OpenAIChatCompletions, "responses": OpenAIResponses, "anthropic": AnthropicMessages}[protocol]
    return cls(client=client, **options)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "codex", "grok", "copilot"])
@pytest.mark.parametrize("retry", [None, 0, 1])
async def test_borrowed_configuration_and_explicit_retry(protocol: str, retry: int | None) -> None:
    replies = [httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "rate"}})]
    replies.append(reply(protocol))
    transport = Transport(replies)
    http = httpx.AsyncClient(transport=transport, follow_redirects=True)
    cls = anthropic.AsyncAnthropic if protocol == "anthropic" else openai.AsyncOpenAI
    client = cls(
        api_key="original",
        base_url="https://caller.test/v1",
        max_retries=1,
        default_headers={"x-caller": "retained", "originator": "caller"},
        default_query={"caller": "kept"},
        http_client=http,
    )
    async with adapter(protocol, client, max_retries=retry) as provider:
        req = request()
        if protocol == "anthropic":
            req.options.max_output_tokens = 20
        if retry == 0:
            with pytest.raises(ProviderError):
                await generate(provider, req)
            assert len(transport.requests) == 1
        else:
            await generate(provider, req)
            assert len(transport.requests) == 2
        for sent in transport.requests:
            assert sent.url.host == "caller.test" and sent.url.query == b"caller=kept"
            assert sent.headers["x-caller"] == "retained" and sent.headers["originator"] == "caller"
        assert provider._client.max_retries == (1 if retry is None else retry)
    assert client.max_retries == 1 and client.api_key == "original"
    assert not client.is_closed() and http.follow_redirects
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["codex", "grok", "copilot"])
async def test_oauth_explicit_routing_headers_and_native_parameters(protocol: str) -> None:
    transport = Transport([reply(protocol)])
    async with openai.AsyncOpenAI(
        api_key="old",
        base_url="https://borrowed.test",
        max_retries=3,
        default_headers={"originator": "borrowed", "Authorization": "borrowed-auth"},
        http_client=httpx.AsyncClient(transport=transport),
    ) as client:
        async with adapter(
            protocol,
            client,
            base_url="https://explicit.test/v1",
            max_retries=0,
            headers={"originator": "explicit", "authorization": "Bearer explicit"},
        ) as provider:
            req = request()
            req.options.parallel_tool_calls = False
            native = {
                "store": True,
                "extra_body": {"caller_extension": {"value": 1}},
                "extra_headers": {"originator": "request", "authorization": "Bearer request"},
            }
            if protocol != "copilot":
                native.update(include=[], truncation="auto", previous_response_id="resp_previous")
            req.options.provider_options = native
            await generate(provider, req)
        sent = transport.requests[0]
        assert sent.url.host == "explicit.test"
        assert sent.headers["authorization"] == "Bearer request" and sent.headers["originator"] == "request"
        assert client._custom_headers["originator"] == "borrowed"
        body = json.loads(sent.content)
        assert body["store"] is True and body["caller_extension"] == {"value": 1}
        assert body["parallel_tool_calls"] is False
        if protocol != "copilot":
            assert body["include"] == [] and body["truncation"] == "auto"
            assert body["previous_response_id"] == "resp_previous"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["codex", "grok", "copilot"])
@pytest.mark.parametrize("expiry", [None, 1])
async def test_existing_credentials_need_no_refresh_or_known_future_expiry(protocol, expiry) -> None:
    transport = Transport([reply(protocol)])
    async with openai.AsyncOpenAI(
        api_key="fixture", base_url="https://fixture.test", http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        if protocol == "codex":
            tokens = codex.CodexTokens("access", expires_at=expiry)
            provider = OpenAICodex(tokens, client=client)
        elif protocol == "grok":
            tokens = GrokTokens("access", expires_at=expiry)
            provider = GrokOAuth(tokens, client_version="1.0.41", client=client)
        else:
            tokens = CopilotToken("access", expires_at=expiry)
            provider = GitHubCopilot(tokens, integration_id="fixture", client=client)
        assert tokens.is_expired() is (expiry == 1)
        async with provider:
            await generate(provider, request())
        assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_codex_code_exchange_and_optional_file_helpers(tmp_path: Path) -> None:
    authorization = await codex.create_authorization()
    transport = Transport([httpx.Response(200, json={"access_token": "existing-access", "token_type": "Bearer"})])
    tokens = await codex.exchange_authorization_code(authorization, "manual-code", transport=transport)
    assert tokens.refresh_token is None and tokens.expires_at is None and not tokens.is_expired()
    assert b"code_verifier=" in transport.requests[0].content
    path = tmp_path / "caller-chosen.json"
    codex.write_tokens(path, tokens)
    assert codex.read_tokens(path) == tokens and "existing-access" not in repr(tokens)
    with pytest.raises(codex.CodexAuthError, match="no_refresh_token"):
        await codex.refresh_tokens(tokens, transport=Transport([]))


@pytest.mark.asyncio
async def test_codex_explicit_instructions_conflict_with_system_message() -> None:
    from republic import Message, TextPart, UnsupportedRequestError

    transport = Transport([])
    async with (
        openai.AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(transport=transport)) as client,
        OpenAICodex("access", client=client) as provider,
    ):
        req = request()
        req.messages.insert(0, Message(role="system", parts=[TextPart(text="system")]))
        req.options.provider_options = {"instructions": "native"}
        with pytest.raises(UnsupportedRequestError, match="instructions"):
            await generate(provider, req)
    assert not transport.requests


@pytest.mark.asyncio
async def test_owned_client_explicit_retry_timeout_and_header_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = Transport([httpx.Response(500, headers={"retry-after-ms": "1"}), reply("codex")])
    http = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", lambda **kwargs: http)
    async with OpenAICodex(
        "access", base_url="https://owned.test/v1", max_retries=1, timeout=7, headers={"originator": "caller"}
    ) as provider:
        await generate(provider, request())
        assert len(transport.requests) == 2
        assert all(sent.url.host == "owned.test" for sent in transport.requests)
        assert transport.requests[0].extensions["timeout"]["read"] == 7
        assert transport.requests[0].headers["originator"] == "caller"
    assert http.is_closed
