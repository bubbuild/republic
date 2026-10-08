# ruff: noqa: S105, S106 - all credentials in this module are test data
from __future__ import annotations

import asyncio
import base64
import json
import time
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs

import httpx2
import pytest
from authlib.integrations.httpx_client import OAuth2Client

import republic
from republic.history import InMemoryHistory
from republic.providers import CodexAuth, codex
from tests.conftest import FakeService


def reply(service: FakeService, text: str) -> None:
    service.reply_events([
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "test",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        },
        "[DONE]",
    ])


@pytest.mark.parametrize("streaming", [False, True])
async def test_codex_uses_native_stream_and_preserves_options(service: FakeService, streaming: bool) -> None:
    reply(service, '{"ok":true}')
    history = InMemoryHistory()
    model = republic.get_model(
        "codex:test",
        auth=CodexAuth({"access_token": "provided-token"}, account_id="provided-account"),
        http_client=service.client(),
        history=history,
    )
    options: republic.ChatOptions = {
        "reasoning_effort": "low",
        "extra_body": {"instructions": "Be brief", "include": ["other"]},
    }

    if streaming:
        async with model.stream("hello", output_schema=dict[str, bool], **options) as stream:
            events = [event async for event in stream]
        assert isinstance(events[-1], republic.events.Completed)
        response = stream.response
    else:
        response = await model.chat("hello", output_schema=dict[str, bool], **options)

    assert response.text == '{"ok":true}'
    assert response.output == {"ok": True}
    assert response.id == "resp_1"
    assert response.model == "test"
    assert response.token_usage == republic.TokenUsage(3, 2)
    assert await history.read() == [republic.user("hello"), response.message]
    assert str(service.requests[0].url) == "https://chatgpt.com/backend-api/codex/responses"
    assert service.requests[0].headers["authorization"] == "Bearer provided-token"
    assert service.requests[0].headers["chatgpt-account-id"] == "provided-account"
    body = service.body()
    assert body["stream"] is True
    assert body["store"] is False
    assert body["instructions"] == "Be brief"
    assert body["include"] == ["other", "reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"


async def test_codex_keeps_reasoning_and_tool_round_trip(service: FakeService) -> None:
    reasoning = {"type": "reasoning", "id": "r_1", "encrypted_content": "opaque", "summary": []}
    service.reply_events([
        {"type": "response.output_item.done", "item": reasoning},
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c_1", "name": "weather", "arguments": '{"city":"Paris"}'},
        },
    ])
    reply(service, "sunny")
    model = republic.get_model("codex:test", http_client=service.client())

    first = await model.chat("weather?")
    result = republic.tool_result(first.tool_calls[0], "sunny")
    second = await model.chat(["weather?", first.message, republic.assistant(tool_results=[result])])

    assert first.tool_calls[0].args == {"city": "Paris"}
    assert second.text == "sunny"
    assert service.body()["input"][1:] == [
        reasoning,
        {"type": "function_call", "call_id": "c_1", "name": "weather", "arguments": '{"city":"Paris"}'},
        {"type": "function_call_output", "call_id": "c_1", "output": "sunny"},
    ]


async def test_codex_rejects_token_limit_instead_of_ignoring_it(service: FakeService) -> None:
    model = republic.get_model("codex:test", http_client=service.client())
    with pytest.raises(republic.UnsupportedFeatureError, match="max_tokens"):
        await model.chat("hello", max_tokens=10)
    assert not service.requests


def test_codex_rejects_other_formats_and_embeddings() -> None:
    with pytest.raises(republic.UnsupportedApiFormatError):
        republic.get_model("codex:test", api_format="chat")
    with pytest.raises(republic.UnsupportedApiFormatError):
        republic.get_embedding_model("codex:test")


def credentials(path: Path, **tokens: object) -> Path:
    path.write_text(
        json.dumps({
            "auth_mode": "chatgpt",
            "unrelated": {"keep": True},
            "tokens": {"access_token": "dummy-access", "account_id": "account-1", "id_token": "dummy-id", **tokens},
        })
    )
    return path


async def test_codex_reads_current_file_and_keeps_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = credentials(tmp_path / "auth.json", expires_at="2099-01-01T00:00:00Z")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    auth = CodexAuth.from_file()
    original = path.read_bytes()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        response = await client.post("https://example.test/responses", json={"input": "hello"}, auth=auth)
        assert response.request.headers["authorization"] == "Bearer dummy-access"
        assert response.request.headers["chatgpt-account-id"] == "account-1"
        assert json.loads(response.request.content) == {"input": "hello"}
        assert path.read_bytes() == original

        credentials(path, access_token="external-login", account_id="account-2")
        response = await client.post("https://example.test/responses", json={}, auth=auth)
        assert response.request.headers["authorization"] == "Bearer external-login"
        assert response.request.headers["chatgpt-account-id"] == "account-2"


@pytest.mark.parametrize("source", ["file", "provided"])
@pytest.mark.parametrize("expiry_source", ["field", "jwt"])
async def test_codex_refreshes_once_and_preserves_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expiry_source: str, source: str
) -> None:
    tokens: dict[str, object] = {"refresh_token": "dummy-refresh"}
    if expiry_source == "field":
        tokens["expires_at"] = 1
    else:
        payload = base64.urlsafe_b64encode(b'{"exp":1}').decode().rstrip("=")
        tokens["access_token"] = f"header.{payload}.signature"
    path = credentials(tmp_path / "auth.json", **tokens)
    requests = []

    def refresh(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"access_token": "refreshed", "refresh_token": "rotated", "expires_in": 3600})

    monkeypatch.setattr(codex, "OAuth2Client", partial(OAuth2Client, transport=httpx2.MockTransport(refresh)))
    original = path.read_bytes()
    provided = json.loads(original)["tokens"]
    auth = CodexAuth.from_file(path) if source == "file" else CodexAuth(provided, account_id="account-1")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        responses = await asyncio.gather(*[
            client.post("https://example.test/responses", json={}, auth=auth) for _ in range(3)
        ])

    assert len(requests) == 1
    assert str(requests[0].url) == "https://auth.openai.com/oauth/token"
    assert parse_qs(requests[0].content.decode()) == {
        "grant_type": ["refresh_token"],
        "refresh_token": ["dummy-refresh"],
        "client_id": ["app_EMoamEEZ73f0CkXaXp7hrann"],
    }
    assert "authorization" not in requests[0].headers
    assert all(response.request.headers["authorization"] == "Bearer refreshed" for response in responses)
    assert auth.token["access_token"] == "refreshed"
    assert auth.token["refresh_token"] == "rotated"
    assert auth.token["id_token"] == "dummy-id"
    assert auth.token["expires_at"] > time.time()
    assert provided == json.loads(original)["tokens"]
    if source == "provided":
        assert path.read_bytes() == original
        return
    saved = json.loads(path.read_text())
    assert saved["unrelated"] == {"keep": True}
    assert saved["tokens"]["id_token"] == "dummy-id"
    assert saved["tokens"]["account_id"] == "account-1"
    assert saved["tokens"]["access_token"] == "refreshed"
    assert saved["tokens"]["refresh_token"] == "rotated"
    assert "expires_at" in saved["tokens"]
    assert "last_refresh" in saved
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


async def test_codex_refresh_failure_preserves_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = credentials(tmp_path / "auth.json", expires_at=1, refresh_token="private-refresh")
    original = path.read_bytes()
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(400, json={"error": "invalid_grant", "error_description": "private-refresh"})
    )
    monkeypatch.setattr(codex, "OAuth2Client", partial(OAuth2Client, transport=transport))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: pytest.fail("must not send inference"))
    ) as client:
        with pytest.raises(republic.AuthenticationError) as error:
            await client.post("https://example.test", json={}, auth=CodexAuth.from_file(path))

    assert "private-refresh" not in str(error.value)
    assert error.value.__suppress_context__
    assert path.read_bytes() == original


@pytest.mark.parametrize("contents", [None, "not-json", "[]", '{"tokens":{}}'])
async def test_codex_bad_credentials_fail_before_request(tmp_path: Path, contents: str | None) -> None:
    path = tmp_path / "auth.json"
    if contents is not None:
        path.write_text(contents)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: pytest.fail("must not send inference"))
    ) as client:
        with pytest.raises(republic.AuthenticationError):
            await client.post("https://example.test", json={}, auth=CodexAuth.from_file(path))


def test_codex_auth_also_signs_sync_requests(tmp_path: Path) -> None:
    path = credentials(tmp_path / "auth.json", expires_at=int(time.time()) + 3600)
    with httpx2.Client(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        response = client.post("https://example.test", json={}, auth=CodexAuth.from_file(path))
    assert response.request.headers["authorization"] == "Bearer dummy-access"


def test_codex_does_not_restart_a_persisted_relative_expiry(tmp_path: Path) -> None:
    # A saved expires_in has no meaning without the time the token was issued.
    path = credentials(tmp_path / "auth.json", expires_in=1)
    original = path.read_bytes()
    with httpx2.Client(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        response = client.post("https://example.test", json={}, auth=CodexAuth.from_file(path))
    assert response.request.headers["authorization"] == "Bearer dummy-access"
    assert path.read_bytes() == original


@pytest.mark.parametrize("expiry", [{}, {"expires_in": 3600}])
def test_codex_provided_token_does_not_use_local_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expiry: dict[str, int]
) -> None:
    path = tmp_path / "auth.json"
    path.write_text("invalid local credentials")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    provided = {"access_token": "provided-token", **expiry}
    original = provided.copy()
    auth = CodexAuth(provided, account_id="provided-account")
    with httpx2.Client(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        response = client.post("https://example.test", json={}, auth=auth)

    assert response.request.headers["authorization"] == "Bearer provided-token"
    assert response.request.headers["chatgpt-account-id"] == "provided-account"
    assert provided == original
    assert path.read_text() == "invalid local credentials"
    if expiry:
        assert auth.token["expires_at"] > time.time()
    else:
        assert auth.token["expires_at"] is None


async def test_codex_provided_token_refresh_failure_keeps_current_token(monkeypatch: pytest.MonkeyPatch) -> None:
    provided = {"access_token": "provided-token", "refresh_token": "private-refresh", "expires_at": 1}
    auth = CodexAuth(provided, account_id="provided-account")
    original = dict(auth.token)
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(400, json={"error": "invalid_grant", "error_description": "private-refresh"})
    )
    monkeypatch.setattr(codex, "OAuth2Client", partial(OAuth2Client, transport=transport))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: pytest.fail("must not send inference"))
    ) as client:
        with pytest.raises(republic.AuthenticationError) as error:
            await client.post("https://example.test", json={}, auth=auth)

    assert "private-refresh" not in str(error.value)
    assert auth.token == original
    assert provided == {"access_token": "provided-token", "refresh_token": "private-refresh", "expires_at": 1}


async def test_codex_expired_provided_token_requires_refresh_token() -> None:
    auth = CodexAuth({"access_token": "provided-token", "expires_at": 1}, account_id="provided-account")
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: pytest.fail("must not send inference"))
    ) as client:
        with pytest.raises(republic.AuthenticationError, match="no refresh token"):
            await client.post("https://example.test", json={}, auth=auth)
