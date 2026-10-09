from __future__ import annotations

import asyncio
import base64
import hashlib
import traceback
from functools import partial
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest

import republic
from republic.providers import OpenRouterAuth, openrouter
from tests.conftest import FakeService


@pytest.fixture
def login_service(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> FakeService:
    monkeypatch.setattr(
        openrouter.httpx2, "AsyncClient", partial(httpx2.AsyncClient, transport=httpx2.MockTransport(service._handle))
    )
    return service


async def test_login_uses_pkce_and_returns_reusable_auth(login_service: FakeService) -> None:
    service = login_service
    service.reply_json({"key": "new-api-key"})
    authorization_urls = []

    async def authorize(url: str) -> str:
        authorization_urls.append(url)
        return "  private-code\n"

    auth = await OpenRouterAuth.login(on_authorize=authorize)

    url = urlsplit(authorization_urls[0])
    assert f"{url.scheme}://{url.netloc}{url.path}" == "https://openrouter.ai/auth"
    query = parse_qs(url.query)
    body = service.body()
    verifier = body["code_verifier"]
    assert 43 <= len(verifier) <= 128
    assert query == {
        "code_challenge": [base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()],
        "code_challenge_method": ["S256"],
    }
    request = service.requests[0]
    assert request.method == "POST"
    assert request.url == "https://openrouter.ai/api/v1/auth/keys"
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    assert body == {"code": "private-code", "code_verifier": verifier, "code_challenge_method": "S256"}
    for selected in [auth, OpenRouterAuth(auth.api_key)]:
        service.reply_json({"choices": [{"message": {"content": "hello"}}]})
        model = republic.get_model("openrouter:test", auth=selected, api_format="chat")
        response = await model.chat("Hi")
        assert response.text == "hello"
        assert service.requests[-1].url == "https://openrouter.ai/api/v1/chat/completions"
        assert service.requests[-1].headers["authorization"] == "Bearer new-api-key"


async def test_logins_use_distinct_challenges(login_service: FakeService) -> None:
    urls = []

    async def authorize(url: str) -> str:
        urls.append(url)
        return "code"

    for _ in range(2):
        login_service.reply_json({"key": "api-key"})
        await OpenRouterAuth.login(on_authorize=authorize)
    assert urls[0] != urls[1]
    assert login_service.body(0)["code_verifier"] != login_service.body(1)["code_verifier"]


@pytest.mark.parametrize("failure", ["http", "json", "null", "missing", "empty", "wrong_type", "network"])
async def test_exchange_errors_do_not_expose_credentials(
    login_service: FakeService, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "http":
        login_service.reply_json({"error": "private-code"}, status_code=403)
    elif failure == "json":
        login_service.reply_bytes(b"private-code", content_type="text/plain")
    elif failure == "network":

        def handle(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("private-code", request=request)

        monkeypatch.setattr(
            openrouter.httpx2, "AsyncClient", partial(httpx2.AsyncClient, transport=httpx2.MockTransport(handle))
        )
    else:
        login_service.reply_json(
            {"null": None, "missing": {}, "empty": {"key": ""}, "wrong_type": {"key": 42}}[failure]
        )

    async def authorize(url: str) -> str:
        return "private-code"

    with pytest.raises(republic.errors.AuthenticationError) as caught:
        await OpenRouterAuth.login(on_authorize=authorize)
    assert "private-code" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("cancelled", [False, True])
async def test_no_exchange_without_authorization(login_service: FakeService, cancelled: bool) -> None:
    async def authorize(url: str) -> str:
        if cancelled:
            raise asyncio.CancelledError
        return " "

    with pytest.raises(asyncio.CancelledError if cancelled else republic.errors.AuthenticationError):
        await OpenRouterAuth.login(on_authorize=authorize)
    assert login_service.requests == []


def test_empty_key_is_rejected() -> None:
    with pytest.raises(republic.errors.AuthenticationError):
        OpenRouterAuth("")
