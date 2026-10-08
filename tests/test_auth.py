from __future__ import annotations

import republic
from republic.auth import OAuth2Auth
from tests.conftest import FakeService


async def test_authlib_auth_works_with_existing_provider(service: FakeService) -> None:
    service.reply_json({"output": []})
    model = republic.get_model(
        "openai:test", api_key="ignored", auth=OAuth2Auth({"access_token": "oauth-token"}), http_client=service.client()
    )

    await model.chat("hello")

    assert service.requests[0].headers["authorization"] == "Bearer oauth-token"
