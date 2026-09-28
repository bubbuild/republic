"""Direct Cohere v2 wire fixtures; no Gateway, SDK mock or live network."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from republic import ProviderError, RerankRequest, RerankResponse, UnsupportedRequestError, rerank
from republic.providers.cohere import CohereRerank
from tests.http_fixtures import Bytes, Transport


def request(**overrides):
    return RerankRequest.model_validate({
        "model": "rerank-fixture",
        "query": "tokyo",
        "documents": ["pie", "finance", "Japan"],
        **overrides,
    })


def result(**overrides):
    return {
        "results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}],
        "id": "rank-1",
        "meta": {"billed_units": {"search_units": 1}, "api_version": {"version": "2"}},
        **overrides,
    }


@pytest.mark.asyncio
async def test_wire_indexes_score_billing_identity_and_native_options():
    raw = result()
    raw["results"][0]["vendor"] = "preserved"
    transport = Transport([httpx.Response(200, json=raw)])
    async with (
        httpx.AsyncClient(
            transport=transport,
            base_url="https://compatible.test/v2",
            headers={"x-caller": "keep"},
            params={"route": "fixture"},
            follow_redirects=True,
            timeout=19,
        ) as client,
        CohereRerank(api_key="synthetic", client=client, headers={"X-Call": "default"}) as provider,
    ):
        req = request(
            top_n=2,
            provider_options={
                "max_tokens_per_doc": 512,
                "priority": 7,
                "extra_body": {"vendor_option": 1},
                "extra_headers": {"x-call": "per-call"},
            },
        )
        before = req.model_dump_json()
        response = await rerank(provider, req)
        assert req.model_dump_json() == before
        assert [r.index for r in response.ranking] == [2, 0]
        assert [r.score for r in response.ranking] == [0.9, 0.4]
        assert response.ranking[0].provider_metadata == {"cohere": {"vendor": "preserved"}}
        assert response.provider_metadata == {"cohere": {"meta": raw["meta"]}}
        assert response.response_id == "rank-1" and response.usage is None
        assert RerankResponse.model_validate_json(response.model_dump_json()) == response
        assert len(transport.requests) == 1
        sent = transport.requests[0]
        assert str(sent.url) == "https://compatible.test/v2/rerank?route=fixture"
        assert sent.headers["authorization"] == "Bearer synthetic"
        assert sent.headers["x-caller"] == "keep" and sent.headers["x-call"] == "per-call"
        assert json.loads(sent.content) == {
            "model": req.model,
            "query": "tokyo",
            "documents": req.documents,
            "top_n": 2,
            "max_tokens_per_doc": 512,
            "priority": 7,
            "vendor_option": 1,
        }
        assert sent.extensions["timeout"]["read"] == 19
        await provider.aclose()
        assert not client.is_closed and client.follow_redirects and "authorization" not in client.headers
        with pytest.raises(ProviderError, match="closed"):
            await provider.rerank(req)


@pytest.mark.asyncio
async def test_empty_input_defaults_and_explicit_timeout_base_url():
    transport = Transport([httpx.Response(200, json={"results": []})])
    async with (
        httpx.AsyncClient(transport=transport, base_url="https://unused.test") as client,
        CohereRerank(client=client, base_url="https://explicit.test/v2", timeout=4) as provider,
    ):
        assert (await rerank(provider, request(documents=[]))).ranking == []
        assert not transport.requests
        response = await rerank(provider, request(provider_options={"timeout": 2}))
        assert response.response_id is None and response.usage is None
        sent = transport.requests[0]
        assert str(sent.url) == "https://explicit.test/v2/rerank"
        assert sent.extensions["timeout"]["read"] == 2
        assert "top_n" not in json.loads(sent.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_http_errors_do_not_retry(status):
    body = Bytes([b'{"message":"fixture"}'])
    transport = Transport([httpx.Response(status, stream=body, headers={"x-request-id": "req-fixture"})])
    async with httpx.AsyncClient(transport=transport) as client, CohereRerank(client=client) as provider:
        with pytest.raises(ProviderError) as caught:
            await rerank(provider, request())
        assert caught.value.status_code == status and caught.value.request_id == "req-fixture"
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
        assert len(transport.requests) == body.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        {},
        result(results=[{"index": -1, "relevance_score": 0.5}]),
        result(results=[{"index": 3, "relevance_score": 0.5}]),
        result(results=[{"index": True, "relevance_score": 0.5}]),
        result(results=[{"index": 0, "relevance_score": "0.5"}]),
        result(results=[{"index": 0, "relevance_score": 0.5}] * 2),
        result(),
    ],
)
async def test_invalid_rankings_and_top_n(raw):
    transport = Transport([httpx.Response(200, json=raw)])
    async with httpx.AsyncClient(transport=transport) as client, CohereRerank(client=client) as provider:
        with pytest.raises(ProviderError, match="invalid_response"):
            await rerank(provider, request(top_n=1))


@pytest.mark.parametrize("changes", [{"documents": ["text", {"title": "mixed"}]}, {"top_n": 0}, {"top_n": True}])
def test_bad_request_data(changes):
    with pytest.raises(ValidationError):
        request(**changes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"documents": [{"title": "not serialized"}]},
        {"provider_options": {"model": "x"}},
        {"provider_options": {"extra_body": {"documents": []}}},
        {"provider_options": {"extra_headers": {"x": 3}}},
        {"provider_options": {"max_tokens_per_doc": True}},
        {"provider_options": {"priority": 1000}},
        {"provider_options": {"timeout": False}},
    ],
)
async def test_unrepresentable_documents_and_conflicts_never_sent(changes):
    transport = Transport([])
    async with httpx.AsyncClient(transport=transport) as client, CohereRerank(client=client) as provider:
        with pytest.raises(UnsupportedRequestError):
            await rerank(provider, request(**changes))
        assert not transport.requests


@pytest.mark.asyncio
async def test_cancel_and_owned_client_cleanup(monkeypatch):
    body = Bytes([b'{"results":'], wait=True)
    transport = Transport([httpx.Response(200, stream=body)])
    http = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: http)
    async with CohereRerank(api_key="synthetic") as provider:
        task = asyncio.create_task(rerank(provider, request()))
        await body.waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed == 1 and not http.is_closed
        assert str(transport.requests[0].url) == "https://api.cohere.com/v2/rerank"
    assert http.is_closed and transport.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code", [(httpx.ConnectError("fixture"), "connection_error"), (httpx.ReadTimeout("fixture"), "timeout")]
)
async def test_transport_errors(error, code):
    transport = Transport([error])
    async with httpx.AsyncClient(transport=transport) as client, CohereRerank(client=client) as provider:
        with pytest.raises(ProviderError) as caught:
            await rerank(provider, request())
        assert caught.value.code == code and caught.value.__cause__ is error
        assert not client.is_closed and len(transport.requests) == 1
