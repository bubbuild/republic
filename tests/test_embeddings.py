"""Real official-client /embeddings fixtures, with no live requests or batching."""

import asyncio
import base64
import json
import struct

import httpx
import openai
import pytest
from pydantic import ValidationError

from republic import EmbeddingRequest, EmbeddingResponse, ProviderError, UnsupportedRequestError, embed
from republic.providers.openai_embeddings import OpenAIEmbeddings
from tests.http_fixtures import Bytes, Transport


def payload(**extra):
    return {
        "object": "list",
        "model": "resolved-embedding-model",
        "request_label": "retained",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [2.0, 3.0], "tag": "second"},
            {"object": "embedding", "index": 0, "embedding": [0.5, -0.5]},
        ],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
        **extra,
    }


def request(**extra):
    return EmbeddingRequest(model="embed-fixture", inputs=["first", "second"], **extra)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["float", "base64"])
async def test_order_indexes_dimensions_usage_and_round_trip(encoding):
    raw = payload()
    if encoding == "base64":
        for item in raw["data"]:
            item["embedding"] = base64.b64encode(struct.pack("<2f", *item["embedding"])).decode()
    transport = Transport([httpx.Response(200, json=raw)])
    async with openai.AsyncOpenAI(
        api_key="synthetic",
        base_url="https://compatible.test/v1",
        max_retries=3,
        default_headers={"x-caller": "kept"},
        default_query={"route": "kept"},
        http_client=httpx.AsyncClient(transport=transport, follow_redirects=True),
    ) as client:
        req = request(
            dimensions=2,
            provider_options={
                "encoding_format": encoding,
                "user": "caller",
                "extra_body": {"vendor_flag": 1},
                "extra_headers": {"x-call": "yes"},
            },
        )
        before = req.model_dump_json()
        async with OpenAIEmbeddings(client=client, max_retries=0) as provider:
            result = await embed(provider, req)
        assert not client.is_closed() and client.max_retries == 3
        assert req.model_dump_json() == before and len(transport.requests) == 1
        sent = transport.requests[0]
        assert str(sent.url) == "https://compatible.test/v1/embeddings?route=kept"
        assert sent.headers["x-caller"] == "kept" and sent.headers["x-call"] == "yes"
        assert json.loads(sent.content) == {
            "model": req.model,
            "input": req.inputs,
            "dimensions": 2,
            "encoding_format": encoding,
            "user": "caller",
            "vendor_flag": 1,
        }
        assert [item.index for item in result.embeddings] == [0, 1]
        assert [item.vector for item in result.embeddings] == [[0.5, -0.5], [2.0, 3.0]]
        assert result.embeddings[1].provider_metadata == {"openai": {"object": "embedding", "tag": "second"}}
        assert result.provider_metadata == {"openai": {"object": "list", "request_label": "retained"}}
        assert result.response_model == "resolved-embedding-model"
        assert result.usage is not None
        assert result.usage.input_tokens == result.usage.total_tokens == 5
        assert result.usage.output_tokens == 0 and result.usage.raw == raw["usage"]
        assert EmbeddingResponse.model_validate_json(result.model_dump_json()) == result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 0, "embedding": [1]}],
        [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"index": 0, "embedding": [1]}, {"index": 2, "embedding": [2]}],
        [{"index": True, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"index": 0, "embedding": []}, {"index": 1, "embedding": [2]}],
        [{"index": 0, "embedding": ["secret"]}, {"index": 1, "embedding": [2]}],
        [{"index": 0, "embedding": [1, 2]}, {"index": 1, "embedding": [2]}],
        [{"index": 0, "embedding": "!!!!"}, {"index": 1, "embedding": [2]}],
        [{"index": 0, "embedding": "YQ=="}, {"index": 1, "embedding": [2]}],
    ],
)
async def test_bad_alignment_or_vectors_are_errors(data):
    transport = Transport([httpx.Response(200, json=payload(data=data))])
    async with (
        openai.AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(transport=transport)) as client,
        OpenAIEmbeddings(client=client) as provider,
    ):
        with pytest.raises(ProviderError, match="invalid_response"):
            await embed(provider, request())
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_dimensions_are_checked_and_missing_usage_stays_unknown():
    transport = Transport([httpx.Response(200, json=payload(usage={})), httpx.Response(200, json=payload(usage=None))])
    async with (
        openai.AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(transport=transport)) as client,
        OpenAIEmbeddings(client=client) as provider,
    ):
        result = await embed(provider, request())
        assert result.usage is not None
        assert result.usage.input_tokens is None and result.usage.total_tokens is None
        with pytest.raises(ProviderError):
            await embed(provider, request(dimensions=3))


@pytest.mark.parametrize("extra", [{"inputs": []}, {"inputs": [""]}, {"dimensions": 0}, {"dimensions": True}])
def test_invalid_inputs_are_rejected(extra):
    with pytest.raises(ValidationError):
        EmbeddingRequest.model_validate({"model": "m", "inputs": ["valid"], **extra})
    assert EmbeddingRequest.model_validate({"model": "m", "inputs": "one"}).inputs == ["one"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"model": "x"},
        {"dimensions": 1},
        {"encoding_format": "unknown"},
        {"extra_body": {"input": ["x"]}},
        {"extra_headers": {"x": 3}},
    ],
)
async def test_conflicting_native_options_do_not_send(options):
    transport = Transport([])
    async with (
        openai.AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(transport=transport)) as client,
        OpenAIEmbeddings(client=client) as provider,
    ):
        with pytest.raises(UnsupportedRequestError):
            await embed(provider, request(provider_options=options))
    assert not transport.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [0, 1])
async def test_owned_client_errors_and_explicit_retry(monkeypatch, retry):
    transport = Transport([
        httpx.Response(429, json={"error": {"message": "rate"}}, headers={"retry-after-ms": "1"}),
        httpx.Response(200, json=payload()),
    ])
    http = httpx.AsyncClient(transport=transport)
    real = openai.AsyncOpenAI
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: real(**kwargs, http_client=http))
    async with OpenAIEmbeddings(api_key="fixture", base_url="https://owned.test", max_retries=retry) as provider:
        if retry:
            await embed(provider, request())
        else:
            with pytest.raises(ProviderError) as caught:
                await embed(provider, request())
            assert isinstance(caught.value.__cause__, openai.RateLimitError)
        assert len(transport.requests) == retry + 1
    assert http.is_closed


@pytest.mark.asyncio
async def test_cancel_releases_response_without_closing_borrowed_client():
    body = Bytes([b'{"data":'], wait=True)
    transport = Transport([httpx.Response(200, headers={"content-type": "application/json"}, stream=body)])
    async with openai.AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(transport=transport)) as client:
        async with OpenAIEmbeddings(client=client) as provider:
            task = asyncio.create_task(embed(provider, request()))
            await body.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert body.closed == 1 and not client.is_closed()
        with pytest.raises(ProviderError, match="closed"):
            await embed(provider, request())
    assert len(transport.requests) == 1
