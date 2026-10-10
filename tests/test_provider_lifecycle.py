from __future__ import annotations

import asyncio
from typing import Any

import httpx2
import pytest

import republic
from republic.providers import base
from tests.conftest import FakeService


@pytest.fixture(params=["embedding", "decision"])
def other_model(request: pytest.FixtureRequest) -> republic.EmbeddingModel | republic.DecisionModel:
    if request.param == "embedding":
        return republic.get_embedding_model("openai:test")
    return republic.get_decision_model("typesafe:test", api_key="key")


async def _call_other_model(model: republic.EmbeddingModel | republic.DecisionModel, service: FakeService) -> None:
    if isinstance(model, republic.EmbeddingModel):
        service.reply_json({"data": [{"index": 0, "embedding": [0.1]}]})
        assert (await model.embed("Hi")).vector == [0.1]
    else:
        service.reply_json({"answers": {}, "usage": {}})
        assert (await model.decide("Hi", questions={})).answers == {}


async def test_other_model_context_reuses_client_and_closes_provider(
    other_model: republic.EmbeddingModel | republic.DecisionModel,
    clients: list[httpx2.AsyncClient],
    service: FakeService,
) -> None:
    async with other_model as entered:
        assert entered is other_model
        assert clients == []
        await _call_other_model(other_model, service)
        await _call_other_model(other_model, service)
        assert len(clients) == 1
        assert not clients[0].is_closed
    assert other_model.provider.is_closed
    assert clients[0].is_closed
    with pytest.raises(RuntimeError, match="Provider is closed"):
        async with other_model:
            pytest.fail("A model with a closed provider must not be entered")


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_other_model_context_closes_on_failure(
    other_model: republic.EmbeddingModel | republic.DecisionModel,
    failure: type[BaseException],
    clients: list[httpx2.AsyncClient],
    service: FakeService,
) -> None:
    with pytest.raises(failure):
        async with other_model:
            await _call_other_model(other_model, service)
            raise failure
    assert other_model.provider.is_closed
    assert clients[0].is_closed


async def test_unused_other_model_context_does_not_create_client(
    other_model: republic.EmbeddingModel | republic.DecisionModel, clients: list[httpx2.AsyncClient]
) -> None:
    async with other_model:
        assert clients == []
    assert other_model.provider.is_closed
    assert clients == []


async def test_other_model_context_leaves_external_client_open(
    other_model: republic.EmbeddingModel | republic.DecisionModel, service: FakeService
) -> None:
    async with service.client() as client:
        provider = republic.get_provider(other_model.provider.name, api_key="key", http_client=client)
        model = (
            provider.get_embedding_model("test")
            if isinstance(other_model, republic.EmbeddingModel)
            else provider.get_decision_model("test")
        )
        async with model:
            await _call_other_model(model, service)
        assert provider.is_closed
        assert not client.is_closed
    assert client.is_closed


async def test_nested_contexts_for_same_model_keep_provider_open(
    other_model: republic.EmbeddingModel | republic.DecisionModel,
    clients: list[httpx2.AsyncClient],
    service: FakeService,
) -> None:
    async with other_model:
        async with other_model:
            await _call_other_model(other_model, service)
        assert not other_model.provider.is_closed
        assert not clients[0].is_closed
        await _call_other_model(other_model, service)
    assert other_model.provider.is_closed
    assert clients[0].is_closed


async def test_provider_context_keeps_other_model_client_open_after_model_exit(
    other_model: republic.EmbeddingModel | republic.DecisionModel,
    clients: list[httpx2.AsyncClient],
    service: FakeService,
) -> None:
    async with other_model.provider as provider:
        async with other_model:
            await _call_other_model(other_model, service)
        assert not provider.is_closed
        assert not clients[0].is_closed
        await _call_other_model(other_model, service)
    assert provider.is_closed
    assert clients[0].is_closed


async def test_provider_context_allows_sequential_chat_model_contexts(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    async with republic.get_provider("openai") as provider:
        for name in ["first", "second"]:
            service.reply_json({"output": []})
            async with provider.get_model(name) as model:
                await model.chat("Hi")
            assert not provider.is_closed
            assert not clients[0].is_closed
        service.reply_json({"data": [{"id": "first"}]})
        assert (await provider.list_models())[0].id == "first"
        assert len(clients) == 1
    assert provider.is_closed
    assert clients[0].is_closed


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_nested_provider_context_failure_keeps_outer_context_open(
    failure: type[BaseException], clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    async with republic.get_provider("openai") as provider:
        model = provider.get_model("test")
        with pytest.raises(failure):
            async with provider:
                service.reply_json({"output": []})
                await model.chat("Hi")
                raise failure
        assert not provider.is_closed
        assert not clients[0].is_closed
        service.reply_json({"output": []})
        await model.chat("Still open")
    assert provider.is_closed
    assert clients[0].is_closed


async def test_provider_context_exit_keeps_outer_model_context_open(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    async with republic.get_model("openai:test") as model:
        async with model.provider as provider:
            service.reply_json({"output": []})
            await model.chat("Hi")
        assert not provider.is_closed
        assert not clients[0].is_closed
        service.reply_json({"output": []})
        await model.chat("Still open")
    assert provider.is_closed
    assert clients[0].is_closed


async def test_explicit_close_overrides_nested_provider_and_model_contexts(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    async with republic.get_provider("openai") as provider, provider, provider.get_model("test") as model:
        service.reply_json({"output": []})
        await model.chat("Hi")
        await provider.close()
        assert provider.is_closed
        assert clients[0].is_closed
        with pytest.raises(RuntimeError, match="Provider is closed"):
            await model.chat("Again")
    await provider.close()
    assert provider.is_closed


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_other_model_failure_keeps_sibling_context_open(
    other_model: republic.EmbeddingModel | republic.DecisionModel,
    failure: type[BaseException],
    clients: list[httpx2.AsyncClient],
    service: FakeService,
) -> None:
    provider = other_model.provider
    sibling = (
        provider.get_embedding_model("sibling")
        if isinstance(other_model, republic.EmbeddingModel)
        else provider.get_decision_model("sibling")
    )
    async with sibling:
        with pytest.raises(failure):
            async with other_model:
                await _call_other_model(other_model, service)
                raise failure
        assert not provider.is_closed
        assert not clients[0].is_closed
        await _call_other_model(sibling, service)
    assert provider.is_closed
    assert clients[0].is_closed


async def test_chat_and_embedding_contexts_share_provider_until_last_exit(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    provider = republic.get_provider("openai")
    async with provider.get_model("chat") as chat:
        async with provider.get_embedding_model("embedding") as embeddings:
            await _call_other_model(embeddings, service)
        assert not provider.is_closed
        assert not clients[0].is_closed
        # Re-enter an exited model while another model keeps its Provider open.
        async with embeddings:
            await _call_other_model(embeddings, service)
        service.reply_json({"output": []})
        await chat.chat("Still open")
    assert len(clients) == 1
    assert clients[0].is_closed
    assert provider.is_closed


async def test_concurrent_model_contexts_close_only_after_both_exit(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    provider = republic.get_provider("openai")
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def hold() -> None:
        async with provider.get_model("first"):
            entered.set()
            await leave.wait()

    task = asyncio.create_task(hold())
    try:
        await entered.wait()
        async with provider.get_model("second") as model:
            service.reply_json({"output": []})
            await model.chat("Hi")
        assert not provider.is_closed
        assert not clients[0].is_closed
    finally:
        leave.set()
        await task
    assert provider.is_closed
    assert clients[0].is_closed


async def test_explicit_close_overrides_open_model_contexts(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    provider = republic.get_provider("openai")
    async with provider.get_model("first") as first, provider.get_model("second"):
        service.reply_json({"output": []})
        await first.chat("Hi")
        await provider.close()
        assert provider.is_closed
        assert clients[0].is_closed
        with pytest.raises(RuntimeError, match="Provider is closed"):
            await first.chat("Again")
    assert provider.is_closed


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> list[httpx2.AsyncClient]:
    client_type = httpx2.AsyncClient
    created: list[httpx2.AsyncClient] = []

    def create(**options: Any) -> httpx2.AsyncClient:
        options.setdefault("transport", httpx2.MockTransport(service._handle))
        client = client_type(**options)
        created.append(client)
        return client

    monkeypatch.setattr(base.httpx2, "AsyncClient", create)
    return created


async def test_lazy_client_shared_by_models_listing_and_streams(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    provider = republic.get_provider("openai", timeout=17)
    first = provider.get_model("first")
    second = provider.get_model("second")
    embeddings = provider.get_embedding_model("embedding")
    assert clients == []

    async with provider as entered:
        assert entered is provider
        assert clients == []
        service.reply_json({"output": []})
        await first.chat("Hi")
        assert len(clients) == 1
        assert clients[0].timeout.read == 17
        assert not clients[0].is_closed

        service.reply_json({"output": []})
        await second.chat("Hi")
        service.reply_json({"data": [{"index": 0, "embedding": [0.1]}]})
        assert (await embeddings.embed("Hi")).vector == [0.1]
        service.reply_json({"data": [{"id": "first"}]})
        assert (await provider.list_models())[0].id == "first"
        service.reply_events([{"type": "response.output_text.delta", "delta": "hello"}, "[DONE]"])
        async with first.stream("Hi") as stream:
            [event async for event in stream]
        assert stream.text == "hello"
        assert not clients[0].is_closed
        assert len(clients) == 1
        assert not provider.is_closed

    assert clients[0].is_closed
    assert provider.is_closed


async def test_model_context_reuses_client_and_closes_shared_provider(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    model = republic.get_model("openai:first")
    sibling = model.provider.get_model("second")
    async with model as entered:
        assert entered is model
        assert clients == []
        service.reply_json({"output": []})
        await model.chat("Hi")
        service.reply_json({"output": []})
        await sibling.chat("Hi")
        service.reply_events([{"type": "response.output_text.delta", "delta": "hello"}, "[DONE]"])
        async with model.stream("Hi") as stream:
            [event async for event in stream]
        assert stream.text == "hello"
        assert len(clients) == 1
        assert not clients[0].is_closed
    assert model.provider.is_closed
    assert clients[0].is_closed
    with pytest.raises(RuntimeError, match="Provider is closed"):
        await sibling.chat("Again")
    with pytest.raises(RuntimeError, match="Provider is closed"):
        async with model:
            pytest.fail("A model with a closed provider must not be entered")


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_model_context_closes_client_when_body_fails(
    failure: type[BaseException], clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    service.reply_json({"output": []})
    with pytest.raises(failure):
        async with republic.get_model("openai:test") as model:
            await model.chat("Hi")
            raise failure
    assert model.provider.is_closed
    assert clients[0].is_closed


async def test_unused_model_context_does_not_create_client(clients: list[httpx2.AsyncClient]) -> None:
    async with republic.get_model("openai:test") as model:
        assert clients == []
    assert model.provider.is_closed
    assert clients == []


async def test_model_context_leaves_external_client_open(service: FakeService) -> None:
    async with service.client() as client:
        service.reply_json({"output": []})
        async with republic.get_model("openai:test", http_client=client) as model:
            await model.chat("Hi")
        assert model.provider.is_closed
        assert not client.is_closed
    assert client.is_closed


async def test_concurrent_first_requests_create_one_client(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    async with republic.get_provider("openai") as provider:
        for _ in range(8):
            service.reply_json({"output": []})
        await asyncio.gather(*(provider.get_model(str(index)).chat("Hi") for index in range(8)))
        assert len(clients) == 1
        assert len(service.requests) == 8
        assert not clients[0].is_closed
    assert clients[0].is_closed


async def test_manual_close_is_idempotent_and_prevents_reopening(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    service.reply_json({"output": []})
    model = republic.get_model("openai:test")
    await model.chat("Hi")
    await model.provider.close()
    await model.provider.close()

    assert model.provider.is_closed
    assert clients[0].is_closed
    with pytest.raises(RuntimeError, match="Provider is closed"):
        await model.chat("Again")
    with pytest.raises(RuntimeError, match="Provider is closed"):
        await model.provider.list_models()
    with pytest.raises(RuntimeError, match="Provider is closed"):
        async with model.provider:
            pytest.fail("A closed provider must not be entered")
    assert len(clients) == 1
    assert len(service.requests) == 1


async def test_close_unused_provider_does_not_create_client(clients: list[httpx2.AsyncClient]) -> None:
    provider = republic.get_provider("openai")
    await provider.close()
    await provider.close()
    assert clients == []
    assert provider.is_closed
    with pytest.raises(RuntimeError, match="Provider is closed"):
        await provider.get_model("test").chat("Hi")
    assert clients == []


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_provider_context_closes_client_when_body_fails(
    failure: type[BaseException], clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    service.reply_json({"output": []})
    provider = republic.get_provider("openai")
    with pytest.raises(failure):
        async with provider:
            await provider.get_model("test").chat("Hi")
            raise failure
    assert provider.is_closed
    assert clients[0].is_closed


async def test_request_failure_leaves_owned_client_available_until_provider_closes(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    async with republic.get_provider("openai") as provider:
        model = provider.get_model("test")
        service.reply_json({"error": "bad request"}, status_code=400)
        with pytest.raises(republic.errors.APIStatusError):
            await model.chat("Hi")
        assert not clients[0].is_closed
        service.reply_json({"output": []})
        await model.chat("Try again")
        assert len(clients) == 1
    assert clients[0].is_closed


async def test_early_stream_exit_closes_response_but_keeps_client(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    service.reply_events([{"type": "response.output_text.delta", "delta": "partial"}])
    response = service._responses[0]
    async with republic.get_provider("openai") as provider:
        model = provider.get_model("test")
        async with model.stream("Hi") as stream:
            async for event in stream:
                assert isinstance(event, republic.events.TextDelta)
                break
        assert response.is_closed
        assert not clients[0].is_closed
        service.reply_json({"output": []})
        await model.chat("Again")
        assert len(clients) == 1
    assert clients[0].is_closed


async def test_external_client_survives_provider_close_and_can_be_shared(service: FakeService) -> None:
    async with service.client() as client:
        async with republic.get_provider("openai", http_client=client, timeout=17) as first:
            service.reply_json({"output": []})
            await first.get_model("one").chat("Hi")
        await first.close()
        assert first.is_closed
        assert not client.is_closed
        assert client.timeout.read != 17
        with pytest.raises(RuntimeError, match="Provider is closed"):
            await first.get_model("one").chat("Again")
        async with republic.get_provider("openai", http_client=client) as second:
            service.reply_json({"output": []})
            await second.get_model("two").chat("Hi")
        assert not client.is_closed
    assert client.is_closed


async def test_owned_client_rejects_other_event_loop_without_closing(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    service.reply_json({"output": []})
    async with republic.get_provider("openai") as provider:
        model = provider.get_model("test")
        await model.chat("Hi")

        async def elsewhere() -> None:
            with pytest.raises(RuntimeError, match="event loop"):
                await model.chat("Wrong loop")
            with pytest.raises(RuntimeError, match="event loop"):
                await provider.close()
            with pytest.raises(RuntimeError, match="event loop"):
                async with provider:
                    pytest.fail("A used provider must not be entered on another loop")

        await asyncio.to_thread(lambda: asyncio.run(elsewhere()))
        assert not provider.is_closed
        assert not clients[0].is_closed
        service.reply_json({"output": []})
        await model.chat("Original loop")
        assert len(clients) == 1
        assert len(service.requests) == 2
    assert clients[0].is_closed


def test_provider_can_be_constructed_before_event_loop_starts(
    clients: list[httpx2.AsyncClient], service: FakeService
) -> None:
    provider = republic.get_provider("openai")
    assert clients == []
    service.reply_json({"output": []})

    async def use() -> None:
        async with provider:
            await provider.get_model("test").chat("Hi")

    asyncio.run(use())
    assert provider.is_closed
    assert clients[0].is_closed
