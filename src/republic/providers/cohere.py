"""Direct Cohere v2 reranking through HTTPX, with explicit compatible endpoints."""

from typing import Any, Self

import httpx

from republic.errors import ProviderError, UnsupportedRequestError
from republic.reranking import RankedDocument, RerankRequest, RerankResponse

_OPTIONS = {"max_tokens_per_doc", "priority", "extra_body", "extra_headers", "timeout"}
_MANAGED = {"model", "query", "documents", "top_n"}


def _payload(request: RerankRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    if any(not isinstance(document, str) for document in request.documents):
        raise UnsupportedRequestError("documents", "Cohere v2 requires strings; explicitly encode structured documents")
    options = dict(request.provider_options)
    if options.keys() - _OPTIONS:
        raise UnsupportedRequestError("provider_options", "unsupported or managed reranking options")
    extra = options.pop("extra_body", {})
    if not isinstance(extra, dict) or extra.keys() & (_MANAGED | _OPTIONS):
        raise UnsupportedRequestError("extra_body", "cannot override managed or standard fields")
    headers = options.pop("extra_headers", {})
    if not isinstance(headers, dict) or any(not isinstance(value, str) for value in headers.values()):
        raise UnsupportedRequestError("extra_headers", "expected string header values")
    wire: dict[str, Any] = {"headers": headers}
    if "timeout" in options:
        timeout = options.pop("timeout")
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or timeout <= 0:
            raise UnsupportedRequestError("timeout", "expected positive seconds")
        wire["timeout"] = timeout
    _integer_options(options)
    body = {"model": request.model, "query": request.query, "documents": request.documents, **options, **extra}
    if request.top_n is not None:
        body["top_n"] = request.top_n
    return body, wire


def _integer_options(options: dict[str, Any]) -> None:
    for name in ("max_tokens_per_doc", "priority"):
        if name in options:
            value = options[name]
            if (
                type(value) is not int
                or value < (0 if name == "priority" else 1)
                or (name == "priority" and value > 999)
            ):
                raise UnsupportedRequestError(name, "invalid integer option")


def _response(raw: dict[str, Any], request: RerankRequest) -> RerankResponse:
    ranking = [
        RankedDocument(
            index=item["index"],
            score=item["relevance_score"],
            provider_metadata={
                "cohere": {key: value for key, value in item.items() if key not in {"index", "relevance_score"}}
            }
            if item.keys() - {"index", "relevance_score"}
            else None,
        )
        for item in raw["results"]
    ]
    indexes = [item.index for item in ranking]
    if len(set(indexes)) != len(indexes) or any(index >= len(request.documents) for index in indexes):
        raise ValueError("invalid_ranking_indexes")
    if request.top_n is not None and len(ranking) > request.top_n:
        raise ValueError("ranking_exceeds_top_n")
    return RerankResponse(
        ranking=ranking,
        response_id=raw.get("id"),
        # billed_units.search_units is billing data, not input/output token usage.
        provider_metadata={"cohere": {key: value for key, value in raw.items() if key not in {"results", "id"}}},
    )


class CohereRerank:
    """The v2 /rerank wire; owns created clients and borrows injected clients.

    No SDK dependency, retries or automatic chunking. Inject an HTTPX client for
    caller-selected transport/retry/redirect policy; its object is never modified.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        if client is None and api_key is None:
            raise UnsupportedRequestError("client", "api_key or an explicitly configured client is required")
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))
        self._owns_client = client is None
        endpoint = base_url or (
            str(client.base_url) if client is not None and client.base_url.host else "https://api.cohere.com/v2"
        )
        self._url = endpoint.rstrip("/") + "/rerank"
        self._headers = httpx.Headers()
        if api_key is not None:
            self._headers["authorization"] = f"Bearer {api_key}"
        self._headers.update(headers or {})
        self._timeout = timeout
        self._closed = False

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        if self._closed:
            raise ProviderError("closed", provider="cohere", code="closed")
        body, options = _payload(request)
        if not request.documents:
            return RerankResponse(ranking=[])
        headers = httpx.Headers(self._headers)
        headers.update(options.pop("headers"))
        if self._timeout is not None:
            options.setdefault("timeout", self._timeout)
        try:
            async with self._client.stream("POST", self._url, json=body, headers=headers, **options) as response:
                await response.aread()
                response.raise_for_status()
                return _response(response.json(), request)
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                "http_error",
                provider="cohere",
                status_code=exc.response.status_code,
                code="http_error",
                request_id=exc.response.headers.get("x-request-id"),
            ) from exc
        except httpx.HTTPError as exc:
            code = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection_error"
            raise ProviderError(code, provider="cohere", code=code) from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError("invalid_response", provider="cohere", code="invalid_response") from exc

    async def aclose(self) -> None:
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        if self._closed:
            raise ProviderError("closed", provider="cohere", code="closed")
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
