from __future__ import annotations

import os
from typing import TYPE_CHECKING, Unpack

from republic.auth import Auth, HeaderAuth

from .openai import OpenAI

if TYPE_CHECKING:
    from republic._registry import ProviderOptions


class AzureOpenAI(OpenAI):
    """Azure OpenAI through its v1 API. Models are named by deployment.

    Give the resource name, read from ``{env_prefix}_RESOURCE`` when omitted,
    or a full ``api_base`` such as ``https://RESOURCE.openai.azure.com/openai/v1``.
    API keys use the ``api-key`` header; pass ``auth=`` for Microsoft Entra ID tokens.
    """

    name = "azure-openai"
    DEFAULT_API_BASE = ""

    def __init__(self, *, resource: str | None = None, **options: Unpack[ProviderOptions]) -> None:
        if resource and not options.get("api_base"):
            options["api_base"] = _resource_api_base(resource)
        super().__init__(**options)
        env_prefix = options.get("env_prefix") or "REPUBLIC_AZURE_OPENAI"
        if not self.api_base and (resource := os.getenv(f"{env_prefix}_RESOURCE")):
            self.api_base = _resource_api_base(resource)
        if not self.api_base:
            raise ValueError(
                f"Azure OpenAI needs a resource name or api_base; set {env_prefix}_RESOURCE or {env_prefix}_API_BASE"
            )

    def _api_key_auth(self, api_key: str) -> Auth:
        return HeaderAuth("api-key", api_key)


def _resource_api_base(resource: str) -> str:
    return f"https://{resource}.openai.azure.com/openai/v1"
