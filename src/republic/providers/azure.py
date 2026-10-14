from __future__ import annotations

import os
from typing import TYPE_CHECKING, Unpack

from republic.auth import Auth, HeaderAuth
from republic.formats import ImagesFormat, MultipartImagesFormat

from .base import _FormatT
from .openai import OpenAI

if TYPE_CHECKING:
    from republic._registry import ProviderOptions


class AzureOpenAI(OpenAI):
    """Azure OpenAI through its v1 API. Models are named by deployment.

    Give the resource name, read from ``{env_prefix}_RESOURCE`` when omitted,
    or a full ``api_base`` such as ``https://RESOURCE.openai.azure.com/openai/v1``.
    API keys use the ``api-key`` header; pass ``auth=`` for Microsoft Entra ID tokens.
    Image edits are sent as multipart form data, which Azure requires.
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

    def select_api_format(self, format_kind: type[_FormatT], model: str) -> _FormatT:
        api_format = super().select_api_format(format_kind, model)
        if isinstance(api_format, ImagesFormat) and isinstance(_IMAGES_FORMAT, format_kind):
            return _IMAGES_FORMAT
        return api_format


_IMAGES_FORMAT = MultipartImagesFormat()


def _resource_api_base(resource: str) -> str:
    return f"https://{resource}.openai.azure.com/openai/v1"
