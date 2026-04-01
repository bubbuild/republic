"""Provider runtime built on pluggy."""

from __future__ import annotations

import importlib.metadata
import logging

import pluggy

from republic.providers.hookspecs import REPUBLIC_PROVIDER_NAMESPACE, RepublicProviderSpecs, providerimpl
from republic.providers.types import ProviderHook

logger = logging.getLogger(__name__)


class _BuiltinProvidersPlugin:
    @providerimpl
    def provide_providers(self) -> list[object]:
        from republic.providers.anthropic import AnthropicProviderHook
        from republic.providers.gemini import GeminiProviderHook
        from republic.providers.github_copilot import GitHubCopilotProviderHook
        from republic.providers.mistral import MistralProviderHook
        from republic.providers.openai import OpenAIProviderHook

        return [
            OpenAIProviderHook(),
            AnthropicProviderHook(),
            GeminiProviderHook(),
            MistralProviderHook(),
            GitHubCopilotProviderHook(),
        ]


class ProviderRuntime:
    """Load provider hooks and resolve builtin backends."""

    def __init__(self) -> None:
        self._plugin_manager = pluggy.PluginManager(REPUBLIC_PROVIDER_NAMESPACE)
        self._plugin_manager.add_hookspecs(RepublicProviderSpecs)
        self._plugin_manager.register(_BuiltinProvidersPlugin(), name="builtin")
        self._loaded = False
        self._providers: dict[str, ProviderHook] = {}

    def load(self) -> None:
        if self._loaded:
            return

        for entry_point in importlib.metadata.entry_points(group=REPUBLIC_PROVIDER_NAMESPACE):
            plugin = self._load_plugin(entry_point)
            if plugin is None:
                continue
            try:
                self._plugin_manager.register(plugin, name=entry_point.name)
            except ValueError:
                logger.warning("Skipping provider plugin %s because the name is already registered.", entry_point.name)

        providers: dict[str, ProviderHook] = {}
        for plugin_name, plugin in self._plugin_manager.list_name_plugin():
            if plugin is None:
                continue
            for provider in self._collect_providers(plugin_name, plugin):
                providers[provider.name] = provider
        self._providers = providers
        self._loaded = True

    def _load_plugin(self, entry_point: importlib.metadata.EntryPoint) -> object | None:
        try:
            plugin = entry_point.load()
        except Exception:
            logger.warning("Failed to load provider plugin %s.", entry_point.name, exc_info=True)
            return None

        if not callable(plugin):
            return plugin

        try:
            return plugin()
        except Exception:
            logger.warning("Failed to instantiate provider plugin %s.", entry_point.name, exc_info=True)
            return None

    def _collect_providers(self, plugin_name: str, plugin: object) -> list[ProviderHook]:
        provide_providers = getattr(plugin, "provide_providers", None)
        if not callable(provide_providers):
            return []

        try:
            raw_providers = provide_providers()
        except Exception:
            logger.warning("Failed to collect providers from plugin %s.", plugin_name, exc_info=True)
            return []

        providers: list[ProviderHook] = []
        for provider in raw_providers or []:
            if isinstance(provider, ProviderHook) or hasattr(provider, "create_backend"):
                providers.append(provider)
                continue
            logger.warning("Skipping invalid provider object from plugin %s: %r", plugin_name, provider)
        return providers

    def provider_for(self, provider_name: str) -> ProviderHook | None:
        self.load()
        provider = self._providers.get(provider_name)
        if provider is not None and provider.matches(provider_name):
            return provider
        for candidate in self._providers.values():
            if candidate.matches(provider_name):
                return candidate
        return None

    def hook_report(self) -> dict[str, str]:
        self.load()
        return {name: type(provider).__name__ for name, provider in sorted(self._providers.items())}


_DEFAULT_RUNTIME: ProviderRuntime | None = None


def default_provider_runtime() -> ProviderRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = ProviderRuntime()
    return _DEFAULT_RUNTIME


def reset_default_provider_runtime(runtime: ProviderRuntime | None = None) -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = runtime
