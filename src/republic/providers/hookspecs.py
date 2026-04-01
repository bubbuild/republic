"""Pluggy hookspecs for Republic provider loading."""

from __future__ import annotations

import pluggy

REPUBLIC_PROVIDER_NAMESPACE = "republic.providers"
providerspec = pluggy.HookspecMarker(REPUBLIC_PROVIDER_NAMESPACE)
providerimpl = pluggy.HookimplMarker(REPUBLIC_PROVIDER_NAMESPACE)


class RepublicProviderSpecs:
    """Hook contract for registering provider hooks."""

    @providerspec
    def provide_providers(self) -> list[object]:
        """Return provider hook objects."""
        raise NotImplementedError
