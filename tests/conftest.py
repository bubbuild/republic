from __future__ import annotations

import pytest

from .fakes import FakeProviderFactory, install_fake_provider_runtime


@pytest.fixture
def fake_provider_factory(monkeypatch: pytest.MonkeyPatch) -> FakeProviderFactory:
    return install_fake_provider_runtime(monkeypatch)
