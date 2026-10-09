from __future__ import annotations

import pytest

import republic
from republic import errors


@pytest.mark.parametrize("name", errors.__all__)
def test_exceptions_are_exported_only_from_errors_module(name: str) -> None:
    error_type = getattr(errors, name)

    assert issubclass(error_type, errors.RepublicError)
    assert error_type.__module__ == "republic.errors"
    assert not hasattr(republic, name)
    assert name not in republic.__all__
    assert "errors" in republic.__all__
