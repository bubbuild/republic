"""Keep the package scaffold importable during reconstruction."""

import republic


def test_package_import() -> None:
    assert republic.__name__ == "republic"
