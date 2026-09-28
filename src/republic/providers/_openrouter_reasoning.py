"""Preserve documented Chat reasoning_details, including opaque video context."""

from copy import deepcopy
from typing import Any

from republic.errors import ProviderError

_TEXT = {"reasoning.summary": "summary", "reasoning.text": "text", "reasoning.encrypted": "data"}


def _invalid() -> ProviderError:
    return ProviderError("invalid_reasoning_details", provider="openai", code="invalid_response")


def details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise _invalid()
    for item in value:
        if not isinstance(item, dict) or item.get("type") not in _TEXT:
            raise _invalid()
        field = _TEXT[item["type"]]
        if not isinstance(item.get(field), str) and not (field == "text" and item.get(field) is None):
            raise _invalid()
        for key in ("id", "format", "signature"):
            if item.get(key) is not None and not isinstance(item[key], str):
                raise _invalid()
        if "index" in item and (type(item["index"]) is not int or item["index"] < 0):
            raise _invalid()
    return deepcopy(value)


class ReasoningDetails:
    def __init__(self) -> None:
        self.items: dict[tuple[str, Any], dict[str, Any]] = {}

    def feed(self, values: Any) -> None:
        if not isinstance(values, list):
            raise _invalid()
        for item in values:
            if not isinstance(item, dict):
                raise _invalid()
            if "index" in item:
                if type(item["index"]) is not int or item["index"] < 0:
                    raise _invalid()
                key = ("index", item["index"])
            elif isinstance(item.get("id"), str):
                key = ("id", item["id"])
            else:
                # Unidentified entries must be complete. Keep them separately;
                # neither invent a wire index nor guess how to join fragments.
                details([item])
                key = ("entry", len(self.items))
            target = self.items.setdefault(key, {})
            self._merge(target, item)

    @staticmethod
    def _merge(target: dict[str, Any], item: dict[str, Any]) -> None:
        for name, value in item.items():
            if name in {"text", "summary", "data", "signature"} and value is not None:
                if not isinstance(value, str):
                    raise _invalid()
                target[name] = (target.get(name) or "") + value
            elif value is not None:
                if target.get(name) is not None and target[name] != value:
                    raise _invalid()
                target[name] = deepcopy(value)
            else:
                target.setdefault(name, None)

    def result(self) -> list[dict[str, Any]]:
        return details(list(self.items.values()))
