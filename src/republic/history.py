"""Optional conversation history attached to a chat model."""

from __future__ import annotations

from typing import Protocol

from republic._content import Message

__all__ = ["HistoryProtocol", "InMemoryHistory"]


class HistoryProtocol(Protocol):
    async def read(self) -> list[Message]:
        """Read the conversation history."""
        ...

    async def write(self, messages: list[Message]) -> None:
        """Append messages to the conversation history."""
        ...


class InMemoryHistory:
    """Keep the conversation in memory, optionally bounded to the latest messages.

    When trimming, the history always restarts at a plain user message so that a
    tool result is never separated from the call that produced it.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._messages: list[Message] = []

    async def read(self) -> list[Message]:
        return list(self._messages)

    async def write(self, messages: list[Message]) -> None:
        self._messages.extend(messages)
        if self.max_entries is None or len(self._messages) <= self.max_entries:
            return
        kept = self._messages[-self.max_entries :]
        start = next((index for index, message in enumerate(kept) if message.role == "user"), len(kept))
        self._messages = kept[start:]
