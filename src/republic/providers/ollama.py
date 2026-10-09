from __future__ import annotations

from ._chat import MaxTokensChat
from .openai import OpenAICompatible


class Ollama(OpenAICompatible):
    """A local Ollama server, or Ollama Cloud with ``api_base="https://ollama.com/v1"`` and an API key.

    Chat is the default; Responses is stateless and needs Ollama 0.13.3 or later.
    """

    name = "ollama"
    DEFAULT_API_BASE = "http://localhost:11434/v1"
    SUPPORTED_API_FORMATS = ("chat", "responses", "embeddings")
    CHAT_FORMAT = MaxTokensChat()
