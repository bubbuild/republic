"""Built-in providers, their authentication, and the base class for custom ones."""

from .anthropic import Anthropic
from .azure import AzureOpenAI
from .base import Provider
from .codex import Codex, CodexAuth
from .deepseek import DeepSeek
from .github import CopilotAuth, GitHubCLIAuth, GitHubCopilot
from .google import Google
from .grok import Grok, GrokAuth
from .magpie import Magpie
from .minimax import MiniMax
from .mistral import Mistral
from .moonshot import Moonshot
from .ollama import Ollama
from .openai import OpenAI, OpenAICompatible
from .openrouter import OpenRouter, OpenRouterAuth
from .together import Together
from .typesafe import TypeSafe
from .zai import ZAI

__all__ = [
    "ZAI",
    "Anthropic",
    "AzureOpenAI",
    "Codex",
    "CodexAuth",
    "CopilotAuth",
    "DeepSeek",
    "GitHubCLIAuth",
    "GitHubCopilot",
    "Google",
    "Grok",
    "GrokAuth",
    "Magpie",
    "MiniMax",
    "Mistral",
    "Moonshot",
    "Ollama",
    "OpenAI",
    "OpenAICompatible",
    "OpenRouter",
    "OpenRouterAuth",
    "Provider",
    "Together",
    "TypeSafe",
]
