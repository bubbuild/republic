"""Built-in providers, their authentication, and the base class for custom ones."""

from .anthropic import Anthropic
from .base import Provider
from .codex import Codex, CodexAuth
from .github import CopilotAuth, GitHubCLIAuth, GitHubCopilot
from .google import Google
from .openai import OpenAI, OpenAICompatible
from .openrouter import OpenRouter, OpenRouterAuth
from .typesafe import TypeSafe

__all__ = [
    "Anthropic",
    "Codex",
    "CodexAuth",
    "CopilotAuth",
    "GitHubCLIAuth",
    "GitHubCopilot",
    "Google",
    "OpenAI",
    "OpenAICompatible",
    "OpenRouter",
    "OpenRouterAuth",
    "Provider",
    "TypeSafe",
]
