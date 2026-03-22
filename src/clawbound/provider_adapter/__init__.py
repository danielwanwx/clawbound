"""Provider adapter module."""

from .anthropic import AnthropicAdapter, create_anthropic_adapter
from .openai_compat import (
    GEMINI_OPENAI_BASE_URL,
    MINIMAX_CN_OPENAI_BASE_URL,
    MINIMAX_OPENAI_BASE_URL,
    OpenAICompatAdapter,
    create_gemini_adapter,
    create_minimax_adapter,
)
from .resolver import (
    is_supported_provider,
    resolve_effective_overrides,
    resolve_provider_adapter,
)
from .types import AnthropicConfig, OpenAICompatConfig

__all__ = [
    "AnthropicAdapter",
    "AnthropicConfig",
    "GEMINI_OPENAI_BASE_URL",
    "MINIMAX_CN_OPENAI_BASE_URL",
    "MINIMAX_OPENAI_BASE_URL",
    "OpenAICompatAdapter",
    "OpenAICompatConfig",
    "create_anthropic_adapter",
    "create_gemini_adapter",
    "create_minimax_adapter",
    "is_supported_provider",
    "resolve_effective_overrides",
    "resolve_provider_adapter",
]
