"""Provider Adapter Resolver.

Resolves a ModelAdapter for a given provider string. Handles:
- CLAWBOUND_PROVIDER / CLAWBOUND_MODEL env overrides
- API key isolation (override -> ignore host key)
- Provider-specific adapter creation (Anthropic, Gemini, MiniMax)
"""

from __future__ import annotations


from .anthropic import AnthropicAdapter, create_anthropic_adapter
from .openai_compat import (
    MINIMAX_CN_OPENAI_BASE_URL,
    OpenAICompatAdapter,
    create_gemini_adapter,
    create_minimax_adapter,
)
from .types import AnthropicConfig


def resolve_provider_adapter(
    provider: str,
    api_key: str | None,
    model_id: str,
    env: dict[str, str | None] | None = None,
) -> AnthropicAdapter | OpenAICompatAdapter | None:
    """Resolve a model adapter for the given provider, or None if unsupported.

    Supported providers:
    - anthropic / anthropic-* -> AnthropicAdapter
    - google / gemini / google-* -> GeminiAdapter via OpenAI compat
    - minimax / minimax-* -> MinimaxAdapter via OpenAI compat

    Returns None for all other providers.
    """
    if env is None:
        import os
        env = dict(os.environ)

    override_provider = (env.get("CLAWBOUND_PROVIDER") or "").strip() or None
    override_model = (env.get("CLAWBOUND_MODEL") or "").strip() or None
    effective_provider = override_provider or provider
    effective_model_id = override_model or model_id

    # When provider is overridden, host-provided apiKey belongs to the host's
    # provider — ignore it.
    host_api_key = None if override_provider else api_key

    normalized = effective_provider.lower().strip()

    # Anthropic path
    if normalized == "anthropic" or normalized.startswith("anthropic-"):
        key = host_api_key or (env.get("ANTHROPIC_API_KEY") or "").strip() or None
        if not key:
            return None
        return create_anthropic_adapter(AnthropicConfig(api_key=key, model=effective_model_id))

    # Gemini path (via OpenAI compatibility endpoint)
    if normalized in ("google", "gemini") or normalized.startswith(("google-", "google/")):
        gemini_key = (env.get("GEMINI_API_KEY") or "").strip() or None
        if not gemini_key:
            return None
        model = effective_model_id.split("/")[-1] if "/" in effective_model_id else effective_model_id
        return create_gemini_adapter(gemini_key, model)

    # MiniMax path (via OpenAI compatibility endpoint)
    if normalized == "minimax" or normalized.startswith(("minimax-", "minimax/")):
        minimax_key = host_api_key or (env.get("MINIMAX_API_KEY") or "").strip() or None
        if not minimax_key:
            return None
        model = effective_model_id.split("/")[-1] if "/" in effective_model_id else effective_model_id
        base_url_override = (env.get("CLAWBOUND_MINIMAX_BASE_URL") or "").strip() or None
        base_url = (
            MINIMAX_CN_OPENAI_BASE_URL
            if base_url_override == "cn"
            else base_url_override or "https://api.minimax.io/v1"
        )
        return create_minimax_adapter(minimax_key, model, base_url)

    return None


def resolve_effective_overrides(
    env: dict[str, str | None] | None = None,
) -> dict[str, str]:
    """Read CLAWBOUND_PROVIDER / CLAWBOUND_MODEL env overrides."""
    if env is None:
        import os
        env = dict(os.environ)

    p = (env.get("CLAWBOUND_PROVIDER") or "").strip()
    m = (env.get("CLAWBOUND_MODEL") or "").strip()
    result: dict[str, str] = {}
    if p:
        result["effective_provider"] = p
    if m:
        result["effective_model"] = m
    return result


def is_supported_provider(
    provider: str,
    env: dict[str, str | None] | None = None,
) -> bool:
    """Check if a provider is supported by the new runtime."""
    if env is None:
        import os
        env = dict(os.environ)

    override_provider = (env.get("CLAWBOUND_PROVIDER") or "").strip()
    effective = override_provider or provider
    normalized = effective.lower().strip()

    return (
        normalized == "anthropic"
        or normalized.startswith("anthropic-")
        or normalized in ("google", "gemini")
        or normalized.startswith(("google-", "google/"))
        or normalized == "minimax"
        or normalized.startswith(("minimax-", "minimax/"))
    )
