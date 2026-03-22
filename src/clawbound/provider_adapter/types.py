"""Provider-specific configuration and API types.

These are NOT part of the ClawBound runtime contracts.
They isolate provider-specific message/response shapes from the core runtime.
Only the adapters read/write these.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Anthropic types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: str
    model: str
    base_url: str = "https://api.anthropic.com"
    max_tokens: int = 4096
    anthropic_version: str = "2023-06-01"


# ─── OpenAI-compat types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpenAICompatConfig:
    api_key: str
    model: str
    base_url: str
    max_tokens: int = 4096
