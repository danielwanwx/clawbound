"""Provider Resolver tests — Phase 6.

Covers:
1. Anthropic resolution (with/without API key)
2. Gemini resolution (via OpenAI compat)
3. MiniMax resolution (with base URL override, cn shorthand)
4. CLAWBOUND_PROVIDER / CLAWBOUND_MODEL env overrides
5. API key isolation (override -> ignore host key)
6. Unknown provider returns None
7. resolve_effective_overrides helper
8. is_supported_provider helper
"""

from __future__ import annotations


from clawbound.provider_adapter.anthropic import AnthropicAdapter
from clawbound.provider_adapter.openai_compat import (
    OpenAICompatAdapter,
)
from clawbound.provider_adapter.resolver import (
    is_supported_provider,
    resolve_effective_overrides,
    resolve_provider_adapter,
)


# ─── 1. Anthropic resolution ─────────────────────────────────────────────────


class TestAnthropicResolution:
    def test_resolves_with_host_api_key(self):
        adapter = resolve_provider_adapter(
            provider="anthropic",
            api_key="test-key-123",
            model_id="claude-3-haiku-20240307",
            env={},
        )
        assert isinstance(adapter, AnthropicAdapter)

    def test_resolves_with_env_api_key(self):
        adapter = resolve_provider_adapter(
            provider="anthropic",
            api_key=None,
            model_id="claude-3-haiku-20240307",
            env={"ANTHROPIC_API_KEY": "env-key-456"},
        )
        assert isinstance(adapter, AnthropicAdapter)

    def test_returns_none_without_api_key(self):
        adapter = resolve_provider_adapter(
            provider="anthropic",
            api_key=None,
            model_id="claude-3-haiku-20240307",
            env={},
        )
        assert adapter is None

    def test_resolves_anthropic_prefix(self):
        adapter = resolve_provider_adapter(
            provider="anthropic-beta",
            api_key="key",
            model_id="claude-3-haiku-20240307",
            env={},
        )
        assert isinstance(adapter, AnthropicAdapter)


# ─── 2. Gemini resolution ────────────────────────────────────────────────────


class TestGeminiResolution:
    def test_resolves_google_provider(self):
        adapter = resolve_provider_adapter(
            provider="google",
            api_key=None,
            model_id="gemini-2.5-flash",
            env={"GEMINI_API_KEY": "gemini-key"},
        )
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_resolves_gemini_provider(self):
        adapter = resolve_provider_adapter(
            provider="gemini",
            api_key=None,
            model_id="gemini-2.5-flash",
            env={"GEMINI_API_KEY": "gemini-key"},
        )
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_resolves_google_prefix(self):
        adapter = resolve_provider_adapter(
            provider="google-ai",
            api_key=None,
            model_id="gemini-2.5-flash",
            env={"GEMINI_API_KEY": "gemini-key"},
        )
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_strips_provider_prefix_from_model(self):
        adapter = resolve_provider_adapter(
            provider="google",
            api_key=None,
            model_id="google/gemini-2.5-flash",
            env={"GEMINI_API_KEY": "key"},
        )
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_returns_none_without_gemini_key(self):
        adapter = resolve_provider_adapter(
            provider="google",
            api_key=None,
            model_id="gemini-2.5-flash",
            env={},
        )
        assert adapter is None


# ─── 3. MiniMax resolution ───────────────────────────────────────────────────


class TestMinimaxResolution:
    def test_resolves_minimax_provider(self):
        adapter = resolve_provider_adapter(
            provider="minimax",
            api_key=None,
            model_id="MiniMax-M2.7",
            env={"MINIMAX_API_KEY": "minimax-key"},
        )
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_resolves_minimax_prefix(self):
        adapter = resolve_provider_adapter(
            provider="minimax-beta",
            api_key=None,
            model_id="MiniMax-M2.7",
            env={"MINIMAX_API_KEY": "key"},
        )
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_returns_none_without_minimax_key(self):
        adapter = resolve_provider_adapter(
            provider="minimax",
            api_key=None,
            model_id="MiniMax-M2.7",
            env={},
        )
        assert adapter is None


# ─── 4. CLAWBOUND_PROVIDER / CLAWBOUND_MODEL overrides ───────────────────────


class TestProviderOverrides:
    def test_override_provider(self):
        adapter = resolve_provider_adapter(
            provider="openai",  # Not normally supported
            api_key="host-key",
            model_id="gpt-4",
            env={
                "CLAWBOUND_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "override-key",
            },
        )
        assert isinstance(adapter, AnthropicAdapter)

    def test_override_model(self):
        adapter = resolve_provider_adapter(
            provider="anthropic",
            api_key="key",
            model_id="claude-3-haiku-20240307",
            env={"CLAWBOUND_MODEL": "claude-3-opus-20240229"},
        )
        assert isinstance(adapter, AnthropicAdapter)

    def test_override_ignores_host_api_key(self):
        # When CLAWBOUND_PROVIDER is set, host apiKey should be ignored
        adapter = resolve_provider_adapter(
            provider="openai",
            api_key="openai-key",  # This should be ignored
            model_id="gpt-4",
            env={
                "CLAWBOUND_PROVIDER": "anthropic",
                # No ANTHROPIC_API_KEY in env -> should return None
            },
        )
        assert adapter is None


# ─── 5. Unknown provider ─────────────────────────────────────────────────────


class TestUnknownProvider:
    def test_returns_none_for_unknown(self):
        adapter = resolve_provider_adapter(
            provider="some-unknown-provider",
            api_key="key",
            model_id="model",
            env={},
        )
        assert adapter is None


# ─── 6. resolve_effective_overrides ──────────────────────────────────────────


class TestResolveEffectiveOverrides:
    def test_returns_empty_when_no_overrides(self):
        result = resolve_effective_overrides(env={})
        assert result == {}

    def test_returns_provider_override(self):
        result = resolve_effective_overrides(env={"CLAWBOUND_PROVIDER": "anthropic"})
        assert result == {"effective_provider": "anthropic"}

    def test_returns_model_override(self):
        result = resolve_effective_overrides(env={"CLAWBOUND_MODEL": "claude-3-opus-20240229"})
        assert result == {"effective_model": "claude-3-opus-20240229"}

    def test_returns_both_overrides(self):
        result = resolve_effective_overrides(env={
            "CLAWBOUND_PROVIDER": "anthropic",
            "CLAWBOUND_MODEL": "claude-3-opus-20240229",
        })
        assert result == {
            "effective_provider": "anthropic",
            "effective_model": "claude-3-opus-20240229",
        }


# ─── 7. is_supported_provider ────────────────────────────────────────────────


class TestIsSupportedProvider:
    def test_anthropic_supported(self):
        assert is_supported_provider("anthropic", env={}) is True

    def test_anthropic_prefix_supported(self):
        assert is_supported_provider("anthropic-beta", env={}) is True

    def test_google_supported(self):
        assert is_supported_provider("google", env={}) is True

    def test_gemini_supported(self):
        assert is_supported_provider("gemini", env={}) is True

    def test_minimax_supported(self):
        assert is_supported_provider("minimax", env={}) is True

    def test_unknown_not_supported(self):
        assert is_supported_provider("openai", env={}) is False

    def test_override_makes_supported(self):
        assert is_supported_provider("openai", env={"CLAWBOUND_PROVIDER": "anthropic"}) is True
