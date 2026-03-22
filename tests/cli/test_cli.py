"""ClawBound CLI tests — Phase 7.

Covers:
1. parseArgs — required and optional argument parsing
2. formatResult — text and JSON output formatting
3. runClawBound — direct invocation with injected test engine
4. End-to-end CLI pipeline (parseArgs → runClawBound → formatResult)
5. Public API imports

All tests use createTestEngine — no live provider dependency.
"""

from __future__ import annotations

import json

import pytest

from clawbound.cli import (
    RunOptions,
    format_result,
    parse_args,
    run_clawbound,
)
from clawbound.contracts.types import (
    FinalAnswer,
    OrchestratorDiagnostics,
)
from clawbound.engine import (
    EngineResponse,
    create_test_engine,
)


# ─── 1. parseArgs ────────────────────────────────────────────────────────────


class TestParseArgs:
    def test_parses_required_args(self):
        opts = parse_args([
            "--provider", "anthropic",
            "--model", "claude-sonnet-4-20250514",
            "--message", "What is 2+2?",
        ])

        assert opts.provider == "anthropic"
        assert opts.model == "claude-sonnet-4-20250514"
        assert opts.message == "What is 2+2?"

    def test_parses_optional_session_id(self):
        opts = parse_args([
            "--provider", "minimax",
            "--model", "MiniMax-M2.7",
            "--message", "Hi",
            "--session-id", "s-123",
        ])

        assert opts.session_id == "s-123"

    def test_parses_optional_api_key(self):
        opts = parse_args([
            "--provider", "anthropic",
            "--model", "test",
            "--message", "Hi",
            "--api-key", "sk-test-key",
        ])

        assert opts.api_key == "sk-test-key"

    def test_defaults_optional_to_none(self):
        opts = parse_args([
            "--provider", "anthropic",
            "--model", "test",
            "--message", "Hi",
        ])

        assert opts.session_id is None
        assert opts.api_key is None

    def test_throws_on_missing_provider(self):
        with pytest.raises(ValueError, match="--provider is required"):
            parse_args(["--model", "test", "--message", "Hi"])

    def test_throws_on_missing_model(self):
        with pytest.raises(ValueError, match="--model is required"):
            parse_args(["--provider", "anthropic", "--message", "Hi"])

    def test_throws_on_missing_message(self):
        with pytest.raises(ValueError, match="--message is required"):
            parse_args(["--provider", "anthropic", "--model", "test"])


# ─── 2. formatResult ─────────────────────────────────────────────────────────


class TestFormatResult:
    base_response = EngineResponse(
        content="The answer is 42.",
        run_id="run-1",
        trace_id="engine-run-1",
        termination="final_answer",
        iterations=1,
        diagnostics=OrchestratorDiagnostics(
            task_type="answer",
            execution_mode="answer",
            complexity="trivial",
            risk="low",
            prompt_token_estimate=100,
            tools_resolved=0,
            segments_admitted=3,
            segments_rejected=0,
        ),
        duration_ms=150,
    )

    def test_text_format_returns_content_only(self):
        output = format_result(self.base_response, "text")
        assert output == "The answer is 42."

    def test_json_format_returns_full_response(self):
        output = format_result(self.base_response, "json")
        parsed = json.loads(output)
        assert parsed["content"] == "The answer is 42."
        assert parsed["run_id"] == "run-1"
        assert parsed["termination"] == "final_answer"
        assert parsed["diagnostics"]["task_type"] == "answer"


# ─── 3. runClawBound — direct invocation ─────────────────────────────────────


class TestRunClawBound:
    async def test_single_turn_with_injected_engine(self):
        engine = create_test_engine([
            FinalAnswer(content="Direct response from ClawBound."),
        ])

        response = await run_clawbound(
            RunOptions(provider="test", model="test", message="Hi"),
            engine,
        )

        assert response.content == "Direct response from ClawBound."
        assert response.termination == "final_answer"
        assert response.iterations == 1

    async def test_multi_turn_with_session_continuity(self):
        engine = create_test_engine([
            FinalAnswer(content="I see the parser bug."),
            FinalAnswer(content="It's on line 42."),
        ])

        r1 = await run_clawbound(
            RunOptions(provider="test", model="test", message="Find the bug", session_id="s1"),
            engine,
        )
        assert r1.content == "I see the parser bug."

        r2 = await run_clawbound(
            RunOptions(provider="test", model="test", message="Where exactly?", session_id="s1"),
            engine,
        )
        assert r2.content == "It's on line 42."

        session = engine.get_session("s1")
        assert session is not None
        assert len(session.turns) == 2

    async def test_ephemeral_run_without_session(self):
        engine = create_test_engine([
            FinalAnswer(content="Ephemeral response."),
        ])

        response = await run_clawbound(
            RunOptions(provider="test", model="test", message="Quick question"),
            engine,
        )

        assert response.content == "Ephemeral response."

    async def test_propagates_error_for_unsupported_provider(self):
        with pytest.raises(RuntimeError, match="Unsupported provider"):
            await run_clawbound(
                RunOptions(
                    provider="nonexistent",
                    model="fake",
                    message="Hi",
                    provider_env={},
                ),
            )


# ─── 4. End-to-end CLI pipeline ──────────────────────────────────────────────


class TestEndToEndPipeline:
    async def test_parse_run_format(self):
        opts = parse_args([
            "--provider", "test",
            "--model", "deterministic",
            "--message", "What is ClawBound?",
        ])

        engine = create_test_engine([
            FinalAnswer(content="ClawBound is an independent runtime."),
        ])

        response = await run_clawbound(
            RunOptions(
                provider=opts.provider,
                model=opts.model,
                message=opts.message,
                api_key=opts.api_key,
                session_id=opts.session_id,
            ),
            engine,
        )

        text_output = format_result(response, "text")
        json_output = format_result(response, "json")

        assert text_output == "ClawBound is an independent runtime."
        assert json.loads(json_output)["content"] == "ClawBound is an independent runtime."


# ─── 5. Public API imports ───────────────────────────────────────────────────


class TestPublicAPI:
    def test_imports_from_clawbound_package(self):
        from clawbound import (
            ClawBoundEngine,
            EngineConfig,
            create_engine,
            create_test_engine,
            run_clawbound,
        )

        assert ClawBoundEngine is not None
        assert EngineConfig is not None
        assert create_engine is not None
        assert create_test_engine is not None
        assert run_clawbound is not None
