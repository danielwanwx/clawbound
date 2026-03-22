"""ClawBoundEngine tests — Phase 5.

Covers:
1. Single-turn execution with scripted responses
2. Multi-turn session continuity
3. Session operations (get, compact)
4. Provider resolution and error handling
5. Event emission

All tests use createTestEngine with DeterministicAdapter.
"""

from __future__ import annotations

import pytest

from clawbound.contracts.types import (
    FinalAnswer,
)
from clawbound.engine import (
    ClawBoundEngine,
    EngineConfig,
    EngineEvent,
    EngineRequest,
    create_engine,
    create_test_engine,
)


# ─── 1. Single-turn ──────────────────────────────────────────────────────────


class TestSingleTurn:
    async def test_returns_content_matching_scripted_response(self):
        engine = create_test_engine([
            FinalAnswer(content="Hello from ClawBound!"),
        ])

        response = await engine.run(EngineRequest(message="Hi"))

        assert response.content == "Hello from ClawBound!"
        assert response.termination == "final_answer"
        assert response.iterations == 1
        assert response.run_id
        assert response.trace_id.startswith("engine-")
        assert response.duration_ms >= 0

    async def test_uses_provided_run_id(self):
        engine = create_test_engine([
            FinalAnswer(content="ok"),
        ])

        response = await engine.run(EngineRequest(
            message="Hi",
            run_id="custom-run-123",
        ))

        assert response.run_id == "custom-run-123"
        assert response.trace_id == "engine-custom-run-123"

    async def test_returns_diagnostics_with_task_classification(self):
        engine = create_test_engine([
            FinalAnswer(content="The answer is 42."),
        ])

        response = await engine.run(EngineRequest(message="What is 6 * 7?"))

        assert response.diagnostics is not None
        assert response.diagnostics.task_type
        assert response.diagnostics.complexity
        assert response.diagnostics.risk


# ─── 2. Multi-turn session continuity ─────────────────────────────────────────


class TestMultiTurnSessions:
    async def test_second_call_with_same_session_receives_prior_context(self):
        engine = create_test_engine([
            FinalAnswer(content="I can help with parsers."),
            FinalAnswer(content="The bug is on line 42."),
        ])

        r1 = await engine.run(EngineRequest(
            message="Help me fix the parser bug.",
            session_id="session-1",
        ))
        assert r1.content == "I can help with parsers."

        r2 = await engine.run(EngineRequest(
            message="Where exactly is the bug?",
            session_id="session-1",
        ))
        assert r2.content == "The bug is on line 42."

    async def test_different_session_ids_are_independent(self):
        engine = create_test_engine([
            FinalAnswer(content="Response A"),
            FinalAnswer(content="Response B"),
        ])

        await engine.run(EngineRequest(message="Hello", session_id="s1"))
        await engine.run(EngineRequest(message="World", session_id="s2"))

        session1 = engine.get_session("s1")
        session2 = engine.get_session("s2")

        assert session1 is not None
        assert session2 is not None
        assert session1.session_id == "s1"
        assert session2.session_id == "s2"
        assert len(session1.turns) == 1
        assert len(session2.turns) == 1

    async def test_session_accumulates_turns(self):
        engine = create_test_engine([
            FinalAnswer(content="R1"),
            FinalAnswer(content="R2"),
            FinalAnswer(content="R3"),
        ])

        await engine.run(EngineRequest(message="M1", session_id="s1"))
        await engine.run(EngineRequest(message="M2", session_id="s1"))
        await engine.run(EngineRequest(message="M3", session_id="s1"))

        session = engine.get_session("s1")
        assert len(session.turns) == 3

        assert session.turns[0].messages[0].content == "M1"
        assert session.turns[0].messages[1].content == "R1"
        assert session.turns[2].messages[0].content == "M3"
        assert session.turns[2].messages[1].content == "R3"


# ─── 3. Session operations ───────────────────────────────────────────────────


class TestSessionOperations:
    def test_get_session_returns_none_before_first_call(self):
        engine = create_test_engine([
            FinalAnswer(content="ok"),
        ])

        assert engine.get_session("nonexistent") is None

    async def test_get_session_returns_snapshot_after_first_call(self):
        engine = create_test_engine([
            FinalAnswer(content="ok"),
        ])

        await engine.run(EngineRequest(message="Hi", session_id="s1"))

        session = engine.get_session("s1")
        assert session is not None
        assert session.session_id == "s1"
        assert len(session.turns) == 1

    async def test_compact_session_drops_old_turns(self):
        engine = create_test_engine([
            FinalAnswer(content="R1"),
            FinalAnswer(content="R2"),
            FinalAnswer(content="R3"),
        ])

        await engine.run(EngineRequest(message="M1", session_id="s1"))
        await engine.run(EngineRequest(message="M2", session_id="s1"))
        await engine.run(EngineRequest(message="M3", session_id="s1"))

        compacted = engine.compact_session("s1", 1)
        assert len(compacted.turns) == 1
        assert compacted.bounds.was_compacted is True
        assert compacted.turns[0].messages[0].content == "M3"

    async def test_no_session_created_when_session_id_not_provided(self):
        engine = create_test_engine([
            FinalAnswer(content="ephemeral"),
        ])

        await engine.run(EngineRequest(message="Hi"))

        assert engine.get_session("undefined") is None


# ─── 4. Provider resolution / error handling ──────────────────────────────────


class TestProviderResolution:
    def test_create_engine_with_unknown_provider(self):
        engine = create_engine(EngineConfig(
            provider="unknown-provider",
            model="some-model",
        ))
        assert engine is not None

    async def test_unknown_provider_throws_on_run(self):
        engine = create_engine(EngineConfig(
            provider="unknown-provider",
            model="some-model",
        ))

        with pytest.raises(RuntimeError, match='Unsupported provider "unknown-provider"'):
            await engine.run(EngineRequest(message="Hi"))


# ─── 5. Event emission ───────────────────────────────────────────────────────


class TestEngineEvents:
    async def test_on_event_called_on_run_complete(self):
        events: list[EngineEvent] = []

        engine = ClawBoundEngine(
            adapter=__import__("clawbound.execution_loop.adapter", fromlist=["DeterministicAdapter"]).DeterministicAdapter([
                FinalAnswer(content="done"),
            ]),
            config=EngineConfig(
                provider="test",
                model="deterministic",
                on_event=lambda e: events.append(e),
            ),
        )

        response = await engine.run(EngineRequest(message="Hi"))

        assert response.termination == "final_answer"
        assert len(events) == 1
        assert events[0].kind == "engine_run_complete"
        assert events[0].detail["run_id"] == response.run_id
        assert events[0].detail["trace_id"] == response.trace_id
        assert events[0].detail["duration_ms"] >= 0
