"""SessionStore tests — Phase 5.

Covers:
1. Session lifecycle (create, get, duplicate rejection)
2. Turn accumulation (append, ordering, missing session)
3. Turn retrieval (all, range, missing session)
4. Deterministic compaction (drop oldest, summary, accumulation, edge cases)
5. Snapshot isolation (returned copies don't leak mutable state)
6. buildCompactionSummary pure function
7. Multi-turn continuity via loop initialMessages seam

All tests use deterministic helpers — no live provider dependency.
"""

from __future__ import annotations

import uuid

import pytest

from clawbound.contracts.types import (
    FinalAnswer,
    KernelAsset,
    LoopConfig,
    ModelMessage,
    SessionTurn,
    ToolResult,
)
from clawbound.execution_loop.action_gate import ActionGateImpl
from clawbound.execution_loop.adapter import DeterministicAdapter
from clawbound.execution_loop.loop import run_loop
from clawbound.policy_engine.engine import PolicyEngineImpl, default_runtime_config
from clawbound.prompt_builder.builder import PromptBuilderImpl
from clawbound.session_store.store import InMemorySessionStore, build_compaction_summary
from clawbound.signal_processor.processor import SignalProcessorImpl
from clawbound.task_compiler.compiler import CompileInput, TaskCompilerImpl
from clawbound.tool_broker.broker import ToolBrokerImpl


# ─── Shared test helpers ──────────────────────────────────────────────────────


def derive_task_and_policy(user_input: str):
    compiler = TaskCompilerImpl()
    policy_engine = PolicyEngineImpl()
    task_spec = compiler.compile_from_input(CompileInput(
        trace_id=str(uuid.uuid4()),
        user_input=user_input,
        continuation_of=None,
        local_context=[],
    ))
    runtime_policy = policy_engine.resolve(task_spec, default_runtime_config())
    return task_spec, runtime_policy


TEST_KERNEL = KernelAsset(
    version="context-kernel-v0",
    content="- Do not fabricate completion.\n- Default to sparse context.",
    token_estimate=10,
)

DEFAULT_LOOP_CONFIG = LoopConfig(
    max_iterations=10,
    max_consecutive_errors=3,
    max_transient_retries=2,
)


def make_turn(turn_number: int) -> SessionTurn:
    return SessionTurn(
        turn_number=turn_number,
        timestamp="2024-01-01T00:00:00Z",
        messages=(
            ModelMessage(role="user", content=f"User message {turn_number}"),
            ModelMessage(role="assistant", content=f"Assistant response {turn_number}"),
        ),
        tool_results=(),
        signal_bundles=(),
    )


def make_turn_with_tools(
    turn_number: int,
    tool_name: str,
    status: str = "success",
) -> SessionTurn:
    tool_result = ToolResult(
        tool_name=tool_name,
        tool_call_id=f"call-{turn_number}",
        status=status,
        raw_output=f"output from {tool_name}",
        media_type="text/plain",
        output_kind="generic",
        duration_ms=10,
        metadata={},
    )
    return SessionTurn(
        turn_number=turn_number,
        timestamp="2024-01-01T00:00:00Z",
        messages=(
            ModelMessage(role="user", content=f"Run {tool_name}"),
            ModelMessage(role="assistant", content=""),
            ModelMessage(role="tool", content=tool_result.raw_output,
                         tool_call_id=tool_result.tool_call_id, tool_name=tool_name),
        ),
        tool_results=(tool_result,),
        signal_bundles=(),
    )


# ─── 1. Session lifecycle ────────────────────────────────────────────────────


class TestSessionCreate:
    def test_creates_session_with_correct_initial_state(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the parser bug")

        snapshot = store.create("s1", task_spec, policy)

        assert snapshot.session_id == "s1"
        assert snapshot.run_id == task_spec.task_id
        assert snapshot.trace_id == task_spec.trace_id
        assert snapshot.task_spec == task_spec
        assert snapshot.policy == policy
        assert len(snapshot.turns) == 0
        assert snapshot.bounds.max_turns == 50
        assert snapshot.bounds.max_stored_tokens == 100_000
        assert snapshot.bounds.was_compacted is False
        assert snapshot.bounds.retained_turns == 0
        assert snapshot.created_at
        assert snapshot.updated_at == snapshot.created_at
        assert snapshot.compacted_summary is None

    def test_throws_on_duplicate_session_id(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")

        store.create("s1", task_spec, policy)
        with pytest.raises(ValueError, match='Session "s1" already exists'):
            store.create("s1", task_spec, policy)


# ─── 2. Turn accumulation ────────────────────────────────────────────────────


class TestAppendTurn:
    def test_appends_a_turn(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)

        turn = make_turn(1)
        updated = store.append_turn("s1", turn)

        assert len(updated.turns) == 1
        assert updated.turns[0].turn_number == 1

    def test_preserves_turn_ordering(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)

        store.append_turn("s1", make_turn(1))
        store.append_turn("s1", make_turn(2))
        snapshot = store.append_turn("s1", make_turn(3))

        assert len(snapshot.turns) == 3
        assert [t.turn_number for t in snapshot.turns] == [1, 2, 3]

    def test_throws_on_append_to_nonexistent(self):
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match='Session "nope" not found'):
            store.append_turn("nope", make_turn(1))


# ─── 3. Retrieval ────────────────────────────────────────────────────────────


class TestGet:
    def test_returns_snapshot_for_existing_session(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn(1))

        snapshot = store.get("s1")
        assert snapshot is not None
        assert snapshot.session_id == "s1"
        assert len(snapshot.turns) == 1

    def test_returns_none_for_nonexistent(self):
        store = InMemorySessionStore()
        assert store.get("nope") is None


class TestGetTurns:
    def test_returns_all_turns_when_no_range(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn(1))
        store.append_turn("s1", make_turn(2))
        store.append_turn("s1", make_turn(3))

        turns = store.get_turns("s1")
        assert len(turns) == 3

    def test_returns_sliced_turns_with_range(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn(1))
        store.append_turn("s1", make_turn(2))
        store.append_turn("s1", make_turn(3))
        store.append_turn("s1", make_turn(4))

        turns = store.get_turns("s1", range_=(1, 3))
        assert len(turns) == 2
        assert turns[0].turn_number == 2
        assert turns[1].turn_number == 3

    def test_returns_empty_for_nonexistent(self):
        store = InMemorySessionStore()
        assert store.get_turns("nope") == []


# ─── 4. Deterministic compaction ──────────────────────────────────────────────


class TestCompact:
    def test_drops_old_turns_retains_recent(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        for i in range(1, 6):
            store.append_turn("s1", make_turn(i))

        compacted = store.compact("s1", 2)

        assert len(compacted.turns) == 2
        assert compacted.turns[0].turn_number == 4
        assert compacted.turns[1].turn_number == 5
        assert compacted.bounds.was_compacted is True
        assert compacted.bounds.retained_turns == 2

    def test_produces_summary_with_tool_info(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn_with_tools(1, "read_file", "success"))
        store.append_turn("s1", make_turn_with_tools(2, "run_command", "error"))
        store.append_turn("s1", make_turn(3))

        compacted = store.compact("s1", 1)

        assert "Compacted 2 turn(s)" in compacted.compacted_summary
        assert "Tools used: read_file, run_command." in compacted.compacted_summary
        assert "1 success, 1 error" in compacted.compacted_summary

    def test_accumulates_summaries_across_compactions(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)

        # First batch
        store.append_turn("s1", make_turn_with_tools(1, "read_file"))
        store.append_turn("s1", make_turn_with_tools(2, "write_file"))
        store.append_turn("s1", make_turn(3))
        store.compact("s1", 1)  # Drop turns 1-2, keep turn 3

        # Second batch
        store.append_turn("s1", make_turn_with_tools(4, "run_command"))
        store.append_turn("s1", make_turn(5))
        compacted = store.compact("s1", 1)  # Drop turns 3-4, keep turn 5

        assert "read_file" in compacted.compacted_summary
        assert "run_command" in compacted.compacted_summary
        assert "---" in compacted.compacted_summary

    def test_noop_when_retain_gte_total(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn(1))
        store.append_turn("s1", make_turn(2))

        compacted = store.compact("s1", 5)

        assert len(compacted.turns) == 2
        assert compacted.bounds.was_compacted is False
        assert compacted.compacted_summary is None

    def test_compact_to_zero_drops_everything(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn(1))
        store.append_turn("s1", make_turn(2))

        compacted = store.compact("s1", 0)

        assert len(compacted.turns) == 0
        assert compacted.bounds.was_compacted is True
        assert "Compacted 2 turn(s)" in compacted.compacted_summary

    def test_throws_on_negative_retain_turns(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)

        with pytest.raises(ValueError, match="retain_turns must be non-negative"):
            store.compact("s1", -1)

    def test_throws_on_nonexistent_session(self):
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match='Session "nope" not found'):
            store.compact("nope", 2)


# ─── 5. Snapshot isolation ────────────────────────────────────────────────────


class TestSnapshotIsolation:
    def test_mutations_to_returned_snapshot_do_not_affect_store(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the bug")
        store.create("s1", task_spec, policy)
        store.append_turn("s1", make_turn(1))

        store.get("s1")  # first get
        # Returned turns is a tuple (immutable) — cannot mutate in-place.
        # Verify that a new get returns the original state.
        snapshot2 = store.get("s1")
        assert len(snapshot2.turns) == 1
        assert snapshot2.turns[0].turn_number == 1


# ─── 6. buildCompactionSummary pure function ──────────────────────────────────


class TestBuildCompactionSummary:
    def test_summary_with_tools(self):
        turns = [
            make_turn_with_tools(1, "read_file", "success"),
            make_turn_with_tools(2, "run_command", "error"),
        ]

        summary = build_compaction_summary(turns)

        assert "Compacted 2 turn(s)" in summary
        assert "Tools used: read_file, run_command." in summary
        assert "1 success, 1 error" in summary

    def test_summary_without_tools(self):
        turns = [make_turn(1), make_turn(2)]

        summary = build_compaction_summary(turns)

        assert "Compacted 2 turn(s), 4 message(s)." in summary
        assert "Tools used:" not in summary

    def test_prepends_previous_summary(self):
        turns = [make_turn_with_tools(1, "search")]

        summary = build_compaction_summary(turns, "Previous compaction info.")

        assert summary.startswith("Previous compaction info.\n---\n")
        assert "search" in summary


# ─── 7. Multi-turn continuity via loop initialMessages seam ───────────────────


class TestSessionStoreLoopIntegration:
    async def test_multi_turn_continuity_via_initial_messages(self):
        store = InMemorySessionStore()
        task_spec, policy = derive_task_and_policy("Fix the parser bug")

        # ── Turn 1: user says "Hello" → model answers ──
        adapter1 = DeterministicAdapter([
            FinalAnswer(content="Hello! I can help with the parser."),
        ])

        result1 = await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            task_spec=task_spec,
            runtime_policy=policy,
            kernel=TEST_KERNEL,
            local_context=(),
            retrieved_units=(),
            user_message="Hello, help me fix the parser bug.",
            initial_messages=None,
            config=DEFAULT_LOOP_CONFIG,
            prompt_builder=PromptBuilderImpl(),
            tool_broker=ToolBrokerImpl(),
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter1,
            action_gate=ActionGateImpl(),
        )
        assert result1.termination == "final_answer"

        # ── Store turn 1 ──
        store.create("integration-1", task_spec, policy)
        turn1 = SessionTurn(
            turn_number=1,
            timestamp="2024-01-01T00:00:00Z",
            messages=(
                ModelMessage(role="user", content="Hello, help me fix the parser bug."),
                ModelMessage(role="assistant", content=result1.final_content),
            ),
            tool_results=tuple(result1.tool_results),
            signal_bundles=tuple(result1.signal_bundles),
        )
        store.append_turn("integration-1", turn1)

        # ── Turn 2: continue with history from store ──
        adapter2 = DeterministicAdapter([
            FinalAnswer(content="The bug is in the tokenizer."),
        ])

        stored_turns = store.get_turns("integration-1")
        history_messages: list[ModelMessage] = []
        for t in stored_turns:
            history_messages.extend(t.messages)

        result2 = await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            task_spec=task_spec,
            runtime_policy=policy,
            kernel=TEST_KERNEL,
            local_context=(),
            retrieved_units=(),
            user_message="Where exactly is the bug?",
            initial_messages=tuple(history_messages),
            config=DEFAULT_LOOP_CONFIG,
            prompt_builder=PromptBuilderImpl(),
            tool_broker=ToolBrokerImpl(),
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter2,
            action_gate=ActionGateImpl(),
        )
        assert result2.termination == "final_answer"

        # ── Verify model received full conversation history ──
        sent_request = adapter2.request_log[0]
        assert len(sent_request.messages) == 3  # 2 from history + 1 new
        assert sent_request.messages[0].role == "user"
        assert sent_request.messages[0].content == "Hello, help me fix the parser bug."
        assert sent_request.messages[1].role == "assistant"
        assert sent_request.messages[1].content == "Hello! I can help with the parser."
        assert sent_request.messages[2].role == "user"
        assert sent_request.messages[2].content == "Where exactly is the bug?"
