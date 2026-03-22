"""ExecutionLoop parity tests.

Covers:
1. Final-answer path with no tools
2. One-tool cycle
3. Multiple-iteration bounded cycle
4. Unauthorized tool request -> gate denied
5. Transient adapter error -> loop-owned retry
6. Non-transient error -> structured failure
7. Max-iteration enforcement
8. Trace/diagnostic visibility for failure localization

All tests use DeterministicAdapter for exact control over model behavior.
All tests verify which module caused which event via the events trace.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


from clawbound.contracts import (
    FinalAnswer,
    KernelAsset,
    LoopConfig,
    ModelError,
    ModelToolCall,
    ToolCalls,
    ToolDefinition,
)
from clawbound.execution_loop import ActionGateImpl, DeterministicAdapter, run_loop
from clawbound.prompt_builder import PromptBuilderImpl
from clawbound.signal_processor import SignalProcessorImpl
from clawbound.task_compiler import TaskCompilerImpl, CompileInput
from clawbound.policy_engine import PolicyEngineImpl, default_runtime_config
from clawbound.tool_broker import ToolBrokerImpl


# ─── Shared test helpers ──────────────────────────────────────────────────────


TEST_KERNEL = KernelAsset(
    version="context-kernel-v0",
    content="- Do not fabricate completion.\n- Default to sparse context.",
    token_estimate=10,
)

DEFAULT_CONFIG = LoopConfig(
    max_iterations=10,
    max_consecutive_errors=3,
    max_transient_retries=2,
)


def _derive_task_and_policy(user_input: str) -> tuple[Any, Any]:
    """Compile task and resolve policy, matching the TS deriveTaskAndPolicy helper."""
    compiler = TaskCompilerImpl()
    engine = PolicyEngineImpl()
    task_spec = compiler.compile_from_input(CompileInput(
        trace_id=str(uuid.uuid4()),
        user_input=user_input,
        continuation_of=None,
        local_context=[],
    ))
    runtime_policy = engine.resolve(task_spec, default_runtime_config())
    return task_spec, runtime_policy


async def _echo_execute(args: dict[str, Any]) -> dict[str, Any]:
    return {"output": f"echo: {json.dumps(args)}"}


async def _test_result_execute(_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": " Tests  5 passed (5)\n Duration  42ms",
        "output_kind": "test_results",
    }


async def _run(
    user_message: str,
    adapter: DeterministicAdapter,
    broker: ToolBrokerImpl | None = None,
    config: LoopConfig | None = None,
) -> Any:
    """Convenience wrapper that compiles task, resolves policy, and runs the loop."""
    task_spec, runtime_policy = _derive_task_and_policy(user_message)
    return await run_loop(
        run_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        task_spec=task_spec,
        runtime_policy=runtime_policy,
        kernel=TEST_KERNEL,
        local_context=(),
        retrieved_units=(),
        user_message=user_message,
        initial_messages=None,
        config=config or DEFAULT_CONFIG,
        prompt_builder=PromptBuilderImpl(),
        tool_broker=broker or ToolBrokerImpl(),
        signal_processor=SignalProcessorImpl(),
        model_adapter=adapter,
        action_gate=ActionGateImpl(),
    )


# ─── 1. Final-answer path (no tools) ─────────────────────────────────────────


class TestFinalAnswerPath:
    async def test_returns_final_answer_on_first_iteration(self) -> None:
        adapter = DeterministicAdapter([
            FinalAnswer(content="The parser splits input by commas."),
        ])

        result = await _run("Explain what the parser does.", adapter)

        assert result.termination == "final_answer"
        assert result.final_content == "The parser splits input by commas."
        assert result.iterations == 1
        assert len(result.tool_results) == 0
        assert len(result.signal_bundles) == 0

    async def test_emits_model_request_and_final_answer_events(self) -> None:
        adapter = DeterministicAdapter([
            FinalAnswer(content="Done."),
        ])

        result = await _run("Explain the code.", adapter)

        event_kinds = [e.kind for e in result.events]
        assert "model_request" in event_kinds
        assert "final_answer" in event_kinds

    async def test_sends_correct_system_prompt_from_prompt_builder(self) -> None:
        adapter = DeterministicAdapter([
            FinalAnswer(content="Answer."),
        ])

        await _run("Explain the code.", adapter)

        assert len(adapter.request_log) == 1
        request = adapter.request_log[0]
        assert "Do not fabricate completion" in request.system_prompt
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"


# ─── 2. One-tool cycle ────────────────────────────────────────────────────────


class TestOneToolCycle:
    async def test_executes_tool_call_and_returns_final_answer(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="run_command",
                    args={"cmd": "vitest"},
                ),),
                reasoning="I need to run the tests.",
            ),
            FinalAnswer(content="All tests pass."),
        ])

        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(name="run_command", category="execution", risk_level="side_effect"),
            _test_result_execute,
        )

        task_spec, runtime_policy = _derive_task_and_policy("Fix the failing test.")
        result = await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            task_spec=task_spec,
            runtime_policy=runtime_policy,
            kernel=TEST_KERNEL,
            local_context=(),
            retrieved_units=(),
            user_message="Fix the failing test.",
            initial_messages=None,
            config=DEFAULT_CONFIG,
            prompt_builder=PromptBuilderImpl(),
            tool_broker=broker,
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter,
            action_gate=ActionGateImpl(),
        )

        assert result.termination == "final_answer"
        assert result.final_content == "All tests pass."
        assert result.iterations == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].status == "success"
        assert len(result.signal_bundles) == 1

    async def test_sends_tool_result_message_to_model(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="run_command",
                    args={},
                ),),
            ),
            FinalAnswer(content="Done."),
        ])

        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(name="run_command", category="execution", risk_level="side_effect"),
            _test_result_execute,
        )

        task_spec, runtime_policy = _derive_task_and_policy("Fix the broken test.")
        await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            task_spec=task_spec,
            runtime_policy=runtime_policy,
            kernel=TEST_KERNEL,
            local_context=(),
            retrieved_units=(),
            user_message="Fix the broken test.",
            initial_messages=None,
            config=DEFAULT_CONFIG,
            prompt_builder=PromptBuilderImpl(),
            tool_broker=broker,
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter,
            action_gate=ActionGateImpl(),
        )

        # Second request should have tool result message
        assert len(adapter.request_log) == 2
        second_request = adapter.request_log[1]
        tool_msgs = [m for m in second_request.messages if m.role == "tool"]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0].tool_call_id == "tc-1"


# ─── 3. Multiple-iteration bounded cycle ──────────────────────────────────────


class TestMultipleIterationCycle:
    async def test_handles_multi_step_tool_usage(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="read_file",
                    args={"path": "src/a.ts"},
                ),),
            ),
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-2",
                    tool_name="read_file",
                    args={"path": "src/b.ts"},
                ),),
            ),
            FinalAnswer(content="Both files analyzed."),
        ])

        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(name="read_file", category="filesystem", risk_level="read_only"),
            _echo_execute,
        )

        result = await _run("Explain the code.", adapter, broker=broker)

        assert result.termination == "final_answer"
        assert result.iterations == 3
        assert len(result.tool_results) == 2
        assert len(result.signal_bundles) == 2


# ─── 4. Unauthorized tool -> gate denied ──────────────────────────────────────


class TestGateDenied:
    async def test_emits_gate_denied_event_and_continues_loop(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="delete_file",
                    args={"path": "/etc/passwd"},
                ),),
            ),
            FinalAnswer(content="I cannot delete that file."),
        ])

        # Answer mode: delete_file is not in allowed tools
        result = await _run("Explain the code.", adapter)

        assert result.termination == "final_answer"

        gate_events = [e for e in result.events if e.kind == "gate_denied"]
        assert len(gate_events) == 1
        assert gate_events[0].detail["tool_name"] == "delete_file"
        assert "not in allowed list" in gate_events[0].detail["reason"]

        # No tool results (denied at gate, never reached broker)
        assert len(result.tool_results) == 0

    async def test_sends_denial_message_to_model(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="write_file",
                    args={},
                ),),
            ),
            FinalAnswer(content="Understood."),
        ])

        await _run("Explain the code.", adapter)

        # Second request should contain a tool denial message
        second_request = adapter.request_log[1]
        tool_msgs = [m for m in second_request.messages if m.role == "tool"]
        assert len(tool_msgs) >= 1
        assert "denied by action gate" in tool_msgs[0].content


# ─── 5. Transient error -> retry ──────────────────────────────────────────────


class TestTransientRetry:
    async def test_retries_transient_errors_and_succeeds(self) -> None:
        adapter = DeterministicAdapter([
            ModelError(error="Rate limit exceeded", is_transient=True),
            FinalAnswer(content="Recovered after retry."),
        ])

        result = await _run("Explain the code.", adapter)

        assert result.termination == "final_answer"
        assert result.final_content == "Recovered after retry."

        retry_events = [e for e in result.events if e.kind == "retry"]
        assert len(retry_events) > 0

        # Adapter received 2 requests (original + retry)
        assert len(adapter.request_log) == 2

    async def test_exhausts_retries_and_increments_error_count(self) -> None:
        adapter = DeterministicAdapter([
            ModelError(error="Rate limit", is_transient=True),
            ModelError(error="Rate limit", is_transient=True),
            ModelError(error="Rate limit", is_transient=True),
            # After maxTransientRetries (2), should stop retrying and count as error
            FinalAnswer(content="Finally."),
        ])

        result = await _run(
            "Explain the code.",
            adapter,
            config=LoopConfig(
                max_iterations=10,
                max_consecutive_errors=3,
                max_transient_retries=2,
            ),
        )

        error_events = [e for e in result.events if e.kind == "error"]
        assert len(error_events) > 0


# ─── 6. Non-transient error -> structured failure ─────────────────────────────


class TestNonTransientError:
    async def test_terminates_on_max_consecutive_errors(self) -> None:
        adapter = DeterministicAdapter([
            ModelError(error="Invalid response format", is_transient=False),
            ModelError(error="Invalid response format", is_transient=False),
            ModelError(error="Invalid response format", is_transient=False),
        ])

        result = await _run(
            "Explain the code.",
            adapter,
            config=LoopConfig(
                max_iterations=10,
                max_consecutive_errors=3,
                max_transient_retries=2,
            ),
        )

        assert result.termination == "max_errors"
        assert "consecutive errors" in result.final_content
        assert "Invalid response format" in result.final_content

    async def test_resets_consecutive_error_count_on_successful_tool_cycle(self) -> None:
        adapter = DeterministicAdapter([
            ModelError(error="Glitch", is_transient=False),
            ModelError(error="Glitch", is_transient=False),
            # Error count = 2, then a successful tool cycle resets it
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="read_file",
                    args={},
                ),),
            ),
            # Another error -- count restarts from 0
            ModelError(error="Glitch again", is_transient=False),
            FinalAnswer(content="Made it."),
        ])

        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(name="read_file", category="filesystem", risk_level="read_only"),
            _echo_execute,
        )

        result = await _run(
            "Explain the code.",
            adapter,
            broker=broker,
            config=LoopConfig(
                max_iterations=10,
                max_consecutive_errors=3,
                max_transient_retries=2,
            ),
        )

        # Should succeed because error count was reset by the tool call
        assert result.termination == "final_answer"
        assert result.final_content == "Made it."


# ─── 7. Max-iteration enforcement ─────────────────────────────────────────────


class TestMaxIterationEnforcement:
    async def test_terminates_when_iteration_limit_reached(self) -> None:
        # Model always requests tools, never gives a final answer
        infinite_tool_calls = [
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id=f"tc-{i}",
                    tool_name="read_file",
                    args={},
                ),),
            )
            for i in range(5)
        ]

        adapter = DeterministicAdapter(infinite_tool_calls)
        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(name="read_file", category="filesystem", risk_level="read_only"),
            _echo_execute,
        )

        result = await _run(
            "Explain the code.",
            adapter,
            broker=broker,
            config=LoopConfig(
                max_iterations=3,
                max_consecutive_errors=3,
                max_transient_retries=2,
            ),
        )

        assert result.termination == "max_iterations"
        assert result.iterations == 3
        assert len(result.tool_results) == 3


# ─── 8. Trace/diagnostic visibility ───────────────────────────────────────────


class TestTraceDiagnosticVisibility:
    async def test_events_trace_localizes_each_stage_of_tool_cycle(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="run_command",
                    args={"cmd": "tsc"},
                ),),
            ),
            FinalAnswer(content="Build passed."),
        ])

        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(
                name="run_command",
                category="execution",
                risk_level="side_effect",
                declared_output_kind="build_output",
            ),
            lambda _args: _build_output_execute(_args),
        )

        task_spec, runtime_policy = _derive_task_and_policy("Fix the build error.")
        result = await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            task_spec=task_spec,
            runtime_policy=runtime_policy,
            kernel=TEST_KERNEL,
            local_context=(),
            retrieved_units=(),
            user_message="Fix the build error.",
            initial_messages=None,
            config=DEFAULT_CONFIG,
            prompt_builder=PromptBuilderImpl(),
            tool_broker=broker,
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter,
            action_gate=ActionGateImpl(),
        )

        event_kinds = [e.kind for e in result.events]

        # Iteration 1: model_request -> tool_execution -> signal_processed
        assert event_kinds[0] == "model_request"
        assert event_kinds[1] == "tool_execution"
        assert event_kinds[2] == "signal_processed"

        # Iteration 2: model_request -> final_answer
        assert event_kinds[3] == "model_request"
        assert event_kinds[4] == "final_answer"

        # Tool execution event has localization detail
        tool_event = next(e for e in result.events if e.kind == "tool_execution")
        assert tool_event.detail["tool_name"] == "run_command"
        assert tool_event.detail["status"] == "success"
        assert tool_event.detail["output_kind"] == "build_output"

        # Signal event has localization detail
        signal_event = next(e for e in result.events if e.kind == "signal_processed")
        assert signal_event.detail["signal_kind"] == "build_output"
        assert isinstance(signal_event.detail["compression_ratio"], float)

    async def test_events_distinguish_adapter_errors_from_gate_denials(self) -> None:
        adapter = DeterministicAdapter([
            ModelError(error="Timeout", is_transient=False),
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="delete_file",
                    args={},
                ),),
            ),
            FinalAnswer(content="Done."),
        ])

        result = await _run("Explain the code.", adapter)

        error_event = next((e for e in result.events if e.kind == "error"), None)
        assert error_event is not None
        assert error_event.detail["error"] == "Timeout"

        gate_event = next((e for e in result.events if e.kind == "gate_denied"), None)
        assert gate_event is not None
        assert gate_event.detail["tool_name"] == "delete_file"


# ─── Full pipeline integration ────────────────────────────────────────────────


class TestFullPipelineIntegration:
    async def test_task_compiler_to_policy_engine_to_execution_loop(self) -> None:
        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="read_file",
                    args={"path": "src/main.ts"},
                ),),
            ),
            FinalAnswer(content="The main module exports the app entry point."),
        ])

        broker = ToolBrokerImpl()
        broker.register(
            ToolDefinition(name="read_file", category="filesystem", risk_level="read_only"),
            lambda args: _main_file_execute(args),
        )

        # Full pipeline: compile -> resolve policy -> run loop
        compiler = TaskCompilerImpl()
        policy_engine = PolicyEngineImpl()

        user_input = "Explain what the main module does."
        task_spec = compiler.compile_from_input(CompileInput(
            trace_id=str(uuid.uuid4()),
            user_input=user_input,
            continuation_of=None,
            local_context=[],
        ))
        runtime_policy = policy_engine.resolve(task_spec, default_runtime_config())

        result = await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=task_spec.trace_id,
            task_spec=task_spec,
            runtime_policy=runtime_policy,
            kernel=TEST_KERNEL,
            local_context=(),
            retrieved_units=(),
            user_message=user_input,
            initial_messages=None,
            config=DEFAULT_CONFIG,
            prompt_builder=PromptBuilderImpl(),
            tool_broker=broker,
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter,
            action_gate=ActionGateImpl(),
        )

        assert result.termination == "final_answer"
        assert "main module" in result.final_content
        assert len(result.tool_results) == 1
        assert len(result.signal_bundles) == 1
        assert len(result.events) > 0

        # Trace confirms loop-managed orchestration
        assert result.events[0].kind == "model_request"


# ─── Async helpers for tool execute functions ─────────────────────────────────


async def _build_output_execute(_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": "Compilation complete. No errors found.",
        "output_kind": "build_output",
    }


async def _main_file_execute(_args: dict[str, Any]) -> dict[str, Any]:
    return {"output": 'export function main() { console.log("hello"); }'}
