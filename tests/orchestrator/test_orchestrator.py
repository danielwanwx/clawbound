"""Orchestrator tests — Phase 5.

Covers:
1. End-to-end pipeline returns structured result
2. Diagnostics include task classification
3. Diagnostics include prompt and tool stats
4. Tool cycle works through orchestrator with registered tools
5. Propagates loop termination reasons
6. LoopResult included for full diagnostic access

All tests use DeterministicAdapter — no live provider dependency.
The switchover router/divergence tests from TS are dropped (migration artifacts).
"""

from __future__ import annotations

import uuid
from typing import Any


from clawbound.contracts.types import (
    FinalAnswer,
    ModelError,
    ModelToolCall,
    ToolCalls,
    ToolDefinition,
)
from clawbound.execution_loop.adapter import DeterministicAdapter
from clawbound.orchestrator import ToolRegistration, run_orchestrator
from clawbound.tool_broker.broker import ToolBrokerExecuteFn


# ─── Test helpers ─────────────────────────────────────────────────────────────


async def echo_fn(args: dict[str, Any]) -> dict[str, str]:
    return {"output": f"echo: {args}"}


def make_tool_registration(
    name: str = "run_command",
    execute_fn: ToolBrokerExecuteFn | None = None,
) -> ToolRegistration:
    return ToolRegistration(
        definition=ToolDefinition(
            name=name,
            category="execution",
            risk_level="side_effect",
            description=f"Execute {name}",
        ),
        execute_fn=execute_fn or echo_fn,
    )


# ─── Orchestrator tests ──────────────────────────────────────────────────────


class TestRunOrchestrator:
    async def test_end_to_end_returns_structured_result(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="The parser uses recursive descent."),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Explain what the parser does.",
            model_adapter=adapter,
        )

        assert result.termination == "final_answer"
        assert result.final_content == "The parser uses recursive descent."
        assert result.iterations == 1
        assert result.run_id
        assert result.trace_id

    async def test_diagnostics_include_task_classification(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="Done."),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Fix the failing parser test and verify it passes.",
            model_adapter=adapter,
        )

        assert result.diagnostics.task_type
        assert result.diagnostics.execution_mode
        assert result.diagnostics.complexity
        assert result.diagnostics.risk

    async def test_diagnostics_include_prompt_and_tool_stats(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="Done."),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Read the README file.",
            model_adapter=adapter,
        )

        assert result.diagnostics.prompt_token_estimate > 0
        assert isinstance(result.diagnostics.tools_resolved, int)
        assert isinstance(result.diagnostics.segments_admitted, int)
        assert isinstance(result.diagnostics.segments_rejected, int)

    async def test_tool_cycle_works_with_registered_tools(self):
        tool_reg = make_tool_registration()

        adapter = DeterministicAdapter([
            ToolCalls(
                calls=(ModelToolCall(
                    tool_call_id="tc-1",
                    tool_name="run_command",
                    args={"cmd": "echo hello"},
                ),),
                reasoning="Let me run this command.",
            ),
            FinalAnswer(content="Command executed successfully."),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Fix the build by running echo hello.",
            model_adapter=adapter,
            tool_registrations=[tool_reg],
        )

        assert result.termination == "final_answer"
        assert result.final_content == "Command executed successfully."
        assert result.iterations == 2
        assert len(result.loop_result.tool_results) == 1
        assert result.loop_result.tool_results[0].tool_name == "run_command"
        assert len(result.loop_result.signal_bundles) == 1

    async def test_propagates_loop_termination_reasons(self):
        adapter = DeterministicAdapter([
            ModelError(error="Service unavailable", is_transient=False),
            ModelError(error="Service unavailable", is_transient=False),
            ModelError(error="Service unavailable", is_transient=False),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Explain parsers.",
            model_adapter=adapter,
            max_iterations=10,
        )

        assert result.termination == "max_errors"
        assert "consecutive errors" in result.final_content

    async def test_loop_result_included_for_diagnostics(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="Done."),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Explain what the parser does.",
            model_adapter=adapter,
        )

        assert result.loop_result.events
        assert len(result.loop_result.events) > 0
        assert result.loop_result.events[0].kind == "model_request"

    async def test_preserves_run_id_and_trace_id(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="ok"),
        ])

        run_id = "custom-run-456"
        trace_id = "custom-trace-789"

        result = await run_orchestrator(
            run_id=run_id,
            trace_id=trace_id,
            user_message="Hello",
            model_adapter=adapter,
        )

        assert result.run_id == run_id
        assert result.trace_id == trace_id

    async def test_task_spec_and_policy_included_in_result(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="ok"),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="What is the weather?",
            model_adapter=adapter,
        )

        assert result.task_spec is not None
        assert result.task_spec.task_type
        assert result.runtime_policy is not None

    async def test_local_context_passed_through(self):
        from clawbound.contracts.types import LocalContextItem

        adapter = DeterministicAdapter([
            FinalAnswer(content="ok"),
        ])

        ctx = LocalContextItem(
            kind="file",
            ref="/src/main.py",
            content="print('hello')",
            token_estimate=5,
        )

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Read this file",
            model_adapter=adapter,
            local_context=(ctx,),
        )

        assert result.termination == "final_answer"

    async def test_max_iterations_override(self):
        adapter = DeterministicAdapter([
            FinalAnswer(content="ok"),
        ])

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Hello",
            model_adapter=adapter,
            max_iterations=1,
        )

        assert result.iterations == 1

    async def test_initial_messages_passed_through(self):
        from clawbound.contracts.types import ModelMessage

        adapter = DeterministicAdapter([
            FinalAnswer(content="Continuing from before."),
        ])

        history = (
            ModelMessage(role="user", content="First message"),
            ModelMessage(role="assistant", content="First response"),
        )

        result = await run_orchestrator(
            run_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            user_message="Continue please",
            model_adapter=adapter,
            initial_messages=history,
        )

        assert result.termination == "final_answer"
        # Verify adapter received history + new message
        sent = adapter.request_log[0]
        assert len(sent.messages) == 3  # 2 history + 1 new
