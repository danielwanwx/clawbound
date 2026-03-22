"""PolicyEngine parity tests.

Verify context budgets match contextBudgetFor and policy fields resolve correctly.
"""

from clawbound.contracts import (
    ContextBudget,
    DecisionTrace,
    TaskSpec,
)
from clawbound.policy_engine import PolicyEngineImpl, default_runtime_config


def _engine() -> PolicyEngineImpl:
    return PolicyEngineImpl()


def _task_spec(**overrides: object) -> TaskSpec:
    defaults: dict[str, object] = {
        "task_id": "test-task-001",
        "trace_id": "test-trace-001",
        "task_type": "code_change",
        "complexity": "bounded",
        "risk": "medium",
        "domain_specificity": "generic",
        "execution_mode": "executor",
        "output_kind": "code_patch",
        "side_effect_intent": "immediate",
        "target_artifacts": (),
        "raw_input": "Fix a bug.",
        "decision_trace": DecisionTrace(
            trace_id="test-trace-001",
            summary="test",
            steps=(),
        ),
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)  # type: ignore[arg-type]


class TestContextBudgetParity:
    def test_answer_mode(self) -> None:
        policy = _engine().resolve(_task_spec(execution_mode="answer"), default_runtime_config())
        assert policy.context_budget.always_on_max_tokens == 180
        assert policy.context_budget.retrieval_max_units == 0
        assert policy.context_budget.retrieval_max_tokens == 0

    def test_executor_mode(self) -> None:
        policy = _engine().resolve(_task_spec(execution_mode="executor"), default_runtime_config())
        assert policy.context_budget.always_on_max_tokens == 200
        assert policy.context_budget.retrieval_max_units == 2
        assert policy.context_budget.retrieval_max_tokens == 120

    def test_executor_then_reviewer_mode(self) -> None:
        policy = _engine().resolve(
            _task_spec(execution_mode="executor_then_reviewer"),
            default_runtime_config(),
        )
        assert policy.context_budget.always_on_max_tokens == 220
        assert policy.context_budget.retrieval_max_units == 3
        assert policy.context_budget.retrieval_max_tokens == 180

    def test_reviewer_mode(self) -> None:
        policy = _engine().resolve(_task_spec(execution_mode="reviewer"), default_runtime_config())
        assert policy.context_budget.always_on_max_tokens == 200
        assert policy.context_budget.retrieval_max_units == 2
        assert policy.context_budget.retrieval_max_tokens == 140

    def test_architect_like_plan_mode(self) -> None:
        policy = _engine().resolve(
            _task_spec(execution_mode="architect_like_plan"),
            default_runtime_config(),
        )
        assert policy.context_budget.always_on_max_tokens == 220
        assert policy.context_budget.retrieval_max_units == 2
        assert policy.context_budget.retrieval_max_tokens == 160


class TestToolProfileResolution:
    def test_answer_read_only(self) -> None:
        policy = _engine().resolve(_task_spec(execution_mode="answer"), default_runtime_config())
        assert policy.tool_profile.profile_name == "answer"
        assert "read_file" in policy.tool_profile.allowed_tools
        assert "write_file" in policy.tool_profile.denied_tools

    def test_executor_full_access(self) -> None:
        policy = _engine().resolve(_task_spec(execution_mode="executor"), default_runtime_config())
        assert policy.tool_profile.profile_name == "executor"
        assert "write_file" in policy.tool_profile.allowed_tools
        assert "run_command" in policy.tool_profile.allowed_tools

    def test_reviewer_denies_mutations(self) -> None:
        policy = _engine().resolve(_task_spec(execution_mode="reviewer"), default_runtime_config())
        assert "write_file" in policy.tool_profile.denied_tools
        assert "edit_file" in policy.tool_profile.denied_tools

    def test_architect_denies_execution(self) -> None:
        policy = _engine().resolve(
            _task_spec(execution_mode="architect_like_plan"),
            default_runtime_config(),
        )
        assert "run_command" in policy.tool_profile.denied_tools


class TestRiskSensitivePolicy:
    def test_high_risk_disables_network_and_subagent(self) -> None:
        policy = _engine().resolve(_task_spec(risk="high"), default_runtime_config())
        assert policy.scope_bounds.allow_network_access is False
        assert policy.scope_bounds.allow_subagent_delegation is False

    def test_high_risk_requires_approval(self) -> None:
        policy = _engine().resolve(_task_spec(risk="high"), default_runtime_config())
        assert policy.approval_policy.require_approval_for_all_side_effects is True

    def test_low_risk_no_blanket_approval(self) -> None:
        policy = _engine().resolve(_task_spec(risk="low"), default_runtime_config())
        assert policy.approval_policy.require_approval_for_all_side_effects is False


class TestComplexitySensitiveIteration:
    def test_trivial_reduced_budget(self) -> None:
        policy = _engine().resolve(_task_spec(complexity="trivial"), default_runtime_config())
        assert policy.iteration_policy.max_turns <= 3
        assert policy.iteration_policy.max_tool_calls_per_turn <= 2

    def test_multi_step_expanded_budget(self) -> None:
        policy = _engine().resolve(_task_spec(complexity="multi_step"), default_runtime_config())
        assert policy.iteration_policy.max_turns >= 15

    def test_bounded_default_limits(self) -> None:
        policy = _engine().resolve(_task_spec(complexity="bounded"), default_runtime_config())
        assert policy.iteration_policy.max_turns == 10


class TestHostOverrides:
    def test_applies_context_budget_override(self) -> None:
        config = default_runtime_config(
            host_overrides={
                "context_budget": ContextBudget(
                    always_on_max_tokens=500,
                    retrieval_max_units=10,
                    retrieval_max_tokens=1000,
                    host_injection_max_tokens=200,
                    signal_max_tokens_per_result=150,
                ),
            },
        )
        policy = _engine().resolve(_task_spec(execution_mode="executor"), config)
        assert policy.context_budget.always_on_max_tokens == 500
        assert policy.context_budget.retrieval_max_units == 10


class TestDecisionTrace:
    def test_complete_trace(self) -> None:
        policy = _engine().resolve(_task_spec(), default_runtime_config())
        assert policy.decision_trace.trace_id == "test-trace-001"
        assert len(policy.decision_trace.steps) == 5
        codes = [s.code for s in policy.decision_trace.steps]
        assert codes == [
            "CONTEXT_BUDGET",
            "TOOL_PROFILE",
            "SCOPE_BOUNDS",
            "APPROVAL_POLICY",
            "ITERATION_POLICY",
        ]


class TestStructuralInvariants:
    def test_execution_mode_matches(self) -> None:
        for mode in ("answer", "executor", "executor_then_reviewer", "reviewer", "architect_like_plan"):
            policy = _engine().resolve(_task_spec(execution_mode=mode), default_runtime_config())
            assert policy.execution_mode == mode
