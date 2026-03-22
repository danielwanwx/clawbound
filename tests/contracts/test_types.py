"""Tests for clawbound.contracts.types — verify Pydantic models."""

import pytest
from pydantic import ValidationError

from clawbound.contracts import (
    DecisionStep,
    DecisionTrace,
    TargetArtifact,
    TaskSpec,
    ContextBudget,
    ToolProfilePolicy,
    ScopeBounds,
    ApprovalPolicy,
    IterationPolicy,
    RuntimePolicy,
    AdmittedFull,
    AdmittedTrimmed,
    Rejected,
    PromptSegment,
    KernelAsset,
    FinalAnswer,
    ToolCalls,
    ModelError,
    ModelToolCall,
    ModelMessage,
    ActionAllowed,
    ActionDenied,
    GenericSignal,
    TestResultsSignal,
    TestSummary,
    CompressionMetrics,
    SignalBundle,
    ToolDefinition,
    SessionBounds,
    SessionSnapshot,
)


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_decision_trace(trace_id: str = "t-1") -> DecisionTrace:
    return DecisionTrace(
        trace_id=trace_id,
        summary="test",
        steps=(),
    )


def _make_task_spec(**overrides: object) -> TaskSpec:
    defaults: dict[str, object] = {
        "task_id": "task-1",
        "trace_id": "trace-1",
        "task_type": "answer",
        "complexity": "trivial",
        "risk": "low",
        "domain_specificity": "generic",
        "execution_mode": "answer",
        "output_kind": "explanation",
        "side_effect_intent": "none",
        "target_artifacts": (),
        "raw_input": "test input",
        "decision_trace": _make_decision_trace(),
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)  # type: ignore[arg-type]


def _make_runtime_policy(**overrides: object) -> RuntimePolicy:
    defaults: dict[str, object] = {
        "execution_mode": "answer",
        "context_budget": ContextBudget(
            always_on_max_tokens=4000,
            retrieval_max_units=5,
            retrieval_max_tokens=2000,
            host_injection_max_tokens=1000,
            signal_max_tokens_per_result=500,
        ),
        "tool_profile": ToolProfilePolicy(
            profile_name="default",
            allowed_tools=(),
            denied_tools=(),
            notes=(),
            requires_review=False,
        ),
        "scope_bounds": ScopeBounds(
            include_paths=(),
            exclude_paths=(),
            allow_network_access=False,
            allow_subagent_delegation=False,
        ),
        "approval_policy": ApprovalPolicy(
            require_approval_for_categories=(),
            require_approval_for_patterns=(),
            require_approval_for_all_side_effects=False,
        ),
        "iteration_policy": IterationPolicy(
            max_turns=5,
            max_tool_calls_per_turn=3,
            max_consecutive_errors=2,
            allow_retry_on_transient_error=True,
            max_transient_retries=1,
        ),
        "decision_trace": _make_decision_trace(),
    }
    defaults.update(overrides)
    return RuntimePolicy(**defaults)  # type: ignore[arg-type]


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestTaskSpec:
    def test_creates_with_valid_fields(self) -> None:
        spec = _make_task_spec()
        assert spec.task_type == "answer"
        assert spec.complexity == "trivial"

    def test_frozen(self) -> None:
        spec = _make_task_spec()
        with pytest.raises(ValidationError):
            spec.task_type = "review"  # type: ignore[misc]

    def test_12_fields(self) -> None:
        """Guardrail #1: TaskSpec ≤ 12 fields."""
        _make_task_spec()  # verify it can be constructed
        assert len(TaskSpec.model_fields) == 12

    def test_invalid_task_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_task_spec(task_type="invalid")

    def test_target_artifacts(self) -> None:
        artifact = TargetArtifact(kind="file", ref="src/main.ts")
        spec = _make_task_spec(target_artifacts=(artifact,))
        assert len(spec.target_artifacts) == 1
        assert spec.target_artifacts[0].ref == "src/main.ts"


class TestRuntimePolicy:
    def test_creates_with_valid_fields(self) -> None:
        policy = _make_runtime_policy()
        assert policy.execution_mode == "answer"
        assert policy.iteration_policy.max_turns == 5

    def test_frozen(self) -> None:
        policy = _make_runtime_policy()
        with pytest.raises(ValidationError):
            policy.execution_mode = "reviewer"  # type: ignore[misc]


class TestDecisionTrace:
    def test_empty_steps(self) -> None:
        trace = _make_decision_trace()
        assert len(trace.steps) == 0

    def test_with_steps(self) -> None:
        step = DecisionStep(
            code="classify",
            evidence="contains question mark",
            effect={"task_type": "answer"},
            reason="user asked a question",
            timestamp="2024-01-01T00:00:00Z",
        )
        trace = DecisionTrace(trace_id="t-1", summary="test", steps=(step,))
        assert len(trace.steps) == 1
        assert trace.steps[0].code == "classify"


class TestSegmentAdmissionOutcome:
    def test_admitted_full(self) -> None:
        outcome = AdmittedFull()
        assert outcome.status == "admitted_full"

    def test_admitted_trimmed(self) -> None:
        outcome = AdmittedTrimmed(
            original_token_estimate=100,
            trimmed_token_estimate=50,
            trim_reason="budget_cap",
        )
        assert outcome.status == "admitted_trimmed"
        assert outcome.trimmed_token_estimate == 50

    def test_rejected(self) -> None:
        outcome = Rejected(rejection_reason="budget_exhausted")
        assert outcome.status == "rejected"


class TestPromptSegment:
    def test_creates_segment(self) -> None:
        segment = PromptSegment(
            segment_id="seg-1",
            owner="runtime",
            purpose="kernel",
            content="test content",
            token_estimate=10,
            budget_cap=100,
            candidate_reason="always_on",
            admission_outcome=AdmittedFull(),
            provenance="test",
            trace_visible=True,
            order=0,
        )
        assert segment.owner == "runtime"
        assert segment.admission_outcome.status == "admitted_full"


class TestModelResponse:
    def test_final_answer(self) -> None:
        resp = FinalAnswer(content="42")
        assert resp.kind == "final_answer"
        assert resp.content == "42"

    def test_tool_calls(self) -> None:
        call = ModelToolCall(tool_call_id="tc-1", tool_name="read_file", args={"path": "."})
        resp = ToolCalls(calls=(call,))
        assert resp.kind == "tool_calls"
        assert len(resp.calls) == 1

    def test_error(self) -> None:
        resp = ModelError(error="timeout", is_transient=True)
        assert resp.kind == "error"
        assert resp.is_transient is True


class TestModelMessage:
    def test_user_message(self) -> None:
        msg = ModelMessage(role="user", content="hello")
        assert msg.role == "user"

    def test_assistant_with_tool_calls(self) -> None:
        call = ModelToolCall(tool_call_id="tc-1", tool_name="read", args={})
        msg = ModelMessage(role="assistant", content="", tool_calls=(call,))
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1


class TestActionGateDecision:
    def test_allowed(self) -> None:
        d = ActionAllowed()
        assert d.allowed is True

    def test_denied(self) -> None:
        d = ActionDenied(reason="policy blocked")
        assert d.allowed is False
        assert d.reason == "policy blocked"


class TestKernelAsset:
    def test_creates(self) -> None:
        k = KernelAsset(version="v0", content="test", token_estimate=5)
        assert k.version == "v0"


class TestSignalTypes:
    def test_generic_signal(self) -> None:
        s = GenericSignal(extracted={"key": "value"})
        assert s.kind == "generic"

    def test_test_results_signal(self) -> None:
        s = TestResultsSignal(
            summary=TestSummary(total=10, passed=8, failed=2, skipped=0),
            failures=(),
        )
        assert s.kind == "test_results"
        assert s.summary.total == 10


class TestToolDefinition:
    def test_creates(self) -> None:
        td = ToolDefinition(
            name="read_file",
            category="filesystem",
            risk_level="read_only",
        )
        assert td.name == "read_file"


class TestSignalBundle:
    def test_creates(self) -> None:
        bundle = SignalBundle(
            tool_call_id="tc-1",
            tool_name="run_command",
            structured=GenericSignal(extracted={"status": "ok"}),
            compressed_text="status ok",
            compression_metrics=CompressionMetrics(
                original_tokens=100,
                compressed_tokens=10,
                compression_ratio=0.1,
                classified_as="generic",
                filter_applied="none",
                loss_risk="low",
            ),
        )
        assert bundle.tool_name == "run_command"
        assert bundle.compression_metrics.compression_ratio == 0.1


class TestSessionTypes:
    def test_session_snapshot(self) -> None:
        spec = _make_task_spec()
        policy = _make_runtime_policy()
        snapshot = SessionSnapshot(
            session_id="s-1",
            run_id="r-1",
            trace_id="t-1",
            task_spec=spec,
            policy=policy,
            turns=(),
            bounds=SessionBounds(
                max_turns=10,
                max_stored_tokens=10000,
                was_compacted=False,
                retained_turns=0,
            ),
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert snapshot.session_id == "s-1"
        assert snapshot.compacted_summary is None
