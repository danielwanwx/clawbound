"""ActionGate tests — policy-based tool call gating.

Tests cover:
- Allowed when tool is in allowed list
- Denied when tool is in denied list
- Denied when tool is not in allowed list (non-empty allowed list)
- Denied-list takes precedence if tool appears in both
- Empty allowed list allows all non-denied tools
- Reason messages include tool name and profile name
"""

from __future__ import annotations

from clawbound.contracts import (
    ApprovalPolicy,
    ContextBudget,
    DecisionTrace,
    IterationPolicy,
    RuntimePolicy,
    ScopeBounds,
    ToolProfilePolicy,
)
from clawbound.execution_loop import ActionGateImpl


# ─── Test helpers ──────────────────────────────────────────────────────────────


def _gate() -> ActionGateImpl:
    return ActionGateImpl()


def _make_policy(
    allowed: tuple[str, ...] = (),
    denied: tuple[str, ...] = (),
    profile_name: str = "test-profile",
) -> RuntimePolicy:
    return RuntimePolicy(
        execution_mode="executor",
        context_budget=ContextBudget(
            always_on_max_tokens=200,
            retrieval_max_units=0,
            retrieval_max_tokens=0,
            host_injection_max_tokens=0,
            signal_max_tokens_per_result=0,
        ),
        tool_profile=ToolProfilePolicy(
            profile_name=profile_name,
            allowed_tools=allowed,
            denied_tools=denied,
            notes=(),
            requires_review=False,
        ),
        scope_bounds=ScopeBounds(
            include_paths=(),
            exclude_paths=(),
            allow_network_access=False,
            allow_subagent_delegation=False,
        ),
        approval_policy=ApprovalPolicy(
            require_approval_for_categories=(),
            require_approval_for_patterns=(),
            require_approval_for_all_side_effects=False,
        ),
        iteration_policy=IterationPolicy(
            max_turns=10,
            max_tool_calls_per_turn=5,
            max_consecutive_errors=3,
            allow_retry_on_transient_error=True,
            max_transient_retries=2,
        ),
        decision_trace=DecisionTrace(
            trace_id="test-trace", summary="test", steps=(),
        ),
    )


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestActionGateAllowed:
    def test_allowed_when_in_allowed_list(self) -> None:
        policy = _make_policy(allowed=("read_file", "write_file"))
        decision = _gate().check("read_file", {}, policy)
        assert decision.allowed is True

    def test_allowed_when_allowed_list_is_empty_and_not_denied(self) -> None:
        policy = _make_policy(allowed=(), denied=("dangerous_tool",))
        decision = _gate().check("read_file", {}, policy)
        assert decision.allowed is True


class TestActionGateDenied:
    def test_denied_when_in_denied_list(self) -> None:
        policy = _make_policy(
            allowed=("read_file",),
            denied=("write_file",),
        )
        decision = _gate().check("write_file", {}, policy)
        assert decision.allowed is False
        assert "write_file" in decision.reason  # type: ignore[union-attr]

    def test_denied_when_not_in_allowed_list(self) -> None:
        policy = _make_policy(allowed=("read_file",))
        decision = _gate().check("write_file", {}, policy)
        assert decision.allowed is False
        assert "write_file" in decision.reason  # type: ignore[union-attr]

    def test_denied_list_takes_precedence_over_allowed(self) -> None:
        policy = _make_policy(
            allowed=("read_file", "write_file"),
            denied=("write_file",),
        )
        decision = _gate().check("write_file", {}, policy)
        assert decision.allowed is False

    def test_denial_reason_includes_profile_name(self) -> None:
        policy = _make_policy(
            allowed=("read_file",),
            denied=("write_file",),
            profile_name="answer",
        )
        decision = _gate().check("write_file", {}, policy)
        assert decision.allowed is False
        assert "answer" in decision.reason  # type: ignore[union-attr]
