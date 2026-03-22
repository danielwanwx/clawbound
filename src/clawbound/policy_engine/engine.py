"""PolicyEngine — resolves TaskSpec + RuntimeConfig into RuntimePolicy.

Context budget logic is a 1:1 extraction from contextBudgetFor.
"""

from __future__ import annotations

from datetime import datetime, timezone

from clawbound.contracts.types import (
    ApprovalPolicy,
    ContextBudget,
    DecisionStep,
    DecisionTrace,
    ExecutionMode,
    IterationPolicy,
    RuntimeConfig,
    RuntimePolicy,
    ScopeBounds,
    TaskSpec,
    ToolProfilePolicy,
)


def default_runtime_config(**overrides: object) -> RuntimeConfig:
    """Produce a RuntimeConfig with MVP defaults."""
    budget_defaults = overrides.get("budget_defaults") or {
        "answer": ContextBudget(
            always_on_max_tokens=180,
            retrieval_max_units=0,
            retrieval_max_tokens=0,
            host_injection_max_tokens=0,
            signal_max_tokens_per_result=0,
        ),
        "executor": ContextBudget(
            always_on_max_tokens=200,
            retrieval_max_units=2,
            retrieval_max_tokens=120,
            host_injection_max_tokens=100,
            signal_max_tokens_per_result=80,
        ),
        "executor_then_reviewer": ContextBudget(
            always_on_max_tokens=220,
            retrieval_max_units=3,
            retrieval_max_tokens=180,
            host_injection_max_tokens=120,
            signal_max_tokens_per_result=100,
        ),
        "reviewer": ContextBudget(
            always_on_max_tokens=200,
            retrieval_max_units=2,
            retrieval_max_tokens=140,
            host_injection_max_tokens=100,
            signal_max_tokens_per_result=80,
        ),
        "architect_like_plan": ContextBudget(
            always_on_max_tokens=220,
            retrieval_max_units=2,
            retrieval_max_tokens=160,
            host_injection_max_tokens=120,
            signal_max_tokens_per_result=100,
        ),
    }

    tool_profile_defaults = overrides.get("tool_profile_defaults") or {
        "answer": ToolProfilePolicy(
            profile_name="answer",
            allowed_tools=("read_file",),
            denied_tools=("run_command", "write_file", "edit_file"),
            notes=("Read-only mode for answer tasks.",),
            requires_review=False,
        ),
        "executor": ToolProfilePolicy(
            profile_name="executor",
            allowed_tools=("read_file", "write_file", "edit_file", "run_command", "list_dir", "search"),
            denied_tools=(),
            notes=(),
            requires_review=False,
        ),
        "executor_then_reviewer": ToolProfilePolicy(
            profile_name="executor_then_reviewer",
            allowed_tools=("read_file", "write_file", "edit_file", "run_command", "list_dir", "search"),
            denied_tools=(),
            notes=("Post-execution review step will be triggered.",),
            requires_review=True,
        ),
        "reviewer": ToolProfilePolicy(
            profile_name="reviewer",
            allowed_tools=("read_file", "list_dir", "search", "run_command"),
            denied_tools=("write_file", "edit_file"),
            notes=("Reviewer mode: read and verify, no mutations.",),
            requires_review=False,
        ),
        "architect_like_plan": ToolProfilePolicy(
            profile_name="architect_like_plan",
            allowed_tools=("read_file", "list_dir", "search"),
            denied_tools=("write_file", "edit_file", "run_command"),
            notes=("Planning mode: analysis only, no execution.",),
            requires_review=False,
        ),
    }

    default_scope_bounds = overrides.get("default_scope_bounds") or ScopeBounds(
        include_paths=(),
        exclude_paths=("node_modules", ".git", "dist", "build"),
        allow_network_access=False,
        allow_subagent_delegation=False,
    )

    default_approval_policy = overrides.get("default_approval_policy") or ApprovalPolicy(
        require_approval_for_categories=("destructive",),
        require_approval_for_patterns=(),
        require_approval_for_all_side_effects=False,
    )

    default_iteration_policy = overrides.get("default_iteration_policy") or IterationPolicy(
        max_turns=10,
        max_tool_calls_per_turn=5,
        max_consecutive_errors=3,
        allow_retry_on_transient_error=True,
        max_transient_retries=2,
    )

    host_overrides = overrides.get("host_overrides")

    return RuntimeConfig(
        budget_defaults=budget_defaults,  # type: ignore[arg-type]
        tool_profile_defaults=tool_profile_defaults,  # type: ignore[arg-type]
        default_scope_bounds=default_scope_bounds,  # type: ignore[arg-type]
        default_approval_policy=default_approval_policy,  # type: ignore[arg-type]
        default_iteration_policy=default_iteration_policy,  # type: ignore[arg-type]
        host_overrides=host_overrides if host_overrides else None,  # type: ignore[arg-type]
    )


class PolicyEngineImpl:
    """Resolves TaskSpec + RuntimeConfig → RuntimePolicy."""

    def resolve(self, task_spec: TaskSpec, runtime_config: RuntimeConfig) -> RuntimePolicy:
        return _resolve_policy(task_spec, runtime_config)


def _resolve_policy(task_spec: TaskSpec, config: RuntimeConfig) -> RuntimePolicy:
    mode = task_spec.execution_mode
    steps: list[DecisionStep] = []
    now = datetime.now(timezone.utc).isoformat()

    # Context budget
    context_budget = _resolve_context_budget(mode, config)
    steps.append(DecisionStep(
        code="CONTEXT_BUDGET",
        evidence=f"always_on={context_budget.always_on_max_tokens}, retrieval_units={context_budget.retrieval_max_units}, retrieval_tokens={context_budget.retrieval_max_tokens}",
        effect={"context_budget": context_budget.model_dump()},
        reason="Context budget determined by execution mode and sparse-context limits.",
        timestamp=now,
    ))

    # Tool profile
    tool_profile = _resolve_tool_profile(mode, config)
    steps.append(DecisionStep(
        code="TOOL_PROFILE",
        evidence=f"profile={tool_profile.profile_name}, allowed={len(tool_profile.allowed_tools)}, denied={len(tool_profile.denied_tools)}",
        effect={"profile_name": tool_profile.profile_name},
        reason="Tool profile selected from execution mode defaults.",
        timestamp=now,
    ))

    # Scope bounds
    scope_bounds = _resolve_scope_bounds(task_spec, config)
    steps.append(DecisionStep(
        code="SCOPE_BOUNDS",
        evidence=f"network={scope_bounds.allow_network_access}, subagent={scope_bounds.allow_subagent_delegation}",
        effect={"scope_bounds": scope_bounds.model_dump()},
        reason="Scope bounds from config defaults, adjusted by task risk.",
        timestamp=now,
    ))

    # Approval policy
    approval_policy = _resolve_approval_policy(task_spec, config)
    steps.append(DecisionStep(
        code="APPROVAL_POLICY",
        evidence=f"all_side_effects={approval_policy.require_approval_for_all_side_effects}, categories={len(approval_policy.require_approval_for_categories)}",
        effect={"approval_policy": approval_policy.model_dump()},
        reason="Approval policy tightened for high-risk tasks.",
        timestamp=now,
    ))

    # Iteration policy
    iteration_policy = _resolve_iteration_policy(task_spec, config)
    steps.append(DecisionStep(
        code="ITERATION_POLICY",
        evidence=f"max_turns={iteration_policy.max_turns}, max_tool_calls={iteration_policy.max_tool_calls_per_turn}",
        effect={"iteration_policy": iteration_policy.model_dump()},
        reason="Iteration limits from defaults, adjusted by complexity.",
        timestamp=now,
    ))

    decision_trace = DecisionTrace(
        trace_id=task_spec.trace_id,
        summary=f"mode={mode};budget_always_on={context_budget.always_on_max_tokens};profile={tool_profile.profile_name}",
        steps=tuple(steps),
    )

    policy = RuntimePolicy(
        execution_mode=mode,
        context_budget=context_budget,
        tool_profile=tool_profile,
        scope_bounds=scope_bounds,
        approval_policy=approval_policy,
        iteration_policy=iteration_policy,
        decision_trace=decision_trace,
    )

    if config.host_overrides:
        return _apply_host_overrides(policy, config.host_overrides)

    return policy


def _resolve_context_budget(mode: ExecutionMode, config: RuntimeConfig) -> ContextBudget:
    budget = config.budget_defaults.get(mode)
    if budget is not None:
        return budget
    return config.budget_defaults["executor"]


def _resolve_tool_profile(mode: ExecutionMode, config: RuntimeConfig) -> ToolProfilePolicy:
    profile = config.tool_profile_defaults.get(mode)
    if profile is not None:
        return profile
    return config.tool_profile_defaults["executor"]


def _resolve_scope_bounds(task_spec: TaskSpec, config: RuntimeConfig) -> ScopeBounds:
    base = config.default_scope_bounds
    if task_spec.risk == "high":
        return ScopeBounds(
            include_paths=base.include_paths,
            exclude_paths=base.exclude_paths,
            allow_network_access=False,
            allow_subagent_delegation=False,
        )
    return base


def _resolve_approval_policy(task_spec: TaskSpec, config: RuntimeConfig) -> ApprovalPolicy:
    base = config.default_approval_policy
    if task_spec.risk == "high":
        return ApprovalPolicy(
            require_approval_for_categories=base.require_approval_for_categories,
            require_approval_for_patterns=base.require_approval_for_patterns,
            require_approval_for_all_side_effects=True,
        )
    return base


def _resolve_iteration_policy(task_spec: TaskSpec, config: RuntimeConfig) -> IterationPolicy:
    base = config.default_iteration_policy
    if task_spec.complexity == "trivial":
        return IterationPolicy(
            max_turns=min(base.max_turns, 3),
            max_tool_calls_per_turn=min(base.max_tool_calls_per_turn, 2),
            max_consecutive_errors=base.max_consecutive_errors,
            allow_retry_on_transient_error=base.allow_retry_on_transient_error,
            max_transient_retries=base.max_transient_retries,
        )
    if task_spec.complexity in ("multi_step", "ambiguous"):
        return IterationPolicy(
            max_turns=max(base.max_turns, 15),
            max_tool_calls_per_turn=base.max_tool_calls_per_turn,
            max_consecutive_errors=base.max_consecutive_errors,
            allow_retry_on_transient_error=base.allow_retry_on_transient_error,
            max_transient_retries=base.max_transient_retries,
        )
    return base


def _apply_host_overrides(
    policy: RuntimePolicy,
    overrides: dict[str, object],
) -> RuntimePolicy:
    updates: dict[str, object] = {}
    for key in ("execution_mode", "context_budget", "tool_profile", "scope_bounds", "approval_policy", "iteration_policy"):
        if key in overrides and overrides[key] is not None:
            updates[key] = overrides[key]
    if updates:
        return policy.model_copy(update=updates)
    return policy
