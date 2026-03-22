"""PromptBuilder — budget-aware segment admission and deterministic rendering.

Creates structured PromptSegments from inputs, applies budget-aware admission,
and produces a PromptEnvelope with deterministic rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clawbound.contracts.types import (
    AdmittedFull,
    AdmittedTrimmed,
    BudgetUtilization,
    ExecutionMode,
    LocalContextItem,
    PromptAssemblyStats,
    PromptBuildInput,
    PromptEnvelope,
    PromptSegment,
    Rejected,
    RetrievalUnit,
    SegmentAdmissionOutcome,
    SegmentCandidateReason,
    SegmentOwner,
    ToolProfilePolicy,
)
from clawbound.shared.tokens import estimate_tokens_from_items, estimate_tokens_from_text

from .renderer import render_segments_to_system_prompt


@dataclass(frozen=True)
class HostInjection:
    owner: str
    purpose: str
    content: str
    provenance: str
    budget_cap: int | None = None


class PromptBuilderImpl:
    def build(self, input: PromptBuildInput) -> PromptEnvelope:
        return _build_envelope(input)

    def build_with_injections(
        self, input: PromptBuildInput, host_injections: list[HostInjection] | None = None,
    ) -> PromptEnvelope:
        return _build_envelope(input, host_injections or [])


# ─── Core build logic ─────────────────────────────────────────────────────────


def _build_envelope(
    input: PromptBuildInput,
    host_injections: list[HostInjection] | None = None,
) -> PromptEnvelope:
    budget = input.runtime_policy.context_budget
    tool_profile = input.runtime_policy.tool_profile

    candidates = _create_candidate_segments(input, tool_profile, host_injections or [])
    segments = _apply_admission(
        candidates,
        budget.always_on_max_tokens,
        budget.retrieval_max_tokens,
        budget.host_injection_max_tokens,
    )
    assembly_stats = _compute_assembly_stats(segments, budget, input.no_load)
    system_prompt = render_segments_to_system_prompt(segments)

    return PromptEnvelope(
        run_id=input.run_id,
        trace_id=input.trace_id,
        role=_role_for_mode(input.task_spec.execution_mode),
        segments=tuple(segments),
        assembly_stats=assembly_stats,
        system_prompt=system_prompt,
        no_load=input.no_load,
    )


# ─── Candidate segment ─────────────────────────────────────────────────────────


@dataclass
class _CandidateSegment:
    segment_id: str
    owner: SegmentOwner
    purpose: str
    content: str
    token_estimate: int
    candidate_reason: SegmentCandidateReason
    provenance: str
    trace_visible: bool
    order: int
    budget_category: str  # "alwaysOn" | "retrieval" | "hostInjection"
    injection_budget_cap: int | None = None


def _create_candidate_segments(
    input: PromptBuildInput,
    tool_profile: ToolProfilePolicy,
    host_injections: list[HostInjection],
) -> list[_CandidateSegment]:
    candidates: list[_CandidateSegment] = []

    # 0: Kernel
    candidates.append(_CandidateSegment(
        segment_id="kernel",
        owner="runtime",
        purpose="ClawBound Kernel",
        content=f"## ClawBound Kernel\n{input.kernel.content}",
        token_estimate=input.kernel.token_estimate,
        candidate_reason="always_on",
        provenance=input.kernel.version,
        trace_visible=True,
        order=0,
        budget_category="alwaysOn",
    ))

    # 1: Mode instruction
    mode_instruction = _mode_instruction_for(input.task_spec.execution_mode)
    candidates.append(_CandidateSegment(
        segment_id="mode_instruction",
        owner="runtime",
        purpose="ClawBound Mode",
        content=f"## ClawBound Mode\n{mode_instruction}",
        token_estimate=estimate_tokens_from_text(mode_instruction),
        candidate_reason="always_on",
        provenance="runtime/mode_instruction",
        trace_visible=True,
        order=1,
        budget_category="alwaysOn",
    ))

    # 2: Task brief
    candidates.append(_CandidateSegment(
        segment_id="task_brief",
        owner="task",
        purpose="Task Brief",
        content=f"## Task Brief\n{input.task_spec.raw_input}",
        token_estimate=estimate_tokens_from_text(input.task_spec.raw_input),
        candidate_reason="task_derived",
        provenance=f"task/{input.task_spec.task_id}",
        trace_visible=True,
        order=2,
        budget_category="alwaysOn",
    ))

    # 3: Local context
    local_context_content = _format_local_context(list(input.local_context))
    candidates.append(_CandidateSegment(
        segment_id="local_context",
        owner="context",
        purpose="Explicit Local Context",
        content=f"## Explicit Local Context\n{local_context_content}",
        token_estimate=estimate_tokens_from_items([item.content for item in input.local_context]),
        candidate_reason="task_derived",
        provenance="context/local",
        trace_visible=True,
        order=3,
        budget_category="alwaysOn",
    ))

    # 4: Retrieved snippets
    retrieval_content = _format_retrieved_units(list(input.retrieved_units), input.no_load)
    candidates.append(_CandidateSegment(
        segment_id="retrieved_snippets",
        owner="context",
        purpose="Retrieved Snippets",
        content=f"## Retrieved Snippets\n{retrieval_content}",
        token_estimate=sum(u.token_estimate for u in input.retrieved_units),
        candidate_reason="retrieval_gated",
        provenance="context/retrieval",
        trace_visible=True,
        order=4,
        budget_category="retrieval",
    ))

    # 5: Tool contract
    tool_contract_content = _format_tool_contract(tool_profile)
    candidates.append(_CandidateSegment(
        segment_id="tool_contract",
        owner="tool_contract",
        purpose="Tool Contract",
        content=f"## Tool Contract\n{tool_contract_content}",
        token_estimate=estimate_tokens_from_items([
            *tool_profile.allowed_tools, *tool_profile.denied_tools, *tool_profile.notes,
        ]),
        candidate_reason="policy_required",
        provenance=f"policy/{tool_profile.profile_name}",
        trace_visible=True,
        order=5,
        budget_category="alwaysOn",
    ))

    # 6+: Host injections
    for i, injection in enumerate(host_injections):
        candidates.append(_CandidateSegment(
            segment_id=f"host_injection_{i}",
            owner="host_injection",
            purpose=injection.purpose,
            content=f"## {injection.purpose}\n{injection.content}",
            token_estimate=estimate_tokens_from_text(injection.content),
            candidate_reason="host_registered",
            provenance=injection.provenance,
            trace_visible=True,
            order=6 + i,
            budget_category="hostInjection",
            injection_budget_cap=injection.budget_cap,
        ))

    return candidates


# ─── Budget-aware admission ───────────────────────────────────────────────────


def _apply_admission(
    candidates: list[_CandidateSegment],
    always_on_budget: int,
    retrieval_budget: int,
    host_injection_budget: int,
) -> list[PromptSegment]:
    used = {"alwaysOn": 0, "retrieval": 0, "hostInjection": 0}
    budgets = {"alwaysOn": always_on_budget, "retrieval": retrieval_budget, "hostInjection": host_injection_budget}
    segments: list[PromptSegment] = []

    for candidate in candidates:
        cat = candidate.budget_category
        budget_cap = _resolve_budget_cap(candidate, budgets)
        remaining = budgets[cat] - used[cat]

        admission_outcome: SegmentAdmissionOutcome
        content = candidate.content

        if candidate.token_estimate <= remaining:
            admission_outcome = AdmittedFull()
            used[cat] += candidate.token_estimate
        elif remaining > 0 and candidate.token_estimate > 0:
            trimmed_estimate = remaining
            admission_outcome = AdmittedTrimmed(
                original_token_estimate=candidate.token_estimate,
                trimmed_token_estimate=trimmed_estimate,
                trim_reason="budget_cap",
            )
            content = _trim_content(candidate.content, trimmed_estimate)
            used[cat] += trimmed_estimate
        else:
            admission_outcome = Rejected(rejection_reason="budget_exhausted")

        segments.append(PromptSegment(
            segment_id=candidate.segment_id,
            owner=candidate.owner,
            purpose=candidate.purpose,
            content=content,
            token_estimate=candidate.token_estimate,
            budget_cap=budget_cap,
            candidate_reason=candidate.candidate_reason,
            admission_outcome=admission_outcome,
            provenance=candidate.provenance,
            trace_visible=candidate.trace_visible,
            order=candidate.order,
        ))

    return segments


def _resolve_budget_cap(candidate: _CandidateSegment, budgets: dict[str, int]) -> int:
    if candidate.budget_category == "alwaysOn":
        return budgets["alwaysOn"]
    if candidate.budget_category == "retrieval":
        return budgets["retrieval"]
    if candidate.injection_budget_cap is not None:
        return min(budgets["hostInjection"], candidate.injection_budget_cap)
    return budgets["hostInjection"]


def _trim_content(content: str, trimmed_tokens: int) -> str:
    approx_chars = trimmed_tokens * 5
    if len(content) <= approx_chars:
        return content
    return content[:approx_chars] + "\n[trimmed]"


# ─── Assembly stats ─────────────────────────────────────────────────────────


def _compute_assembly_stats(
    segments: list[PromptSegment],
    budget: Any,
    no_load: bool,
) -> PromptAssemblyStats:
    per_segment_tokens: dict[str, int] = {}
    always_on_total = 0
    retrieval_total = 0
    host_injection_total = 0
    total_segment_tokens = 0
    admitted = 0
    rejected = 0

    for seg in segments:
        effective = _effective_token_count(seg)
        per_segment_tokens[seg.segment_id] = effective
        total_segment_tokens += effective

        if seg.admission_outcome.status == "rejected":
            rejected += 1
        else:
            admitted += 1
            if seg.owner == "host_injection":
                host_injection_total += effective
            elif seg.candidate_reason == "retrieval_gated":
                retrieval_total += effective
            else:
                always_on_total += effective

    return PromptAssemblyStats(
        per_segment_tokens=per_segment_tokens,
        total_segment_tokens=total_segment_tokens,
        total_estimated_tokens=total_segment_tokens,
        budget_utilization=BudgetUtilization(
            always_on=always_on_total / budget.always_on_max_tokens if budget.always_on_max_tokens > 0 else 0,
            retrieval=retrieval_total / budget.retrieval_max_tokens if budget.retrieval_max_tokens > 0 else 0,
            host_injection=host_injection_total / budget.host_injection_max_tokens if budget.host_injection_max_tokens > 0 else 0,
        ),
        no_load=no_load,
        segments_admitted=admitted,
        segments_rejected=rejected,
    )


def _effective_token_count(seg: PromptSegment) -> int:
    if seg.admission_outcome.status == "rejected":
        return 0
    if seg.admission_outcome.status == "admitted_trimmed":
        return seg.admission_outcome.trimmed_token_estimate
    return seg.token_estimate


# ─── Formatting helpers ─────────────────────────────────────────────────────


def _format_local_context(items: list[LocalContextItem]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- [{item.kind}] {item.ref}: {item.content}" for item in items)


def _format_retrieved_units(units: list[RetrievalUnit], no_load: bool) -> str:
    if not units:
        return f"- none (no_load={str(no_load).lower()})"
    return "\n".join(
        f"- [{unit.id}] {unit.scope} ({unit.source_ref})\n{unit.content}"
        for unit in units
    )


def _format_tool_contract(tool_profile: ToolProfilePolicy) -> str:
    allowed = ", ".join(tool_profile.allowed_tools) or "none"
    denied = ", ".join(tool_profile.denied_tools) or "none"
    notes = " ".join(tool_profile.notes) or "none"
    return f"Allowed: {allowed}\nDenied: {denied}\nNotes: {notes}"


# ─── Mode helpers ─────────────────────────────────────────────────────────────


def _mode_instruction_for(execution_mode: ExecutionMode) -> str:
    match execution_mode:
        case "answer":
            return "Answer directly and avoid unnecessary context loading."
        case "reviewer":
            return "Inspect for scope drift, compatibility risks, and regressions without taking over implementation."
        case "architect_like_plan":
            return "Produce a scoped plan with explicit tradeoffs and constraints."
        case "executor_then_reviewer":
            return "Execute cautiously, then inspect for regression and scope drift."
        case _:
            return "Make a bounded change and preserve external behavior unless told otherwise."


def _role_for_mode(execution_mode: ExecutionMode) -> str:
    match execution_mode:
        case "reviewer":
            return "clawbound-reviewer"
        case "architect_like_plan":
            return "clawbound-architect"
        case "answer":
            return "clawbound-answer"
        case _:
            return "clawbound-executor"
