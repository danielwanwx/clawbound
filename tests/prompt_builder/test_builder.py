"""PromptBuilder tests — ported from builder.test.ts.

Tests cover:
- Segment creation (all 6 core segments present)
- Admission outcomes (admitted_full, admitted_trimmed, rejected)
- Rejection visibility (rejected segments remain in envelope)
- Trimming behavior
- Stable segment ordering
- Deterministic rendering
- Host injection admission/rejection
- Semantic parity with legacy prompt behavior
- Assembly stats
- Budget behavior parity
- Envelope structure
- Factory
"""

from __future__ import annotations

from clawbound.contracts import (
    ApprovalPolicy,
    ContextBudget,
    DecisionTrace,
    IterationPolicy,
    KernelAsset,
    LocalContextItem,
    PromptBuildInput,
    RetrievalUnit,
    RuntimePolicy,
    ScopeBounds,
    TaskSpec,
    ToolProfilePolicy,
)
from clawbound.prompt_builder import HostInjection, PromptBuilderImpl


# ─── Test helpers ──────────────────────────────────────────────────────────────


def _make_kernel() -> KernelAsset:
    return KernelAsset(
        version="context-kernel-v0",
        content="\n".join([
            "- Do not fabricate completion.",
            "- Do not expand scope implicitly.",
            "- Default to sparse context.",
            "- Additional context must be justified by task risk or specificity.",
            "- Prefer snippet retrieval over document loading.",
            "- Explicit current task constraints override weak historical hints.",
            "- Review mode should inspect, not take over implementation.",
            "- If extra context is not clearly useful, no-load is a valid outcome.",
        ]),
        token_estimate=50,
    )


def _make_task_spec(**overrides: object) -> TaskSpec:
    defaults: dict[str, object] = {
        "task_id": "test-task-001",
        "trace_id": "test-trace-001",
        "task_type": "answer",
        "complexity": "trivial",
        "risk": "low",
        "domain_specificity": "generic",
        "execution_mode": "answer",
        "output_kind": "explanation",
        "side_effect_intent": "none",
        "target_artifacts": (),
        "raw_input": "Explain what this parser function does.",
        "decision_trace": DecisionTrace(
            trace_id="test-trace-001", summary="test", steps=(),
        ),
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)  # type: ignore[arg-type]


def _make_policy(**overrides: object) -> RuntimePolicy:
    defaults: dict[str, object] = {
        "execution_mode": "answer",
        "context_budget": ContextBudget(
            always_on_max_tokens=180,
            retrieval_max_units=0,
            retrieval_max_tokens=0,
            host_injection_max_tokens=0,
            signal_max_tokens_per_result=0,
        ),
        "tool_profile": ToolProfilePolicy(
            profile_name="answer",
            allowed_tools=("read_file",),
            denied_tools=("write_file", "edit_file", "run_command"),
            notes=("Read-only mode for answer tasks.",),
            requires_review=False,
        ),
        "scope_bounds": ScopeBounds(
            include_paths=(),
            exclude_paths=("node_modules", ".git"),
            allow_network_access=False,
            allow_subagent_delegation=False,
        ),
        "approval_policy": ApprovalPolicy(
            require_approval_for_categories=(),
            require_approval_for_patterns=(),
            require_approval_for_all_side_effects=False,
        ),
        "iteration_policy": IterationPolicy(
            max_turns=3,
            max_tool_calls_per_turn=2,
            max_consecutive_errors=3,
            allow_retry_on_transient_error=True,
            max_transient_retries=2,
        ),
        "decision_trace": DecisionTrace(
            trace_id="test-trace-001", summary="test", steps=(),
        ),
    }
    defaults.update(overrides)
    return RuntimePolicy(**defaults)  # type: ignore[arg-type]


def _make_executor_policy() -> RuntimePolicy:
    return _make_policy(
        execution_mode="executor_then_reviewer",
        context_budget=ContextBudget(
            always_on_max_tokens=220,
            retrieval_max_units=3,
            retrieval_max_tokens=180,
            host_injection_max_tokens=120,
            signal_max_tokens_per_result=100,
        ),
        tool_profile=ToolProfilePolicy(
            profile_name="executor_then_reviewer",
            allowed_tools=("read_file", "write_file", "edit_file", "run_command"),
            denied_tools=(),
            notes=("Post-execution review step will be triggered.",),
            requires_review=True,
        ),
    )


def _make_build_input(**overrides: object) -> PromptBuildInput:
    defaults: dict[str, object] = {
        "run_id": "run-001",
        "trace_id": "trace-001",
        "task_spec": _make_task_spec(),
        "runtime_policy": _make_policy(),
        "kernel": _make_kernel(),
        "local_context": (),
        "retrieved_units": (),
        "no_load": True,
    }
    defaults.update(overrides)
    return PromptBuildInput(**defaults)  # type: ignore[arg-type]


def _make_retrieval_unit(unit_id: str, **overrides: object) -> RetrievalUnit:
    defaults: dict[str, object] = {
        "id": unit_id,
        "type": "law",
        "scope": "test-scope",
        "content": "Test retrieval content.",
        "confidence": 0.9,
        "priority": 1,
        "token_estimate": 10,
        "source_ref": "test-source",
        "tags": ("test",),
    }
    defaults.update(overrides)
    return RetrievalUnit(**defaults)  # type: ignore[arg-type]


def _builder() -> PromptBuilderImpl:
    return PromptBuilderImpl()


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestSegmentCreation:
    def test_creates_all_6_core_segments(self) -> None:
        envelope = _builder().build(_make_build_input())
        segment_ids = [s.segment_id for s in envelope.segments]
        assert segment_ids == [
            "kernel",
            "mode_instruction",
            "task_brief",
            "local_context",
            "retrieved_snippets",
            "tool_contract",
        ]

    def test_assigns_correct_owners_to_segments(self) -> None:
        envelope = _builder().build(_make_build_input())
        owner_map = {s.segment_id: s.owner for s in envelope.segments}
        assert owner_map == {
            "kernel": "runtime",
            "mode_instruction": "runtime",
            "task_brief": "task",
            "local_context": "context",
            "retrieved_snippets": "context",
            "tool_contract": "tool_contract",
        }

    def test_assigns_correct_candidate_reasons(self) -> None:
        envelope = _builder().build(_make_build_input())
        reasons = {s.segment_id: s.candidate_reason for s in envelope.segments}
        assert reasons == {
            "kernel": "always_on",
            "mode_instruction": "always_on",
            "task_brief": "task_derived",
            "local_context": "task_derived",
            "retrieved_snippets": "retrieval_gated",
            "tool_contract": "policy_required",
        }


class TestAdmissionOutcomes:
    def test_admits_all_segments_when_within_budget(self) -> None:
        envelope = _builder().build(_make_build_input())
        outcomes = [s.admission_outcome.status for s in envelope.segments]
        assert all(o == "admitted_full" for o in outcomes)

    def test_rejects_segments_when_budget_is_exhausted(self) -> None:
        envelope = _builder().build(_make_build_input(
            runtime_policy=_make_policy(
                context_budget=ContextBudget(
                    always_on_max_tokens=1,
                    retrieval_max_units=0,
                    retrieval_max_tokens=0,
                    host_injection_max_tokens=0,
                    signal_max_tokens_per_result=0,
                ),
            ),
        ))
        rejected = [s for s in envelope.segments if s.admission_outcome.status == "rejected"]
        assert len(rejected) > 0

    def test_trims_segments_that_partially_fit_budget(self) -> None:
        envelope = _builder().build(_make_build_input(
            runtime_policy=_make_policy(
                context_budget=ContextBudget(
                    always_on_max_tokens=55,
                    retrieval_max_units=0,
                    retrieval_max_tokens=0,
                    host_injection_max_tokens=0,
                    signal_max_tokens_per_result=0,
                ),
            ),
        ))
        non_full = [
            s for s in envelope.segments if s.admission_outcome.status != "admitted_full"
        ]
        assert len(non_full) > 0


class TestRejectionVisibility:
    def test_rejected_segments_remain_in_envelope(self) -> None:
        envelope = _builder().build(_make_build_input(
            runtime_policy=_make_policy(
                context_budget=ContextBudget(
                    always_on_max_tokens=1,
                    retrieval_max_units=0,
                    retrieval_max_tokens=0,
                    host_injection_max_tokens=0,
                    signal_max_tokens_per_result=0,
                ),
            ),
        ))
        # All 6 segments must still be present
        assert len(envelope.segments) == 6

        # Rejected segments should have original content for tracing
        rejected_with_content = [
            s for s in envelope.segments
            if s.admission_outcome.status == "rejected" and len(s.content) > 0
        ]
        assert len(rejected_with_content) > 0

    def test_rejected_segments_not_in_system_prompt(self) -> None:
        envelope = _builder().build(_make_build_input(
            runtime_policy=_make_policy(
                context_budget=ContextBudget(
                    always_on_max_tokens=1,
                    retrieval_max_units=0,
                    retrieval_max_tokens=0,
                    host_injection_max_tokens=0,
                    signal_max_tokens_per_result=0,
                ),
            ),
        ))
        rejected_purposes = [
            s.purpose for s in envelope.segments
            if s.admission_outcome.status == "rejected"
        ]
        for purpose in rejected_purposes:
            assert f"## {purpose}" not in envelope.system_prompt


class TestStableSegmentOrdering:
    def test_renders_segments_in_order_field_order(self) -> None:
        envelope = _builder().build(_make_build_input())
        orders = [s.order for s in envelope.segments]
        assert orders == [0, 1, 2, 3, 4, 5]

    def test_system_prompt_sections_appear_in_correct_order(self) -> None:
        envelope = _builder().build(_make_build_input())
        prompt = envelope.system_prompt

        kernel_idx = prompt.index("## ClawBound Kernel")
        mode_idx = prompt.index("## ClawBound Mode")
        task_idx = prompt.index("## Task Brief")
        context_idx = prompt.index("## Explicit Local Context")
        retrieval_idx = prompt.index("## Retrieved Snippets")
        tool_idx = prompt.index("## Tool Contract")

        assert kernel_idx < mode_idx
        assert mode_idx < task_idx
        assert task_idx < context_idx
        assert context_idx < retrieval_idx
        assert retrieval_idx < tool_idx


class TestDeterministicRendering:
    def test_produces_identical_output_for_identical_inputs(self) -> None:
        builder = _builder()
        input_ = _make_build_input()
        envelope1 = builder.build(input_)
        envelope2 = builder.build(input_)
        assert envelope1.system_prompt == envelope2.system_prompt


class TestHostInjectionAdmission:
    def test_admits_host_injections_within_budget(self) -> None:
        builder = _builder()
        injection = HostInjection(
            owner="host-test",
            purpose="Test Injection",
            content="Some host-specific guidance.",
            provenance="test/host",
        )
        envelope = builder.build_with_injections(
            _make_build_input(runtime_policy=_make_executor_policy()),
            host_injections=[injection],
        )
        injection_seg = next(
            (s for s in envelope.segments if s.segment_id == "host_injection_0"), None,
        )
        assert injection_seg is not None
        assert injection_seg.owner == "host_injection"
        assert injection_seg.candidate_reason == "host_registered"
        assert injection_seg.admission_outcome.status == "admitted_full"
        assert "## Test Injection" in envelope.system_prompt

    def test_rejects_host_injections_when_host_budget_is_zero(self) -> None:
        builder = _builder()
        injection = HostInjection(
            owner="host-test",
            purpose="Rejected Injection",
            content="This should not appear.",
            provenance="test/host",
        )
        # answer policy has host_injection_max_tokens=0
        envelope = builder.build_with_injections(
            _make_build_input(),
            host_injections=[injection],
        )
        injection_seg = next(
            (s for s in envelope.segments if s.segment_id == "host_injection_0"), None,
        )
        assert injection_seg is not None
        assert injection_seg.admission_outcome.status == "rejected"
        assert "## Rejected Injection" not in envelope.system_prompt

    def test_rejected_host_injections_remain_trace_visible(self) -> None:
        builder = _builder()
        envelope = builder.build_with_injections(
            _make_build_input(),
            host_injections=[
                HostInjection(
                    owner="host-test",
                    purpose="Trace Visible",
                    content="trace content",
                    provenance="test/host",
                ),
            ],
        )
        injection_seg = next(
            (s for s in envelope.segments if s.segment_id == "host_injection_0"), None,
        )
        assert injection_seg is not None
        assert injection_seg.trace_visible is True
        assert "trace content" in injection_seg.content


class TestSemanticParityWithLegacyPrompt:
    def test_answer_mode_produces_expected_semantic_sections(self) -> None:
        envelope = _builder().build(_make_build_input())

        # Required sections present
        assert "## ClawBound Kernel" in envelope.system_prompt
        assert "## ClawBound Mode" in envelope.system_prompt
        assert "## Task Brief" in envelope.system_prompt
        assert "## Explicit Local Context" in envelope.system_prompt
        assert "## Retrieved Snippets" in envelope.system_prompt
        assert "## Tool Contract" in envelope.system_prompt

        # Mode instruction matches legacy modeInstructionFor("answer")
        assert "Answer directly and avoid unnecessary context loading." in envelope.system_prompt

        # No-load indicator present
        assert "no_load=true" in envelope.system_prompt

        # Role correct
        assert envelope.role == "clawbound-answer"

    def test_executor_then_reviewer_mode_produces_correct_mode_instruction(self) -> None:
        envelope = _builder().build(_make_build_input(
            task_spec=_make_task_spec(
                execution_mode="executor_then_reviewer",
                raw_input="Fix the failing parser test without changing public API.",
            ),
            runtime_policy=_make_executor_policy(),
            no_load=False,
        ))
        assert "Execute cautiously, then inspect for regression and scope drift." in envelope.system_prompt
        assert envelope.role == "clawbound-executor"

    def test_reviewer_mode_uses_reviewer_role(self) -> None:
        envelope = _builder().build(_make_build_input(
            task_spec=_make_task_spec(
                execution_mode="reviewer",
                raw_input="Review the changes.",
            ),
            runtime_policy=_make_policy(
                execution_mode="reviewer",
                context_budget=ContextBudget(
                    always_on_max_tokens=200,
                    retrieval_max_units=2,
                    retrieval_max_tokens=140,
                    host_injection_max_tokens=100,
                    signal_max_tokens_per_result=80,
                ),
                tool_profile=ToolProfilePolicy(
                    profile_name="reviewer",
                    allowed_tools=("read_file", "run_tests"),
                    denied_tools=("write_file", "edit_file"),
                    notes=("Review mode.",),
                    requires_review=True,
                ),
            ),
        ))
        assert envelope.role == "clawbound-reviewer"
        assert "Inspect for scope drift, compatibility risks, and regressions" in envelope.system_prompt

    def test_local_context_formats_as_legacy_format(self) -> None:
        local_context = (
            LocalContextItem(
                kind="diff_summary",
                ref="working-tree",
                content="Parser tests are failing.",
                token_estimate=5,
            ),
        )
        envelope = _builder().build(_make_build_input(
            local_context=local_context,
            runtime_policy=_make_executor_policy(),
            task_spec=_make_task_spec(execution_mode="executor_then_reviewer"),
        ))
        assert "- [diff_summary] working-tree: Parser tests are failing." in envelope.system_prompt

    def test_retrieved_units_format_as_legacy_format(self) -> None:
        units = (
            _make_retrieval_unit(
                "law_001",
                scope="test-scope",
                source_ref="laws/law_001.json",
                content="Important law content.",
                token_estimate=8,
            ),
        )
        envelope = _builder().build(_make_build_input(
            retrieved_units=units,
            no_load=False,
            runtime_policy=_make_executor_policy(),
            task_spec=_make_task_spec(execution_mode="executor_then_reviewer"),
        ))
        assert "- [law_001] test-scope (laws/law_001.json)" in envelope.system_prompt
        assert "Important law content." in envelope.system_prompt

    def test_tool_contract_formats_as_legacy_format(self) -> None:
        envelope = _builder().build(_make_build_input())
        assert "Allowed: read_file" in envelope.system_prompt
        assert "Denied: write_file, edit_file, run_command" in envelope.system_prompt
        assert "Notes: Read-only mode for answer tasks." in envelope.system_prompt

    def test_empty_local_context_shows_none(self) -> None:
        envelope = _builder().build(_make_build_input())
        assert "## Explicit Local Context\n- none" in envelope.system_prompt

    def test_no_load_retrieval_shows_none_no_load(self) -> None:
        envelope = _builder().build(_make_build_input(no_load=True))
        assert "## Retrieved Snippets\n- none (no_load=true)" in envelope.system_prompt


class TestAssemblyStats:
    def test_tracks_per_segment_tokens(self) -> None:
        envelope = _builder().build(_make_build_input())
        assert len(envelope.assembly_stats.per_segment_tokens) == 6
        assert "kernel" in envelope.assembly_stats.per_segment_tokens

    def test_counts_admitted_and_rejected_segments(self) -> None:
        envelope = _builder().build(_make_build_input())
        assert envelope.assembly_stats.segments_admitted == 6
        assert envelope.assembly_stats.segments_rejected == 0

    def test_rejected_segments_count_as_zero_in_stats(self) -> None:
        envelope = _builder().build(_make_build_input(
            runtime_policy=_make_policy(
                context_budget=ContextBudget(
                    always_on_max_tokens=1,
                    retrieval_max_units=0,
                    retrieval_max_tokens=0,
                    host_injection_max_tokens=0,
                    signal_max_tokens_per_result=0,
                ),
            ),
        ))
        assert envelope.assembly_stats.segments_rejected > 0

        # Total tokens should only count admitted segments
        admitted_tokens = 0
        for s in envelope.segments:
            if s.admission_outcome.status == "rejected":
                continue
            if s.admission_outcome.status == "admitted_trimmed":
                admitted_tokens += s.admission_outcome.trimmed_token_estimate  # type: ignore[union-attr]
            else:
                admitted_tokens += s.token_estimate
        assert envelope.assembly_stats.total_estimated_tokens == admitted_tokens

    def test_no_load_flag_propagates_to_stats(self) -> None:
        envelope = _builder().build(_make_build_input(no_load=True))
        assert envelope.assembly_stats.no_load is True

        envelope2 = _builder().build(_make_build_input(no_load=False))
        assert envelope2.assembly_stats.no_load is False


class TestBudgetBehaviorParity:
    def test_answer_mode_utilization_stays_within_bounds(self) -> None:
        envelope = _builder().build(_make_build_input())
        assert envelope.assembly_stats.budget_utilization.always_on <= 1
        assert envelope.assembly_stats.budget_utilization.retrieval == 0

    def test_executor_mode_with_retrieval_tracks_retrieval_utilization(self) -> None:
        units = (
            _make_retrieval_unit("u1", token_estimate=15),
            _make_retrieval_unit("u2", token_estimate=20),
        )
        envelope = _builder().build(_make_build_input(
            task_spec=_make_task_spec(execution_mode="executor_then_reviewer"),
            runtime_policy=_make_executor_policy(),
            retrieved_units=units,
            no_load=False,
        ))
        assert envelope.assembly_stats.budget_utilization.retrieval > 0
        assert envelope.assembly_stats.budget_utilization.retrieval <= 1


class TestEnvelopeStructure:
    def test_populates_all_required_envelope_fields(self) -> None:
        envelope = _builder().build(_make_build_input())

        assert envelope.run_id == "run-001"
        assert envelope.trace_id == "trace-001"
        assert envelope.role
        assert isinstance(envelope.segments, tuple)
        assert envelope.assembly_stats is not None
        assert isinstance(envelope.system_prompt, str)
        assert len(envelope.system_prompt) > 0
        assert isinstance(envelope.no_load, bool)


class TestFactory:
    def test_creates_a_working_instance(self) -> None:
        builder = PromptBuilderImpl()
        assert isinstance(builder, PromptBuilderImpl)

        envelope = builder.build(_make_build_input())
        assert len(envelope.segments) == 6
