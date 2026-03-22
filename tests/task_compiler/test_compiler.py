"""TaskCompiler parity tests.

Verify classification outcomes match the original buildRoutePlan.
"""

from clawbound.contracts import LocalContextItem
from clawbound.task_compiler import TaskCompilerImpl, CompileInput


def _compiler() -> TaskCompilerImpl:
    return TaskCompilerImpl()


def _input(user_input: str, **overrides: object) -> CompileInput:
    kwargs: dict[str, object] = {
        "trace_id": "test-trace-001",
        "user_input": user_input,
        "continuation_of": None,
        "local_context": [],
    }
    kwargs.update(overrides)
    return CompileInput(**kwargs)  # type: ignore[arg-type]


class TestTaskTypeClassification:
    def test_explain_tasks_as_answer(self) -> None:
        spec = _compiler().compile_from_input(_input("Explain what this parser function does."))
        assert spec.task_type == "answer"
        assert spec.execution_mode == "answer"

    def test_fix_tasks_as_code_change(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the failing parser test without changing public API.")
        )
        assert spec.task_type == "code_change"

    def test_review_tasks_as_review(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Review the recent changes for regression risk.")
        )
        assert spec.task_type == "review"
        assert spec.execution_mode == "reviewer"

    def test_architecture_tasks(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Design an approach for restructuring the module.")
        )
        assert spec.task_type == "architecture"
        assert spec.execution_mode == "architect_like_plan"

    def test_debug_tasks(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Debug and troubleshoot the failing tests.")
        )
        assert spec.task_type == "debug"

    def test_verification_like_tasks_as_review(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Verify the tests still passes and report without editing.")
        )
        assert spec.task_type == "review"

    def test_defaults_to_answer(self) -> None:
        spec = _compiler().compile_from_input(_input("Hello world"))
        assert spec.task_type == "answer"


class TestComplexityClassification:
    def test_short_answer_tasks_trivial(self) -> None:
        spec = _compiler().compile_from_input(_input("What is this?"))
        assert spec.task_type == "answer"
        assert spec.complexity == "trivial"

    def test_review_tasks_multi_step(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Review the changes for regression risk.")
        )
        assert spec.complexity == "multi_step"

    def test_ambiguity_markers(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix this safely with a careful strategy for the refactor.")
        )
        assert spec.complexity == "ambiguous"

    def test_normal_code_tasks_bounded(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Rename the function getUser to fetchUser.")
        )
        assert spec.complexity == "bounded"


class TestRiskClassification:
    def test_public_api_high_risk(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the failing parser test without changing public api.")
        )
        assert spec.risk == "high"

    def test_payment_high_risk(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the payment processing module.")
        )
        assert spec.risk == "high"

    def test_code_change_medium_risk(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Rename the internal helper function.")
        )
        assert spec.risk == "medium"

    def test_simple_answer_low_risk(self) -> None:
        spec = _compiler().compile_from_input(
            _input("What is the meaning of life?")
        )
        assert spec.risk == "low"


class TestDomainSpecificity:
    def test_continuation_tasks(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Continue from the previous run.", continuation_of="prev-task-id")
        )
        assert spec.domain_specificity == "continuation_sensitive"

    def test_tasks_with_local_context(self) -> None:
        ctx = [LocalContextItem(kind="diff_summary", ref="working-tree", content="some diff", token_estimate=10)]
        spec = _compiler().compile_from_input(
            _input("Fix this function.", local_context=ctx)
        )
        assert spec.domain_specificity == "repo_specific"

    def test_context_free_generic(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Explain what TypeScript generics are.")
        )
        assert spec.domain_specificity == "generic"


class TestExecutionModeRouting:
    def test_high_risk_code_change_executor_then_reviewer(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the failing parser test without changing public api.")
        )
        assert spec.execution_mode == "executor_then_reviewer"

    def test_continuation_sensitive_executor_then_reviewer(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Implement the next step.", continuation_of="prev-task-id")
        )
        assert spec.execution_mode == "executor_then_reviewer"

    def test_simple_code_change_executor(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Rename the internal helper to newName.")
        )
        assert spec.execution_mode == "executor"


class TestPhase3IntentFields:
    def test_output_kind_derivation(self) -> None:
        c = _compiler()
        assert c.compile_from_input(_input("Explain this.")).output_kind == "explanation"
        assert c.compile_from_input(_input("Review the changes.")).output_kind == "review_comments"
        assert c.compile_from_input(_input("Fix this bug.")).output_kind == "code_patch"
        assert c.compile_from_input(_input("Design the architecture.")).output_kind == "plan"
        assert c.compile_from_input(_input("Debug the failing test.")).output_kind == "diagnostic"

    def test_side_effect_intent_derivation(self) -> None:
        c = _compiler()
        assert c.compile_from_input(_input("Explain this.")).side_effect_intent == "none"
        assert c.compile_from_input(_input("Review the changes.")).side_effect_intent == "none"
        assert c.compile_from_input(_input("Design the architecture.")).side_effect_intent == "proposed"
        assert c.compile_from_input(_input("Rename the internal helper.")).side_effect_intent == "immediate"

    def test_extract_file_artifacts(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the bug in src/foo/bar.ts and check tests/baz.test.ts")
        )
        refs = {a.ref for a in spec.target_artifacts if a.kind == "file"}
        assert "src/foo/bar.ts" in refs
        assert "tests/baz.test.ts" in refs

    def test_extract_url_artifacts(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Check https://example.com/api/docs for info.")
        )
        urls = [a for a in spec.target_artifacts if a.kind == "url"]
        assert any(a.ref == "https://example.com/api/docs" for a in urls)

    def test_extract_pr_and_issue_refs(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the issue from PR #123 and #456.")
        )
        artifacts = {(a.kind, a.ref) for a in spec.target_artifacts}
        assert ("pr", "#123") in artifacts
        assert ("issue", "#456") in artifacts


class TestStructuralInvariants:
    def test_all_required_fields(self) -> None:
        spec = _compiler().compile_from_input(
            _input("Fix the bug in the parser.")
        )
        assert spec.task_id
        assert spec.trace_id == "test-trace-001"
        assert spec.task_type
        assert spec.complexity
        assert spec.risk
        assert spec.domain_specificity
        assert spec.execution_mode
        assert spec.output_kind
        assert spec.side_effect_intent
        assert spec.raw_input == "Fix the bug in the parser."
        assert spec.decision_trace
        assert len(spec.decision_trace.steps) > 0

    def test_unique_task_ids(self) -> None:
        c = _compiler()
        inp = _input("Do something.")
        spec1 = c.compile_from_input(inp)
        spec2 = c.compile_from_input(inp)
        assert spec1.task_id != spec2.task_id

    def test_compile_string_shorthand(self) -> None:
        spec = _compiler().compile("Explain this function.")
        assert spec.task_type == "answer"
        assert spec.raw_input == "Explain this function."
        assert spec.trace_id
