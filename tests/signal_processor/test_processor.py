"""SignalProcessor tests — ported from processor.test.ts.

Covers: deterministic transformation, required output kinds (test_results,
build_output, generic), optional kinds (lint, directory, json), compression
behavior, failure-first extraction, repetition collapse, metrics reporting.
"""

from __future__ import annotations

import json
from typing import Any

from clawbound.contracts import ToolResult
from clawbound.signal_processor import SignalProcessorImpl


# ─── Test helpers ──────────────────────────────────────────────────────────────


def _make_tool_result(**overrides: Any) -> ToolResult:
    defaults: dict[str, Any] = {
        "tool_name": "test_tool",
        "tool_call_id": "call-001",
        "status": "success",
        "raw_output": "some output",
        "media_type": "text/plain",
        "output_kind": "generic",
        "duration_ms": 100,
        "metadata": {},
    }
    defaults.update(overrides)
    return ToolResult(**defaults)


def _processor() -> SignalProcessorImpl:
    return SignalProcessorImpl()


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestDeterministicTransformation:
    def test_produces_identical_output_for_identical_input(self) -> None:
        processor = _processor()
        result = _make_tool_result()
        bundle1 = processor.process(result)
        bundle2 = processor.process(result)

        assert bundle1.structured == bundle2.structured
        assert bundle1.compressed_text == bundle2.compressed_text
        assert bundle1.compression_metrics == bundle2.compression_metrics

    def test_preserves_tool_call_id_and_tool_name(self) -> None:
        processor = _processor()
        bundle = processor.process(
            _make_tool_result(tool_call_id="call-xyz", tool_name="my_tool"),
        )
        assert bundle.tool_call_id == "call-xyz"
        assert bundle.tool_name == "my_tool"


class TestTestResultsFilter:
    def test_extracts_pass_fail_counts_from_vitest_style_output(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="test_results",
            raw_output="\n".join([
                " \u2713 src/foo.test.ts (10 tests) 5ms",
                " \u2717 src/bar.test.ts (2 tests) 3ms",
                "   \u00d7 should handle empty input",
                "     Error: expected true got false",
                "     at src/bar.test.ts:15:5",
                "",
                " Test Files  1 passed, 1 failed (2)",
                "      Tests  10 passed, 2 failed (12)",
                "   Duration  42ms",
            ]),
        ))
        assert bundle.structured.kind == "test_results"
        assert bundle.structured.summary.passed == 10  # type: ignore[union-attr]
        assert bundle.structured.summary.failed == 2  # type: ignore[union-attr]
        assert bundle.structured.summary.total == 12  # type: ignore[union-attr]
        assert bundle.structured.summary.duration_ms == 42  # type: ignore[union-attr]

    def test_extracts_failure_details_with_file_locations(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="test_results",
            raw_output="\n".join([
                " \u2717 should validate input",
                "   Error: expected 1 to equal 2",
                "     at tests/validate.test.ts:25:10",
                "",
                " Tests  0 passed, 1 failed (1)",
            ]),
        ))
        assert bundle.structured.kind == "test_results"
        failures = bundle.structured.failures  # type: ignore[union-attr]
        assert len(failures) > 0
        failure = failures[0]
        assert "validate input" in failure.name
        assert failure.file is not None
        assert "validate.test.ts" in failure.file
        assert failure.line == 25

    def test_prioritizes_failures_in_compressed_text(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="test_results",
            raw_output="\n".join([
                " \u2717 should not crash",
                "   Error: segfault",
                "",
                " Tests  5 passed, 1 failed (6)",
            ]),
        ))
        assert "Failures:" in bundle.compressed_text
        assert "should not crash" in bundle.compressed_text
        assert "5 passed" in bundle.compressed_text
        assert "1 failed" in bundle.compressed_text

    def test_uses_metadata_counts_when_provided(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="test_results",
            raw_output="some test output",
            metadata={"total": 50, "passed": 48, "failed": 2},
        ))
        assert bundle.structured.kind == "test_results"
        assert bundle.structured.summary.total == 50  # type: ignore[union-attr]
        assert bundle.structured.summary.passed == 48  # type: ignore[union-attr]
        assert bundle.structured.summary.failed == 2  # type: ignore[union-attr]


class TestBuildOutputFilter:
    def test_extracts_typescript_errors(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="build_output",
            raw_output="\n".join([
                "src/foo.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'.",
                "src/bar.ts(20,3): error TS6133: 'x' is declared but its value is never read.",
                "src/baz.ts(5,1): warning TS6133: 'y' is declared but its value is never read.",
            ]),
        ))
        assert bundle.structured.kind == "build_output"
        assert bundle.structured.success is False  # type: ignore[union-attr]
        assert len(bundle.structured.errors) == 2  # type: ignore[union-attr]
        assert bundle.structured.errors[0].file == "src/foo.ts"  # type: ignore[union-attr]
        assert bundle.structured.errors[0].line == 10  # type: ignore[union-attr]
        assert bundle.structured.errors[0].code == "TS2322"  # type: ignore[union-attr]
        assert len(bundle.structured.warnings) == 1  # type: ignore[union-attr]

    def test_marks_build_as_success_when_no_errors(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="build_output",
            raw_output="Compilation complete. No errors found.",
        ))
        assert bundle.structured.kind == "build_output"
        assert bundle.structured.success is True  # type: ignore[union-attr]
        assert len(bundle.structured.errors) == 0  # type: ignore[union-attr]

    def test_collapses_warnings_in_compressed_text(self) -> None:
        processor = _processor()
        warnings = [
            f"src/mod{i}.ts({i},1): warning TS6133: unused" for i in range(15)
        ]
        bundle = processor.process(_make_tool_result(
            output_kind="build_output",
            raw_output="\n".join(warnings),
        ))
        # Compressed text should not list all 15 warnings
        assert "Warnings (" in bundle.compressed_text
        assert "... and" in bundle.compressed_text

    def test_uses_exit_code_from_metadata_for_success(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="build_output",
            raw_output="some output with no error patterns",
            metadata={"exitCode": 1},
        ))
        assert bundle.structured.kind == "build_output"
        assert bundle.structured.success is False  # type: ignore[union-attr]


class TestGenericFilter:
    def test_produces_generic_signal_with_basic_metrics(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="generic",
            raw_output="line one\nline two\nline three",
        ))
        assert bundle.structured.kind == "generic"
        assert bundle.structured.extracted["lineCount"] == 3  # type: ignore[union-attr]
        assert isinstance(bundle.structured.extracted["charCount"], int)  # type: ignore[union-attr]

    def test_truncates_large_output_with_head_tail_preservation(self) -> None:
        processor = _processor()
        big_output = "\n".join(
            f"Line {i}: {'x' * 50}" for i in range(1000)
        )
        bundle = processor.process(_make_tool_result(
            output_kind="generic",
            raw_output=big_output,
        ))
        assert len(bundle.compressed_text) < len(big_output)
        assert "lines omitted" in bundle.compressed_text

    def test_preserves_short_output_without_truncation(self) -> None:
        processor = _processor()
        short_output = "Hello, world!"
        bundle = processor.process(_make_tool_result(
            output_kind="generic",
            raw_output=short_output,
        ))
        assert bundle.compressed_text == short_output


class TestLintOutputFilter:
    def test_extracts_eslint_style_violations(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="lint_output",
            raw_output="\n".join([
                "src/foo.ts:10:5  error  Missing return type  @typescript-eslint/explicit-function-return-type",
                "src/foo.ts:20:1  warning  Unexpected console statement  no-console",
                "src/bar.ts:5:3  error  Missing return type  @typescript-eslint/explicit-function-return-type",
                "",
                "3 problems (2 errors, 1 warning). 1 fixable",
            ]),
        ))
        assert bundle.structured.kind == "lint_output"
        assert bundle.structured.total_violations == 3  # type: ignore[union-attr]
        assert bundle.structured.fixable == 1  # type: ignore[union-attr]
        assert len(bundle.structured.by_rule) == 2  # type: ignore[union-attr]

    def test_falls_back_to_generic_when_no_lint_patterns(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="lint_output",
            raw_output="All good, no issues.",
        ))
        assert bundle.structured.kind == "generic"


class TestDirectoryListingFilter:
    def test_extracts_directory_structure(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="directory_listing",
            raw_output="\n".join([
                "src",
                "\u251c\u2500\u2500 index.ts",
                "\u251c\u2500\u2500 utils",
                "\u2502   \u251c\u2500\u2500 helper.ts",
                "\u2502   \u2514\u2500\u2500 format.ts",
                "\u2514\u2500\u2500 tests",
                "    \u2514\u2500\u2500 index.test.ts",
            ]),
        ))
        assert bundle.structured.kind == "directory_listing"
        assert bundle.structured.total_files > 0  # type: ignore[union-attr]


class TestJsonResponseFilter:
    def test_extracts_schema_and_summary_from_json(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="json_response",
            raw_output=json.dumps({
                "status": 200,
                "data": [1, 2, 3],
                "message": "ok",
                "nested": {"key": "value"},
            }),
        ))
        assert bundle.structured.kind == "json_response"
        assert bundle.structured.http_status == 200  # type: ignore[union-attr]
        assert "array" in bundle.structured.schema_["data"]  # type: ignore[union-attr]
        assert bundle.structured.schema_["message"] == "string"  # type: ignore[union-attr]
        assert bundle.structured.summary["message"] == "ok"  # type: ignore[union-attr]

    def test_falls_back_to_generic_for_invalid_json(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="json_response",
            raw_output="not valid json {{{",
        ))
        assert bundle.structured.kind == "generic"


class TestCompressionMetrics:
    def test_reports_compression_ratio(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="test_results",
            raw_output="\n".join([
                " Tests  50 passed, 0 failed (50)",
                " Duration  100ms",
            ]),
        ))
        assert bundle.compression_metrics.original_tokens > 0
        assert bundle.compression_metrics.compressed_tokens > 0
        assert bundle.compression_metrics.compression_ratio > 0
        assert isinstance(bundle.compression_metrics.compression_ratio, float)

    def test_reports_classified_as_matching_output_kind(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="build_output", raw_output="ok",
        ))
        assert bundle.compression_metrics.classified_as == "build_output"

    def test_reports_filter_applied_matching_structured_kind(self) -> None:
        processor = _processor()
        bundle = processor.process(_make_tool_result(
            output_kind="generic", raw_output="text",
        ))
        assert bundle.compression_metrics.filter_applied == "generic"

    def test_assesses_loss_risk_based_on_compression_ratio(self) -> None:
        processor = _processor()

        # Short output: no loss
        short_bundle = processor.process(_make_tool_result(raw_output="ok"))
        assert short_bundle.compression_metrics.loss_risk == "none"

        # Large generic output: higher risk
        large_bundle = processor.process(_make_tool_result(
            raw_output="x\n" * 5000,
            output_kind="generic",
        ))
        assert large_bundle.compression_metrics.loss_risk in ("medium", "high")


class TestPipelineIntegration:
    def test_tool_result_to_signal_bundle_pipeline(self) -> None:
        processor = _processor()
        tool_result = _make_tool_result(
            tool_call_id="call-pipeline",
            tool_name="run_tests",
            output_kind="test_results",
            raw_output="\n".join([
                " \u2713 should pass (3 tests) 5ms",
                " \u2717 should fail",
                "   Error: assertion failed",
                "",
                " Tests  3 passed, 1 failed (4)",
            ]),
        )
        bundle = processor.process(tool_result)

        # Structural validity
        assert bundle.tool_call_id == "call-pipeline"
        assert bundle.tool_name == "run_tests"
        assert bundle.structured.kind == "test_results"
        assert isinstance(bundle.compressed_text, str)
        assert len(bundle.compressed_text) > 0
        assert bundle.compression_metrics.original_tokens > 0


class TestFactory:
    def test_creates_a_working_instance(self) -> None:
        processor = SignalProcessorImpl()
        assert isinstance(processor, SignalProcessorImpl)
