"""SignalProcessor — deterministic transformation from ToolResult to SignalBundle.

Never calls an LLM. Never depends on provider-specific logic.
GenericSignal is the fallback (guardrail #3).
"""

from __future__ import annotations

import json
import re
from typing import Any

from clawbound.contracts.types import (
    BuildError,
    BuildOutputSignal,
    BuildWarning,
    CompressionMetrics,
    DirectoryEntry,
    DirectoryListingSignal,
    GenericSignal,
    JsonResponseSignal,
    LintExample,
    LintOutputSignal,
    LintRule,
    SignalBundle,
    SignalLossRisk,
    StructuredOutputKind,
    StructuredSignal,
    TestFailure,
    TestResultsSignal,
    TestSummary,
    ToolResult,
)
from clawbound.shared.tokens import estimate_tokens_from_text

MAX_COMPRESSED_TOKENS = 200
MAX_FAILURE_STACK_LINES = 8
MAX_BUILD_ERRORS = 20
MAX_BUILD_WARNINGS = 10
MAX_LINT_RULES = 15


class SignalProcessorImpl:
    def process(self, result: ToolResult) -> SignalBundle:
        original_tokens = estimate_tokens_from_text(result.raw_output)
        structured, compressed_text = _route_to_filter(
            result.output_kind, result.raw_output, result.metadata,
        )
        compressed_tokens = estimate_tokens_from_text(compressed_text)

        compression_metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            classified_as=result.output_kind,
            filter_applied=structured.kind,
            loss_risk=_assess_loss_risk(original_tokens, compressed_tokens, structured),
        )

        return SignalBundle(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            structured=structured,
            compressed_text=compressed_text,
            compression_metrics=compression_metrics,
        )


# ─── Filter router ─────────────────────────────────────────────────────────────


def _route_to_filter(
    output_kind: StructuredOutputKind,
    raw_output: str,
    metadata: dict[str, Any],
) -> tuple[StructuredSignal, str]:
    match output_kind:
        case "test_results":
            return _filter_test_results(raw_output, metadata)
        case "build_output":
            return _filter_build_output(raw_output, metadata)
        case "lint_output":
            return _filter_lint_output(raw_output)
        case "directory_listing":
            return _filter_directory_listing(raw_output)
        case "json_response" | "api_response":
            return _filter_json_response(raw_output)
        case _:
            return _filter_generic(raw_output, output_kind)


# ─── Filter: test_results ─────────────────────────────────────────────────────


def _filter_test_results(
    raw_output: str,
    metadata: dict[str, Any],
) -> tuple[TestResultsSignal, str]:
    lines = raw_output.split("\n")

    total = passed = failed = skipped = 0
    duration_ms: int | None = None

    for line in lines:
        tests_match = re.search(
            r"Tests?\s+(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+skipped)?(?:.*?\((\d+)\))?",
            line, re.IGNORECASE,
        )
        if tests_match:
            passed = int(tests_match.group(1) or 0)
            failed = int(tests_match.group(2) or 0)
            skipped = int(tests_match.group(3) or 0)
            total = int(tests_match.group(4) or 0) or (passed + failed + skipped)

        dur_match = re.search(r"Duration\s+(\d[\d,.]*)\s*m?s", line, re.IGNORECASE)
        if dur_match:
            duration_ms = int(float(dur_match.group(1).replace(",", "")))

    # Fallback: count individual markers
    if total == 0:
        for line in lines:
            if re.match(r"^\s*[✓✔√]|^\s*PASS\s", line, re.IGNORECASE):
                passed += 1
            if re.match(r"^\s*[✗✘×]|^\s*FAIL\s", line, re.IGNORECASE):
                failed += 1
            if re.match(r"^\s*[-⊘].*skip", line, re.IGNORECASE):
                skipped += 1
        total = passed + failed + skipped

    # Use metadata if available
    if isinstance(metadata.get("total"), int):
        total = metadata["total"]
    if isinstance(metadata.get("passed"), int):
        passed = metadata["passed"]
    if isinstance(metadata.get("failed"), int):
        failed = metadata["failed"]

    failures = _extract_test_failures(lines)

    structured = TestResultsSignal(
        summary=TestSummary(
            total=total, passed=passed, failed=failed,
            skipped=skipped, duration_ms=duration_ms,
        ),
        failures=tuple(failures),
    )

    parts = [f"Tests: {passed} passed, {failed} failed, {skipped} skipped ({total} total)"]
    if duration_ms is not None:
        parts.append(f"Duration: {duration_ms}ms")
    if failures:
        parts.append("Failures:")
        for f in failures:
            loc = f" ({f.file}{f':{f.line}' if f.line else ''})" if f.file else ""
            parts.append(f"  × {f.name}{loc}: {f.message}")
            if f.stack:
                for sl in f.stack.split("\n")[:MAX_FAILURE_STACK_LINES]:
                    parts.append(f"    {sl.strip()}")

    return structured, "\n".join(parts)


def _extract_test_failures(lines: list[str]) -> list[TestFailure]:
    failures: list[TestFailure] = []
    in_failure = False
    name = message = ""
    file: str | None = None
    line_num: int | None = None
    stack: list[str] = []

    def flush() -> None:
        nonlocal in_failure, name, message, file, line_num, stack
        if in_failure and name:
            failures.append(TestFailure(
                name=name,
                file=file,
                line=line_num,
                message=message or "Test failed",
                stack="\n".join(stack) if stack else None,
            ))
        in_failure = False
        name = message = ""
        file = None
        line_num = None
        stack = []

    for raw_line in lines:
        fail_match = re.match(r"^\s*[✗✘×]\s+(.+)|^\s*FAIL\s+(.+)", raw_line)
        if fail_match:
            flush()
            in_failure = True
            name = (fail_match.group(1) or fail_match.group(2)).strip()
            continue

        if in_failure:
            err_match = re.match(
                r"^\s*(?:Error|AssertionError|AssertError|Expected|expect).*?:\s*(.+)", raw_line, re.IGNORECASE,
            )
            if err_match and not message:
                message = err_match.group(1).strip()
                continue

            loc_match = re.search(r"(?:at\s+)?([^\s(]+\.[cm]?[jt]sx?)[:(](\d+)", raw_line)
            if loc_match and not file:
                file = loc_match.group(1)
                line_num = int(loc_match.group(2))

            if re.match(r"^\s+at\s+", raw_line):
                stack.append(raw_line)

            if raw_line.strip() == "" and stack:
                flush()

    flush()
    return failures


# ─── Filter: build_output ─────────────────────────────────────────────────────


def _filter_build_output(
    raw_output: str,
    metadata: dict[str, Any],
) -> tuple[BuildOutputSignal, str]:
    lines = raw_output.split("\n")
    errors: list[BuildError] = []
    warnings: list[BuildWarning] = []

    for line in lines:
        ts_err = re.match(r"^(.+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)", line)
        if ts_err:
            if len(errors) < MAX_BUILD_ERRORS:
                errors.append(BuildError(
                    file=ts_err.group(1), line=int(ts_err.group(2)),
                    column=int(ts_err.group(3)), code=ts_err.group(4), message=ts_err.group(5),
                ))
            continue

        ts_warn = re.match(r"^(.+?)\((\d+),(\d+)\):\s*warning\s+(TS\d+):\s*(.+)", line)
        if ts_warn:
            if len(warnings) < MAX_BUILD_WARNINGS:
                warnings.append(BuildWarning(
                    file=ts_warn.group(1), line=int(ts_warn.group(2)),
                    message=f"{ts_warn.group(4)}: {ts_warn.group(5)}",
                ))
            continue

        gen_err = re.match(r"^(.+?\.[a-z]+):(\d+)(?::\d+)?:\s*error[:\s]+(.+)", line, re.IGNORECASE)
        if gen_err:
            if len(errors) < MAX_BUILD_ERRORS:
                errors.append(BuildError(
                    file=gen_err.group(1), line=int(gen_err.group(2)), message=gen_err.group(3),
                ))
            continue

        gen_warn = re.match(r"^(.+?\.[a-z]+):(\d+)(?::\d+)?:\s*warning[:\s]+(.+)", line, re.IGNORECASE)
        if gen_warn:
            if len(warnings) < MAX_BUILD_WARNINGS:
                warnings.append(BuildWarning(
                    file=gen_warn.group(1), line=int(gen_warn.group(2)), message=gen_warn.group(3),
                ))

    exit_code = metadata.get("exitCode")
    success = len(errors) == 0 and (exit_code == 0 if isinstance(exit_code, int) else True)

    structured = BuildOutputSignal(success=success, errors=tuple(errors), warnings=tuple(warnings))

    parts = [f"Build: {'SUCCESS' if success else 'FAILED'}"]
    if errors:
        parts.append(f"Errors ({len(errors)}):")
        for e in errors:
            loc = f"{e.file}:{e.line}" if e.line else e.file
            parts.append(f"  {loc}{f' [{e.code}]' if e.code else ''}: {e.message}")
    if warnings:
        parts.append(f"Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            parts.append(f"  {w.file}:{w.line or '?'}: {w.message}")
        if len(warnings) > 5:
            parts.append(f"  ... and {len(warnings) - 5} more")

    return structured, "\n".join(parts)


# ─── Filter: generic ──────────────────────────────────────────────────────────


def _filter_generic(
    raw_output: str,
    classified_as: StructuredOutputKind,
) -> tuple[GenericSignal, str]:
    lines = raw_output.split("\n")
    line_count = len(lines)
    char_count = len(raw_output)
    token_estimate = estimate_tokens_from_text(raw_output)

    structured = GenericSignal(extracted={
        "lineCount": line_count,
        "charCount": char_count,
        "tokenEstimate": token_estimate,
        "classifiedAs": classified_as,
    })

    if token_estimate <= MAX_COMPRESSED_TOKENS:
        compressed_text = raw_output
    else:
        keep_chars = MAX_COMPRESSED_TOKENS * 5
        head_chars = int(keep_chars * 0.7)
        tail_chars = int(keep_chars * 0.3)
        head = raw_output[:head_chars]
        tail = raw_output[-tail_chars:]
        omitted = line_count - head.count("\n") - tail.count("\n") - 2
        compressed_text = f"{head}\n\n[... {omitted} lines omitted ...]\n\n{tail}"

    return structured, compressed_text


# ─── Filter: lint_output ──────────────────────────────────────────────────────


def _filter_lint_output(raw_output: str) -> tuple[StructuredSignal, str]:
    lines = raw_output.split("\n")
    rule_map: dict[str, dict[str, Any]] = {}
    total_violations = 0
    fixable = 0

    for line in lines:
        eslint_match = re.match(
            r"^(.+?):(\d+):\d+\s+(error|warning)\s+.+?\s+([@\w/-]+)\s*$", line,
        )
        if eslint_match:
            total_violations += 1
            file = eslint_match.group(1)
            line_num = int(eslint_match.group(2))
            severity = eslint_match.group(3)
            rule = eslint_match.group(4)

            if rule not in rule_map:
                rule_map[rule] = {"count": 0, "severity": severity, "examples": []}
            rule_map[rule]["count"] += 1
            if len(rule_map[rule]["examples"]) < 3:
                rule_map[rule]["examples"].append(LintExample(file=file, line=line_num))
            continue

        fix_match = re.search(r"(\d+)\s+fixable", line, re.IGNORECASE)
        if fix_match:
            fixable = int(fix_match.group(1))

    if total_violations == 0:
        return _filter_generic(raw_output, "lint_output")

    by_rule = [
        LintRule(
            rule=rule,
            count=data["count"],
            severity=data["severity"],
            examples=tuple(data["examples"]),
        )
        for rule, data in list(rule_map.items())[:MAX_LINT_RULES]
    ]

    structured = LintOutputSignal(
        total_violations=total_violations,
        fixable=fixable,
        by_rule=tuple(by_rule),
    )

    parts = [f"Lint: {total_violations} violations ({fixable} fixable)"]
    for r in by_rule[:10]:
        parts.append(f"  {r.rule}: {r.count} {r.severity}(s)")

    return structured, "\n".join(parts)


# ─── Filter: directory_listing ────────────────────────────────────────────────


def _filter_directory_listing(raw_output: str) -> tuple[StructuredSignal, str]:
    lines = [line for line in raw_output.split("\n") if line.strip()]
    total_files = 0
    total_dirs = 0
    tree: list[DirectoryEntry] = []
    root = ""

    for line in lines:
        trimmed = re.sub(r"[├└│─\s]+", "", line).strip()
        if not trimmed:
            continue
        if not root and "├" not in line and "└" not in line:
            root = trimmed
            continue
        if "." in trimmed or "/" in trimmed:
            total_files += 1
        else:
            total_dirs += 1
            tree.append(DirectoryEntry(path=trimmed, file_count=0))

    if total_files == 0 and total_dirs == 0:
        return _filter_generic(raw_output, "directory_listing")

    structured = DirectoryListingSignal(
        root=root or ".",
        total_files=total_files,
        total_dirs=total_dirs,
        tree=tuple(tree),
    )

    parts = [f"Directory: {root or '.'}", f"{total_files} files, {total_dirs} directories"]
    if tree:
        for entry in tree[:10]:
            parts.append(f"  {entry.path}/")
        if len(tree) > 10:
            parts.append(f"  ... and {len(tree) - 10} more directories")

    return structured, "\n".join(parts)


# ─── Filter: json_response ───────────────────────────────────────────────────


def _filter_json_response(raw_output: str) -> tuple[StructuredSignal, str]:
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, ValueError):
        return _filter_generic(raw_output, "json_response")

    if not isinstance(parsed, dict):
        return _filter_generic(raw_output, "json_response")

    schema: dict[str, str] = {}
    summary: dict[str, str | int | bool] = {}

    for key, value in parsed.items():
        if isinstance(value, list):
            schema[key] = f"array[{len(value)}]"
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            schema[key] = f"object{{{len(value)}}}"
        elif isinstance(value, (str, int, float, bool)):
            schema[key] = type(value).__name__
            if isinstance(value, float):
                schema[key] = "number"
            elif isinstance(value, bool):
                schema[key] = "boolean"
                summary[key] = value
            elif isinstance(value, int):
                schema[key] = "number"
                summary[key] = value
            elif isinstance(value, str):
                schema[key] = "string"
                summary[key] = value
        elif value is None:
            schema[key] = "null"

    http_status: int | None = None
    if isinstance(parsed.get("status"), int):
        http_status = parsed["status"]
    elif isinstance(parsed.get("statusCode"), int):
        http_status = parsed["statusCode"]

    structured = JsonResponseSignal(
        http_status=http_status,
        schema_=schema,
        summary=summary,
    )

    parts: list[str] = []
    if http_status is not None:
        parts.append(f"HTTP {http_status}")
    schema_str = ", ".join(f"{k}: {v}" for k, v in schema.items())
    parts.append(f"Schema: {{{schema_str}}}")
    if summary:
        vals = ", ".join(f"{k}={str(v)[:50]}" for k, v in list(summary.items())[:5])
        parts.append(f"Values: {vals}")

    return structured, "\n".join(parts)


# ─── Loss risk assessment ─────────────────────────────────────────────────────


def _assess_loss_risk(
    original_tokens: int,
    compressed_tokens: int,
    structured: StructuredSignal,
) -> SignalLossRisk:
    if original_tokens == 0:
        return "none"

    ratio = compressed_tokens / original_tokens

    if ratio >= 0.95:
        return "none"

    if structured.kind != "generic":
        if ratio >= 0.5:
            return "low"
        return "medium"

    if ratio >= 0.7:
        return "low"
    if ratio >= 0.3:
        return "medium"
    return "high"
