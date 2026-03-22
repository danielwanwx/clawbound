"""Text analysis utilities.

Extracted from runtime.ts lines 1084-1179. Pure functions
with no side effects — safe to call from any module.
"""

from __future__ import annotations

import re


def matches_any(text: str, keywords: list[str]) -> bool:
    """Return True if text contains any of the keywords."""
    return any(keyword in text for keyword in keywords)


def is_verification_like_task(text: str) -> bool:
    """Detect verification-like tasks that should not trigger edits."""
    verification_cue = matches_any(text, [
        "verify",
        "verification",
        "confirm",
        "validate",
        "check whether",
        "whether",
        "what changed",
        "now passes",
        "still passes",
        "passes",
    ])
    if not verification_cue:
        return False

    return matches_any(text, [
        "do not make further edits",
        "do not edit",
        "without editing",
        "no further edits",
        "in two bullets",
        "report",
        "confirm",
    ])


_TEST_FILE_PATTERN = re.compile(r"[\w./-]+\.(?:test|spec)\.[cm]?[jt]sx?", re.IGNORECASE)


def extract_explicit_test_files(text: str) -> list[str]:
    """Extract test file paths from user input."""
    matches = _TEST_FILE_PATTERN.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def canonical_tool_name(tool_name: str) -> str:
    """Normalize tool names to canonical form."""
    normalized = tool_name.strip().lower()
    if not normalized:
        return ""
    match normalized:
        case "read":
            return "read_file"
        case "edit":
            return "edit_file"
        case "write":
            return "write_file"
        case "exec" | "bash":
            return "run_command"
        case _:
            return normalized


def canonicalize_candidate_tools(candidate_tools: list[str]) -> list[str]:
    """Normalize and deduplicate candidate tool names."""
    seen: set[str] = set()
    result: list[str] = []
    for tool in candidate_tools:
        canonical = canonical_tool_name(tool)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def build_focused_test_discipline_notes(user_input: str) -> list[str]:
    """Build focused test discipline notes for explicit test files."""
    test_files = extract_explicit_test_files(user_input)
    if not test_files:
        return []
    files_str = ", ".join(test_files)
    return [
        f"For named test files ({files_str}), use the narrowest verification "
        f"command available. Prefer an existing focused script or direct runner "
        f"such as node --test <file>, and avoid npm test -- <file> unless a "
        f"test script is known to exist.",
    ]


def size_of_intersection(left: set[str], right: set[str]) -> int:
    """Count the number of elements in the intersection of two sets."""
    count = 0
    for value in left:
        if value in right:
            count += 1
    return count


def ratio(numerator: int | float, denominator: int | float) -> float:
    """Safe division that returns 0 when denominator is non-positive."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator
