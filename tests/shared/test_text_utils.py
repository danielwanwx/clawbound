"""Tests for clawbound.shared.text_utils."""

from clawbound.shared.text_utils import (
    matches_any,
    is_verification_like_task,
    extract_explicit_test_files,
    canonical_tool_name,
    canonicalize_candidate_tools,
    build_focused_test_discipline_notes,
    size_of_intersection,
    ratio,
)


class TestMatchesAny:
    def test_matches_single_keyword(self) -> None:
        assert matches_any("hello world", ["hello"]) is True

    def test_no_match(self) -> None:
        assert matches_any("hello world", ["foo", "bar"]) is False

    def test_empty_keywords(self) -> None:
        assert matches_any("hello", []) is False

    def test_empty_text(self) -> None:
        assert matches_any("", ["hello"]) is False

    def test_partial_match(self) -> None:
        assert matches_any("verification required", ["verify"]) is False
        assert matches_any("verification required", ["verification"]) is True


class TestIsVerificationLikeTask:
    def test_verification_with_no_edit_cue(self) -> None:
        assert is_verification_like_task("verify that tests pass, do not edit") is True

    def test_verification_with_report_cue(self) -> None:
        assert is_verification_like_task("confirm what changed and report") is True

    def test_no_verification_cue(self) -> None:
        assert is_verification_like_task("add a new feature") is False

    def test_verification_cue_but_no_action_cue(self) -> None:
        assert is_verification_like_task("verify it works") is False

    def test_passes_with_confirm(self) -> None:
        assert is_verification_like_task("check whether tests passes, confirm") is True


class TestExtractExplicitTestFiles:
    def test_single_test_file(self) -> None:
        result = extract_explicit_test_files("run src/foo.test.ts")
        assert result == ["src/foo.test.ts"]

    def test_multiple_test_files(self) -> None:
        result = extract_explicit_test_files("run foo.test.ts and bar.spec.js")
        assert result == ["foo.test.ts", "bar.spec.js"]

    def test_no_test_files(self) -> None:
        result = extract_explicit_test_files("run the build")
        assert result == []

    def test_deduplicates(self) -> None:
        result = extract_explicit_test_files("run foo.test.ts then foo.test.ts again")
        assert result == ["foo.test.ts"]

    def test_various_extensions(self) -> None:
        text = "files: a.test.tsx b.spec.cjs c.test.mts"
        result = extract_explicit_test_files(text)
        assert len(result) == 3


class TestCanonicalToolName:
    def test_read_maps_to_read_file(self) -> None:
        assert canonical_tool_name("read") == "read_file"

    def test_edit_maps_to_edit_file(self) -> None:
        assert canonical_tool_name("Edit") == "edit_file"

    def test_write_maps_to_write_file(self) -> None:
        assert canonical_tool_name("WRITE") == "write_file"

    def test_exec_maps_to_run_command(self) -> None:
        assert canonical_tool_name("exec") == "run_command"

    def test_bash_maps_to_run_command(self) -> None:
        assert canonical_tool_name("bash") == "run_command"

    def test_unknown_lowercased(self) -> None:
        assert canonical_tool_name("CustomTool") == "customtool"

    def test_empty_string(self) -> None:
        assert canonical_tool_name("") == ""

    def test_whitespace_only(self) -> None:
        assert canonical_tool_name("  ") == ""


class TestCanonicalizeCandidateTools:
    def test_normalizes_and_deduplicates(self) -> None:
        result = canonicalize_candidate_tools(["read", "Read", "edit", "custom"])
        assert result == ["read_file", "edit_file", "custom"]

    def test_filters_empty(self) -> None:
        result = canonicalize_candidate_tools(["read", "", "edit"])
        assert result == ["read_file", "edit_file"]

    def test_empty_input(self) -> None:
        result = canonicalize_candidate_tools([])
        assert result == []


class TestBuildFocusedTestDisciplineNotes:
    def test_with_test_files(self) -> None:
        result = build_focused_test_discipline_notes("run src/foo.test.ts")
        assert len(result) == 1
        assert "src/foo.test.ts" in result[0]

    def test_without_test_files(self) -> None:
        result = build_focused_test_discipline_notes("build the project")
        assert result == []


class TestSizeOfIntersection:
    def test_full_overlap(self) -> None:
        assert size_of_intersection({"a", "b"}, {"a", "b"}) == 2

    def test_partial_overlap(self) -> None:
        assert size_of_intersection({"a", "b", "c"}, {"b", "c", "d"}) == 2

    def test_no_overlap(self) -> None:
        assert size_of_intersection({"a"}, {"b"}) == 0

    def test_empty_sets(self) -> None:
        assert size_of_intersection(set(), set()) == 0


class TestRatio:
    def test_normal_division(self) -> None:
        assert ratio(3, 6) == 0.5

    def test_zero_denominator(self) -> None:
        assert ratio(5, 0) == 0.0

    def test_negative_denominator(self) -> None:
        assert ratio(5, -1) == 0.0

    def test_zero_numerator(self) -> None:
        assert ratio(0, 10) == 0.0
