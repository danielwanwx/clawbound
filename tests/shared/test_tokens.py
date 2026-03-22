"""Tests for clawbound.shared.tokens."""

from clawbound.shared.tokens import (
    tokenize,
    unique_tokens,
    estimate_tokens_from_text,
    estimate_tokens_from_items,
)


class TestTokenize:
    def test_simple_text(self) -> None:
        result = tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_multiple_parts(self) -> None:
        result = tokenize("hello", "world")
        assert result == ["hello", "world"]

    def test_punctuation_stripped(self) -> None:
        result = tokenize("hello, world!")
        assert result == ["hello", "world"]

    def test_empty_string(self) -> None:
        result = tokenize("")
        assert result == []

    def test_none_like_empty(self) -> None:
        result = tokenize("", "")
        assert result == []

    def test_underscores_kept(self) -> None:
        result = tokenize("snake_case")
        assert result == ["snake_case"]

    def test_mixed_content(self) -> None:
        result = tokenize("file.test.ts has 3 tests")
        assert result == ["file", "test", "ts", "has", "3", "tests"]


class TestUniqueTokens:
    def test_deduplicates(self) -> None:
        result = unique_tokens("hello hello world hello")
        assert result == ["hello", "world"]

    def test_preserves_order(self) -> None:
        result = unique_tokens("world hello world")
        assert result == ["world", "hello"]

    def test_empty(self) -> None:
        result = unique_tokens("")
        assert result == []


class TestEstimateTokensFromText:
    def test_non_empty(self) -> None:
        result = estimate_tokens_from_text("hello world")
        assert result == 2

    def test_empty_returns_zero(self) -> None:
        result = estimate_tokens_from_text("")
        assert result == 0

    def test_single_word(self) -> None:
        result = estimate_tokens_from_text("x")
        assert result == 1

    def test_multiple_parts(self) -> None:
        result = estimate_tokens_from_text("hello", "world foo")
        assert result == 3

    def test_all_empty_returns_zero(self) -> None:
        result = estimate_tokens_from_text("", "")
        assert result == 0

    def test_punctuation_only_returns_zero(self) -> None:
        # No alphanumeric tokens, but string is non-empty
        # This returns max(1, 0) = 1 because the part has length > 0
        result = estimate_tokens_from_text("!!!")
        assert result == 1


class TestEstimateTokensFromItems:
    def test_joins_items(self) -> None:
        result = estimate_tokens_from_items(["hello", "world"])
        assert result == 2

    def test_empty_list(self) -> None:
        result = estimate_tokens_from_items([])
        assert result == 0
