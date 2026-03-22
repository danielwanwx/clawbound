"""Token estimation utilities.

Lightweight, regex-based token estimates — not full BPE tokenization.
Good enough for budget gating; not for billing.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(*parts: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        for match in _TOKEN_PATTERN.finditer(part):
            tokens.append(match.group(0).lower())
    return tokens


def unique_tokens(*parts: str) -> list[str]:
    """Return deduplicated tokens preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(*parts):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def estimate_tokens_from_text(*parts: str) -> int:
    """Estimate token count from text. Returns 0 for empty input, min 1 otherwise."""
    if any(part and len(part) > 0 for part in parts):
        return max(1, len(tokenize(*parts)))
    return 0


def estimate_tokens_from_items(items: list[str]) -> int:
    """Estimate token count from a list of strings joined by spaces."""
    return estimate_tokens_from_text(" ".join(items))
