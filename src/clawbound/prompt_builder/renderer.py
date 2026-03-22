"""Deterministic segment renderer.

Only segments with admitted_full or admitted_trimmed are included.
Ordered by `order` field (stable sort), joined with single newline.
"""

from __future__ import annotations

from clawbound.contracts.types import PromptSegment


def render_segments_to_system_prompt(segments: tuple[PromptSegment, ...] | list[PromptSegment]) -> str:
    admitted = [seg for seg in segments if seg.admission_outcome.status != "rejected"]
    admitted.sort(key=lambda seg: seg.order)
    return "\n".join(seg.content for seg in admitted).strip()
