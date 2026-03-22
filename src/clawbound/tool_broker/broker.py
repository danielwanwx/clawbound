"""ToolBroker — tool registration, capability filtering, and typed execution.

Unauthorized tools produce structured denial results, not exceptions.
declaredOutputKind on ToolDefinition takes precedence over heuristics.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from clawbound.contracts.types import (
    StructuredOutputKind,
    ToolDefinition,
    ToolExecuteParams,
    ToolOutputMediaType,
    ToolProfilePolicy,
    ToolResult,
)

ToolBrokerExecuteFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolBrokerImpl:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolBrokerExecuteFn]] = {}

    def register(self, definition: ToolDefinition, execute_fn: ToolBrokerExecuteFn) -> None:
        self._tools[definition.name] = (definition, execute_fn)

    def resolve_for_turn(self, policy: ToolProfilePolicy) -> list[ToolDefinition]:
        allowed = set(policy.allowed_tools)
        denied = set(policy.denied_tools)
        resolved: list[ToolDefinition] = []
        for name, (defn, _) in self._tools.items():
            if name in denied:
                continue
            if allowed and name not in allowed:
                continue
            resolved.append(defn)
        return resolved

    async def execute(self, params: ToolExecuteParams) -> ToolResult:
        entry = self._tools.get(params.tool_name)

        if entry is None:
            return ToolResult(
                tool_name=params.tool_name,
                tool_call_id=params.tool_call_id,
                status="not_found",
                raw_output=f'Tool "{params.tool_name}" is not registered.',
                media_type="text/plain",
                output_kind="generic",
                duration_ms=0,
                metadata={},
            )

        defn, execute_fn = entry

        if not _is_authorized(params.tool_name, params.policy):
            return ToolResult(
                tool_name=params.tool_name,
                tool_call_id=params.tool_call_id,
                status="denied",
                raw_output=f'Tool "{params.tool_name}" is not authorized by current policy (profile: {params.policy.profile_name}).',
                media_type="text/plain",
                output_kind="generic",
                duration_ms=0,
                metadata={"denied_by": params.policy.profile_name},
            )

        start = time.monotonic()
        try:
            raw = await execute_fn(params.args)
            duration_ms = int((time.monotonic() - start) * 1000)

            output_kind = _resolve_output_kind(
                defn.declared_output_kind, raw.get("output_kind"), raw.get("output", ""),
            )
            media_type = _resolve_media_type(
                defn.declared_media_type, raw.get("media_type"), raw.get("output", ""),
            )

            return ToolResult(
                tool_name=params.tool_name,
                tool_call_id=params.tool_call_id,
                status="success",
                raw_output=raw.get("output", ""),
                media_type=media_type,
                output_kind=output_kind,
                duration_ms=duration_ms,
                metadata=raw.get("metadata", {}),
            )
        except Exception as err:
            duration_ms = int((time.monotonic() - start) * 1000)
            is_timeout = "timeout" in str(err).lower() or "ETIMEDOUT" in str(err)

            return ToolResult(
                tool_name=params.tool_name,
                tool_call_id=params.tool_call_id,
                status="timeout" if is_timeout else "error",
                raw_output=str(err),
                media_type="text/plain",
                output_kind="generic",
                duration_ms=duration_ms,
                metadata={"error_type": type(err).__name__},
            )


def _is_authorized(tool_name: str, policy: ToolProfilePolicy) -> bool:
    if tool_name in policy.denied_tools:
        return False
    if not policy.allowed_tools:
        return True
    return tool_name in policy.allowed_tools


def _resolve_output_kind(
    declared: StructuredOutputKind | None,
    from_execution: str | None,
    raw_output: str,
) -> StructuredOutputKind:
    if declared:
        return declared
    if from_execution:
        return from_execution  # type: ignore[return-value]
    return _classify_output_kind_heuristic(raw_output)


def _resolve_media_type(
    declared: ToolOutputMediaType | None,
    from_execution: str | None,
    raw_output: str,
) -> ToolOutputMediaType:
    if declared:
        return declared
    if from_execution:
        return from_execution  # type: ignore[return-value]
    return _classify_media_type_heuristic(raw_output)


def _classify_output_kind_heuristic(output: str) -> StructuredOutputKind:
    lower = output.lower()

    if any(kw in lower for kw in ("tests passed", "tests failed", "test files", "✓", "✗")) or re.search(r"\d+\s+pass(?:ed|ing)", lower):
        return "test_results"

    if re.search(r"error\s+ts\d+", output, re.IGNORECASE) or ("error:" in lower and any(kw in lower for kw in ("compile", ".ts(", ".js("))):
        return "build_output"

    if "eslint" in lower or ("warning" in lower and "rule" in lower) or re.search(r"\d+\s+problems?", lower):
        return "lint_output"

    if "directory" in lower or (("├" in lower or "└" in lower) and "│" in lower):
        return "directory_listing"

    trimmed = output.lstrip()
    if trimmed.startswith("{") or trimmed.startswith("["):
        try:
            json.loads(output)
            return "json_response"
        except (json.JSONDecodeError, ValueError):
            pass

    return "generic"


def _classify_media_type_heuristic(output: str) -> ToolOutputMediaType:
    trimmed = output.lstrip()
    if trimmed.startswith("{") or trimmed.startswith("["):
        try:
            json.loads(output)
            return "application/json"
        except (json.JSONDecodeError, ValueError):
            pass
    if "diff --git" in trimmed or (trimmed.startswith("---") and "+++" in trimmed):
        return "text/diff"
    return "text/plain"
