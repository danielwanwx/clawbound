"""ToolBroker parity tests.

Covers: registration, capability filtering, execute() returning typed
ToolResult, unauthorized tool denial, metadata precedence over heuristics.
"""

from __future__ import annotations

import json
from typing import Any


from clawbound.contracts import (
    ToolDefinition,
    ToolExecuteParams,
    ToolProfilePolicy,
)
from clawbound.tool_broker import ToolBrokerImpl


# ─── Test helpers ──────────────────────────────────────────────────────────────


def _policy(**overrides: Any) -> ToolProfilePolicy:
    defaults: dict[str, Any] = {
        "profile_name": "test-policy",
        "allowed_tools": (),
        "denied_tools": (),
        "notes": (),
        "requires_review": False,
    }
    defaults.update(overrides)
    return ToolProfilePolicy(**defaults)


def _def(name: str, **overrides: Any) -> ToolDefinition:
    defaults: dict[str, Any] = {
        "name": name,
        "category": "filesystem",
        "risk_level": "read_only",
    }
    defaults.update(overrides)
    return ToolDefinition(**defaults)


async def _echo_execute(args: dict[str, Any]) -> dict[str, Any]:
    return {"output": f"echo: {json.dumps(args)}"}


async def _test_result_execute(_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": "Tests  3 passed (3)\nDuration  42ms",
        "output_kind": "test_results",
    }


# ─── Registration ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_registers_a_tool_definition(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)

        resolved = broker.resolve_for_turn(_policy())
        assert len(resolved) == 1
        assert resolved[0].name == "read_file"

    def test_registers_multiple_tools(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)
        broker.register(_def("write_file"), _echo_execute)
        broker.register(_def("run_command"), _echo_execute)

        resolved = broker.resolve_for_turn(_policy())
        assert len(resolved) == 3

    def test_overwrites_tool_with_same_name(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file", risk_level="read_only"), _echo_execute)
        broker.register(_def("read_file", risk_level="side_effect"), _echo_execute)

        resolved = broker.resolve_for_turn(_policy())
        assert len(resolved) == 1
        assert resolved[0].risk_level == "side_effect"


# ─── Capability filtering ─────────────────────────────────────────────────────


class TestCapabilityFiltering:
    def test_returns_all_tools_when_policy_has_no_allow_deny(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)
        broker.register(_def("write_file"), _echo_execute)

        resolved = broker.resolve_for_turn(_policy())
        assert len(resolved) == 2

    def test_filters_to_allowed_tools_when_specified(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)
        broker.register(_def("write_file"), _echo_execute)
        broker.register(_def("run_command"), _echo_execute)

        resolved = broker.resolve_for_turn(
            _policy(allowed_tools=("read_file", "write_file")),
        )
        assert len(resolved) == 2
        names = [t.name for t in resolved]
        assert "read_file" in names
        assert "write_file" in names

    def test_excludes_denied_tools(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)
        broker.register(_def("write_file"), _echo_execute)

        resolved = broker.resolve_for_turn(
            _policy(denied_tools=("write_file",)),
        )
        assert len(resolved) == 1
        assert resolved[0].name == "read_file"

    def test_deny_takes_precedence_over_allow(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)

        resolved = broker.resolve_for_turn(
            _policy(
                allowed_tools=("read_file",),
                denied_tools=("read_file",),
            ),
        )
        assert len(resolved) == 0


# ─── Execute — typed ToolResult ────────────────────────────────────────────────


class TestExecuteSuccess:
    async def test_returns_success_tool_result(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)

        result = await broker.execute(ToolExecuteParams(
            tool_name="read_file",
            tool_call_id="call-001",
            args={"path": "/tmp/test.txt"},
            policy=_policy(allowed_tools=("read_file",)),
        ))

        assert result.status == "success"
        assert result.tool_name == "read_file"
        assert result.tool_call_id == "call-001"
        assert "echo:" in result.raw_output
        assert result.duration_ms >= 0

    async def test_returns_structured_media_type_and_output_kind(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("read_file"), _echo_execute)

        result = await broker.execute(ToolExecuteParams(
            tool_name="read_file",
            tool_call_id="call-002",
            args={},
            policy=_policy(allowed_tools=("read_file",)),
        ))

        assert isinstance(result.media_type, str)
        assert isinstance(result.output_kind, str)


# ─── Unauthorized tool denial ──────────────────────────────────────────────────


class TestUnauthorizedToolDenial:
    async def test_returns_denied_for_unauthorized_tool(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("write_file"), _echo_execute)

        result = await broker.execute(ToolExecuteParams(
            tool_name="write_file",
            tool_call_id="call-denied",
            args={},
            policy=_policy(allowed_tools=("read_file",)),
        ))

        assert result.status == "denied"
        assert "not authorized" in result.raw_output
        assert "denied_by" in result.metadata

    async def test_returns_denied_for_explicitly_denied_tool(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("write_file"), _echo_execute)

        result = await broker.execute(ToolExecuteParams(
            tool_name="write_file",
            tool_call_id="call-denied-2",
            args={},
            policy=_policy(denied_tools=("write_file",)),
        ))

        assert result.status == "denied"

    async def test_returns_not_found_for_unregistered_tool(self) -> None:
        broker = ToolBrokerImpl()

        result = await broker.execute(ToolExecuteParams(
            tool_name="nonexistent_tool",
            tool_call_id="call-404",
            args={},
            policy=_policy(),
        ))

        assert result.status == "not_found"
        assert "not registered" in result.raw_output


# ─── Metadata precedence over heuristics ───────────────────────────────────────


class TestMetadataPrecedence:
    async def test_uses_declared_output_kind_from_definition(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(
            _def("run_tests", declared_output_kind="test_results"),
            _echo_execute,
        )

        result = await broker.execute(ToolExecuteParams(
            tool_name="run_tests",
            tool_call_id="call-meta",
            args={},
            policy=_policy(),
        ))

        assert result.output_kind == "test_results"

    async def test_uses_declared_media_type_from_definition(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(
            _def("api_call", declared_media_type="application/json"),
            _echo_execute,
        )

        result = await broker.execute(ToolExecuteParams(
            tool_name="api_call",
            tool_call_id="call-media",
            args={},
            policy=_policy(),
        ))

        assert result.media_type == "application/json"

    async def test_uses_execute_fn_return_output_kind_when_no_declaration(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(_def("run_tests"), _test_result_execute)

        result = await broker.execute(ToolExecuteParams(
            tool_name="run_tests",
            tool_call_id="call-fn-meta",
            args={},
            policy=_policy(),
        ))

        assert result.output_kind == "test_results"

    async def test_declaration_takes_precedence_over_execute_fn_return(self) -> None:
        broker = ToolBrokerImpl()
        broker.register(
            _def("run_tests", declared_output_kind="build_output"),
            _test_result_execute,
        )

        result = await broker.execute(ToolExecuteParams(
            tool_name="run_tests",
            tool_call_id="call-precedence",
            args={},
            policy=_policy(),
        ))

        # Declaration wins over executeFn return
        assert result.output_kind == "build_output"

    async def test_falls_back_to_heuristic_when_no_metadata(self) -> None:
        broker = ToolBrokerImpl()

        async def json_execute(_args: dict[str, Any]) -> dict[str, Any]:
            return {"output": '{"key": "value"}'}

        broker.register(_def("run_cmd"), json_execute)

        result = await broker.execute(ToolExecuteParams(
            tool_name="run_cmd",
            tool_call_id="call-heuristic",
            args={},
            policy=_policy(),
        ))

        assert result.output_kind == "json_response"
        assert result.media_type == "application/json"


# ─── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    async def test_returns_error_status_when_execute_fn_throws(self) -> None:
        broker = ToolBrokerImpl()

        async def failing_fn(_args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Something went wrong")

        broker.register(_def("failing_tool"), failing_fn)

        result = await broker.execute(ToolExecuteParams(
            tool_name="failing_tool",
            tool_call_id="call-error",
            args={},
            policy=_policy(),
        ))

        assert result.status == "error"
        assert "Something went wrong" in result.raw_output
        assert result.metadata.get("error_type") == "RuntimeError"

    async def test_returns_timeout_status_for_timeout_errors(self) -> None:
        broker = ToolBrokerImpl()

        async def timeout_fn(_args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Operation timeout exceeded")

        broker.register(_def("slow_tool"), timeout_fn)

        result = await broker.execute(ToolExecuteParams(
            tool_name="slow_tool",
            tool_call_id="call-timeout",
            args={},
            policy=_policy(),
        ))

        assert result.status == "timeout"


# ─── Factory ───────────────────────────────────────────────────────────────────


class TestFactory:
    def test_constructor_returns_working_instance(self) -> None:
        broker = ToolBrokerImpl()
        assert isinstance(broker, ToolBrokerImpl)
