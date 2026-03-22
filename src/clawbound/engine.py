"""ClawBoundEngine — independent runtime entrypoint.

Single public API for ClawBound. Handles provider resolution,
session continuity, and orchestration internally.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from clawbound.contracts.types import (
    FinalAnswer,
    LocalContextItem,
    LoopTermination,
    ModelError,
    ModelMessage,
    OrchestratorDiagnostics,
    SessionSnapshot,
    ToolCalls,
)
from clawbound.execution_loop.adapter import DeterministicAdapter
from clawbound.orchestrator import ToolRegistration, run_orchestrator
from clawbound.session_store.store import InMemorySessionStore


@dataclass(frozen=True)
class EngineConfig:
    provider: str
    model: str
    api_key: str | None = None
    provider_env: dict[str, str | None] | None = None
    on_event: Callable[[EngineEvent], None] | None = None


@dataclass(frozen=True)
class EngineRequest:
    message: str
    session_id: str | None = None
    run_id: str | None = None
    local_context: tuple[LocalContextItem, ...] | None = None
    tool_registrations: list[ToolRegistration] | None = None
    max_iterations: int | None = None


@dataclass(frozen=True)
class EngineResponse:
    content: str
    run_id: str
    trace_id: str
    termination: LoopTermination
    iterations: int
    diagnostics: OrchestratorDiagnostics
    duration_ms: int


@dataclass(frozen=True)
class EngineEvent:
    kind: str
    detail: dict[str, Any]


class ClawBoundEngine:
    def __init__(
        self,
        adapter: Any,
        config: EngineConfig,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._store = InMemorySessionStore()

    async def run(self, request: EngineRequest) -> EngineResponse:
        if self._adapter is None:
            raise RuntimeError(
                f'Unsupported provider "{self._config.provider}" — cannot resolve adapter for model "{self._config.model}"'
            )

        started = time.monotonic()
        run_id = request.run_id or str(uuid.uuid4())
        trace_id = f"engine-{run_id}"
        session_id = request.session_id

        # Resolve session history
        initial_messages: tuple[ModelMessage, ...] | None = None
        if session_id:
            existing = self._store.get(session_id)
            if existing:
                turns = self._store.get_turns(session_id)
                msgs: list[ModelMessage] = []
                for t in turns:
                    msgs.extend(t.messages)
                initial_messages = tuple(msgs)

        # Run orchestrator
        result = await run_orchestrator(
            run_id=run_id,
            trace_id=trace_id,
            user_message=request.message,
            model_adapter=self._adapter,
            local_context=request.local_context,
            tool_registrations=request.tool_registrations,
            max_iterations=request.max_iterations,
            initial_messages=initial_messages,
        )

        # Store turn for session continuity
        if session_id:
            existing = self._store.get(session_id)
            if not existing:
                self._store.create(session_id, result.task_spec, result.runtime_policy)

            from clawbound.contracts.types import SessionTurn
            from datetime import datetime, timezone

            turn_messages = (
                ModelMessage(role="user", content=request.message),
                ModelMessage(role="assistant", content=result.final_content),
            )

            turns = self._store.get_turns(session_id)
            self._store.append_turn(session_id, SessionTurn(
                turn_number=len(turns) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                messages=turn_messages,
                tool_results=result.loop_result.tool_results,
                signal_bundles=result.loop_result.signal_bundles,
            ))

        duration_ms = int((time.monotonic() - started) * 1000)

        if self._config.on_event:
            self._config.on_event(EngineEvent(
                kind="engine_run_complete",
                detail={
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "termination": result.termination,
                    "iterations": result.iterations,
                    "duration_ms": duration_ms,
                },
            ))

        return EngineResponse(
            content=result.final_content,
            run_id=run_id,
            trace_id=trace_id,
            termination=result.termination,
            iterations=result.iterations,
            diagnostics=result.diagnostics,
            duration_ms=duration_ms,
        )

    def get_session(self, session_id: str) -> SessionSnapshot | None:
        return self._store.get(session_id)

    def compact_session(self, session_id: str, retain_turns: int) -> SessionSnapshot:
        return self._store.compact(session_id, retain_turns)


def create_engine(config: EngineConfig) -> ClawBoundEngine:
    """Create a real engine with provider resolution."""
    from clawbound.provider_adapter.resolver import resolve_provider_adapter

    adapter = resolve_provider_adapter(
        provider=config.provider,
        api_key=config.api_key,
        model_id=config.model,
        env=config.provider_env,
    )
    return ClawBoundEngine(adapter, config)


def create_test_engine(
    responses: list[FinalAnswer | ToolCalls | ModelError],
) -> ClawBoundEngine:
    """Create a test engine with scripted responses."""
    adapter = DeterministicAdapter(responses)
    return ClawBoundEngine(adapter, EngineConfig(provider="test", model="deterministic"))
