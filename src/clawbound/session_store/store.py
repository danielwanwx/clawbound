"""InMemorySessionStore — bounded operational session state.

Compaction is deterministic — no LLM summarization (guardrail #5).
"""

from __future__ import annotations

from datetime import datetime, timezone

from clawbound.contracts.types import (
    RuntimePolicy,
    SessionBounds,
    SessionSnapshot,
    SessionTurn,
    TaskSpec,
)

DEFAULT_MAX_TURNS = 50
DEFAULT_MAX_STORED_TOKENS = 100_000


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, _MutableSnapshot] = {}

    def create(
        self,
        session_id: str,
        task_spec: TaskSpec,
        policy: RuntimePolicy,
    ) -> SessionSnapshot:
        if session_id in self._sessions:
            raise ValueError(f'Session "{session_id}" already exists')

        now = datetime.now(timezone.utc).isoformat()
        snapshot = _MutableSnapshot(
            session_id=session_id,
            run_id=task_spec.task_id,
            trace_id=task_spec.trace_id,
            task_spec=task_spec,
            policy=policy,
            turns=[],
            bounds=SessionBounds(
                max_turns=DEFAULT_MAX_TURNS,
                max_stored_tokens=DEFAULT_MAX_STORED_TOKENS,
                was_compacted=False,
                retained_turns=0,
            ),
            created_at=now,
            updated_at=now,
        )
        self._sessions[session_id] = snapshot
        return snapshot.to_snapshot()

    def append_turn(self, session_id: str, turn: SessionTurn) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        snapshot.turns.append(turn)
        snapshot.updated_at = datetime.now(timezone.utc).isoformat()
        return snapshot.to_snapshot()

    def get(self, session_id: str) -> SessionSnapshot | None:
        snapshot = self._sessions.get(session_id)
        return snapshot.to_snapshot() if snapshot else None

    def get_turns(
        self,
        session_id: str,
        range_: tuple[int, int] | None = None,
    ) -> list[SessionTurn]:
        snapshot = self._sessions.get(session_id)
        if not snapshot:
            return []
        if range_ is None:
            return list(snapshot.turns)
        start, end = range_
        return snapshot.turns[start:end]

    def compact(self, session_id: str, retain_turns: int) -> SessionSnapshot:
        snapshot = self._require_session(session_id)

        if retain_turns < 0:
            raise ValueError(f"retain_turns must be non-negative, got {retain_turns}")

        total = len(snapshot.turns)
        if retain_turns >= total:
            return snapshot.to_snapshot()

        dropped = snapshot.turns[: total - retain_turns]
        retained = snapshot.turns[total - retain_turns :]

        snapshot.compacted_summary = build_compaction_summary(
            dropped, snapshot.compacted_summary,
        )
        snapshot.turns = retained
        snapshot.bounds = SessionBounds(
            max_turns=snapshot.bounds.max_turns,
            max_stored_tokens=snapshot.bounds.max_stored_tokens,
            was_compacted=True,
            retained_turns=len(retained),
        )
        snapshot.updated_at = datetime.now(timezone.utc).isoformat()
        return snapshot.to_snapshot()

    def _require_session(self, session_id: str) -> _MutableSnapshot:
        snapshot = self._sessions.get(session_id)
        if not snapshot:
            raise ValueError(f'Session "{session_id}" not found')
        return snapshot


class _MutableSnapshot:
    __slots__ = (
        "session_id", "run_id", "trace_id", "task_spec", "policy",
        "turns", "bounds", "created_at", "updated_at", "compacted_summary",
    )

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        trace_id: str,
        task_spec: TaskSpec,
        policy: RuntimePolicy,
        turns: list[SessionTurn],
        bounds: SessionBounds,
        created_at: str,
        updated_at: str,
        compacted_summary: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.run_id = run_id
        self.trace_id = trace_id
        self.task_spec = task_spec
        self.policy = policy
        self.turns = turns
        self.bounds = bounds
        self.created_at = created_at
        self.updated_at = updated_at
        self.compacted_summary = compacted_summary

    def to_snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.session_id,
            run_id=self.run_id,
            trace_id=self.trace_id,
            task_spec=self.task_spec,
            policy=self.policy,
            turns=tuple(self.turns),
            bounds=self.bounds,
            created_at=self.created_at,
            updated_at=self.updated_at,
            compacted_summary=self.compacted_summary,
        )


def build_compaction_summary(
    dropped_turns: list[SessionTurn],
    previous_summary: str | None = None,
) -> str:
    tool_names: set[str] = set()
    success_count = 0
    error_count = 0
    message_count = 0

    for turn in dropped_turns:
        message_count += len(turn.messages)
        for tr in turn.tool_results:
            tool_names.add(tr.tool_name)
            if tr.status == "success":
                success_count += 1
            else:
                error_count += 1

    lines: list[str] = []
    if previous_summary:
        lines.append(previous_summary)
        lines.append("---")
    lines.append(f"Compacted {len(dropped_turns)} turn(s), {message_count} message(s).")
    if tool_names:
        lines.append(f"Tools used: {', '.join(sorted(tool_names))}.")
        lines.append(f"Results: {success_count} success, {error_count} error.")

    return "\n".join(lines)
