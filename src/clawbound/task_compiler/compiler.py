"""TaskCompiler — deterministic keyword-based classification.

Classifies raw user input into a TaskSpec using deterministic keyword routing.
1:1 extraction from runtime.ts buildRoutePlan.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from clawbound.contracts.types import (
    Complexity,
    DecisionStep,
    DecisionTrace,
    DomainSpecificity,
    ExecutionMode,
    LocalContextItem,
    OutputKind,
    RiskLevel,
    SideEffectIntent,
    TargetArtifact,
    TaskSpec,
    TaskType,
)
from clawbound.shared.text_utils import is_verification_like_task, matches_any


class CompileInput:
    """Structured input for compile."""

    __slots__ = ("trace_id", "user_input", "continuation_of", "local_context")

    def __init__(
        self,
        *,
        trace_id: str,
        user_input: str,
        continuation_of: str | None = None,
        local_context: list[LocalContextItem] | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.user_input = user_input
        self.continuation_of = continuation_of
        self.local_context = local_context or []


class TaskCompilerImpl:
    """Deterministic task compiler."""

    def compile(
        self,
        input_text: str,
        session_context: dict[str, object] | None = None,
    ) -> TaskSpec:
        continuation_of: str | None = None
        if session_context and "previous_task_spec" in session_context:
            prev = session_context["previous_task_spec"]
            if isinstance(prev, TaskSpec):
                continuation_of = prev.task_id
        return _compile_from_input(
            CompileInput(
                trace_id=str(uuid.uuid4()),
                user_input=input_text,
                continuation_of=continuation_of,
            ),
        )

    def compile_from_input(self, ci: CompileInput) -> TaskSpec:
        return _compile_from_input(ci)


def _compile_from_input(task: CompileInput) -> TaskSpec:
    text = task.user_input.lower()

    task_type = classify_task_type(text)
    complexity = classify_complexity(text, task_type)
    domain_specificity = classify_domain_specificity(
        text, task.continuation_of, task.local_context,
    )
    risk = classify_risk(text, task_type, domain_specificity)
    execution_mode = classify_execution_mode(task_type, risk, complexity, domain_specificity)
    output_kind = derive_output_kind(task_type)
    side_effect_intent = derive_side_effect_intent(task_type, execution_mode)
    target_artifacts = extract_target_artifacts(task.user_input)

    now = datetime.now(timezone.utc).isoformat()
    steps: list[DecisionStep] = [
        DecisionStep(
            code="TASK_TYPE",
            evidence=task.user_input,
            effect={"task_type": task_type},
            reason="Task type inferred from explicit request keywords.",
            timestamp=now,
        ),
        DecisionStep(
            code="COMPLEXITY",
            evidence=task.user_input,
            effect={"complexity": complexity},
            reason="Complexity is inferred from task shape and ambiguity markers.",
            timestamp=now,
        ),
        DecisionStep(
            code="DOMAIN_SPECIFICITY",
            evidence=f"continuation_of={task.continuation_of}, local_context={len(task.local_context)}",
            effect={"domain_specificity": domain_specificity},
            reason="Domain specificity reflects continuation state and repo-local signals.",
            timestamp=now,
        ),
        DecisionStep(
            code="RISK",
            evidence=task.user_input,
            effect={"risk": risk},
            reason="Risk rises for compatibility-sensitive and externally sensitive work.",
            timestamp=now,
        ),
        DecisionStep(
            code="EXECUTION_MODE",
            evidence=f"risk={risk}, complexity={complexity}, domain={domain_specificity}",
            effect={"execution_mode": execution_mode},
            reason="Execution mode follows the deterministic MVP routing matrix.",
            timestamp=now,
        ),
        DecisionStep(
            code="OUTPUT_KIND",
            evidence=f"task_type={task_type}",
            effect={"output_kind": output_kind},
            reason="Output kind derived from task type classification.",
            timestamp=now,
        ),
        DecisionStep(
            code="SIDE_EFFECT_INTENT",
            evidence=f"task_type={task_type}, execution_mode={execution_mode}",
            effect={"side_effect_intent": side_effect_intent},
            reason="Side-effect intent captures whether task intends to mutate state.",
            timestamp=now,
        ),
    ]

    decision_trace = DecisionTrace(
        trace_id=task.trace_id,
        summary=f"task_type={task_type};complexity={complexity};domain={domain_specificity};risk={risk};mode={execution_mode}",
        steps=tuple(steps),
    )

    return TaskSpec(
        task_id=str(uuid.uuid4()),
        trace_id=task.trace_id,
        task_type=task_type,
        complexity=complexity,
        risk=risk,
        domain_specificity=domain_specificity,
        execution_mode=execution_mode,
        output_kind=output_kind,
        side_effect_intent=side_effect_intent,
        target_artifacts=tuple(target_artifacts),
        raw_input=task.user_input,
        decision_trace=decision_trace,
    )


# ─── Classification functions ─────────────────────────────────────────────────


def classify_task_type(text: str) -> TaskType:
    if is_verification_like_task(text) or matches_any(
        text, ["review", "regression risk", "compatibility audit"]
    ):
        return "review"
    if matches_any(text, ["architecture", "design", "approach", "restructure", "proposal"]):
        return "architecture"
    if matches_any(text, ["debug", "troubleshoot", "investigate"]):
        return "debug"
    if matches_any(text, ["explain", "what does", "summarize", "answer"]):
        return "answer"
    if matches_any(text, ["fix", "rename", "change", "refactor", "implement", "add"]):
        return "code_change"
    return "answer"


def classify_complexity(text: str, task_type: TaskType) -> Complexity:
    if task_type == "answer" and len(text.split()) <= 10:
        return "trivial"
    if task_type in ("review", "architecture"):
        return "multi_step"
    if matches_any(text, ["safely", "approach", "plan", "strategy", "ambiguous"]):
        return "ambiguous"
    return "bounded"


def classify_domain_specificity(
    text: str,
    continuation_of: str | None,
    local_context: list[LocalContextItem],
) -> DomainSpecificity:
    if continuation_of or matches_any(text, ["previous run", "continue"]):
        return "continuation_sensitive"
    if len(local_context) > 0 or matches_any(
        text, ["parser", "payment", "sync", "module", ".py", "public api", "diff"]
    ):
        return "repo_specific"
    return "generic"


def classify_risk(
    text: str,
    task_type: TaskType,
    domain_specificity: DomainSpecificity,
) -> RiskLevel:
    if matches_any(
        text, ["public api", "external behavior", "payment", "security", "delete", "migrate"]
    ):
        return "high"
    if (
        task_type in ("review", "architecture", "debug", "code_change")
        or domain_specificity != "generic"
    ):
        return "medium"
    return "low"


def classify_execution_mode(
    task_type: TaskType,
    risk: RiskLevel,
    complexity: Complexity,
    domain_specificity: DomainSpecificity,
) -> ExecutionMode:
    if task_type == "answer":
        return "answer"
    if task_type == "review":
        return "reviewer"
    if task_type == "architecture":
        return "architect_like_plan"
    if (
        risk == "high"
        or complexity in ("multi_step", "ambiguous")
        or domain_specificity == "continuation_sensitive"
    ):
        return "executor_then_reviewer"
    return "executor"


# ─── Phase 3 intent derivation ────────────────────────────────────────────────


def derive_output_kind(task_type: TaskType) -> OutputKind:
    match task_type:
        case "answer":
            return "explanation"
        case "review":
            return "review_comments"
        case "code_change":
            return "code_patch"
        case "architecture":
            return "plan"
        case "debug":
            return "diagnostic"


def derive_side_effect_intent(
    task_type: TaskType,
    execution_mode: ExecutionMode,
) -> SideEffectIntent:
    if task_type in ("answer", "review"):
        return "none"
    if task_type == "architecture":
        return "proposed"
    if execution_mode in ("executor", "executor_then_reviewer"):
        return "immediate"
    return "proposed"


# ─── Target artifact extraction ────────────────────────────────────────────────

_FILE_PATH_RE = re.compile(
    r"(?:\.{0,2}/)?[\w./-]+\.(?:[cm]?[jt]sx?|py|rs|go|md|json|yaml|yml|toml|sh)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s)\"']+", re.IGNORECASE)
_ISSUE_REF_RE = re.compile(r"(?:PR\s*)?#(\d{1,6})", re.IGNORECASE)


def extract_target_artifacts(raw_input: str) -> list[TargetArtifact]:
    artifacts: list[TargetArtifact] = []
    seen: set[str] = set()

    def add_unique(kind: str, ref: str) -> None:
        key = f"{kind}:{ref}"
        if key not in seen:
            seen.add(key)
            artifacts.append(TargetArtifact(kind=kind, ref=ref))  # type: ignore[arg-type]

    for fp in _FILE_PATH_RE.findall(raw_input):
        add_unique("file", fp)

    for url in _URL_RE.findall(raw_input):
        add_unique("url", url)

    for match in _ISSUE_REF_RE.finditer(raw_input):
        num = match.group(1)
        full = match.group(0)
        kind = "pr" if full.lower().startswith("pr") else "issue"
        add_unique(kind, f"#{num}")

    return artifacts
