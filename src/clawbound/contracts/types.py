"""ClawBound Runtime Contracts.

Canonical type definitions for the ClawBound runtime pipeline.
These types are the shared language between all runtime modules.

Tier 1: Foundational data shapes (TaskSpec, RuntimePolicy, PromptSegment, etc.)
Tier 2: Gate and validation decisions
Tier 3: Provider-facing types
Tier 4: Composite types
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Discriminator

# ─── Literal Types (unions) ────────────────────────────────────────────────────

ExecutionMode = Literal[
    "answer",
    "reviewer",
    "executor",
    "executor_then_reviewer",
    "architect_like_plan",
]

TaskType = Literal["answer", "review", "code_change", "architecture", "debug"]

Complexity = Literal["trivial", "bounded", "multi_step", "ambiguous"]

RiskLevel = Literal["low", "medium", "high"]

DomainSpecificity = Literal["generic", "repo_specific", "continuation_sensitive"]

OutputKind = Literal["explanation", "code_patch", "review_comments", "plan", "diagnostic"]

SideEffectIntent = Literal["none", "proposed", "immediate"]


# ─── Decision Trace ───────────────────────────────────────────────────────────


class DecisionStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    evidence: str
    effect: dict[str, Any]
    reason: str
    timestamp: str


class DecisionTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace_id: str
    summary: str
    steps: tuple[DecisionStep, ...]


# ─── Tier 1: TaskSpec ──────────────────────────────────────────────────────────

TargetArtifactKind = Literal["file", "directory", "url", "symbol", "pr", "issue"]


class TargetArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: TargetArtifactKind
    ref: str


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    trace_id: str
    task_type: TaskType
    complexity: Complexity
    risk: RiskLevel
    domain_specificity: DomainSpecificity
    execution_mode: ExecutionMode
    output_kind: OutputKind
    side_effect_intent: SideEffectIntent
    target_artifacts: tuple[TargetArtifact, ...]
    raw_input: str
    decision_trace: DecisionTrace


# ─── Tier 1: RuntimePolicy ────────────────────────────────────────────────────


class ContextBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    always_on_max_tokens: int
    retrieval_max_units: int
    retrieval_max_tokens: int
    host_injection_max_tokens: int
    signal_max_tokens_per_result: int


class ToolProfilePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_name: str
    allowed_tools: tuple[str, ...]
    denied_tools: tuple[str, ...]
    notes: tuple[str, ...]
    requires_review: bool


class ScopeBounds(BaseModel):
    model_config = ConfigDict(frozen=True)

    include_paths: tuple[str, ...]
    exclude_paths: tuple[str, ...]
    allow_network_access: bool
    allow_subagent_delegation: bool


class ApprovalPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    require_approval_for_categories: tuple[str, ...]
    require_approval_for_patterns: tuple[str, ...]
    require_approval_for_all_side_effects: bool


class IterationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_turns: int
    max_tool_calls_per_turn: int
    max_consecutive_errors: int
    allow_retry_on_transient_error: bool
    max_transient_retries: int


class RuntimePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_mode: ExecutionMode
    context_budget: ContextBudget
    tool_profile: ToolProfilePolicy
    scope_bounds: ScopeBounds
    approval_policy: ApprovalPolicy
    iteration_policy: IterationPolicy
    decision_trace: DecisionTrace


# ─── Tier 1: PromptSegment & PromptEnvelope ────────────────────────────────────

SegmentOwner = Literal[
    "runtime", "task", "context", "tool_contract", "host_injection", "session_history"
]

SegmentCandidateReason = Literal[
    "always_on",
    "task_derived",
    "retrieval_gated",
    "policy_required",
    "host_registered",
    "continuation",
]

TrimReason = Literal["budget_cap", "content_too_large"]

RejectionReason = Literal[
    "budget_exhausted", "policy_denied", "no_load", "below_confidence", "duplicate"
]


class AdmittedFull(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["admitted_full"] = "admitted_full"


class AdmittedTrimmed(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["admitted_trimmed"] = "admitted_trimmed"
    original_token_estimate: int
    trimmed_token_estimate: int
    trim_reason: TrimReason


class Rejected(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["rejected"] = "rejected"
    rejection_reason: RejectionReason


SegmentAdmissionOutcome = Annotated[
    AdmittedFull | AdmittedTrimmed | Rejected,
    Discriminator("status"),
]


class PromptSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment_id: str
    owner: SegmentOwner
    purpose: str
    content: str
    token_estimate: int
    budget_cap: int
    candidate_reason: SegmentCandidateReason
    admission_outcome: SegmentAdmissionOutcome
    provenance: str
    trace_visible: bool
    order: int


class BudgetUtilization(BaseModel):
    model_config = ConfigDict(frozen=True)

    always_on: float
    retrieval: float
    host_injection: float


class PromptAssemblyStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_segment_tokens: dict[str, int]
    total_segment_tokens: int
    total_estimated_tokens: int
    budget_utilization: BudgetUtilization
    no_load: bool
    segments_admitted: int
    segments_rejected: int


# ─── Tier 1: Local Context & Retrieval ─────────────────────────────────────────


class LocalContextItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    ref: str
    content: str
    token_estimate: int


class RetrievalUnit(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: str
    scope: str
    content: str
    confidence: float
    priority: int
    token_estimate: int
    source_ref: str
    tags: tuple[str, ...]


class KernelAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    content: str
    token_estimate: int


# ─── Tier 1: Tool Types ────────────────────────────────────────────────────────

ToolRiskLevel = Literal["read_only", "side_effect", "destructive", "network"]

ToolCategory = Literal[
    "filesystem",
    "execution",
    "memory",
    "web",
    "messaging",
    "ui",
    "runtime",
    "session",
    "automation",
    "plugin",
]

ToolExecutionStatus = Literal["success", "error", "timeout", "denied", "not_found"]

ToolOutputMediaType = Literal[
    "text/plain",
    "text/structured",
    "application/json",
    "text/diff",
    "text/directory_listing",
    "image/base64",
    "binary/opaque",
]

StructuredOutputKind = Literal[
    "test_results",
    "build_output",
    "lint_output",
    "diff_output",
    "log_output",
    "directory_listing",
    "json_response",
    "api_response",
    "file_content",
    "search_results",
    "generic",
]


# ─── Tier 1: Signal Types ──────────────────────────────────────────────────────


class TestFailure(BaseModel):
    __test__ = False  # prevent pytest collection
    model_config = ConfigDict(frozen=True)

    name: str
    file: str | None = None
    line: int | None = None
    message: str
    stack: str | None = None


class TestSummary(BaseModel):
    __test__ = False  # prevent pytest collection
    model_config = ConfigDict(frozen=True)

    total: int
    passed: int
    failed: int
    skipped: int
    duration_ms: int | None = None


class TestResultsSignal(BaseModel):
    __test__ = False  # prevent pytest collection
    model_config = ConfigDict(frozen=True)

    kind: Literal["test_results"] = "test_results"
    summary: TestSummary
    failures: tuple[TestFailure, ...]


class BuildError(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str
    line: int | None = None
    column: int | None = None
    code: str | None = None
    message: str


class BuildWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str
    line: int | None = None
    message: str


class BuildOutputSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["build_output"] = "build_output"
    success: bool
    errors: tuple[BuildError, ...]
    warnings: tuple[BuildWarning, ...]


class LintExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str
    line: int | None = None


class LintRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    count: int
    severity: Literal["error", "warning"]
    examples: tuple[LintExample, ...]


class LintOutputSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["lint_output"] = "lint_output"
    total_violations: int
    fixable: int
    by_rule: tuple[LintRule, ...]


class DirectoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    file_count: int
    notable: tuple[str, ...] | None = None


class DirectoryListingSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["directory_listing"] = "directory_listing"
    root: str
    total_files: int
    total_dirs: int
    tree: tuple[DirectoryEntry, ...]


class JsonResponseSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["json_response"] = "json_response"
    http_status: int | None = None
    schema_: dict[str, str]
    summary: dict[str, str | int | bool]


class GenericSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["generic"] = "generic"
    extracted: dict[str, str | int | bool]


StructuredSignal = Annotated[
    TestResultsSignal
    | BuildOutputSignal
    | LintOutputSignal
    | DirectoryListingSignal
    | JsonResponseSignal
    | GenericSignal,
    Discriminator("kind"),
]

SignalLossRisk = Literal["none", "low", "medium", "high"]


class CompressionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    classified_as: StructuredOutputKind
    filter_applied: str
    loss_risk: SignalLossRisk


# ─── Tier 4: PromptEnvelope ───────────────────────────────────────────────────


class PromptEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    trace_id: str
    role: str
    segments: tuple[PromptSegment, ...]
    assembly_stats: PromptAssemblyStats
    system_prompt: str
    no_load: bool


# ─── Tier 2: ToolDefinition & ToolResult ──────────────────────────────────────


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    category: ToolCategory
    risk_level: ToolRiskLevel
    declared_output_kind: StructuredOutputKind | None = None
    declared_media_type: ToolOutputMediaType | None = None
    description: str | None = None


class ToolExecuteParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    policy: ToolProfilePolicy


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_call_id: str
    status: ToolExecutionStatus
    raw_output: str
    media_type: ToolOutputMediaType
    output_kind: StructuredOutputKind
    duration_ms: int
    metadata: dict[str, Any]


# ─── Tier 2: SignalBundle ─────────────────────────────────────────────────────


class SignalBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    structured: StructuredSignal
    compressed_text: str
    compression_metrics: CompressionMetrics


# ─── Module Interfaces ────────────────────────────────────────────────────────


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget_defaults: dict[str, ContextBudget]  # keyed by ExecutionMode
    tool_profile_defaults: dict[str, ToolProfilePolicy]  # keyed by ExecutionMode
    default_scope_bounds: ScopeBounds
    default_approval_policy: ApprovalPolicy
    default_iteration_policy: IterationPolicy
    host_overrides: dict[str, Any] | None = None


class PromptBuildInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    trace_id: str
    task_spec: TaskSpec
    runtime_policy: RuntimePolicy
    kernel: KernelAsset
    local_context: tuple[LocalContextItem, ...]
    retrieved_units: tuple[RetrievalUnit, ...]
    no_load: bool


# ─── Provider-facing types (Execution Loop) ───────────────────────────────────


class ModelMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: tuple[ModelToolCall, ...] | None = None


class ModelToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str
    messages: tuple[ModelMessage, ...]
    tool_definitions: tuple[ToolDefinition, ...]


class FinalAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["final_answer"] = "final_answer"
    content: str


class ToolCalls(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_calls"] = "tool_calls"
    calls: tuple[ModelToolCall, ...]
    reasoning: str | None = None


class ModelError(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["error"] = "error"
    error: str
    is_transient: bool


ModelResponse = Annotated[
    FinalAnswer | ToolCalls | ModelError,
    Discriminator("kind"),
]


class ModelAdapter(Protocol):
    """Thin model adapter interface. Translates request/response only."""

    async def send(self, request: ModelRequest) -> FinalAnswer | ToolCalls | ModelError: ...


# ─── Action Gate ──────────────────────────────────────────────────────────────


class ActionAllowed(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: Literal[True] = True


class ActionDenied(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: Literal[False] = False
    reason: str


ActionGateDecision = ActionAllowed | ActionDenied


# ─── Loop Types ────────────────────────────────────────────────────────────────


class LoopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iterations: int
    max_consecutive_errors: int
    max_transient_retries: int


LoopTermination = Literal["final_answer", "max_iterations", "max_errors", "adapter_failure"]


class LoopStepEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    iteration: int
    kind: Literal[
        "model_request",
        "final_answer",
        "tool_execution",
        "signal_processed",
        "gate_denied",
        "retry",
        "error",
    ]
    detail: dict[str, Any]


class LoopResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    trace_id: str
    termination: LoopTermination
    final_content: str
    iterations: int
    tool_results: tuple[ToolResult, ...]
    signal_bundles: tuple[SignalBundle, ...]
    events: tuple[LoopStepEvent, ...]


class LoopInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    trace_id: str
    task_spec: TaskSpec
    runtime_policy: RuntimePolicy
    kernel: KernelAsset
    local_context: tuple[LocalContextItem, ...]
    retrieved_units: tuple[RetrievalUnit, ...]
    user_message: str
    initial_messages: tuple[ModelMessage, ...] | None = None
    config: LoopConfig


# ─── Session Types ─────────────────────────────────────────────────────────────


class SessionTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_number: int
    timestamp: str
    messages: tuple[ModelMessage, ...]
    tool_results: tuple[ToolResult, ...]
    signal_bundles: tuple[SignalBundle, ...]


class SessionBounds(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_turns: int
    max_stored_tokens: int
    was_compacted: bool
    retained_turns: int


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    run_id: str
    trace_id: str
    task_spec: TaskSpec
    policy: RuntimePolicy
    turns: tuple[SessionTurn, ...]
    bounds: SessionBounds
    created_at: str
    updated_at: str
    compacted_summary: str | None = None


# ─── Orchestrator Types ───────────────────────────────────────────────────────


class ToolRegistration(BaseModel):
    model_config = ConfigDict(frozen=True)

    definition: ToolDefinition
    # execute_fn is not serializable — stored separately at runtime


class OrchestratorDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    execution_mode: ExecutionMode
    complexity: Complexity
    risk: RiskLevel
    prompt_token_estimate: int
    tools_resolved: int
    segments_admitted: int
    segments_rejected: int


class OrchestratorInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    trace_id: str
    user_message: str
    candidate_tools: tuple[str, ...] | None = None
    local_context: tuple[LocalContextItem, ...] | None = None
    session_id: str | None = None
    max_iterations: int | None = None
    initial_messages: tuple[ModelMessage, ...] | None = None


class OrchestratorResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    trace_id: str
    path: Literal["new_runtime"] = "new_runtime"
    termination: LoopTermination
    final_content: str
    iterations: int
    diagnostics: OrchestratorDiagnostics
    loop_result: LoopResult
    task_spec: TaskSpec
    runtime_policy: RuntimePolicy
