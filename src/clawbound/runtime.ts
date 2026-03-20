import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { emitAgentEvent } from "../infra/agent-events.js";
import { isSubagentSessionKey } from "../sessions/session-key-utils.js";

const TOKEN_PATTERN = /[A-Za-z0-9_]+/g;
const MAX_SNIPPET_TOKENS = 48;
const MAX_SNIPPET_CHARS = 280;
const MAX_SNIPPET_LINES = 4;
const DEFAULT_SOURCE_HOST = "clawbound-shadow/openclaw";
const SNAPSHOT_VERSION = "clawbound-shadow-v1";

export type ClawBoundLocalContextItem = {
  kind: string;
  ref: string;
  content: string;
};

export type ClawBoundDecisionStep = {
  code: string;
  evidence: string;
  effect: Record<string, unknown>;
  reason: string;
};

export type ClawBoundDecisionTrace = {
  traceId: string;
  summary: string;
  steps: ClawBoundDecisionStep[];
};

export type ClawBoundContextBudget = {
  alwaysOnMaxTokens: number;
  retrievalMaxUnits: number;
  retrievalMaxTokens: number;
};

export type ClawBoundRoutePlan = {
  taskType: string;
  complexity: string;
  risk: string;
  domainSpecificity: string;
  executionMode: string;
  contextBudget: ClawBoundContextBudget;
  decisionTrace: ClawBoundDecisionTrace;
};

export type ClawBoundRetrievalPlan = {
  loadLaws: boolean;
  loadProjectNotes: boolean;
  loadEpisodeSummary: boolean;
  lawQueries: string[];
  projectQueries: string[];
  episodeQueries: string[];
  maxUnits: number;
  maxTokens: number;
  noLoad: boolean;
  gateTrace: ClawBoundDecisionTrace;
};

export type ClawBoundRetrievalUnit = {
  id: string;
  type: string;
  scope: string;
  applicability: string;
  content: string;
  confidence: number;
  priority: number;
  tokenEstimate: number;
  sourceRef: string;
  lastValidatedAt: string;
  tags: string[];
  enabled: boolean;
  snippetStatus: "ok" | "truncated";
  snippetDiagnostics: string[];
  sourceTokenEstimate: number;
};

export type ClawBoundToolProfile = {
  profile: string;
  allowedTools: string[];
  deniedTools: string[];
  notes: string[];
  candidateToolsObserved: string[];
  requiresReview: boolean;
};

export type ClawBoundPromptPackage = {
  runId: string;
  traceId: string;
  role: string;
  kernelVersion: string;
  kernelText: string;
  assemblyOrder: string[];
  modeInstruction: string;
  taskBrief: string;
  localContext: ClawBoundLocalContextItem[];
  retrievedUnits: ClawBoundRetrievalUnit[];
  toolContract: Pick<ClawBoundToolProfile, "allowedTools" | "deniedTools" | "notes">;
  assemblyStats: {
    alwaysOnTokens: number;
    retrievedTokens: number;
    retrievalUnitsCount: number;
    segmentBreakdown: Record<string, number>;
    budget: {
      alwaysOnMaxTokens: number;
      retrievalMaxUnits: number;
      retrievalMaxTokens: number;
      alwaysOnUtilization: number;
      retrievalTokenUtilization: number;
      retrievalUnitUtilization: number;
    };
    totalEstimatedTokens: number;
    noLoad: boolean;
  };
  noLoad: boolean;
};

export type ClawBoundParityResult = {
  fixtureId: string | null;
  available: boolean;
  checks: {
    route: boolean;
    noLoad: boolean;
    retrievalUnits: boolean;
    toolProfile: boolean;
  };
  expected?: Record<string, unknown>;
  actual?: Record<string, unknown>;
};

export type ClawBoundShadowEvent = {
  sequenceNo: number;
  eventType: string;
  createdAt: string;
  payload: Record<string, unknown>;
};

export type ClawBoundShadowSnapshot = {
  version: string;
  hostRunId: string | null;
  runId: string;
  traceId: string;
  sourceHost: string;
  sessionId: string;
  conversationId: string;
  userInput: string;
  continuationOf: string | null;
  routePlan: ClawBoundRoutePlan;
  retrievalPlan: ClawBoundRetrievalPlan;
  promptPackage: ClawBoundPromptPackage;
  toolProfile: ClawBoundToolProfile;
  parity: ClawBoundParityResult;
  events: ClawBoundShadowEvent[];
  createdAt: string;
  completedAt: string;
};

export type ClawBoundPlanRunInput = {
  runId?: string;
  traceId?: string;
  hostRunId?: string;
  sessionId: string;
  sessionKey?: string;
  conversationId?: string;
  userInput: string;
  continuationOf?: string;
  intentHints?: string[];
  localContext?: ClawBoundLocalContextItem[];
  candidateTools?: string[];
  sourceHost?: string;
  emitToHostEvents?: boolean;
};

export type ClawBoundPlanRunResult = {
  runId: string;
  traceId: string;
  routePlan: ClawBoundRoutePlan;
  retrievalPlan: ClawBoundRetrievalPlan;
  promptPackage: ClawBoundPromptPackage;
  toolProfile: ClawBoundToolProfile;
  parity: ClawBoundParityResult;
  events: ClawBoundShadowEvent[];
  persistedPath: string;
};

export type ClawBoundKernelAsset = {
  version: string;
  content: string;
  tokenEstimate: number;
};

type RawSeedUnit = {
  id: string;
  type: string;
  scope: string;
  applicability: string;
  content: string;
  confidence: number;
  priority: number;
  token_estimate: number;
  source_ref: string;
  last_validated_at: string;
  tags: string[];
  enabled: boolean;
  run_id?: string;
  session_id?: string;
};

type OracleFixture = {
  id: string;
  userInput: string;
  expected: {
    taskType: string;
    executionMode: string;
    noLoad: boolean;
    retrievedUnitIds: string[];
    toolProfile: string;
    allowedTools: string[];
    deniedTools: string[];
  };
};

type ClawBoundRuntimeOptions = {
  shadowRootDir: string;
  idFactory?: () => string;
  kernelAssetPath?: string;
  lawsDir?: string;
  projectNotesDir?: string;
  episodeSummariesDir?: string;
  oracleFixturePath?: string;
};

const defaultAssetPath = (...parts: string[]) => fileURLToPath(new URL(parts.join("/"), import.meta.url));

export class ClawBoundRuntime {
  private readonly shadowRootDir: string;
  private readonly idFactory: () => string;
  private readonly kernelAssetPath: string;
  private readonly lawsDir: string;
  private readonly projectNotesDir: string;
  private readonly episodeSummariesDir: string;
  private readonly oracleFixturePath: string;

  constructor(options: ClawBoundRuntimeOptions) {
    this.shadowRootDir = options.shadowRootDir;
    this.idFactory = options.idFactory ?? randomUUID;
    this.kernelAssetPath =
      options.kernelAssetPath ?? defaultAssetPath("./assets/kernel/context-kernel-v0.md");
    this.lawsDir = options.lawsDir ?? defaultAssetPath("./assets/seeds/laws");
    this.projectNotesDir = options.projectNotesDir ?? defaultAssetPath("./assets/seeds/project_notes");
    this.episodeSummariesDir =
      options.episodeSummariesDir ?? defaultAssetPath("./assets/seeds/episode_summaries");
    this.oracleFixturePath =
      options.oracleFixturePath ?? defaultAssetPath("./assets/oracle/phase1-fixtures.json");
  }

  createRunId() {
    return `clawbound-run-${this.idFactory()}`;
  }

  createTraceId() {
    return `clawbound-trace-${this.idFactory()}`;
  }

  async loadKernelAsset() {
    return loadKernelAsset(this.kernelAssetPath);
  }

  async planRun(input: ClawBoundPlanRunInput): Promise<ClawBoundPlanRunResult> {
    const runId = input.runId?.trim() || this.createRunId();
    const traceId = input.traceId?.trim() || this.createTraceId();
    const events: ClawBoundShadowEvent[] = [];
    const sourceHost = input.sourceHost ?? DEFAULT_SOURCE_HOST;
    const localContext = input.localContext ?? [];
    const conversationId = input.conversationId ?? input.sessionKey ?? input.sessionId;
    const hostRunId = input.hostRunId?.trim() || null;

    const recordEvent = (eventType: string, payload: Record<string, unknown>) => {
      const event: ClawBoundShadowEvent = {
        sequenceNo: events.length + 1,
        eventType,
        createdAt: new Date().toISOString(),
        payload,
      };
      events.push(event);
      if (input.emitToHostEvents) {
        emitAgentEvent({
          runId: hostRunId ?? runId,
          sessionKey: input.sessionKey,
          stream: "clawbound",
          data: {
            nativeRunId: runId,
            nativeTraceId: traceId,
            eventType,
            ...payload,
          },
        });
      }
      return event;
    };

    try {
      recordEvent("task_received", {
        hostRunId,
        sourceHost,
        sessionId: input.sessionId,
        conversationId,
        userInput: input.userInput,
      });

      const task = {
        taskId: hostRunId ? `task-${hostRunId}` : `task-${this.idFactory()}`,
        traceId,
        sourceHost,
        sessionId: input.sessionId,
        conversationId,
        userInput: input.userInput,
        continuationOf: input.continuationOf ?? null,
        intentHints: input.intentHints ?? [],
        localContext,
        candidateTools: canonicalizeCandidateTools(input.candidateTools ?? []),
      };

      const kernel = await this.loadKernelAsset();
      const routePlan = buildRoutePlan(task);
      recordEvent("route_decided", { routePlan });

      const retrievalPlan = buildRetrievalPlan(task, routePlan);
      recordEvent("gate_decided", { retrievalPlan });

      const retrievedUnits = await retrieveUnits({
        lawsDir: this.lawsDir,
        projectNotesDir: this.projectNotesDir,
        episodeSummariesDir: this.episodeSummariesDir,
        task,
        retrievalPlan,
      });
      recordEvent("retrieval_executed", {
        selectedUnitIds: retrievedUnits.map((unit) => unit.id),
        selectedUnits: retrievedUnits,
        totalRetrievedTokens: retrievedUnits.reduce((total, unit) => total + unit.tokenEstimate, 0),
      });

      const toolProfile = buildToolProfile(
        routePlan.executionMode,
        input.sessionKey,
        task.userInput,
        task.candidateTools,
      );
      const promptPackage = buildPromptPackage({
        runId,
        traceId,
        kernel,
        task,
        routePlan,
        retrievalPlan,
        retrievedUnits,
        toolProfile,
      });
      recordEvent("prompt_built", { promptPackage });
      recordEvent("tool_profile_built", { toolProfile });

      const parity = await compareToOracle(
        this.oracleFixturePath,
        input.userInput,
        routePlan,
        retrievalPlan,
        retrievedUnits,
        toolProfile,
      );
      recordEvent("run_completed", {
        parity,
        retrievalUnitIds: retrievedUnits.map((unit) => unit.id),
      });

      const snapshot: ClawBoundShadowSnapshot = {
        version: SNAPSHOT_VERSION,
        hostRunId,
        runId,
        traceId,
        sourceHost,
        sessionId: input.sessionId,
        conversationId,
        userInput: input.userInput,
        continuationOf: input.continuationOf ?? null,
        routePlan,
        retrievalPlan,
        promptPackage,
        toolProfile,
        parity,
        events,
        createdAt: events[0]?.createdAt ?? new Date().toISOString(),
        completedAt: events.at(-1)?.createdAt ?? new Date().toISOString(),
      };

      const persistedPath = await persistSnapshot(this.shadowRootDir, hostRunId ?? runId, snapshot);
      return {
        runId,
        traceId,
        routePlan,
        retrievalPlan,
        promptPackage,
        toolProfile,
        parity,
        events,
        persistedPath,
      };
    } catch (error) {
      recordEvent("run_failed", {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }
}

function tokenize(...parts: string[]) {
  const tokens: string[] = [];
  for (const part of parts) {
    if (!part) {
      continue;
    }
    for (const match of part.matchAll(TOKEN_PATTERN)) {
      tokens.push(match[0].toLowerCase());
    }
  }
  return tokens;
}

function uniqueTokens(...parts: string[]) {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const token of tokenize(...parts)) {
    if (seen.has(token)) {
      continue;
    }
    seen.add(token);
    ordered.push(token);
  }
  return ordered;
}

function estimateTokensFromText(...parts: string[]) {
  return parts.some((part) => part && part.length > 0) ? Math.max(1, tokenize(...parts).length) : 0;
}

function estimateTokensFromItems(items: string[]) {
  return estimateTokensFromText(items.join(" "));
}

async function loadKernelAsset(kernelAssetPath: string): Promise<ClawBoundKernelAsset> {
  const content = (await fs.readFile(kernelAssetPath, "utf8")).trim();
  return {
    version: path.basename(kernelAssetPath, path.extname(kernelAssetPath)),
    content,
    tokenEstimate: estimateTokensFromText(content),
  };
}

function buildRoutePlan(task: {
  traceId: string;
  userInput: string;
  continuationOf: string | null;
  localContext: ClawBoundLocalContextItem[];
}) {
  const text = task.userInput.toLowerCase();
  let taskType = "answer";
  if (
    isVerificationLikeTask(text) ||
    matchesAny(text, ["review", "regression risk", "compatibility audit"])
  ) {
    taskType = "review";
  } else if (matchesAny(text, ["architecture", "design", "approach", "restructure", "proposal"])) {
    taskType = "architecture";
  } else if (matchesAny(text, ["debug", "troubleshoot", "investigate"])) {
    taskType = "debug";
  } else if (matchesAny(text, ["explain", "what does", "summarize", "answer"])) {
    taskType = "answer";
  } else if (matchesAny(text, ["fix", "rename", "change", "refactor", "implement", "add"])) {
    taskType = "code_change";
  }

  let complexity = "bounded";
  if (taskType === "answer" && text.split(/\s+/).length <= 10) {
    complexity = "trivial";
  } else if (taskType === "review" || taskType === "architecture") {
    complexity = "multi_step";
  } else if (matchesAny(text, ["safely", "approach", "plan", "strategy", "ambiguous"])) {
    complexity = "ambiguous";
  }

  let domainSpecificity = "generic";
  if (task.continuationOf || matchesAny(text, ["previous run", "continue"])) {
    domainSpecificity = "continuation_sensitive";
  } else if (
    task.localContext.length > 0 ||
    matchesAny(text, ["parser", "payment", "sync", "module", ".py", "public api", "diff"])
  ) {
    domainSpecificity = "repo_specific";
  }

  let risk = "low";
  if (
    matchesAny(text, [
      "public api",
      "external behavior",
      "payment",
      "security",
      "delete",
      "migrate",
    ])
  ) {
    risk = "high";
  } else if (
    taskType === "review" ||
    taskType === "architecture" ||
    taskType === "debug" ||
    taskType === "code_change" ||
    domainSpecificity !== "generic"
  ) {
    risk = "medium";
  }

  let executionMode = "executor";
  if (taskType === "answer") {
    executionMode = "answer";
  } else if (taskType === "review") {
    executionMode = "reviewer";
  } else if (taskType === "architecture") {
    executionMode = "architect_like_plan";
  } else if (
    risk === "high" ||
    complexity === "multi_step" ||
    complexity === "ambiguous" ||
    domainSpecificity === "continuation_sensitive"
  ) {
    executionMode = "executor_then_reviewer";
  }

  const contextBudget = contextBudgetFor(executionMode);
  const steps: ClawBoundDecisionStep[] = [
    {
      code: "TASK_TYPE",
      evidence: task.userInput,
      effect: { taskType },
      reason: "Task type inferred from explicit request keywords.",
    },
    {
      code: "COMPLEXITY",
      evidence: task.userInput,
      effect: { complexity },
      reason: "Complexity is inferred from task shape and ambiguity markers.",
    },
    {
      code: "DOMAIN_SPECIFICITY",
      evidence: `continuation_of=${task.continuationOf}, local_context=${task.localContext.length}`,
      effect: { domainSpecificity },
      reason: "Domain specificity reflects continuation state and repo-local signals.",
    },
    {
      code: "RISK",
      evidence: task.userInput,
      effect: { risk },
      reason: "Risk rises for compatibility-sensitive and externally sensitive work.",
    },
    {
      code: "EXECUTION_MODE",
      evidence: `risk=${risk}, complexity=${complexity}, domain=${domainSpecificity}`,
      effect: { executionMode },
      reason: "Execution mode follows the deterministic MVP routing matrix.",
    },
    {
      code: "CONTEXT_BUDGET",
      evidence: `always_on=${contextBudget.alwaysOnMaxTokens}, retrieval_units=${contextBudget.retrievalMaxUnits}, retrieval_tokens=${contextBudget.retrievalMaxTokens}`,
      effect: { contextBudget },
      reason: "Context budget is determined by execution mode and sparse-context limits.",
    },
  ];

  return {
    taskType,
    complexity,
    risk,
    domainSpecificity,
    executionMode,
    contextBudget,
    decisionTrace: {
      traceId: task.traceId,
      summary: `task_type=${taskType};complexity=${complexity};domain=${domainSpecificity};risk=${risk};mode=${executionMode}`,
      steps,
    },
  } satisfies ClawBoundRoutePlan;
}

function contextBudgetFor(executionMode: string): ClawBoundContextBudget {
  if (executionMode === "answer") {
    return { alwaysOnMaxTokens: 180, retrievalMaxUnits: 0, retrievalMaxTokens: 0 };
  }
  if (executionMode === "executor") {
    return { alwaysOnMaxTokens: 200, retrievalMaxUnits: 2, retrievalMaxTokens: 120 };
  }
  if (executionMode === "executor_then_reviewer") {
    return { alwaysOnMaxTokens: 220, retrievalMaxUnits: 3, retrievalMaxTokens: 180 };
  }
  if (executionMode === "reviewer") {
    return { alwaysOnMaxTokens: 200, retrievalMaxUnits: 2, retrievalMaxTokens: 140 };
  }
  return { alwaysOnMaxTokens: 220, retrievalMaxUnits: 2, retrievalMaxTokens: 160 };
}

function buildRetrievalPlan(
  task: {
    traceId: string;
    userInput: string;
    continuationOf: string | null;
    intentHints: string[];
    localContext: ClawBoundLocalContextItem[];
  },
  routePlan: ClawBoundRoutePlan,
): ClawBoundRetrievalPlan {
  const text = task.userInput.toLowerCase();
  let loadLaws = false;
  let loadProjectNotes = false;
  let loadEpisodeSummary = false;
  let lawQueries: string[] = [];
  let projectQueries: string[] = [];
  let episodeQueries: string[] = [];
  const steps: ClawBoundDecisionStep[] = [];

  if (
    routePlan.executionMode === "reviewer" ||
    routePlan.executionMode === "architect_like_plan" ||
    routePlan.risk === "high" ||
    matchesAny(text, ["public api", "external behavior", "compatibility", "regression", "preserve", "review"])
  ) {
    loadLaws = true;
    lawQueries = uniqueTokens(task.userInput, task.intentHints.join(" "));
    steps.push({
      code: "LAW_GATE",
      evidence: "Task risk, review mode, or explicit compatibility constraint triggered law retrieval.",
      effect: { loadLaws: true },
      reason: "Laws help enforce bounded behavior and review priorities.",
    });
  }

  if (routePlan.domainSpecificity === "repo_specific" && routePlan.executionMode !== "answer") {
    loadProjectNotes = true;
    projectQueries = uniqueTokens(
      task.userInput,
      task.localContext.map((item) => item.content).join(" "),
    );
    steps.push({
      code: "PROJECT_NOTE_GATE",
      evidence: "Repo-specific task or local context is present.",
      effect: { loadProjectNotes: true },
      reason: "Project notes may capture sensitive local conventions.",
    });
  }

  if (routePlan.domainSpecificity === "continuation_sensitive") {
    loadEpisodeSummary = true;
    episodeQueries = uniqueTokens(task.userInput, task.continuationOf ?? "");
    steps.push({
      code: "EPISODE_GATE",
      evidence: "Task is a continuation or explicitly references a prior run.",
      effect: { loadEpisodeSummary: true },
      reason: "Recent episode summaries can preserve bounded continuity.",
    });
  }

  const noLoad = !(loadLaws || loadProjectNotes || loadEpisodeSummary);
  if (noLoad) {
    steps.push({
      code: "NO_LOAD",
      evidence: "No deterministic gate opened for this task.",
      effect: { noLoad: true },
      reason: "Sparse context is the default outcome.",
    });
  }

  return {
    loadLaws,
    loadProjectNotes,
    loadEpisodeSummary,
    lawQueries,
    projectQueries,
    episodeQueries,
    maxUnits: noLoad ? 0 : routePlan.contextBudget.retrievalMaxUnits,
    maxTokens: noLoad ? 0 : routePlan.contextBudget.retrievalMaxTokens,
    noLoad,
    gateTrace: {
      traceId: task.traceId,
      summary: steps.length > 0 ? steps.map((step) => step.code).join("; ") : "NO_DECISION",
      steps,
    },
  };
}

async function retrieveUnits(params: {
  lawsDir: string;
  projectNotesDir: string;
  episodeSummariesDir: string;
  task: {
    userInput: string;
    continuationOf: string | null;
  };
  retrievalPlan: ClawBoundRetrievalPlan;
}) {
  if (params.retrievalPlan.noLoad) {
    return [] satisfies ClawBoundRetrievalUnit[];
  }

  const candidates: Array<{ score: number; reason: string; unit: ClawBoundRetrievalUnit }> = [];
  if (params.retrievalPlan.loadLaws) {
    const units = await loadSeedUnits(params.lawsDir);
    candidates.push(...scoreRetrievalUnits(units, params.retrievalPlan.lawQueries));
  }
  if (params.retrievalPlan.loadProjectNotes) {
    const units = await loadSeedUnits(params.projectNotesDir);
    candidates.push(...scoreRetrievalUnits(units, params.retrievalPlan.projectQueries));
  }
  if (params.retrievalPlan.loadEpisodeSummary) {
    const units = await loadSeedUnits(params.episodeSummariesDir);
    candidates.push(
      ...scoreEpisodeSummaries(units, params.task.continuationOf, params.retrievalPlan.episodeQueries),
    );
  }

  const selected: ClawBoundRetrievalUnit[] = [];
  const seen = new Set<string>();
  let totalTokens = 0;
  for (const candidate of [...candidates].sort((left, right) => right.score - left.score)) {
    if (seen.has(candidate.unit.id) || selected.length >= params.retrievalPlan.maxUnits) {
      continue;
    }
    if (totalTokens + candidate.unit.tokenEstimate > params.retrievalPlan.maxTokens) {
      continue;
    }
    seen.add(candidate.unit.id);
    totalTokens += candidate.unit.tokenEstimate;
    selected.push(candidate.unit);
  }
  return selected;
}

async function loadSeedUnits(dirPath: string) {
  const entries = await fs.readdir(dirPath);
  const units = await Promise.all(
    entries
      .filter((entry) => entry.endsWith(".json"))
      .sort()
      .map(async (entry) => {
        const payload = JSON.parse(await fs.readFile(path.join(dirPath, entry), "utf8")) as RawSeedUnit;
        return normalizeUnit(payload);
      }),
  );
  return units.filter((unit) => unit.enabled);
}

function normalizeUnit(unit: RawSeedUnit): ClawBoundRetrievalUnit {
  const originalContent = unit.content;
  let content = originalContent;
  const diagnostics: string[] = [];

  const lines = content.split("\n");
  if (lines.length > MAX_SNIPPET_LINES) {
    content = lines.slice(0, MAX_SNIPPET_LINES).join("\n").trim();
    diagnostics.push("content_truncated", "line_limit");
  }
  if (content.length > MAX_SNIPPET_CHARS) {
    content = `${content.slice(0, MAX_SNIPPET_CHARS - 3).trimEnd()}...`;
    diagnostics.push("content_truncated", "char_limit");
  }
  if (estimateTokensFromText(content) > MAX_SNIPPET_TOKENS) {
    const words = content.split(/\s+/).slice(0, MAX_SNIPPET_TOKENS).join(" ").trim();
    content = words.endsWith("...") ? words : `${words.replace(/\.*$/, "")}...`;
    diagnostics.push("content_truncated", "token_limit");
  }

  const snippetDiagnostics = [...new Set(diagnostics)];
  const snippetStatus = snippetDiagnostics.length > 0 ? "truncated" : "ok";
  return {
    id: unit.id,
    type: unit.type,
    scope: unit.scope,
    applicability: unit.applicability,
    content: content || `${originalContent.slice(0, MAX_SNIPPET_CHARS - 3).trimEnd()}...`,
    confidence: unit.confidence,
    priority: unit.priority,
    tokenEstimate: Math.min(
      snippetStatus === "ok" ? unit.token_estimate : estimateTokensFromText(content),
      MAX_SNIPPET_TOKENS,
    ),
    sourceRef: unit.source_ref,
    lastValidatedAt: unit.last_validated_at,
    tags: unit.tags ?? [],
    enabled: unit.enabled,
    snippetStatus,
    snippetDiagnostics,
    sourceTokenEstimate: unit.token_estimate,
  };
}

function scoreRetrievalUnits(units: ClawBoundRetrievalUnit[], queries: string[]) {
  const queryTokens = new Set(tokenize(queries.join(" ")));
  const scored: Array<{ score: number; reason: string; unit: ClawBoundRetrievalUnit }> = [];
  for (const unit of units) {
    const haystack = new Set(tokenize(unit.scope, unit.applicability, unit.content, unit.tags.join(" ")));
    const overlap = queryTokens.size > 0 ? sizeOfIntersection(queryTokens, haystack) : 0;
    if (queryTokens.size > 0 && overlap === 0) {
      continue;
    }
    const baseReason = overlap > 0 ? `lexical_overlap:${overlap}` : "fallback_priority";
    scored.push({
      score: overlap * 10 + unit.priority + unit.confidence,
      reason: withSnippetStatus(baseReason, unit),
      unit,
    });
  }
  return scored;
}

function scoreEpisodeSummaries(
  units: ClawBoundRetrievalUnit[],
  continuationOf: string | null,
  queries: string[],
) {
  const queryTokens = new Set(tokenize(queries.join(" ")));
  const scored: Array<{ score: number; reason: string; unit: ClawBoundRetrievalUnit }> = [];
  for (const unit of units) {
    let score = 0;
    let reason = "fallback_priority";
    if (continuationOf && unit.sourceRef === `run:${continuationOf}`) {
      score += 100;
      reason = "continuation_match";
    }
    const haystack = new Set(tokenize(unit.scope, unit.applicability, unit.content, unit.tags.join(" ")));
    const overlap = sizeOfIntersection(queryTokens, haystack);
    score += overlap * 10;
    if (overlap > 0 && reason !== "continuation_match") {
      reason = `lexical_overlap:${overlap}`;
    }
    if (score <= 0 && continuationOf) {
      continue;
    }
    if (score <= 0) {
      score = unit.priority + unit.confidence;
    }
    scored.push({
      score,
      reason: withSnippetStatus(reason, unit),
      unit,
    });
  }
  return scored;
}

function withSnippetStatus(reason: string, unit: ClawBoundRetrievalUnit) {
  if (unit.snippetStatus !== "truncated") {
    return reason;
  }
  return `${reason}|snippet_truncated:${unit.snippetDiagnostics.join(",") || "content_truncated"}`;
}

function buildToolProfile(
  executionMode: string,
  sessionKey: string | undefined,
  userInput: string,
  candidateToolsObserved: string[],
): ClawBoundToolProfile {
  const focusedTestNotes = buildFocusedTestDisciplineNotes(userInput);
  if (executionMode === "answer") {
    return {
      profile: "answer-minimal",
      allowedTools: ["read_file"],
      deniedTools: ["edit_file", "write_file", "delete_file", "run_command", "message"],
      notes: ["Answer mode should stay lightweight and avoid host messaging actions."],
      candidateToolsObserved,
      requiresReview: false,
    };
  }
  if (executionMode === "reviewer") {
    if (
      !isSubagentSessionKey(sessionKey) &&
      matchesAny(userInput, ["delegate", "subagent"]) &&
      candidateToolsObserved.some((tool) => tool.trim().toLowerCase() === "sessions_spawn")
    ) {
      return {
        profile: "review-delegate-bounded",
        allowedTools: ["read_file", "run_tests", "sessions_spawn"],
        deniedTools: ["edit_file", "write_file", "delete_file"],
        notes: [
          "Delegated review may spawn one narrow subagent, but should not mutate files.",
          ...focusedTestNotes,
        ],
        candidateToolsObserved,
        requiresReview: true,
      };
    }
    return {
      profile: "review-read-only",
      allowedTools: ["read_file", "run_tests"],
      deniedTools: ["edit_file", "write_file", "delete_file"],
      notes: ["Review mode should inspect, not mutate production state.", ...focusedTestNotes],
      candidateToolsObserved,
      requiresReview: true,
    };
  }
  if (executionMode === "architect_like_plan") {
    return {
      profile: "plan-bounded",
      allowedTools: ["read_file", "run_tests"],
      deniedTools: ["edit_file", "write_file", "delete_file"],
      notes: [
        "Planning mode should stay bounded and avoid mutating the workspace.",
        ...focusedTestNotes,
      ],
      candidateToolsObserved,
      requiresReview: true,
    };
  }
  return {
    profile: "executor-bounded",
    allowedTools: ["read_file", "edit_file", "run_tests", "run_command"],
    deniedTools: ["delete_file"],
    notes: [
      "Mutating tools may still require host enforcement and review.",
      ...focusedTestNotes,
    ],
    candidateToolsObserved,
    requiresReview: executionMode === "executor_then_reviewer",
  };
}

function buildPromptPackage(params: {
  runId: string;
  traceId: string;
  kernel: KernelAsset;
  task: {
    userInput: string;
    localContext: ClawBoundLocalContextItem[];
  };
  routePlan: ClawBoundRoutePlan;
  retrievalPlan: ClawBoundRetrievalPlan;
  retrievedUnits: ClawBoundRetrievalUnit[];
  toolProfile: ClawBoundToolProfile;
}): ClawBoundPromptPackage {
  const modeInstruction = modeInstructionFor(params.routePlan.executionMode);
  const segmentBreakdown = {
    kernel: params.kernel.tokenEstimate,
    modeInstruction: estimateTokensFromText(modeInstruction),
    taskBrief: estimateTokensFromText(params.task.userInput),
    localContext: estimateTokensFromItems(params.task.localContext.map((item) => item.content)),
    toolContract: estimateTokensFromItems([
      ...params.toolProfile.allowedTools,
      ...params.toolProfile.deniedTools,
      ...params.toolProfile.notes,
    ]),
    retrieval: params.retrievedUnits.reduce((total, unit) => total + unit.tokenEstimate, 0),
  };
  const alwaysOnTokens =
    segmentBreakdown.kernel +
    segmentBreakdown.modeInstruction +
    segmentBreakdown.taskBrief +
    segmentBreakdown.localContext +
    segmentBreakdown.toolContract;
  const retrievedTokens = segmentBreakdown.retrieval;
  return {
    runId: params.runId,
    traceId: params.traceId,
    role: roleForMode(params.routePlan.executionMode),
    kernelVersion: params.kernel.version,
    kernelText: params.kernel.content,
    assemblyOrder: ["kernel", "mode_instruction", "task_brief", "local_context", "retrieved_units", "tool_contract"],
    modeInstruction,
    taskBrief: params.task.userInput,
    localContext: params.task.localContext,
    retrievedUnits: params.retrievedUnits,
    toolContract: {
      allowedTools: params.toolProfile.allowedTools,
      deniedTools: params.toolProfile.deniedTools,
      notes: params.toolProfile.notes,
    },
    assemblyStats: {
      alwaysOnTokens,
      retrievedTokens,
      retrievalUnitsCount: params.retrievedUnits.length,
      segmentBreakdown,
      budget: {
        alwaysOnMaxTokens: params.routePlan.contextBudget.alwaysOnMaxTokens,
        retrievalMaxUnits: params.retrievalPlan.maxUnits,
        retrievalMaxTokens: params.retrievalPlan.maxTokens,
        alwaysOnUtilization: ratio(alwaysOnTokens, params.routePlan.contextBudget.alwaysOnMaxTokens),
        retrievalTokenUtilization: ratio(retrievedTokens, params.retrievalPlan.maxTokens),
        retrievalUnitUtilization: ratio(params.retrievedUnits.length, params.retrievalPlan.maxUnits),
      },
      totalEstimatedTokens: alwaysOnTokens + retrievedTokens,
      noLoad: params.retrievalPlan.noLoad,
    },
    noLoad: params.retrievalPlan.noLoad,
  };
}

function modeInstructionFor(executionMode: string) {
  if (executionMode === "answer") {
    return "Answer directly and avoid unnecessary context loading.";
  }
  if (executionMode === "reviewer") {
    return "Inspect for scope drift, compatibility risks, and regressions without taking over implementation.";
  }
  if (executionMode === "architect_like_plan") {
    return "Produce a scoped plan with explicit tradeoffs and constraints.";
  }
  if (executionMode === "executor_then_reviewer") {
    return "Execute cautiously, then inspect for regression and scope drift.";
  }
  return "Make a bounded change and preserve external behavior unless told otherwise.";
}

function roleForMode(executionMode: string) {
  if (executionMode === "reviewer") {
    return "clawbound-reviewer";
  }
  if (executionMode === "architect_like_plan") {
    return "clawbound-architect";
  }
  if (executionMode === "answer") {
    return "clawbound-answer";
  }
  return "clawbound-executor";
}

async function compareToOracle(
  fixturePath: string,
  userInput: string,
  routePlan: ClawBoundRoutePlan,
  retrievalPlan: ClawBoundRetrievalPlan,
  retrievedUnits: ClawBoundRetrievalUnit[],
  toolProfile: ClawBoundToolProfile,
): Promise<ClawBoundParityResult> {
  const fixtures = JSON.parse(await fs.readFile(fixturePath, "utf8")) as OracleFixture[];
  const fixture =
    fixtures.find((candidate) => candidate.userInput.trim().toLowerCase() === userInput.trim().toLowerCase()) ??
    null;
  if (!fixture) {
    return {
      fixtureId: null,
      available: false,
      checks: { route: false, noLoad: false, retrievalUnits: false, toolProfile: false },
    };
  }
  const actual = {
    taskType: routePlan.taskType,
    executionMode: routePlan.executionMode,
    noLoad: retrievalPlan.noLoad,
    retrievedUnitIds: retrievedUnits.map((unit) => unit.id),
    toolProfile: toolProfile.profile,
    allowedTools: toolProfile.allowedTools,
    deniedTools: toolProfile.deniedTools,
  };
  return {
    fixtureId: fixture.id,
    available: true,
    checks: {
      route:
        fixture.expected.taskType === actual.taskType &&
        fixture.expected.executionMode === actual.executionMode,
      noLoad: fixture.expected.noLoad === actual.noLoad,
      retrievalUnits:
        JSON.stringify(fixture.expected.retrievedUnitIds) === JSON.stringify(actual.retrievedUnitIds),
      toolProfile:
        fixture.expected.toolProfile === actual.toolProfile &&
        JSON.stringify(fixture.expected.allowedTools) === JSON.stringify(actual.allowedTools) &&
        JSON.stringify(fixture.expected.deniedTools) === JSON.stringify(actual.deniedTools),
    },
    expected: fixture.expected,
    actual,
  };
}

async function persistSnapshot(rootDir: string, key: string, snapshot: ClawBoundShadowSnapshot) {
  const runsDir = path.join(rootDir, "runs");
  await fs.mkdir(runsDir, { recursive: true });
  const fileName = `${key.replace(/[^a-zA-Z0-9._-]+/g, "_")}.json`;
  const persistedPath = path.join(runsDir, fileName);
  await fs.writeFile(persistedPath, `${JSON.stringify(snapshot, null, 2)}\n`, "utf8");
  return persistedPath;
}

function canonicalizeCandidateTools(candidateTools: string[]) {
  return [...new Set(candidateTools.map((tool) => canonicalToolName(tool)).filter(Boolean))];
}

function canonicalToolName(toolName: string) {
  const normalized = toolName.trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  if (normalized === "read") {
    return "read_file";
  }
  if (normalized === "edit") {
    return "edit_file";
  }
  if (normalized === "write") {
    return "write_file";
  }
  if (normalized === "exec" || normalized === "bash") {
    return "run_command";
  }
  return normalized;
}

function buildFocusedTestDisciplineNotes(userInput: string) {
  const testFiles = extractExplicitTestFiles(userInput);
  if (testFiles.length === 0) {
    return [];
  }
  return [
    `For named test files (${testFiles.join(", ")}), use the narrowest verification command available. Prefer an existing focused script or direct runner such as node --test <file>, and avoid npm test -- <file> unless a test script is known to exist.`,
  ];
}

function extractExplicitTestFiles(text: string) {
  const matches = text.match(/[\w./-]+\.(?:test|spec)\.[cm]?[jt]sx?/gi) ?? [];
  return [...new Set(matches)];
}

function matchesAny(text: string, keywords: string[]) {
  return keywords.some((keyword) => text.includes(keyword));
}

function isVerificationLikeTask(text: string) {
  const verificationCue = matchesAny(text, [
    "verify",
    "verification",
    "confirm",
    "validate",
    "check whether",
    "whether",
    "what changed",
    "now passes",
    "still passes",
    "passes",
  ]);
  if (!verificationCue) {
    return false;
  }

  return matchesAny(text, [
    "do not make further edits",
    "do not edit",
    "without editing",
    "no further edits",
    "in two bullets",
    "report",
    "confirm",
  ]);
}

function sizeOfIntersection(left: Set<string>, right: Set<string>) {
  let count = 0;
  for (const value of left) {
    if (right.has(value)) {
      count += 1;
    }
  }
  return count;
}

function ratio(numerator: number, denominator: number) {
  if (denominator <= 0) {
    return 0;
  }
  return numerator / denominator;
}
