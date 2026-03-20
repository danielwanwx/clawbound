import type {
  ClawBoundKernelAsset,
  ClawBoundLocalContextItem,
  ClawBoundParityResult,
  ClawBoundPromptPackage,
  ClawBoundRetrievalPlan,
  ClawBoundRetrievalUnit,
  ClawBoundRoutePlan,
  ClawBoundShadowEvent,
  ClawBoundToolProfile,
} from "../runtime.js";
import { ClawBoundRuntime } from "../runtime.js";

export type ClawBoundContextEngineSession = {
  hostRunId?: string;
  sessionId: string;
  sessionKey?: string;
  conversationId?: string;
  sourceHost?: string;
  emitToHostEvents?: boolean;
};

export type ClawBoundContextEngineTask = {
  userInput: string;
  continuationOf?: string | null;
  intentHints?: string[];
  localContext?: ClawBoundLocalContextItem[];
};

export type ClawBoundContextInputEvent =
  | {
      type: "local_context";
      item: ClawBoundLocalContextItem;
    }
  | {
      type: "host_event";
      eventType: string;
      payload: Record<string, unknown>;
    };

export type ClawBoundContextSessionState = {
  runId: string;
  traceId: string;
  hostRunId: string | null;
  sessionId: string;
  sessionKey?: string;
  conversationId: string;
  sourceHost: string;
  emitToHostEvents: boolean;
  kernel: ClawBoundKernelAsset;
  runtimeDefaults: {
    noLoadAllowed: boolean;
    maxSubagentSnippets: number;
    maxSubagentLocalContextItems: number;
  };
  userInput: string;
  continuationOf: string | null;
  intentHints: string[];
  localContext: ClawBoundLocalContextItem[];
  rawEvents: ClawBoundContextInputEvent[];
  lifecycleEvents: ClawBoundShadowEvent[];
  activeContext: {
    localContext: ClawBoundLocalContextItem[];
    retrievedUnits: ClawBoundRetrievalUnit[];
  };
  routePlan?: ClawBoundRoutePlan;
  retrievalPlan?: ClawBoundRetrievalPlan;
  promptPackage?: ClawBoundPromptPackage;
  toolProfile?: ClawBoundToolProfile;
  parity?: ClawBoundParityResult;
  persistedPath?: string;
};

export type ClawBoundContextEventBatch = {
  state: ClawBoundContextSessionState;
  events: ClawBoundContextInputEvent[];
};

export type ClawBoundContextRunPlan = {
  state: ClawBoundContextSessionState;
  candidateTools?: string[];
};

export type ClawBoundContextAssembleResult = ClawBoundPlanRunResult & {
  state: ClawBoundContextSessionState;
  promptText: string;
};

export type ClawBoundContextCompactResult = {
  traceSummary: {
    executionMode: string;
    eventTypes: string[];
    eventCount: number;
    totalEstimatedTokens: number;
  };
  retrievalSummary: {
    noLoad: boolean;
    selectedUnitIds: string[];
    totalRetrievedTokens: number;
  };
  activeContext: {
    localContextRefs: string[];
    localContextCount: number;
    retrievedUnitIds: string[];
    retrievedUnitCount: number;
    promptPackageTokens: number;
  };
  omitted: {
    transcriptIncluded: false;
    reasons: string[];
  };
  boundedness: {
    stayedBounded: boolean;
    reasons: string[];
    budget: {
      alwaysOnMaxTokens: number;
      retrievalMaxUnits: number;
      retrievalMaxTokens: number;
    };
    utilization: {
      alwaysOn: number;
      retrievalTokens: number;
      retrievalUnits: number;
    };
  };
  transcriptIncluded: false;
};

export type ClawBoundSubagentTask = {
  userInput: string;
};

export type ClawBoundSubagentContext = {
  text: string;
  inheritsFullTranscript: false;
  localContext: ClawBoundLocalContextItem[];
  retrievedUnits: ClawBoundRetrievalUnit[];
  executionMode: string;
  noLoad: boolean;
};

export interface ClawBoundContextEngine {
  bootstrap(
    session: ClawBoundContextEngineSession,
    task: ClawBoundContextEngineTask,
  ): Promise<ClawBoundContextSessionState>;
  ingest(eventBatch: ClawBoundContextEventBatch): Promise<ClawBoundContextSessionState>;
  assemble(runPlan: ClawBoundContextRunPlan): Promise<ClawBoundContextAssembleResult>;
  compact(sessionState: ClawBoundContextSessionState): Promise<ClawBoundContextCompactResult>;
  prepareSubagentContext(
    parentState: ClawBoundContextSessionState,
    subtask: ClawBoundSubagentTask,
  ): Promise<ClawBoundSubagentContext>;
}

export type SparseClawBoundContextEngineOptions = {
  rootDir: string;
  idFactory?: () => string;
};

const DEFAULT_SOURCE_HOST = "clawbound-context/openclaw";

export class SparseClawBoundContextEngine implements ClawBoundContextEngine {
  private readonly runtime: ClawBoundRuntime;

  constructor(options: SparseClawBoundContextEngineOptions) {
    this.runtime = new ClawBoundRuntime({
      shadowRootDir: options.rootDir,
      idFactory: options.idFactory,
    });
  }

  async bootstrap(
    session: ClawBoundContextEngineSession,
    task: ClawBoundContextEngineTask,
  ): Promise<ClawBoundContextSessionState> {
    const kernel = await this.runtime.loadKernelAsset();
    const localContext = [...(task.localContext ?? [])];
    return {
      runId: this.runtime.createRunId(),
      traceId: this.runtime.createTraceId(),
      hostRunId: session.hostRunId?.trim() || null,
      sessionId: session.sessionId,
      sessionKey: session.sessionKey,
      conversationId: session.conversationId ?? session.sessionKey ?? session.sessionId,
      sourceHost: session.sourceHost ?? DEFAULT_SOURCE_HOST,
      emitToHostEvents: session.emitToHostEvents === true,
      kernel,
      runtimeDefaults: {
        noLoadAllowed: true,
        maxSubagentSnippets: 2,
        maxSubagentLocalContextItems: 1,
      },
      userInput: task.userInput,
      continuationOf: task.continuationOf ?? null,
      intentHints: [...(task.intentHints ?? [])],
      localContext,
      rawEvents: [],
      lifecycleEvents: [],
      activeContext: {
        localContext,
        retrievedUnits: [],
      },
    };
  }

  async ingest(eventBatch: ClawBoundContextEventBatch): Promise<ClawBoundContextSessionState> {
    const nextLocalContext = [...eventBatch.state.localContext];
    for (const event of eventBatch.events) {
      if (event.type !== "local_context") {
        continue;
      }
      nextLocalContext.push(event.item);
    }
    return {
      ...eventBatch.state,
      localContext: nextLocalContext,
      rawEvents: [...eventBatch.state.rawEvents, ...eventBatch.events],
      activeContext: {
        ...eventBatch.state.activeContext,
        localContext: nextLocalContext,
      },
    };
  }

  async assemble(runPlan: ClawBoundContextRunPlan): Promise<ClawBoundContextAssembleResult> {
    const result = await this.runtime.planRun({
      runId: runPlan.state.runId,
      traceId: runPlan.state.traceId,
      hostRunId: runPlan.state.hostRunId ?? undefined,
      sessionId: runPlan.state.sessionId,
      sessionKey: runPlan.state.sessionKey,
      conversationId: runPlan.state.conversationId,
      userInput: runPlan.state.userInput,
      continuationOf: runPlan.state.continuationOf ?? undefined,
      intentHints: runPlan.state.intentHints,
      localContext: runPlan.state.localContext,
      candidateTools: runPlan.candidateTools,
      sourceHost: runPlan.state.sourceHost,
      emitToHostEvents: runPlan.state.emitToHostEvents,
    });

    const state: ClawBoundContextSessionState = {
      ...runPlan.state,
      activeContext: {
        localContext: result.promptPackage.localContext,
        retrievedUnits: result.promptPackage.retrievedUnits,
      },
      routePlan: result.routePlan,
      retrievalPlan: result.retrievalPlan,
      promptPackage: result.promptPackage,
      toolProfile: result.toolProfile,
      parity: result.parity,
      lifecycleEvents: result.events,
      persistedPath: result.persistedPath,
    };

    return {
      ...result,
      state,
      promptText: renderClawBoundPromptPackage(result.promptPackage),
    };
  }

  async compact(sessionState: ClawBoundContextSessionState): Promise<ClawBoundContextCompactResult> {
    const retrievedUnits = sessionState.activeContext.retrievedUnits;
    const promptBudget = sessionState.promptPackage?.assemblyStats.budget;
    const omittedReasons = [
      "full_transcript_not_loaded",
      "host_workspace_inflation_not_loaded",
      "host_skills_prompt_not_loaded",
    ];
    const boundednessReasons = [
      "no_host_workspace_inflation",
      "no_host_skills_prompt",
    ];
    if ((sessionState.retrievalPlan?.noLoad ?? true) === true) {
      boundednessReasons.push("no_load_selected");
    }
    if (promptBudget && promptBudget.alwaysOnUtilization <= 1) {
      boundednessReasons.push("always_on_within_budget");
    }
    if (promptBudget && promptBudget.retrievalTokenUtilization <= 1) {
      boundednessReasons.push("retrieval_within_budget");
    }

    return {
      traceSummary: {
        executionMode: sessionState.routePlan?.executionMode ?? "unplanned",
        eventTypes: sessionState.lifecycleEvents.map((event) => event.eventType),
        eventCount: sessionState.lifecycleEvents.length,
        totalEstimatedTokens: sessionState.promptPackage?.assemblyStats.totalEstimatedTokens ?? 0,
      },
      retrievalSummary: {
        noLoad: sessionState.retrievalPlan?.noLoad ?? true,
        selectedUnitIds: retrievedUnits.map((unit) => unit.id),
        totalRetrievedTokens: retrievedUnits.reduce((total, unit) => total + unit.tokenEstimate, 0),
      },
      activeContext: {
        localContextRefs: sessionState.activeContext.localContext.map((item) => item.ref),
        localContextCount: sessionState.activeContext.localContext.length,
        retrievedUnitIds: retrievedUnits.map((unit) => unit.id),
        retrievedUnitCount: retrievedUnits.length,
        promptPackageTokens: sessionState.promptPackage?.assemblyStats.totalEstimatedTokens ?? 0,
      },
      omitted: {
        transcriptIncluded: false,
        reasons: omittedReasons,
      },
      boundedness: {
        stayedBounded: true,
        reasons: boundednessReasons,
        budget: {
          alwaysOnMaxTokens: sessionState.routePlan?.contextBudget.alwaysOnMaxTokens ?? 0,
          retrievalMaxUnits: sessionState.retrievalPlan?.maxUnits ?? 0,
          retrievalMaxTokens: sessionState.retrievalPlan?.maxTokens ?? 0,
        },
        utilization: {
          alwaysOn: promptBudget?.alwaysOnUtilization ?? 0,
          retrievalTokens: promptBudget?.retrievalTokenUtilization ?? 0,
          retrievalUnits: promptBudget?.retrievalUnitUtilization ?? 0,
        },
      },
      transcriptIncluded: false,
    };
  }

  async prepareSubagentContext(
    parentState: ClawBoundContextSessionState,
    subtask: ClawBoundSubagentTask,
  ): Promise<ClawBoundSubagentContext> {
    const localContext = parentState.activeContext.localContext.slice(
      0,
      parentState.runtimeDefaults.maxSubagentLocalContextItems,
    );
    const retrievedUnits = parentState.activeContext.retrievedUnits.slice(
      0,
      parentState.runtimeDefaults.maxSubagentSnippets,
    );
    const executionMode = parentState.routePlan?.executionMode ?? "answer";

    return {
      text: renderClawBoundSubagentHandoff({
        executionMode,
        parentTraceId: parentState.traceId,
        userInput: subtask.userInput,
        localContext,
        retrievedUnits,
      }),
      inheritsFullTranscript: false,
      localContext,
      retrievedUnits,
      executionMode,
      noLoad: retrievedUnits.length === 0,
    };
  }
}

export function renderClawBoundPromptPackage(promptPackage: ClawBoundPromptPackage) {
  const localContextLines =
    promptPackage.localContext.length > 0
      ? promptPackage.localContext.map((item) => `- [${item.kind}] ${item.ref}: ${item.content}`)
      : ["- none"];
  const retrievalLines =
    promptPackage.retrievedUnits.length > 0
      ? promptPackage.retrievedUnits.map(
          (unit) => `- [${unit.id}] ${unit.scope} (${unit.sourceRef})\n${unit.content}`,
        )
      : [`- none (no_load=${String(promptPackage.noLoad)})`];

  return [
    "## ClawBound Kernel",
    promptPackage.kernelText,
    "## ClawBound Mode",
    promptPackage.modeInstruction,
    "## Task Brief",
    promptPackage.taskBrief,
    "## Explicit Local Context",
    ...localContextLines,
    "## Retrieved Snippets",
    ...retrievalLines,
    "## Tool Contract",
    `Allowed: ${promptPackage.toolContract.allowedTools.join(", ") || "none"}`,
    `Denied: ${promptPackage.toolContract.deniedTools.join(", ") || "none"}`,
    `Notes: ${promptPackage.toolContract.notes.join(" ") || "none"}`,
  ]
    .join("\n")
    .trim();
}

function renderClawBoundSubagentHandoff(params: {
  executionMode: string;
  parentTraceId: string;
  userInput: string;
  localContext: ClawBoundLocalContextItem[];
  retrievedUnits: ClawBoundRetrievalUnit[];
}) {
  const localContextLines =
    params.localContext.length > 0
      ? params.localContext.map((item) => `- [${item.kind}] ${item.ref}: ${item.content}`)
      : ["- none"];
  const retrievalLines =
    params.retrievedUnits.length > 0
      ? params.retrievedUnits.map((unit) => `- [${unit.id}] ${unit.content}`)
      : ["- none"];

  return [
    "## ClawBound Subagent Handoff",
    `Parent trace: ${params.parentTraceId}`,
    `Execution mode: ${params.executionMode}`,
    `Subtask: ${params.userInput}`,
    "## Local Context",
    ...localContextLines,
    "## Retrieved Snippets",
    ...retrievalLines,
    "Do not inherit the parent transcript wholesale. Use only the bounded handoff above.",
  ]
    .join("\n")
    .trim();
}
