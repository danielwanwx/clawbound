import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { resolveOpenClawAgentDir } from "../agents/agent-paths.js";
import {
  classifyClawBoundBenchmarkLiveRun,
  prepareClawBoundBenchmarkLiveEnvironment,
  readClawBoundBenchmarkSessionMessages,
  resolveClawBoundBenchmarkConfiguredLiveModelBinding,
  summarizeClawBoundBenchmarkLiveRuns,
  summarizeClawBoundBenchmarkLiveModelBinding,
  withClawBoundBenchmarkHarnessTimeout,
  type ClawBoundBenchmarkLiveModelBinding,
  type ClawBoundBenchmarkLiveFailureClassification,
  type ClawBoundBenchmarkLiveRunClassification,
  type ClawBoundBenchmarkLiveValidationOutcome,
} from "./benchmarks/live-harness.js";
import type { OpenClawConfig } from "../config/config.js";
import { isTruthyEnvValue } from "../infra/env.js";
import { extractToolCallNames } from "../utils/transcript-tools.js";

type ModelLike = { api: string; provider: string; id: string };
type ContextLike = {
  systemPrompt?: string;
  messages: Array<{
    role: string;
    content: unknown;
  }>;
  tools?: Array<{ name: string }>;
};

type PromptSnapshot = {
  label: string;
  prompt: string;
  systemPrompt: string;
  toolNames: string[];
  systemPromptChars: number;
  containsClawBoundKernel: boolean;
  containsAgentsMd: boolean;
  containsProjectContext: boolean;
  containsRetrievedSnippets: boolean;
};

type RunSummary = {
  label: string;
  task: string;
  attemptCount: number;
  failureClassification: ClawBoundBenchmarkLiveFailureClassification | null;
  validationOutcome: ClawBoundBenchmarkLiveValidationOutcome;
  productValid: boolean;
  externallyBlocked: boolean;
  retryable: boolean;
  failureReason: string | null;
  hostPathUsed: string;
  clawboundContextEngineModeActive: boolean;
  routePlan: Record<string, unknown> | null;
  retrievalPlan: Record<string, unknown> | null;
  noLoad: boolean | null;
  promptInflationSignals: {
    systemPromptChars: number;
    projectContextChars: number;
    injectedWorkspaceFilesCount: number;
    injectedWorkspaceChars: number;
    skillsPromptChars: number;
    observedContainsClawBoundKernel: boolean;
    observedContainsAgentsMd: boolean;
    observedContainsProjectContext: boolean;
    observedContainsRetrievedSnippets: boolean;
    modelBoundaryCallCount: number;
  };
  finalToolProfile: Record<string, unknown>;
  actualToolNamesSeenByModel: string[];
  actualToolCallNames: string[];
  runOutput: string;
  runOutcome: {
    completed: boolean;
    aborted: boolean;
    stopReason: string;
  };
  workspaceMutated: boolean;
  persistedArtifactsLocation: {
    contextEngineRun?: string | null;
    sessionFile: string;
    workspaceDir: string;
  };
};

const LIVE =
  isTruthyEnvValue(process.env.OPENCLAW_LIVE_TEST) ||
  isTruthyEnvValue(process.env.MINIMAX_LIVE_TEST) ||
  isTruthyEnvValue(process.env.LIVE);

const mainAgentDir = resolveOpenClawAgentDir();
const mainAuthStore = path.join(mainAgentDir, "auth-profiles.json");
const describeLive = LIVE && fs.existsSync(mainAuthStore) ? describe : describe.skip;

const observedPrompts = new Map<string, PromptSnapshot[]>();
const MAX_EXTERNAL_RETRIES = 2;

function messageContentToText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((entry) => {
        if (!entry || typeof entry !== "object") {
          return "";
        }
        const text =
          "text" in (entry as { text?: unknown }) && typeof (entry as { text?: unknown }).text === "string"
            ? (entry as { text: string }).text
            : "";
        return text;
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

vi.mock("@mariozechner/pi-ai", async () => {
  const actual = await vi.importActual<typeof import("@mariozechner/pi-ai")>("@mariozechner/pi-ai");
  return {
    ...actual,
    streamSimple: (model: ModelLike, context: ContextLike, options: Record<string, unknown>) => {
      const label = String((globalThis as { __STEWARD_REVIEW_LABEL__?: string }).__STEWARD_REVIEW_LABEL__ ?? "unknown");
      const systemPrompt = typeof context?.systemPrompt === "string" ? context.systemPrompt : "";
      const prompt = messageContentToText(context.messages.at(-1)?.content);
      const toolNames = (context.tools ?? []).map((tool) => tool.name);
      const snapshot: PromptSnapshot = {
        label,
        prompt,
        systemPrompt,
        toolNames,
        systemPromptChars: systemPrompt.length,
        containsClawBoundKernel: systemPrompt.includes("## ClawBound Kernel"),
        containsAgentsMd: systemPrompt.includes("## AGENTS.md"),
        containsProjectContext: systemPrompt.includes("# Project Context"),
        containsRetrievedSnippets: systemPrompt.includes("## Retrieved Snippets"),
      };
      const existing = observedPrompts.get(label) ?? [];
      existing.push(snapshot);
      observedPrompts.set(label, existing);
      return actual.streamSimple(model as never, context as never, options as never);
    },
  };
});

let runEmbeddedPiAgent: typeof import("/Users/javiswan/Projects/clawbound/src/agents/pi-embedded-runner.ts").runEmbeddedPiAgent;
let ensureAgentWorkspace: typeof import("/Users/javiswan/Projects/clawbound/src/agents/workspace.ts").ensureAgentWorkspace;
let tempRoot = "";
let config: OpenClawConfig;
const intendedLiveModel: ClawBoundBenchmarkLiveModelBinding =
  resolveClawBoundBenchmarkConfiguredLiveModelBinding();
let activeLiveModel: ClawBoundBenchmarkLiveModelBinding = { ...intendedLiveModel };
let actualLiveModel: ClawBoundBenchmarkLiveModelBinding | null = null;
const reportPath = path.join(os.tmpdir(), "clawbound-native-review-live-report.json");

const immediateEnqueue = async <T>(task: () => Promise<T>) => task();

async function seedWorkspace(workspaceDir: string) {
  await ensureAgentWorkspace({
    dir: workspaceDir,
    ensureBootstrapFiles: true,
  });

  await fsp.writeFile(
    path.join(workspaceDir, "parser.js"),
    [
      "export function parseCsvLine(input) {",
      '  return input.split(",");',
      "}",
      "",
    ].join("\n"),
    "utf8",
  );
  await fsp.writeFile(
    path.join(workspaceDir, "parser.test.js"),
    [
      'import test from "node:test";',
      'import assert from "node:assert/strict";',
      'import { parseCsvLine } from "./parser.js";',
      "",
      'test("parseCsvLine trims whitespace without changing public API", () => {',
      '  assert.deepEqual(parseCsvLine("alpha, beta , gamma "), ["alpha", "beta", "gamma"]);',
      "});",
      "",
    ].join("\n"),
    "utf8",
  );
  await fsp.writeFile(
    path.join(workspaceDir, "AGENTS.md"),
    "# AGENTS.md\n- Review parser behavior conservatively.\n- Do not edit files during review tasks.\n",
    "utf8",
  );
  await fsp.writeFile(
    path.join(workspaceDir, "TOOLS.md"),
    "# TOOLS.md\n- Read code before judging it.\n- If needed, run the focused parser test for evidence.\n",
    "utf8",
  );
}

async function copyMainAuthStore(agentDir: string) {
  await fsp.mkdir(agentDir, { recursive: true });
  await fsp.copyFile(mainAuthStore, path.join(agentDir, "auth-profiles.json"));
}

async function readSessionMessages(sessionFile: string) {
  return await readClawBoundBenchmarkSessionMessages(sessionFile);
}

async function runScenario(params: {
  label: string;
  prompt: string;
  nativeMode: boolean;
  attemptCount: number;
}) {
  const runRoot = await fsp.mkdtemp(path.join(tempRoot, `${params.label}-`));
  const agentDir = path.join(runRoot, "agent");
  const workspaceDir = path.join(runRoot, "workspace");
  const sessionFile = path.join(runRoot, "session.jsonl");
  await fsp.mkdir(workspaceDir, { recursive: true });
  await copyMainAuthStore(agentDir);
  await seedWorkspace(workspaceDir);

  const parserBefore = await fsp.readFile(path.join(workspaceDir, "parser.js"), "utf8");
  const testBefore = await fsp.readFile(path.join(workspaceDir, "parser.test.js"), "utf8");

  const runId = `live-${params.label}`;
  const previous = process.env.STEWARD_CONTEXT_ENGINE_MODE;
  if (params.nativeMode) {
    process.env.STEWARD_CONTEXT_ENGINE_MODE = "1";
  } else {
    delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
  }
  (globalThis as { __STEWARD_REVIEW_LABEL__?: string }).__STEWARD_REVIEW_LABEL__ = params.label;
  try {
    let result:
      | Awaited<ReturnType<typeof runEmbeddedPiAgent>>
      | null = null;
    let fatalErrorMessage: string | null = null;
    try {
      result = await withClawBoundBenchmarkHarnessTimeout({
        timeoutMs: 55_000,
        timeoutMessage: `live harness timeout after 55000ms`,
        work: () =>
          runEmbeddedPiAgent({
            sessionId: `session:${params.label}`,
            sessionKey: `agent:main:${params.label}`,
            sessionFile,
            workspaceDir,
            config,
            prompt: params.prompt,
            provider: activeLiveModel.provider,
            model: activeLiveModel.model,
            timeoutMs: 45_000,
            agentDir,
            enqueue: immediateEnqueue,
            runId,
          }),
      });
    } catch (error) {
      fatalErrorMessage = error instanceof Error ? error.message : String(error);
    }

    const parserAfter = await fsp.readFile(path.join(workspaceDir, "parser.js"), "utf8");
    const testAfter = await fsp.readFile(path.join(workspaceDir, "parser.test.js"), "utf8");
    const snapshots = observedPrompts.get(params.label) ?? [];
    const firstSnapshot = snapshots[0];
    const systemPromptReport = result?.meta.systemPromptReport;
    const outputText =
      result?.payloads?.map((payload) => payload.text).filter(Boolean).join("\n") ??
      fatalErrorMessage ??
      "";
    const nativeArtifactPath = params.nativeMode
      ? path.join(workspaceDir, ".clawbound", "context-engine", "runs", `${runId}.json`)
      : null;
    const nativeArtifact =
      nativeArtifactPath && fs.existsSync(nativeArtifactPath)
        ? (JSON.parse(await fsp.readFile(nativeArtifactPath, "utf8")) as Record<string, any>)
        : null;
    const sessionMessages = await readSessionMessages(sessionFile);
    const toolCallNames = Array.from(
      new Set(sessionMessages.flatMap((message) => extractToolCallNames(message))),
    );
    const classification = classifyClawBoundBenchmarkLiveRun({
      attemptCount: params.attemptCount,
      outputQuality: outputText,
      actualToolCalls: toolCallNames,
      workspaceMutated: parserBefore !== parserAfter || testBefore !== testAfter,
      fatalErrorMessage,
    });

    return {
      label: params.label,
      task: params.prompt,
      attemptCount: params.attemptCount,
      failureClassification: classification.failureClassification,
      validationOutcome: classification.validationOutcome,
      productValid: classification.productValid,
      externallyBlocked: classification.externallyBlocked,
      retryable: classification.retryable,
      failureReason: classification.failureReason,
      hostPathUsed: params.nativeMode
        ? "embedded-runner / clawbound-bounded-context-engine"
        : "embedded-runner / default-host-path",
      clawboundContextEngineModeActive: params.nativeMode,
      routePlan: nativeArtifact?.routePlan ?? null,
      retrievalPlan: nativeArtifact?.retrievalPlan ?? null,
      noLoad: nativeArtifact?.retrievalPlan?.noLoad ?? null,
      promptInflationSignals: {
        systemPromptChars: systemPromptReport?.systemPrompt.chars ?? 0,
        projectContextChars: systemPromptReport?.systemPrompt.projectContextChars ?? 0,
        injectedWorkspaceFilesCount: systemPromptReport?.injectedWorkspaceFiles.length ?? 0,
        injectedWorkspaceChars:
          systemPromptReport?.injectedWorkspaceFiles.reduce(
            (total, file) => total + (file.injectedChars ?? 0),
            0,
          ) ?? 0,
        skillsPromptChars: systemPromptReport?.skills.promptChars ?? 0,
        observedContainsClawBoundKernel: firstSnapshot?.containsClawBoundKernel ?? false,
        observedContainsAgentsMd: firstSnapshot?.containsAgentsMd ?? false,
        observedContainsProjectContext: firstSnapshot?.containsProjectContext ?? false,
        observedContainsRetrievedSnippets: firstSnapshot?.containsRetrievedSnippets ?? false,
        modelBoundaryCallCount: snapshots.length,
      },
      finalToolProfile: nativeArtifact
        ? nativeArtifact.toolProfile
        : {
            profile: "host-default",
            allowedTools: firstSnapshot?.toolNames ?? [],
            deniedTools: [],
            notes: ["Default host path; no ClawBound-native bounded profile applied."],
          },
      actualToolNamesSeenByModel: firstSnapshot?.toolNames ?? [],
      actualToolCallNames: toolCallNames,
      runOutput: outputText,
      runOutcome: {
        completed: result ? !(result.meta.aborted ?? false) : false,
        aborted: result?.meta.aborted ?? fatalErrorMessage != null,
        stopReason: result?.meta.stopReason ?? (fatalErrorMessage ? "error" : "completed"),
      },
      workspaceMutated: parserBefore !== parserAfter || testBefore !== testAfter,
      persistedArtifactsLocation: {
        contextEngineRun: nativeArtifactPath,
        sessionFile,
        workspaceDir,
      },
    } satisfies RunSummary;
  } finally {
    if (typeof previous === "string") {
      process.env.STEWARD_CONTEXT_ENGINE_MODE = previous;
    } else {
      delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
    }
    delete (globalThis as { __STEWARD_REVIEW_LABEL__?: string }).__STEWARD_REVIEW_LABEL__;
  }
}

async function runScenarioWithRetries(params: {
  label: string;
  prompt: string;
  nativeMode: boolean;
}) {
  let lastRun: RunSummary | null = null;
  for (let attemptCount = 1; attemptCount <= MAX_EXTERNAL_RETRIES; attemptCount += 1) {
    lastRun = await runScenario({
      ...params,
      attemptCount,
    });
    if (lastRun.productValid || !lastRun.retryable) {
      return lastRun;
    }
    if (attemptCount < MAX_EXTERNAL_RETRIES) {
      await new Promise((resolve) => setTimeout(resolve, 1_500 * attemptCount));
    }
  }

  if (!lastRun) {
    throw new Error(`review retries exhausted without producing a run for ${params.label}`);
  }
  return lastRun;
}

beforeAll(async () => {
  vi.useRealTimers();
  tempRoot = await fsp.mkdtemp(path.join(os.tmpdir(), "clawbound-native-review-live-"));
});

afterAll(async () => {
  if (tempRoot) {
    await fsp.rm(tempRoot, { recursive: true, force: true });
  }
});

describeLive("ClawBound bounded review live", () => {
  it(
    "runs the review task on the default and bounded-native paths and writes a validation report",
    async () => {
      const prompt =
        "Review parseCsvLine in parser.js for correctness and edge cases without editing files. Focus on parser.js and parser.test.js. If needed, run the focused parser test for evidence.";
      await fsp.rm(reportPath, { force: true });
      const runs: RunSummary[] = [];
      let workflowFailure: ClawBoundBenchmarkLiveRunClassification | null = null;
      let restoreEnv: (() => void) | null = null;
      try {
        const liveEnvironment = await prepareClawBoundBenchmarkLiveEnvironment({
          tempRoot,
        });
        restoreEnv = liveEnvironment.restoreEnv;
        config = liveEnvironment.config;
        activeLiveModel = liveEnvironment.activeLiveModel;
        actualLiveModel = liveEnvironment.activeLiveModel;
        ({ runEmbeddedPiAgent } = await import("/Users/javiswan/Projects/clawbound/src/agents/pi-embedded-runner.ts"));
        ({ ensureAgentWorkspace } = await import("/Users/javiswan/Projects/clawbound/src/agents/workspace.ts"));

        runs.push(
          await runScenarioWithRetries({
            label: "review-host",
            prompt,
            nativeMode: false,
          }),
        );
        runs.push(
          await runScenarioWithRetries({
            label: "review-native",
            prompt,
            nativeMode: true,
          }),
        );
      } catch (error) {
        workflowFailure = classifyClawBoundBenchmarkLiveRun({
          attemptCount: 1,
          outputQuality: "",
          actualToolCalls: [],
          workspaceMutated: false,
          fatalErrorMessage: error instanceof Error ? error.message : String(error),
        });
      } finally {
        const meaningfulExecution = runs.some((run) => run.productValid || run.actualToolCallNames.length > 0);
        const summaryInputs = workflowFailure ? [...runs, workflowFailure] : runs;
        const report = {
          createdAt: new Date().toISOString(),
          provider: (actualLiveModel ?? intendedLiveModel).provider,
          model: (actualLiveModel ?? intendedLiveModel).model,
          modelBinding: summarizeClawBoundBenchmarkLiveModelBinding({
            intended: intendedLiveModel,
            actual: actualLiveModel,
            meaningfulExecution,
          }),
          runs,
          summary: summarizeClawBoundBenchmarkLiveRuns(summaryInputs),
          workflowFailure,
        };
        await fsp.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
        restoreEnv?.();
      }

      if (workflowFailure) {
        throw new Error(workflowFailure.failureReason ?? "review harness bootstrap failed");
      }

      expect(runs).toHaveLength(2);
      expect(runs.every((run) => run.validationOutcome.length > 0)).toBe(true);
      expect(runs.every((run) => run.attemptCount >= 1)).toBe(true);
      const nativeRun = runs.find((run) => run.label === "review-native");
      expect(nativeRun?.validationOutcome === "valid_product_result" || nativeRun?.externallyBlocked === true).toBe(true);
      if (nativeRun?.productValid) {
        expect(nativeRun.promptInflationSignals.injectedWorkspaceFilesCount).toBe(0);
        expect(nativeRun.promptInflationSignals.skillsPromptChars).toBe(0);
        expect(nativeRun.routePlan?.executionMode).toBe("reviewer");
        expect(nativeRun.actualToolNamesSeenByModel).toEqual(["read", "exec"]);
      }
      if (runs.every((run) => run.productValid)) {
        expect(runs.every((run) => run.workspaceMutated === false)).toBe(true);
      }
    },
    240_000,
  );
});
