import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
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
  messages: Array<{ role: string; content: unknown }>;
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

type TurnSummary = {
  turn: number;
  label: string;
  prompt: string;
  clawboundContextEngineModeActive: boolean;
  routePlan: Record<string, unknown> | null;
  retrievalPlan: Record<string, unknown> | null;
  noLoad: boolean | null;
  promptInflationIndicators: {
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
  modelBoundaryTools: string[];
  actualToolCalls: string[];
  outputQuality: string;
  contextStayedBounded: boolean;
  persistedArtifactsLocation: {
    contextEngineRun?: string | null;
    sessionFile: string;
    workspaceDir: string;
  };
  continuityArtifacts: {
    compactArtifactPath: string | null;
    compactEventTypes: string[];
    compactRetrievedUnitIds: string[];
    activeContextLocalContextRefs: string[];
    activeContextRetrievedUnitIds: string[];
    omittedReasons: string[];
    boundednessReasons: string[];
    stayedBoundedByCompact: boolean | null;
    promptPackageTokens: number | null;
    retrievedUnitIds: string[];
  };
};

type ScenarioSummary = {
  label: string;
  attemptCount: number;
  failureClassification: ClawBoundBenchmarkLiveFailureClassification | null;
  validationOutcome: ClawBoundBenchmarkLiveValidationOutcome;
  productValid: boolean;
  externallyBlocked: boolean;
  retryable: boolean;
  failureReason: string | null;
  hostPathUsed: string;
  clawboundMode: boolean;
  turns: TurnSummary[];
  sessionLevel: {
    contextEngineArtifactsProduced: boolean;
    activeContextStayedBounded: boolean;
    silentPromptGrowthDetected: boolean;
    systemPromptCharsByTurn: number[];
    promptPackageTokensByTurn: Array<number | null>;
    finalParserTest: {
      exitCode: number | null;
      stdout: string;
      stderr: string;
    };
    workspaceDir: string;
    sessionFile: string;
  };
};

type SessionMessage = {
  role?: string;
  content?: unknown;
  errorMessage?: unknown;
};

const LIVE =
  isTruthyEnvValue(process.env.OPENCLAW_LIVE_TEST) ||
  isTruthyEnvValue(process.env.MINIMAX_LIVE_TEST) ||
  isTruthyEnvValue(process.env.LIVE);

const mainAgentDir = resolveOpenClawAgentDir();
const mainAuthStore = path.join(mainAgentDir, "auth-profiles.json");
const describeLive = LIVE && fs.existsSync(mainAuthStore) ? describe : describe.skip;
const reportPath = path.join(os.tmpdir(), "clawbound-native-multiturn-live-report.json");
const observedPrompts = new Map<string, PromptSnapshot[]>();
const MAX_EXTERNAL_RETRIES = 2;

function messageContentToText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
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

function extractLatestAssistantText(messages: SessionMessage[]) {
  const latestAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  if (!latestAssistant) {
    return "";
  }
  const text = messageContentToText(latestAssistant.content).trim();
  if (text) {
    return text;
  }
  return typeof latestAssistant.errorMessage === "string" ? latestAssistant.errorMessage.trim() : "";
}

function isTurnContextBounded(turn: TurnSummary) {
  if (!turn.clawboundContextEngineModeActive) {
    return false;
  }
  return (
    turn.promptInflationIndicators.injectedWorkspaceFilesCount === 0 &&
    turn.promptInflationIndicators.skillsPromptChars === 0 &&
    !turn.promptInflationIndicators.observedContainsAgentsMd &&
    !turn.promptInflationIndicators.observedContainsProjectContext
  );
}

vi.mock("@mariozechner/pi-ai", async () => {
  const actual = await vi.importActual<typeof import("@mariozechner/pi-ai")>("@mariozechner/pi-ai");
  return {
    ...actual,
    streamSimple: (model: ModelLike, context: ContextLike, options: Record<string, unknown>) => {
      const label = String(
        (globalThis as { __STEWARD_MULTITURN_LABEL__?: string }).__STEWARD_MULTITURN_LABEL__ ?? "unknown",
      );
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
    path.join(workspaceDir, "package.json"),
    ['{', '  "type": "module",', '  "scripts": {', '    "test:parser": "node --test parser.test.js"', "  }", "}", ""].join("\n"),
    "utf8",
  );
  await fsp.writeFile(
    path.join(workspaceDir, "AGENTS.md"),
    "# AGENTS.md\n- Keep parser continuity grounded in parser.js and parser.test.js.\n- Keep fixes narrow and preserve public API.\n",
    "utf8",
  );
  await fsp.writeFile(
    path.join(workspaceDir, "TOOLS.md"),
    "# TOOLS.md\n- Read code before editing.\n- Use the focused parser test instead of broad test runs.\n",
    "utf8",
  );
}

async function copyMainAuthStore(agentDir: string) {
  await fsp.mkdir(agentDir, { recursive: true });
  await fsp.copyFile(mainAuthStore, path.join(agentDir, "auth-profiles.json"));
}

async function readSessionMessages(sessionFile: string) {
  return (await readClawBoundBenchmarkSessionMessages(sessionFile)) as SessionMessage[];
}

async function runFocusedParserTest(workspaceDir: string) {
  return await new Promise<{ exitCode: number | null; stdout: string; stderr: string }>((resolve) => {
    execFile(
      process.execPath,
      ["--test", "parser.test.js"],
      { cwd: workspaceDir, timeout: 30_000 },
      (error, stdout, stderr) => {
        const exitCode =
          error && typeof (error as { code?: unknown }).code === "number"
            ? ((error as { code: number }).code ?? 1)
            : 0;
        resolve({
          exitCode,
          stdout: stdout.trim(),
          stderr: stderr.trim(),
        });
      },
    );
  });
}

async function runWorkflow(params: {
  label: string;
  nativeMode: boolean;
  prompts: string[];
  attemptCount: number;
}) {
  const runRoot = await fsp.mkdtemp(path.join(tempRoot, `${params.label}-`));
  const agentDir = path.join(runRoot, "agent");
  const workspaceDir = path.join(runRoot, "workspace");
  const sessionFile = path.join(runRoot, "session.jsonl");
  const sessionId = `session:${params.label}`;
  const sessionKey = `agent:main:${params.label}`;
  await fsp.mkdir(workspaceDir, { recursive: true });
  await copyMainAuthStore(agentDir);
  await seedWorkspace(workspaceDir);

  const previousMode = process.env.STEWARD_CONTEXT_ENGINE_MODE;
  if (params.nativeMode) {
    process.env.STEWARD_CONTEXT_ENGINE_MODE = "1";
  } else {
    delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
  }

  let seenMessages = 0;
  const turns: TurnSummary[] = [];
  let fatalErrorMessage: string | null = null;
  try {
    try {
      for (const [index, prompt] of params.prompts.entries()) {
        const turn = index + 1;
        const label = `${params.label}:turn-${turn}`;
        const runId = `live-${params.label}-turn-${turn}`;
        (globalThis as { __STEWARD_MULTITURN_LABEL__?: string }).__STEWARD_MULTITURN_LABEL__ = label;

        const result = await withClawBoundBenchmarkHarnessTimeout({
          timeoutMs: 55_000,
          timeoutMessage: `live harness timeout after 55000ms`,
          work: () =>
            runEmbeddedPiAgent({
              sessionId,
              sessionKey,
              sessionFile,
              workspaceDir,
              config,
              prompt,
              provider: activeLiveModel.provider,
              model: activeLiveModel.model,
              timeoutMs: 45_000,
              agentDir,
              enqueue: immediateEnqueue,
              runId,
            }),
        });

        const nativeMeta = (result.meta as Record<string, any>).clawboundContextEngine as
          | {
              compactPersistedPath?: string;
              persistedPath?: string;
              compactEventTypes?: string[];
              compactRetrievedUnitIds?: string[];
            }
          | undefined;
        const nativeArtifactPath = params.nativeMode
          ? path.join(workspaceDir, ".clawbound", "context-engine", "runs", `${runId}.json`)
          : null;
        const nativeArtifact =
          nativeArtifactPath && fs.existsSync(nativeArtifactPath)
            ? (JSON.parse(await fsp.readFile(nativeArtifactPath, "utf8")) as Record<string, any>)
            : null;

        const systemPromptReport = result.meta.systemPromptReport;
        const snapshots = observedPrompts.get(label) ?? [];
        const firstSnapshot = snapshots[0];
        const allMessages = await readSessionMessages(sessionFile);
        const newMessages = allMessages.slice(seenMessages);
        seenMessages = allMessages.length;
        const compactArtifactPath = nativeMeta?.compactPersistedPath ?? null;
        const compactArtifact =
          compactArtifactPath && fs.existsSync(compactArtifactPath)
            ? (JSON.parse(await fsp.readFile(compactArtifactPath, "utf8")) as {
                activeContext?: {
                  localContextRefs?: string[];
                  retrievedUnitIds?: string[];
                };
                omitted?: { reasons?: string[] };
                boundedness?: { stayedBounded?: boolean; reasons?: string[] };
              })
            : null;

        const toolCallNames = Array.from(
          new Set(newMessages.flatMap((message) => extractToolCallNames(message as Record<string, unknown>))),
        );
        const turnSummary: TurnSummary = {
          turn,
          label,
          prompt,
          clawboundContextEngineModeActive: params.nativeMode,
          routePlan: nativeArtifact?.routePlan ?? null,
          retrievalPlan: nativeArtifact?.retrievalPlan ?? null,
          noLoad: nativeArtifact?.retrievalPlan?.noLoad ?? null,
          promptInflationIndicators: {
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
          modelBoundaryTools: firstSnapshot?.toolNames ?? [],
          actualToolCalls: toolCallNames,
          outputQuality: extractLatestAssistantText(newMessages),
          contextStayedBounded: false,
          persistedArtifactsLocation: {
            contextEngineRun: nativeArtifactPath,
            sessionFile,
            workspaceDir,
          },
          continuityArtifacts: {
            compactArtifactPath,
            compactEventTypes: nativeMeta?.compactEventTypes ?? [],
            compactRetrievedUnitIds: nativeMeta?.compactRetrievedUnitIds ?? [],
            activeContextLocalContextRefs: compactArtifact?.activeContext?.localContextRefs ?? [],
            activeContextRetrievedUnitIds: compactArtifact?.activeContext?.retrievedUnitIds ?? [],
            omittedReasons: compactArtifact?.omitted?.reasons ?? [],
            boundednessReasons: compactArtifact?.boundedness?.reasons ?? [],
            stayedBoundedByCompact: compactArtifact?.boundedness?.stayedBounded ?? null,
            promptPackageTokens: nativeArtifact?.promptPackage?.assemblyStats?.totalEstimatedTokens ?? null,
            retrievedUnitIds: (nativeArtifact?.promptPackage?.retrievedUnits ?? []).map(
              (unit: { id: string }) => unit.id,
            ),
          },
        };
        turnSummary.contextStayedBounded = isTurnContextBounded(turnSummary);
        turns.push(turnSummary);
      }
    } catch (error) {
      fatalErrorMessage = error instanceof Error ? error.message : String(error);
    }
  } finally {
    if (typeof previousMode === "string") {
      process.env.STEWARD_CONTEXT_ENGINE_MODE = previousMode;
    } else {
      delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
    }
    delete (globalThis as { __STEWARD_MULTITURN_LABEL__?: string }).__STEWARD_MULTITURN_LABEL__;
  }

  const finalParserTest = await runFocusedParserTest(workspaceDir);
  const systemPromptCharsByTurn = turns.map((turn) => turn.promptInflationIndicators.systemPromptChars);
  const promptPackageTokensByTurn = turns.map((turn) => turn.continuityArtifacts.promptPackageTokens);
  const classification = classifyClawBoundBenchmarkLiveRun({
    attemptCount: params.attemptCount,
    outputQuality: turns.map((turn) => turn.outputQuality).filter(Boolean).join("\n"),
    actualToolCalls: turns.flatMap((turn) => turn.actualToolCalls),
    workspaceMutated: finalParserTest.exitCode === 0,
    finalCheckExitCode: finalParserTest.exitCode,
    fatalErrorMessage,
  });
  return {
    label: params.label,
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
    clawboundMode: params.nativeMode,
    turns,
    sessionLevel: {
      contextEngineArtifactsProduced: params.nativeMode
        ? turns.every((turn) => Boolean(turn.persistedArtifactsLocation.contextEngineRun))
        : false,
      activeContextStayedBounded: params.nativeMode ? turns.every((turn) => turn.contextStayedBounded) : false,
      silentPromptGrowthDetected: systemPromptCharsByTurn.some(
        (chars, index, values) => index > 0 && chars - values[index - 1]! > 2_000,
      ),
      systemPromptCharsByTurn,
      promptPackageTokensByTurn,
      finalParserTest,
      workspaceDir,
      sessionFile,
    },
  } satisfies ScenarioSummary;
}

async function runWorkflowWithRetries(params: {
  label: string;
  nativeMode: boolean;
  prompts: string[];
}) {
  let lastScenario: ScenarioSummary | null = null;
  for (let attemptCount = 1; attemptCount <= MAX_EXTERNAL_RETRIES; attemptCount += 1) {
    lastScenario = await runWorkflow({
      ...params,
      attemptCount,
    });
    if (lastScenario.productValid || !lastScenario.retryable) {
      return lastScenario;
    }
    if (attemptCount < MAX_EXTERNAL_RETRIES) {
      await new Promise((resolve) => setTimeout(resolve, 1_500 * attemptCount));
    }
  }

  if (!lastScenario) {
    throw new Error(`multiturn retries exhausted without producing a scenario for ${params.label}`);
  }
  return lastScenario;
}

beforeAll(async () => {
  vi.useRealTimers();
  tempRoot = await fsp.mkdtemp(path.join(os.tmpdir(), "clawbound-native-multiturn-"));
});

afterAll(async () => {
  if (tempRoot) {
    await fsp.rm(tempRoot, { recursive: true, force: true });
  }
});

describeLive("ClawBound bounded native multi-turn continuity", () => {
  it(
    "runs one narrow multi-turn parser workflow on default and ClawBound-native paths and writes a report",
    async () => {
      await fsp.rm(reportPath, { force: true });
      const prompts = [
        "Explain what parseCsvLine currently does in parser.js. Read the file first. Do not edit files.",
        "Review parseCsvLine edge cases in parser.js and parser.test.js without editing files. Focus on whitespace trimming and quoted fields. Run the focused parser test only if you need evidence.",
        "Make the narrowest fix in parser.js so parser.test.js passes without changing public API. Run the focused parser test before finalizing.",
        "Verify in two bullets what changed in parser.js and whether parser.test.js now passes. Do not make further edits.",
      ];

      const scenarios: ScenarioSummary[] = [];
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

        scenarios.push(
          await runWorkflowWithRetries({
            label: "multiturn-host",
            nativeMode: false,
            prompts,
          }),
        );
        scenarios.push(
          await runWorkflowWithRetries({
            label: "multiturn-native",
            nativeMode: true,
            prompts,
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
        const summaryInputs = workflowFailure ? [...scenarios, workflowFailure] : scenarios;
        const meaningfulExecution = scenarios.some(
          (scenario) =>
            scenario.productValid ||
            scenario.turns.some((turn) => turn.actualToolCalls.length > 0),
        );
        await fsp.writeFile(
          reportPath,
          `${JSON.stringify(
            {
              createdAt: new Date().toISOString(),
              provider: (actualLiveModel ?? intendedLiveModel).provider,
              model: (actualLiveModel ?? intendedLiveModel).model,
              modelBinding: summarizeClawBoundBenchmarkLiveModelBinding({
                intended: intendedLiveModel,
                actual: actualLiveModel,
                meaningfulExecution,
              }),
              scenarios,
              summary: summarizeClawBoundBenchmarkLiveRuns(summaryInputs),
              workflowFailure,
            },
            null,
            2,
          )}\n`,
          "utf8",
        );
        restoreEnv?.();
      }

      if (workflowFailure) {
        throw new Error(workflowFailure.failureReason ?? "multiturn harness bootstrap failed");
      }

      expect(scenarios).toHaveLength(2);
      expect(scenarios.every((scenario) => scenario.validationOutcome.length > 0)).toBe(true);
      expect(scenarios.every((scenario) => scenario.attemptCount >= 1)).toBe(true);
      expect(scenarios[0]?.clawboundMode).toBe(false);
      expect(scenarios[1]?.clawboundMode).toBe(true);
      expect(
        scenarios[1]?.validationOutcome === "valid_product_result" ||
          scenarios[1]?.externallyBlocked === true,
      ).toBe(true);
      if (scenarios[1]?.productValid) {
        expect(scenarios[0]?.turns).toHaveLength(4);
        expect(scenarios[1]?.turns).toHaveLength(4);
        expect(scenarios[1]?.sessionLevel.contextEngineArtifactsProduced).toBe(true);
        expect(scenarios[1]?.turns[0]?.routePlan).not.toBeNull();
      }
    },
    480_000,
  );
});
