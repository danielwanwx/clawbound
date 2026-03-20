import { randomUUID } from "node:crypto";
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
import { getDeterministicFreePortBlock } from "../test-utils/ports.js";
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
  parentRoutePlan: Record<string, unknown> | null;
  parentRetrievalPlan: Record<string, unknown> | null;
  noLoad: boolean | null;
  subagentDelegationOccurred: boolean;
  handoff: {
    childSessionKey: string | null;
    childRunId: string | null;
    handoffArtifactPath: string | null;
    inheritsFullTranscriptPrevented: boolean | null;
    localContextRefs: string[];
    retrievedUnitIds: string[];
  };
  promptInflation: {
    parent: {
      systemPromptChars: number;
      projectContextChars: number;
      injectedWorkspaceFilesCount: number;
      injectedWorkspaceChars: number;
      skillsPromptChars: number;
      containsClawBoundKernel: boolean;
      containsAgentsMd: boolean;
      containsProjectContext: boolean;
    };
    child: {
      systemPromptChars: number;
      projectContextChars: number;
      injectedWorkspaceFilesCount: number;
      injectedWorkspaceChars: number;
      skillsPromptChars: number;
      toolCount: number;
    };
  };
  modelBoundaryTools: {
    parent: string[];
    child: string[];
  };
  actualToolCalls: {
    parent: string[];
    child: string[];
  };
  outputQuality: {
    parent: string;
    child: string;
  };
  taskStayedWithinDelegatedScope: boolean;
  persistedArtifactsLocation: {
    parentContextEngineRun?: string | null;
    childContextEngineRun?: string | null;
    handoffArtifact?: string | null;
    parentSessionFile: string;
    childSessionFile: string | null;
    workspaceDir: string;
  };
};

type SessionMessage = {
  role?: string;
  content?: unknown;
  toolCallId?: string;
};

const LIVE =
  isTruthyEnvValue(process.env.OPENCLAW_LIVE_TEST) ||
  isTruthyEnvValue(process.env.MINIMAX_LIVE_TEST) ||
  isTruthyEnvValue(process.env.LIVE);
const mainAgentDir = resolveOpenClawAgentDir();
const mainAuthStore = path.join(mainAgentDir, "auth-profiles.json");
const describeLive = LIVE && fs.existsSync(mainAuthStore) ? describe : describe.skip;
const reportPath = path.join(os.tmpdir(), "clawbound-native-subagent-live-report.json");
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

vi.mock("@mariozechner/pi-ai", async () => {
  const actual = await vi.importActual<typeof import("@mariozechner/pi-ai")>("@mariozechner/pi-ai");
  return {
    ...actual,
    streamSimple: (model: ModelLike, context: ContextLike, options: Record<string, unknown>) => {
      const label = String(
        (globalThis as { __STEWARD_SUBAGENT_LIVE_LABEL__?: string }).__STEWARD_SUBAGENT_LIVE_LABEL__ ??
          "unknown",
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
let startGatewayServer: typeof import("/Users/javiswan/Projects/clawbound/src/gateway/server.ts").startGatewayServer;
let callGateway: typeof import("/Users/javiswan/Projects/clawbound/src/gateway/call.ts").callGateway;
let loadSessionEntry: typeof import("/Users/javiswan/Projects/clawbound/src/gateway/session-utils.ts").loadSessionEntry;
let resolveSessionTranscriptPath: typeof import("/Users/javiswan/Projects/clawbound/src/config/sessions.ts").resolveSessionTranscriptPath;
let gatewayServer:
  | Awaited<ReturnType<typeof import("/Users/javiswan/Projects/clawbound/src/gateway/server.ts").startGatewayServer>>
  | null = null;
let tempRoot = "";
let gatewayPort = 0;
let config: OpenClawConfig;
const intendedLiveModel: ClawBoundBenchmarkLiveModelBinding =
  resolveClawBoundBenchmarkConfiguredLiveModelBinding();
let activeLiveModel: ClawBoundBenchmarkLiveModelBinding = { ...intendedLiveModel };
let actualLiveModel: ClawBoundBenchmarkLiveModelBinding | null = null;

const previousEnv: Record<string, string | undefined> = {};
const immediateEnqueue = async <T>(task: () => Promise<T>) => task();

function rememberEnv(key: string) {
  previousEnv[key] = process.env[key];
}

function restoreEnv(key: string) {
  const previous = previousEnv[key];
  if (previous === undefined) {
    delete process.env[key];
  } else {
    process.env[key] = previous;
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
    throw new Error(`subagent retries exhausted without producing a run for ${params.label}`);
  }
  return lastRun;
}

async function chooseGatewayPort() {
  return await getDeterministicFreePortBlock({ offsets: [0] });
}

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
}

async function copyMainAuthStore(agentDir: string) {
  await fsp.mkdir(agentDir, { recursive: true });
  await fsp.copyFile(mainAuthStore, path.join(agentDir, "auth-profiles.json"));
}

async function readSessionMessages(sessionFile: string) {
  return (await readClawBoundBenchmarkSessionMessages(sessionFile)) as SessionMessage[];
}

function extractLatestAssistantText(messages: SessionMessage[]) {
  return [...messages]
    .reverse()
    .find((message) => message.role === "assistant" && messageContentToText(message.content).trim())
    ? messageContentToText(
        [...messages]
          .reverse()
          .find((message) => message.role === "assistant" && messageContentToText(message.content).trim())
          ?.content,
      ).trim()
    : "";
}

function extractSessionsSpawnResult(messages: SessionMessage[]) {
  for (const message of messages) {
    if (message.role !== "toolResult") {
      continue;
    }
    const text = messageContentToText(message.content).trim();
    if (!text) {
      continue;
    }
    try {
      const parsed = JSON.parse(text) as {
        childSessionKey?: string;
        runId?: string;
        clawbound?: {
          handoffArtifactPath?: string;
        };
      };
      if (parsed.childSessionKey && parsed.runId) {
        return parsed;
      }
    } catch {
      continue;
    }
  }
  return null;
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
  const parserTestBefore = await fsp.readFile(path.join(workspaceDir, "parser.test.js"), "utf8");

  const runId = `live-${params.label}`;
  const previousMode = process.env.STEWARD_CONTEXT_ENGINE_MODE;
  if (params.nativeMode) {
    process.env.STEWARD_CONTEXT_ENGINE_MODE = "1";
  } else {
    delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
  }
  (globalThis as { __STEWARD_SUBAGENT_LIVE_LABEL__?: string }).__STEWARD_SUBAGENT_LIVE_LABEL__ =
    params.label;

  try {
    let result:
      | Awaited<ReturnType<typeof runEmbeddedPiAgent>>
      | null = null;
    let fatalErrorMessage: string | null = null;
    let parentMessages: SessionMessage[] = [];
    let childMessages: SessionMessage[] = [];
    let childSessionKey: string | null = null;
    let childRunId: string | null = null;
    let childSessionFile: string | null = null;
    let childEntry: Record<string, any> | null = null;
    let handoffArtifactPath: string | null = null;
    let handoffArtifact:
      | {
          inheritsFullTranscript?: boolean;
          localContextRefs?: string[];
          retrievedUnitIds?: string[];
        }
      | null = null;
    let childToolNames: string[] = [];
    let parentSnapshot: PromptSnapshot | undefined;
    let childSnapshot: PromptSnapshot | undefined;
    let parentNativeArtifact:
      | Record<string, any>
      | null = null;
    const parentNativeArtifactPath = params.nativeMode
      ? path.join(workspaceDir, ".clawbound", "context-engine", "runs", `${runId}.json`)
      : null;
    let childNativeArtifactPath: string | null = null;
    try {
      result = await withClawBoundBenchmarkHarnessTimeout({
        timeoutMs: 70_000,
        timeoutMessage: `live harness timeout after 70000ms`,
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
            timeoutMs: 60_000,
            agentDir,
            enqueue: immediateEnqueue,
            runId,
          }),
      });

      parentMessages = await readSessionMessages(sessionFile);
      const snapshots = observedPrompts.get(params.label) ?? [];
      const sessionsSpawnResult = extractSessionsSpawnResult(parentMessages);
      childSessionKey = sessionsSpawnResult?.childSessionKey ?? null;
      childRunId = sessionsSpawnResult?.runId ?? null;
      if (!childSessionKey || !childRunId) {
        throw new Error(`sessions_spawn result missing for ${params.label}`);
      }

      await withClawBoundBenchmarkHarnessTimeout({
        timeoutMs: 80_000,
        timeoutMessage: "live harness timeout after 80000ms",
        work: () =>
          callGateway({
            method: "agent.wait",
            params: {
              runId: childRunId,
              timeoutMs: 60_000,
            },
            timeoutMs: 70_000,
          }),
      });

      ({ entry: childEntry } = loadSessionEntry(childSessionKey));
      const childSessionId = childEntry?.sessionId;
      childSessionFile =
        childSessionId != null ? resolveSessionTranscriptPath(childSessionId, "main") : null;
      if (!childSessionFile) {
        throw new Error(`child session file missing for ${params.label}`);
      }
      childMessages = await readSessionMessages(childSessionFile);
      parentSnapshot = snapshots[0];
      childSnapshot =
        snapshots.find(
          (snapshot, index) => index > 0 && snapshot.systemPrompt.includes(`Your session: ${childSessionKey}.`),
        ) ??
        snapshots.find(
          (snapshot, index) => index > 0 && snapshot.systemPrompt.includes(childSessionKey),
        ) ??
        snapshots.find((snapshot, index) => index > 0 && snapshot.systemPrompt.includes("# Subagent Context")) ??
        snapshots[1];
      parentNativeArtifact =
        parentNativeArtifactPath && fs.existsSync(parentNativeArtifactPath)
          ? (JSON.parse(await fsp.readFile(parentNativeArtifactPath, "utf8")) as Record<string, any>)
          : null;
      childNativeArtifactPath = params.nativeMode
        ? path.join(workspaceDir, ".clawbound", "context-engine", "runs", `${childRunId}.json`)
        : null;
      handoffArtifactPath = sessionsSpawnResult?.clawbound?.handoffArtifactPath ?? null;
      handoffArtifact =
        handoffArtifactPath && fs.existsSync(handoffArtifactPath)
          ? (JSON.parse(await fsp.readFile(handoffArtifactPath, "utf8")) as {
              inheritsFullTranscript?: boolean;
              localContextRefs?: string[];
              retrievedUnitIds?: string[];
            })
          : null;
      childToolNames =
        childEntry?.systemPromptReport?.tools.entries.map((entry) => entry.name) ?? [];
    } catch (error) {
      fatalErrorMessage = error instanceof Error ? error.message : String(error);
      if (fs.existsSync(sessionFile)) {
        parentMessages = await readSessionMessages(sessionFile);
      }
    }
    const parserAfter = await fsp.readFile(path.join(workspaceDir, "parser.js"), "utf8");
    const parserTestAfter = await fsp.readFile(path.join(workspaceDir, "parser.test.js"), "utf8");
    const parentToolCalls = Array.from(new Set(parentMessages.flatMap((message) => extractToolCallNames(message))));
    const childToolCalls = Array.from(new Set(childMessages.flatMap((message) => extractToolCallNames(message))));
    const outputParent =
      result?.payloads?.map((payload) => payload.text).filter(Boolean).join("\n").trim() ??
      fatalErrorMessage ??
      "";
    const outputChild = childMessages.length > 0 ? extractLatestAssistantText(childMessages) : "";
    const classification = classifyClawBoundBenchmarkLiveRun({
      attemptCount: params.attemptCount,
      outputQuality: [outputParent, outputChild].filter(Boolean).join("\n"),
      actualToolCalls: [...parentToolCalls, ...childToolCalls],
      workspaceMutated: parserBefore !== parserAfter || parserTestBefore !== parserTestAfter,
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
        ? "embedded-runner / clawbound-bounded-context-engine + gateway child"
        : "embedded-runner / default-host-path + gateway child",
      clawboundContextEngineModeActive: params.nativeMode,
      parentRoutePlan: parentNativeArtifact?.routePlan ?? null,
      parentRetrievalPlan: parentNativeArtifact?.retrievalPlan ?? null,
      noLoad: parentNativeArtifact?.retrievalPlan?.noLoad ?? null,
      subagentDelegationOccurred: childSessionKey != null && childRunId != null,
      handoff: {
        childSessionKey,
        childRunId,
        handoffArtifactPath,
        inheritsFullTranscriptPrevented:
          handoffArtifact && handoffArtifact.inheritsFullTranscript === false ? true : null,
        localContextRefs: handoffArtifact?.localContextRefs ?? [],
        retrievedUnitIds: handoffArtifact?.retrievedUnitIds ?? [],
      },
      promptInflation: {
        parent: {
          systemPromptChars: result?.meta.systemPromptReport?.systemPrompt.chars ?? 0,
          projectContextChars: result?.meta.systemPromptReport?.systemPrompt.projectContextChars ?? 0,
          injectedWorkspaceFilesCount: result?.meta.systemPromptReport?.injectedWorkspaceFiles.length ?? 0,
          injectedWorkspaceChars:
            result?.meta.systemPromptReport?.injectedWorkspaceFiles.reduce(
              (total, file) => total + (file.injectedChars ?? 0),
              0,
            ) ?? 0,
          skillsPromptChars: result?.meta.systemPromptReport?.skills.promptChars ?? 0,
          containsClawBoundKernel: parentSnapshot?.containsClawBoundKernel ?? false,
          containsAgentsMd: parentSnapshot?.containsAgentsMd ?? false,
          containsProjectContext: parentSnapshot?.containsProjectContext ?? false,
        },
        child: {
          systemPromptChars:
            childSnapshot?.systemPromptChars ?? childEntry?.systemPromptReport?.systemPrompt.chars ?? 0,
          projectContextChars: childEntry?.systemPromptReport?.systemPrompt.projectContextChars ?? 0,
          injectedWorkspaceFilesCount: childEntry?.systemPromptReport?.injectedWorkspaceFiles.length ?? 0,
          injectedWorkspaceChars:
            childEntry?.systemPromptReport?.injectedWorkspaceFiles.reduce(
              (total, file) => total + (file.injectedChars ?? 0),
              0,
            ) ?? 0,
          skillsPromptChars: childEntry?.systemPromptReport?.skills.promptChars ?? 0,
          toolCount: childToolNames.length,
        },
      },
      modelBoundaryTools: {
        parent: parentSnapshot?.toolNames ?? [],
        child: childSnapshot?.toolNames ?? childToolNames,
      },
      actualToolCalls: {
        parent: parentToolCalls,
        child: childToolCalls,
      },
      outputQuality: {
        parent: outputParent,
        child: outputChild,
      },
      taskStayedWithinDelegatedScope: parserBefore === parserAfter && parserTestBefore === parserTestAfter,
      persistedArtifactsLocation: {
        parentContextEngineRun: parentNativeArtifactPath,
        childContextEngineRun: childNativeArtifactPath,
        handoffArtifact: handoffArtifactPath,
        parentSessionFile: sessionFile,
        childSessionFile,
        workspaceDir,
      },
    } satisfies RunSummary;
  } finally {
    if (typeof previousMode === "string") {
      process.env.STEWARD_CONTEXT_ENGINE_MODE = previousMode;
    } else {
      delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
    }
    delete (globalThis as { __STEWARD_SUBAGENT_LIVE_LABEL__?: string })
      .__STEWARD_SUBAGENT_LIVE_LABEL__;
  }
}

beforeAll(async () => {
  vi.useRealTimers();
  tempRoot = await fsp.mkdtemp(path.join(os.tmpdir(), "clawbound-native-subagent-live-"));
});

afterAll(async () => {
  if (gatewayServer) {
    await gatewayServer.close();
  }
  if (tempRoot) {
    await fsp.rm(tempRoot, { recursive: true, force: true });
  }
});

describeLive("ClawBound bounded native subagent live validation", () => {
  it(
    "runs one delegated review task on default and ClawBound-native bounded paths and writes a report",
    async () => {
      const task =
        "Use the sessions_spawn tool exactly once to delegate a narrow review of parseCsvLine quoted-field handling and regression risk. Do not edit files yourself. The child should only inspect parser.js and parser.test.js and report the single highest-risk regression. After spawning, report the delegated scope and child session key only.";
      await fsp.rm(reportPath, { force: true });
      const runs: RunSummary[] = [];
      let workflowFailure: ClawBoundBenchmarkLiveRunClassification | null = null;
      let workflowBootstrapError: string | null = null;
      let restoreLiveEnv: (() => void) | null = null;
      try {
        gatewayPort = await chooseGatewayPort();
        const token = `test-token-${randomUUID()}`;

        for (const key of [
          "OPENCLAW_SKIP_CHANNELS",
          "OPENCLAW_SKIP_GMAIL_WATCHER",
          "OPENCLAW_SKIP_CANVAS_HOST",
          "OPENCLAW_SKIP_BROWSER_CONTROL_SERVER",
          "STEWARD_CONTEXT_ENGINE_MODE",
        ]) {
          rememberEnv(key);
        }

        process.env.OPENCLAW_SKIP_CHANNELS = "1";
        process.env.OPENCLAW_SKIP_GMAIL_WATCHER = "1";
        process.env.OPENCLAW_SKIP_CANVAS_HOST = "1";
        process.env.OPENCLAW_SKIP_BROWSER_CONTROL_SERVER = "1";
        delete process.env.STEWARD_CONTEXT_ENGINE_MODE;

        const liveEnvironment = await prepareClawBoundBenchmarkLiveEnvironment({
          tempRoot,
          workspaceDir: path.join(tempRoot, "gateway-workspace-main"),
          gatewayPort,
          gatewayToken: token,
          includeSessionConfig: true,
          includeSubagentModel: true,
          controlUiEnabled: false,
        });
        restoreLiveEnv = liveEnvironment.restoreEnv;
        activeLiveModel = liveEnvironment.activeLiveModel;
        actualLiveModel = liveEnvironment.activeLiveModel;
        config = liveEnvironment.config;
        ({ runEmbeddedPiAgent } = await import("/Users/javiswan/Projects/clawbound/src/agents/pi-embedded-runner.ts"));
        ({ ensureAgentWorkspace } = await import("/Users/javiswan/Projects/clawbound/src/agents/workspace.ts"));
        ({ startGatewayServer } = await import("/Users/javiswan/Projects/clawbound/src/gateway/server.ts"));
        ({ callGateway } = await import("/Users/javiswan/Projects/clawbound/src/gateway/call.ts"));
        ({ loadSessionEntry } = await import("/Users/javiswan/Projects/clawbound/src/gateway/session-utils.ts"));
        ({ resolveSessionTranscriptPath } = await import("/Users/javiswan/Projects/clawbound/src/config/sessions.ts"));

        gatewayServer = await startGatewayServer(gatewayPort, {
          bind: "loopback",
          host: "127.0.0.1",
          controlUiEnabled: false,
        });

        runs.push(
          await runScenarioWithRetries({
            label: "subagent-host",
            prompt: task,
            nativeMode: false,
          }),
        );
        runs.push(
          await runScenarioWithRetries({
            label: "subagent-native",
            prompt: task,
            nativeMode: true,
          }),
        );
      } catch (error) {
        workflowBootstrapError = error instanceof Error ? error.message : String(error);
        workflowFailure = classifyClawBoundBenchmarkLiveRun({
          attemptCount: 1,
          outputQuality: "",
          actualToolCalls: [],
          workspaceMutated: false,
          fatalErrorMessage: workflowBootstrapError,
        });
      } finally {
        if (gatewayServer) {
          await gatewayServer.close();
          gatewayServer = null;
        }
        restoreLiveEnv?.();
        for (const key of Object.keys(previousEnv)) {
          restoreEnv(key);
        }
        const summaryInputs = workflowFailure ? [...runs, workflowFailure] : runs;
        const meaningfulExecution = runs.some(
          (run) =>
            run.productValid ||
            run.actualToolCalls.parent.length > 0 ||
            run.actualToolCalls.child.length > 0,
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
              runs,
              summary: summarizeClawBoundBenchmarkLiveRuns(summaryInputs),
              workflowFailure,
              workflowBootstrapError,
            },
            null,
            2,
          )}\n`,
          "utf8",
        );
      }

      if (workflowFailure) {
        throw new Error(
          workflowBootstrapError
            ? `${workflowFailure.failureReason ?? "subagent harness bootstrap failed"}\n${workflowBootstrapError}`
            : (workflowFailure.failureReason ?? "subagent harness bootstrap failed"),
        );
      }

      expect(runs).toHaveLength(2);
      expect(runs.every((run) => run.validationOutcome.length > 0)).toBe(true);
      expect(runs.every((run) => run.attemptCount >= 1)).toBe(true);
      const nativeRun = runs[1];
      expect(nativeRun?.validationOutcome === "valid_product_result" || nativeRun?.externallyBlocked === true).toBe(true);
      if (runs.every((run) => run.productValid)) {
        expect(runs[0]?.subagentDelegationOccurred).toBe(true);
        expect(runs[1]?.subagentDelegationOccurred).toBe(true);
        expect(runs[1]?.handoff.inheritsFullTranscriptPrevented).toBe(true);
        expect(runs[1]?.modelBoundaryTools.parent).toEqual(["read", "exec", "sessions_spawn"]);
        expect(runs[1]?.modelBoundaryTools.child).toContain("read");
        expect(runs[1]?.modelBoundaryTools.child).toContain("exec");
        expect(runs[1]?.modelBoundaryTools.child).not.toContain("sessions_spawn");
        expect(runs[1]?.taskStayedWithinDelegatedScope).toBe(true);
      }
    },
    360_000,
  );
});
