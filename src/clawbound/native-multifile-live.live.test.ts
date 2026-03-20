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

type ToolCallDetail = {
  name: string;
  arguments: Record<string, unknown>;
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
  provider: string;
  model: string;
  hostPathUsed: string;
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
  readTargets: string[];
  editTargets: string[];
  execCommands: string[];
  outputQuality: string;
  workspaceMutated: boolean;
  mutatedFiles: string[];
  stayedWithinBoundedFileScope: boolean;
  compactArtifactsInterpretable: boolean;
  compactEventTypes: string[];
  compactOmittedReasons: string[];
  compactBoundednessReasons: string[];
  finalParserTest: {
    exitCode: number | null;
    stdout: string;
    stderr: string;
  };
  persistedArtifactsLocation: {
    contextEngineRun?: string | null;
    compactArtifact?: string | null;
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
const reportPath = path.join(os.tmpdir(), "clawbound-native-multifile-live-report.json");
const TOOL_CALL_TYPES = new Set(["toolcall", "tool_call", "tool_use"]);
const ALLOWED_FILE_SET = new Set(["parser.js", "csv-utils.js", "parser.test.js", "package.json"]);
const MAX_EXTERNAL_RETRIES = 2;

type ActiveLiveModel = {
  provider: string;
  model: string;
};

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

function normalizeToolArguments(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return { raw: value };
    }
    return { raw: value };
  }
  return {};
}

function extractToolCallDetails(messages: Array<Record<string, unknown>>): ToolCallDetail[] {
  const details: ToolCallDetail[] = [];
  for (const message of messages) {
    const content = message.content;
    if (!Array.isArray(content)) {
      continue;
    }
    for (const entry of content) {
      if (!entry || typeof entry !== "object") {
        continue;
      }
      const block = entry as Record<string, unknown>;
      const type =
        typeof block.type === "string" ? block.type.trim().toLowerCase() : "";
      if (!TOOL_CALL_TYPES.has(type)) {
        continue;
      }
      const name = typeof block.name === "string" ? block.name.trim() : "";
      if (!name) {
        continue;
      }
      details.push({
        name,
        arguments: normalizeToolArguments(block.arguments),
      });
    }
  }
  return details;
}

function collectFileTargets(details: ToolCallDetail[], toolName: string): string[] {
  return Array.from(
    new Set(
      details
        .filter((detail) => detail.name === toolName)
        .map((detail) => {
          const raw =
            detail.arguments.path ??
            detail.arguments.file ??
            detail.arguments.target ??
            detail.arguments.pathname;
          return typeof raw === "string" && raw.trim() ? path.basename(raw.trim()) : null;
        })
        .filter((value): value is string => Boolean(value)),
    ),
  );
}

function collectExecCommands(details: ToolCallDetail[]): string[] {
  return Array.from(
    new Set(
      details
        .filter((detail) => detail.name === "exec")
        .map((detail) => detail.arguments.command)
        .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
        .map((value) => value.trim()),
    ),
  );
}

function stayedWithinBoundedScope(readTargets: string[], editTargets: string[], execCommands: string[]) {
  const readsOk = readTargets.every((target) => ALLOWED_FILE_SET.has(target));
  const editsOk = editTargets.every((target) => ALLOWED_FILE_SET.has(target));
  const execOk = execCommands.every(
    (command) =>
      command.includes("node --test parser.test.js") ||
      command.includes("npm run test:parser"),
  );
  return readsOk && editsOk && execOk;
}

vi.mock("@mariozechner/pi-ai", async () => {
  const actual = await vi.importActual<typeof import("@mariozechner/pi-ai")>("@mariozechner/pi-ai");
  return {
    ...actual,
    streamSimple: (model: ModelLike, context: ContextLike, options: Record<string, unknown>) => {
      const label = String(
        (globalThis as { __STEWARD_MULTIFILE_LABEL__?: string }).__STEWARD_MULTIFILE_LABEL__ ??
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
const intendedLiveModel: ActiveLiveModel = resolveClawBoundBenchmarkConfiguredLiveModelBinding();
let activeLiveModel: ActiveLiveModel = { ...intendedLiveModel };
let actualLiveModel: ActiveLiveModel | null = null;

const immediateEnqueue = async <T>(task: () => Promise<T>) => task();

async function seedWorkspace(workspaceDir: string) {
  await ensureAgentWorkspace({
    dir: workspaceDir,
    ensureBootstrapFiles: true,
  });

  await fsp.writeFile(
    path.join(workspaceDir, "parser.js"),
    [
      'import { splitCsvFields } from "./csv-utils.js";',
      "",
      "export function parseCsvLine(input) {",
      "  return splitCsvFields(input);",
      "}",
      "",
    ].join("\n"),
    "utf8",
  );

  await fsp.writeFile(
    path.join(workspaceDir, "csv-utils.js"),
    [
      "export function splitCsvFields(input) {",
      "  return input",
      '    .split(",")',
      "    .map((field) => field.trim())",
      "    .filter(Boolean);",
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
      'test("preserves empty middle fields while trimming whitespace", () => {',
      '  assert.deepEqual(parseCsvLine("alpha, , gamma "), ["alpha", "", "gamma"]);',
      "});",
      "",
      'test("preserves trailing empty fields", () => {',
      '  assert.deepEqual(parseCsvLine("alpha,beta,"), ["alpha", "beta", ""]);',
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
    "# AGENTS.md\n- Keep fixes scoped to parser.js, csv-utils.js, and parser.test.js.\n- Preserve parseCsvLine public API.\n",
    "utf8",
  );

  await fsp.writeFile(
    path.join(workspaceDir, "TOOLS.md"),
    "# TOOLS.md\n- Read the related source and test files before editing.\n- Run only the focused parser test for evidence.\n",
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

async function runFocusedParserTest(workspaceDir: string) {
  return await new Promise<{ exitCode: number | null; stdout: string; stderr: string }>((resolve) => {
    execFile(
      process.execPath,
      ["--test", "parser.test.js"],
      { cwd: workspaceDir },
      (error, stdout, stderr) => {
        resolve({
          exitCode: typeof (error as { code?: unknown } | null)?.code === "number" ? ((error as { code: number }).code ?? null) : 0,
          stdout: stdout.trim(),
          stderr: stderr.trim(),
        });
      },
    );
  });
}

async function runScenario(params: {
  label: string;
  prompt: string;
  nativeMode: boolean;
  attemptIndex?: number;
}) {
  const attemptIndex = params.attemptIndex ?? 1;
  const attemptLabel = `${params.label}-attempt-${attemptIndex}`;
  const runRoot = await fsp.mkdtemp(path.join(tempRoot, `${params.label}-`));
  const agentDir = path.join(runRoot, "agent");
  const workspaceDir = path.join(runRoot, "workspace");
  const sessionFile = path.join(runRoot, "session.jsonl");
  await fsp.mkdir(workspaceDir, { recursive: true });
  await copyMainAuthStore(agentDir);
  await seedWorkspace(workspaceDir);

  const beforeFiles = {
    parser: await fsp.readFile(path.join(workspaceDir, "parser.js"), "utf8"),
    helper: await fsp.readFile(path.join(workspaceDir, "csv-utils.js"), "utf8"),
    test: await fsp.readFile(path.join(workspaceDir, "parser.test.js"), "utf8"),
  };

  const runId = `live-${params.label}`;
  const previous = process.env.STEWARD_CONTEXT_ENGINE_MODE;
  if (params.nativeMode) {
    process.env.STEWARD_CONTEXT_ENGINE_MODE = "1";
  } else {
    delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
  }
  (globalThis as { __STEWARD_MULTIFILE_LABEL__?: string }).__STEWARD_MULTIFILE_LABEL__ = attemptLabel;
  try {
    let result:
      | Awaited<ReturnType<typeof runEmbeddedPiAgent>>
      | null = null;
    let fatalErrorMessage: string | null = null;
    try {
      result = await withClawBoundBenchmarkHarnessTimeout({
        timeoutMs: 70_000,
        timeoutMessage: `live harness timeout after 70000ms`,
        work: () =>
          runEmbeddedPiAgent({
            sessionId: `session:${attemptLabel}`,
            sessionKey: `agent:main:${attemptLabel}`,
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
    } catch (error) {
      fatalErrorMessage = error instanceof Error ? error.message : String(error);
    }

    const afterFiles = {
      parser: await fsp.readFile(path.join(workspaceDir, "parser.js"), "utf8"),
      helper: await fsp.readFile(path.join(workspaceDir, "csv-utils.js"), "utf8"),
      test: await fsp.readFile(path.join(workspaceDir, "parser.test.js"), "utf8"),
    };

    const mutatedFiles = [
      beforeFiles.parser !== afterFiles.parser ? "parser.js" : null,
      beforeFiles.helper !== afterFiles.helper ? "csv-utils.js" : null,
      beforeFiles.test !== afterFiles.test ? "parser.test.js" : null,
    ].filter((value): value is string => Boolean(value));

    const snapshots = observedPrompts.get(attemptLabel) ?? [];
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
    const compactArtifactPath = result?.meta.clawboundContextEngine?.compactPersistedPath ?? null;
    const compactArtifact =
      compactArtifactPath && fs.existsSync(compactArtifactPath)
        ? (JSON.parse(await fsp.readFile(compactArtifactPath, "utf8")) as Record<string, any>)
        : null;
    const sessionMessages = await readSessionMessages(sessionFile);
    const toolCallNames = Array.from(
      new Set(sessionMessages.flatMap((message) => extractToolCallNames(message))),
    );
    const toolCallDetails = extractToolCallDetails(sessionMessages);
    const readTargets = collectFileTargets(toolCallDetails, "read");
    const editTargets = collectFileTargets(toolCallDetails, "edit");
    const execCommands = collectExecCommands(toolCallDetails);
    const finalParserTest = await runFocusedParserTest(workspaceDir);
    const classification = classifyClawBoundBenchmarkLiveRun({
      attemptCount: attemptIndex,
      outputQuality: outputText,
      actualToolCalls: toolCallNames,
      workspaceMutated: mutatedFiles.length > 0,
      finalCheckExitCode: finalParserTest.exitCode,
      fatalErrorMessage,
    });

    return {
      label: params.label,
      task: params.prompt,
      attemptCount: classification.attemptCount,
      failureClassification: classification.failureClassification,
      validationOutcome: classification.validationOutcome,
      productValid: classification.productValid,
      externallyBlocked: classification.externallyBlocked,
      retryable: classification.retryable,
      failureReason: classification.failureReason,
      provider: activeLiveModel.provider,
      model: activeLiveModel.model,
      hostPathUsed: params.nativeMode
        ? "embedded-runner / clawbound-bounded-context-engine"
        : "embedded-runner / default-host-path",
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
      readTargets,
      editTargets,
      execCommands,
      outputQuality: outputText,
      workspaceMutated: mutatedFiles.length > 0,
      mutatedFiles,
      stayedWithinBoundedFileScope: stayedWithinBoundedScope(readTargets, editTargets, execCommands),
      compactArtifactsInterpretable: Boolean(
        compactArtifact &&
          Array.isArray(compactArtifact?.traceSummary?.eventTypes) &&
          Array.isArray(compactArtifact?.omitted?.reasons) &&
          Array.isArray(compactArtifact?.boundedness?.reasons),
      ),
      compactEventTypes: compactArtifact?.traceSummary?.eventTypes ?? [],
      compactOmittedReasons: compactArtifact?.omitted?.reasons ?? [],
      compactBoundednessReasons: compactArtifact?.boundedness?.reasons ?? [],
      finalParserTest,
      persistedArtifactsLocation: {
        contextEngineRun: nativeArtifactPath,
        compactArtifact: compactArtifactPath,
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
    delete (globalThis as { __STEWARD_MULTIFILE_LABEL__?: string }).__STEWARD_MULTIFILE_LABEL__;
  }
}

async function runScenarioWithRetries(params: {
  label: string;
  prompt: string;
  nativeMode: boolean;
}) {
  let lastRun: RunSummary | null = null;
  for (let attemptIndex = 1; attemptIndex <= MAX_EXTERNAL_RETRIES; attemptIndex += 1) {
    const run = await runScenario({
      ...params,
      attemptIndex,
    });
    if (run.productValid || !run.retryable) {
      return run;
    }
    lastRun = run;
    if (attemptIndex < MAX_EXTERNAL_RETRIES) {
      await new Promise((resolve) => setTimeout(resolve, 1_500 * attemptIndex));
    }
  }
  return lastRun as RunSummary;
}

beforeAll(async () => {
  vi.useRealTimers();
  tempRoot = await fsp.mkdtemp(path.join(os.tmpdir(), "clawbound-native-multifile-live-"));
});

afterAll(async () => {
  if (tempRoot) {
    await fsp.rm(tempRoot, { recursive: true, force: true });
  }
});

describeLive("ClawBound bounded multi-file live", () => {
  it(
    "runs one small multi-file workflow on default and ClawBound-native paths and writes a validation report",
    async () => {
      await fsp.rm(reportPath, { force: true });
      const prompt =
        "Fix parseCsvLine so parser.test.js passes without changing public API. Keep scope to parser.js, csv-utils.js, and parser.test.js. Read the relevant files first and run only the focused parser test before finalizing.";

      const runs: RunSummary[] = [];
      let workflowFailure: ClawBoundBenchmarkLiveRunClassification | null = null;
      let restoreEnv: (() => void) | null = null;
      try {
        const liveEnvironment = await prepareClawBoundBenchmarkLiveEnvironment({
          tempRoot,
        });
        restoreEnv = liveEnvironment.restoreEnv;
        activeLiveModel = liveEnvironment.activeLiveModel;
        actualLiveModel = liveEnvironment.activeLiveModel;
        config = liveEnvironment.config;
        ({ runEmbeddedPiAgent } = await import("/Users/javiswan/Projects/clawbound/src/agents/pi-embedded-runner.ts"));
        ({ ensureAgentWorkspace } = await import("/Users/javiswan/Projects/clawbound/src/agents/workspace.ts"));

        runs.push(
          await runScenarioWithRetries({
            label: "multifile-host",
            prompt,
            nativeMode: false,
          }),
        );
        runs.push(
          await runScenarioWithRetries({
            label: "multifile-native",
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
        const summaryInputs = workflowFailure ? [...runs, workflowFailure] : runs;
        const meaningfulExecution = runs.some(
          (run) => run.productValid || run.actualToolCalls.length > 0,
        );
        const report = {
          createdAt: new Date().toISOString(),
          provider: (actualLiveModel ?? intendedLiveModel).provider,
          model: (actualLiveModel ?? intendedLiveModel).model,
          modelBinding: summarizeClawBoundBenchmarkLiveModelBinding({
            intended: intendedLiveModel,
            actual: actualLiveModel,
            meaningfulExecution,
          }),
          summary: summarizeClawBoundBenchmarkLiveRuns(summaryInputs),
          runs,
          workflowFailure,
        };
        await fsp.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
        restoreEnv?.();
      }

      if (workflowFailure) {
        throw new Error(workflowFailure.failureReason ?? "multifile harness bootstrap failed");
      }

      expect(runs).toHaveLength(2);
      expect(runs.every((run) => run.attemptCount >= 1)).toBe(true);
      expect(runs.every((run) => run.validationOutcome.length > 0)).toBe(true);
      expect(runs.find((run) => run.label === "multifile-native")?.promptInflationIndicators.injectedWorkspaceFilesCount).toBe(0);
      expect(runs.find((run) => run.label === "multifile-native")?.promptInflationIndicators.skillsPromptChars).toBe(0);
      expect(runs.find((run) => run.label === "multifile-native")?.routePlan?.executionMode).toBe("executor_then_reviewer");
      expect(runs.find((run) => run.label === "multifile-native")?.modelBoundaryTools).toEqual([
        "read",
        "edit",
        "exec",
      ]);
      expect(runs.find((run) => run.label === "multifile-native")?.stayedWithinBoundedFileScope).toBe(true);
      expect(runs.find((run) => run.label === "multifile-native")?.compactArtifactsInterpretable).toBe(true);
      const nativeRun = runs.find((run) => run.label === "multifile-native");
      expect(nativeRun?.validationOutcome === "valid_product_result" || nativeRun?.externallyBlocked === true).toBe(true);
    },
    420_000,
  );
});
