import fs from "node:fs/promises";
import path from "node:path";
import { isSubagentSessionKey } from "../../sessions/session-key-utils.js";
import type { ClawBoundLocalContextItem } from "../runtime.js";
import {
  type ClawBoundContextSessionState,
  type ClawBoundContextCompactResult,
  type ClawBoundContextAssembleResult,
  type SparseClawBoundContextEngineOptions,
  SparseClawBoundContextEngine,
} from "./engine.js";

export type EmbeddedClawBoundContextEngineInput = {
  env?: NodeJS.ProcessEnv;
  workspaceDir: string;
  hostRunId: string;
  sessionId: string;
  sessionKey?: string;
  prompt: string;
  candidateTools?: string[];
  localContext?: ClawBoundLocalContextItem[];
  continuationOf?: string;
};

export type EmbeddedClawBoundContextEngineResult = ClawBoundContextAssembleResult & {
  compactSummary: ClawBoundContextCompactResult;
  compactPersistedPath: string;
};

type NamedTool = { name: string };
type ExecutableTool = NamedTool & {
  execute?: (...args: unknown[]) => Promise<unknown>;
};
type PromptMode = "full" | "minimal" | "none";
type ClawBoundWrappedToolResult = {
  content?: Array<{ type?: string; text?: string }>;
  details?: Record<string, unknown>;
};

type PackageJsonShape = {
  scripts?: Record<string, unknown>;
};

type FocusedTestExecClassification =
  | {
      kind: "focused_test";
      command: string;
      target: string | null;
      packageScript: string | null;
    }
  | {
      kind: "other";
      command: string;
    };

type FocusedTestExecSummary = {
  kind: "focused_test";
  command: string;
  packageScript: string | null;
  exitCode: number | null;
  status: "passed" | "failed" | "unknown";
  counts: {
    tests: number | null;
    pass: number | null;
    fail: number | null;
  };
  failingTests: string[];
};

const STEWARD_HANDOFF_TEXT_KEY = "__clawbound_handoff_text";
const STEWARD_HANDOFF_ARTIFACT_PATH_KEY = "__clawbound_handoff_artifact_path";
const FOCUSED_NODE_TEST_PATTERN = /\.(?:test|spec)\.[cm]?[jt]sx?$/i;

const STEWARD_TOOL_TO_HOST_TOOL_ALIASES: Record<string, string[]> = {
  read_file: ["read"],
  edit_file: ["edit"],
  write_file: ["write"],
  run_command: ["exec"],
  run_tests: ["exec"],
};

export function isClawBoundContextEngineModeEnabled(env: NodeJS.ProcessEnv = process.env) {
  const raw =
    env.STEWARD_CONTEXT_ENGINE_MODE ??
    env.OPENCLAW_STEWARD_CONTEXT_ENGINE_MODE ??
    env.OPENCLAW_STEWARD_CONTEXT_ENGINE ??
    "";
  const normalized = raw.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

export function expandClawBoundToolNamesToHostToolNames(toolNames: string[]) {
  const expanded = new Set<string>();
  for (const toolName of toolNames) {
    const normalized = toolName.trim();
    if (!normalized) {
      continue;
    }
    expanded.add(normalized);
    for (const alias of STEWARD_TOOL_TO_HOST_TOOL_ALIASES[normalized] ?? []) {
      expanded.add(alias);
    }
  }
  return [...expanded];
}

export async function planClawBoundContextEngineForEmbeddedAttempt(
  input: EmbeddedClawBoundContextEngineInput,
  options?: Pick<SparseClawBoundContextEngineOptions, "idFactory">,
): Promise<EmbeddedClawBoundContextEngineResult | null> {
  if (!isClawBoundContextEngineModeEnabled(input.env)) {
    return null;
  }

  const rootDir = path.join(input.workspaceDir, ".clawbound", "context-engine");
  const engine = new SparseClawBoundContextEngine({
    rootDir,
    idFactory: options?.idFactory,
  });

  let state = await engine.bootstrap(
    {
      hostRunId: input.hostRunId,
      sessionId: input.sessionId,
      sessionKey: input.sessionKey,
      conversationId: input.sessionKey ?? input.sessionId,
      sourceHost: "clawbound-context/openclaw-embedded-runner",
      emitToHostEvents: true,
    },
    {
      userInput: input.prompt,
      continuationOf: input.continuationOf,
    },
  );

  const localContextEvents = (input.localContext ?? []).map((item) => ({
    type: "local_context" as const,
    item,
  }));
  if (localContextEvents.length > 0) {
    state = await engine.ingest({
      state,
      events: localContextEvents,
    });
  }

  const assembled = await engine.assemble({
    state,
    candidateTools: input.candidateTools,
  });
  const compactSummary = await engine.compact(assembled.state);
  const compactPersistedPath = await persistClawBoundCompactArtifact({
    rootDir,
    runId: assembled.runId,
    compactSummary,
  });
  return {
    ...assembled,
    compactSummary,
    compactPersistedPath,
  };
}

export async function wrapEmbeddedClawBoundSubagentTools<TTool extends ExecutableTool>(params: {
  contextEngineState: ClawBoundContextSessionState;
  rootDir: string;
  workspaceDir: string;
  tools: TTool[];
}): Promise<TTool[]> {
  const engine = new SparseClawBoundContextEngine({
    rootDir: params.rootDir,
  });

  return await Promise.all(
    params.tools.map(async (tool) => {
      if (tool.name === "exec" && typeof tool.execute === "function") {
        return {
          ...tool,
          execute: async (...args: unknown[]) => {
            const [toolCallId, rawArgs, signal, onUpdate] = args;
            const paramsRecord =
              rawArgs && typeof rawArgs === "object" ? ({ ...rawArgs } as Record<string, unknown>) : {};
            const command =
              typeof paramsRecord.command === "string" ? paramsRecord.command : "";
            const resolvedWorkspaceDir = resolveClawBoundExecWorkspaceDir(
              params.workspaceDir,
              typeof paramsRecord.workdir === "string" ? paramsRecord.workdir : undefined,
            );
            const normalizedCommand = await normalizeClawBoundExecCommandForWorkspace({
              command,
              workspaceDir: resolvedWorkspaceDir,
            });
            const rawResult = (await tool.execute?.(
              toolCallId,
              normalizedCommand && normalizedCommand !== command
                ? {
                    ...paramsRecord,
                    command: normalizedCommand,
                  }
                : rawArgs,
              signal,
              onUpdate,
            )) as ClawBoundWrappedToolResult | undefined;

            return await maybeShapeClawBoundExecResultForWorkspace({
              command: normalizedCommand || command,
              workspaceDir: resolvedWorkspaceDir,
              result: rawResult,
            });
          },
        } satisfies TTool;
      }

      if (tool.name !== "sessions_spawn" || typeof tool.execute !== "function") {
        return tool;
      }

      return {
        ...tool,
        execute: async (...args: unknown[]) => {
          const [toolCallId, rawArgs, signal, onUpdate] = args;
          const paramsRecord =
            rawArgs && typeof rawArgs === "object" ? ({ ...rawArgs } as Record<string, unknown>) : {};
          const task =
            typeof paramsRecord.task === "string" ? paramsRecord.task.trim() : "";
          if (!task) {
            return await tool.execute?.(toolCallId, rawArgs, signal, onUpdate);
          }

          const handoff = await engine.prepareSubagentContext(params.contextEngineState, {
            userInput: task,
          });
          const artifactPath = await persistClawBoundSubagentHandoffArtifact({
            rootDir: params.rootDir,
            parentState: params.contextEngineState,
            toolCallId: typeof toolCallId === "string" ? toolCallId : "subagent-handoff",
            task,
            handoff,
          });
          const result = (await tool.execute?.(
            toolCallId,
            {
              ...paramsRecord,
              [STEWARD_HANDOFF_TEXT_KEY]: handoff.text,
              [STEWARD_HANDOFF_ARTIFACT_PATH_KEY]: artifactPath,
            },
            signal,
            onUpdate,
          )) as ClawBoundWrappedToolResult | undefined;
          const clawboundDetails = {
            handoffArtifactPath: artifactPath,
            inheritsFullTranscript: handoff.inheritsFullTranscript,
            executionMode: handoff.executionMode,
            noLoad: handoff.noLoad,
            localContextRefs: handoff.localContext.map((item) => item.ref),
            retrievedUnitIds: handoff.retrievedUnits.map((unit) => unit.id),
          };

          if (!result || typeof result !== "object") {
            return result;
          }

          const nextDetails = {
            ...(result.details ?? {}),
            clawbound: clawboundDetails,
          };
          return {
            ...result,
            details: nextDetails,
            content: [
              {
                type: "text",
                text: JSON.stringify(nextDetails, null, 2),
              },
            ],
          } satisfies ClawBoundWrappedToolResult;
        },
      } satisfies TTool;
    }),
  );
}

function resolveClawBoundExecWorkspaceDir(workspaceDir: string, workdir?: string) {
  if (!workdir || !workdir.trim()) {
    return workspaceDir;
  }
  return path.isAbsolute(workdir) ? workdir : path.resolve(workspaceDir, workdir);
}

function normalizeCommandTarget(rawTarget: string) {
  const trimmed = rawTarget.trim();
  if (!trimmed || /[|&;<>`$()]/.test(trimmed)) {
    return null;
  }
  const dealiased = trimmed.startsWith("--") && trimmed.length > 2 ? trimmed.slice(2) : trimmed;
  const unquoted = dealiased.replace(/^['"]|['"]$/g, "");
  if (!unquoted || /\s/.test(unquoted) || unquoted.startsWith("-")) {
    return null;
  }
  return unquoted;
}

function findFocusedTestScript(
  scripts: Record<string, unknown>,
  target: string,
) {
  const normalizedTargets = new Set([target, target.replace(/^\.\//, ""), `./${target.replace(/^\.\//, "")}`]);
  for (const [scriptName, scriptValue] of Object.entries(scripts)) {
    if (scriptName === "test" || typeof scriptValue !== "string") {
      continue;
    }
    const match = /^node\s+--test\s+(.+)$/.exec(scriptValue.trim());
    if (!match) {
      continue;
    }
    const scriptTarget = normalizeCommandTarget(match[1] ?? "");
    if (!scriptTarget) {
      continue;
    }
    if (normalizedTargets.has(scriptTarget)) {
      return scriptName;
    }
  }
  return null;
}

function formatPackageScriptCommand(packageManager: string, scriptName: string) {
  if (packageManager === "yarn") {
    return `yarn ${scriptName}`;
  }
  return `${packageManager} run ${scriptName}`;
}

async function readPackageScriptsForWorkspace(workspaceDir: string) {
  const packageJsonPath = path.join(workspaceDir, "package.json");
  try {
    const packageJson = JSON.parse(await fs.readFile(packageJsonPath, "utf8")) as PackageJsonShape;
    if (!packageJson?.scripts || typeof packageJson.scripts !== "object") {
      return null;
    }
    return packageJson.scripts;
  } catch {
    return null;
  }
}

export async function normalizeClawBoundExecCommandForWorkspace(params: {
  command: string;
  workspaceDir: string;
}) {
  const trimmed = params.command.trim();
  if (!trimmed) {
    return trimmed;
  }

  const match = /^(npm|pnpm|yarn)\s+test\s+--\s+(.+)$/.exec(trimmed);
  if (!match) {
    return trimmed;
  }

  const packageManager = match[1] ?? "npm";
  const target = normalizeCommandTarget(match[2] ?? "");
  if (!target) {
    return trimmed;
  }

  const scripts = await readPackageScriptsForWorkspace(params.workspaceDir);
  if (!scripts) {
    return trimmed;
  }

  if (typeof scripts.test === "string" && scripts.test.trim().length > 0) {
    return trimmed;
  }

  const focusedScript = findFocusedTestScript(scripts, target);
  if (focusedScript) {
    return formatPackageScriptCommand(packageManager, focusedScript);
  }

  if (FOCUSED_NODE_TEST_PATTERN.test(target)) {
    return `node --test ${target}`;
  }

  return trimmed;
}

async function classifyClawBoundExecCommandForWorkspace(params: {
  command: string;
  workspaceDir: string;
}): Promise<FocusedTestExecClassification> {
  const trimmed = params.command.trim();
  if (!trimmed) {
    return { kind: "other", command: trimmed };
  }

  const nodeTestMatch = /^node\s+--test\s+(.+)$/.exec(trimmed);
  if (nodeTestMatch) {
    const target = normalizeCommandTarget(nodeTestMatch[1] ?? "");
    if (target && FOCUSED_NODE_TEST_PATTERN.test(target)) {
      return {
        kind: "focused_test",
        command: trimmed,
        target,
        packageScript: null,
      };
    }
    return { kind: "other", command: trimmed };
  }

  const scriptMatch = /^(npm|pnpm)\s+run\s+([^\s]+)$/.exec(trimmed) ?? /^yarn\s+([^\s]+)$/.exec(trimmed);
  if (!scriptMatch) {
    return { kind: "other", command: trimmed };
  }

  const scriptName = scriptMatch[2] ?? "";
  if (!scriptName) {
    return { kind: "other", command: trimmed };
  }

  const scripts = await readPackageScriptsForWorkspace(params.workspaceDir);
  const scriptValue = scripts?.[scriptName];
  if (typeof scriptValue !== "string") {
    return { kind: "other", command: trimmed };
  }

  const scriptNodeTestMatch = /^node\s+--test\s+(.+)$/.exec(scriptValue.trim());
  if (!scriptNodeTestMatch) {
    return { kind: "other", command: trimmed };
  }

  const target = normalizeCommandTarget(scriptNodeTestMatch[1] ?? "");
  if (!target || !FOCUSED_NODE_TEST_PATTERN.test(target)) {
    return { kind: "other", command: trimmed };
  }

  return {
    kind: "focused_test",
    command: trimmed,
    target,
    packageScript: scriptName,
  };
}

function extractFocusedTestCounts(aggregated: string) {
  const readCount = (label: string) => {
    const match = new RegExp(`(?:^|\\n)[#ℹ]\\s*${label}\\s+(\\d+)`, "i").exec(aggregated);
    return match ? Number.parseInt(match[1] ?? "", 10) : null;
  };
  return {
    tests: readCount("tests"),
    pass: readCount("pass"),
    fail: readCount("fail"),
  };
}

function extractFocusedFailingTests(aggregated: string) {
  return [...new Set(
    aggregated
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.startsWith("✖ "))
      .map((line) => line.replace(/^✖\s+/, "").trim())
      .filter((line) => line.length > 0 && line.toLowerCase() !== "failing tests:")
      .slice(0, 3),
  )];
}

function buildFocusedTestExecSummary(params: {
  classification: Extract<FocusedTestExecClassification, { kind: "focused_test" }>;
  result: ClawBoundWrappedToolResult;
}) {
  const details =
    params.result.details && typeof params.result.details === "object"
      ? params.result.details
      : {};
  const aggregated =
    typeof details.aggregated === "string"
      ? details.aggregated
      : params.result.content
            ?.map((entry) => (typeof entry?.text === "string" ? entry.text : ""))
            .filter(Boolean)
            .join("\n") ?? "";
  const exitCode = typeof details.exitCode === "number" ? details.exitCode : null;
  const counts = extractFocusedTestCounts(aggregated);
  const failingTests = extractFocusedFailingTests(aggregated);
  const status: FocusedTestExecSummary["status"] =
    counts.fail && counts.fail > 0
      ? "failed"
      : exitCode != null && exitCode !== 0
        ? "failed"
        : counts.pass && counts.pass > 0
          ? "passed"
          : exitCode === 0
            ? "passed"
            : "unknown";
  const summary: FocusedTestExecSummary = {
    kind: "focused_test",
    command: params.classification.command,
    packageScript: params.classification.packageScript,
    exitCode,
    status,
    counts,
    failingTests,
  };

  const summaryLines = [
    `Focused test result: ${summary.status}${summary.exitCode == null ? "" : ` (exitCode=${summary.exitCode})`}`,
    `Command: ${summary.command}`,
  ];
  if (summary.counts.tests != null || summary.counts.pass != null || summary.counts.fail != null) {
    summaryLines.push(
      `Counts: tests=${summary.counts.tests ?? "?"} pass=${summary.counts.pass ?? "?"} fail=${summary.counts.fail ?? "?"}`,
    );
  }
  if (summary.failingTests.length > 0) {
    summaryLines.push("Failing tests:");
    for (const failingTest of summary.failingTests) {
      summaryLines.push(`- ${failingTest}`);
    }
  }
  if (summary.counts.tests == null && summary.failingTests.length === 0) {
    const firstSignal = aggregated
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find(Boolean);
    if (firstSignal) {
      summaryLines.push(`First signal: ${firstSignal}`);
    }
  }

  return {
    summary,
    text: summaryLines.join("\n"),
  };
}

async function maybeShapeClawBoundExecResultForWorkspace(params: {
  command: string;
  workspaceDir: string;
  result: ClawBoundWrappedToolResult | undefined;
}) {
  if (!params.result || typeof params.result !== "object") {
    return params.result;
  }

  const classification = await classifyClawBoundExecCommandForWorkspace({
    command: params.command,
    workspaceDir: params.workspaceDir,
  });
  if (classification.kind !== "focused_test") {
    return params.result;
  }

  const { summary, text } = buildFocusedTestExecSummary({
    classification,
    result: params.result,
  });
  return {
    ...params.result,
    details: {
      ...(params.result.details ?? {}),
      clawboundExec: summary,
    },
    content: [
      {
        type: "text",
        text,
      },
    ],
  } satisfies ClawBoundWrappedToolResult;
}

export function applyEmbeddedClawBoundContextEngineOverrides<TTool extends NamedTool, TContextFile>(
  params: {
    contextEngine:
      | null
      | {
          promptText: string;
          routePlan: { executionMode: string };
          retrievalPlan: { noLoad: boolean };
          toolProfile: {
            profile: string;
            allowedTools: string[];
            deniedTools: string[];
            notes: string[];
          };
        };
    sessionKey?: string;
    promptMode: PromptMode;
    extraSystemPrompt?: string;
    skillsPrompt?: string;
    contextFiles: TContextFile[];
    workspaceNotes?: string[];
    tools: TTool[];
  },
) {
  if (!params.contextEngine) {
    return {
      promptMode: params.promptMode,
      extraSystemPrompt: params.extraSystemPrompt,
      skillsPrompt: params.skillsPrompt,
      contextFiles: params.contextFiles,
      workspaceNotes: params.workspaceNotes,
      tools: params.tools,
    };
  }

  const allowedToolNames = expandClawBoundToolNamesToHostToolNames(
    params.contextEngine.toolProfile.allowedTools,
  ).filter((toolName) =>
    !(isSubagentSessionKey(params.sessionKey) && toolName.trim().toLowerCase() === "sessions_spawn"),
  );
  const allowedTools = new Set(allowedToolNames);
  return {
    promptMode: "minimal" as const,
    extraSystemPrompt: [params.extraSystemPrompt, params.contextEngine.promptText]
      .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
      .join("\n\n"),
    skillsPrompt: undefined,
    contextFiles: [] as TContextFile[],
    workspaceNotes: undefined,
    tools: params.tools.filter((tool) => allowedTools.has(tool.name)),
  };
}

async function persistClawBoundSubagentHandoffArtifact(params: {
  rootDir: string;
  parentState: ClawBoundContextSessionState;
  toolCallId: string;
  task: string;
  handoff: Awaited<ReturnType<SparseClawBoundContextEngine["prepareSubagentContext"]>>;
}) {
  const handoffDir = path.join(params.rootDir, "handoffs");
  await fs.mkdir(handoffDir, { recursive: true });
  const artifactPath = path.join(
    handoffDir,
    `${params.parentState.runId}-${params.toolCallId.replace(/[^a-z0-9_-]+/gi, "-")}.json`,
  );
  await fs.writeFile(
    artifactPath,
    `${JSON.stringify(
      {
        parentRunId: params.parentState.runId,
        parentTraceId: params.parentState.traceId,
        task: params.task,
        inheritsFullTranscript: params.handoff.inheritsFullTranscript,
        executionMode: params.handoff.executionMode,
        noLoad: params.handoff.noLoad,
        localContextRefs: params.handoff.localContext.map((item) => item.ref),
        retrievedUnitIds: params.handoff.retrievedUnits.map((unit) => unit.id),
        text: params.handoff.text,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  return artifactPath;
}

async function persistClawBoundCompactArtifact(params: {
  rootDir: string;
  runId: string;
  compactSummary: ClawBoundContextCompactResult;
}) {
  const compactDir = path.join(params.rootDir, "compacts");
  await fs.mkdir(compactDir, { recursive: true });
  const artifactPath = path.join(compactDir, `${params.runId}.json`);
  await fs.writeFile(
    artifactPath,
    `${JSON.stringify(params.compactSummary, null, 2)}\n`,
    "utf8",
  );
  return artifactPath;
}
