import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { resolveOpenClawAgentDir } from "../../agents/agent-paths.js";
import type { OpenClawConfig } from "../../config/config.js";

export type ClawBoundBenchmarkLiveFailureClassification =
  | "provider_connection"
  | "provider_timeout"
  | "auth_failure"
  | "harness_failure"
  | "runtime_failure";

export type ClawBoundBenchmarkLiveValidationOutcome =
  | "valid_product_result"
  | ClawBoundBenchmarkLiveFailureClassification;

export type ClawBoundBenchmarkLiveRunClassificationInput = {
  attemptCount: number;
  outputQuality?: string | null;
  actualToolCalls?: string[];
  workspaceMutated?: boolean;
  finalCheckExitCode?: number | null;
  fatalErrorMessage?: string | null;
};

export type ClawBoundBenchmarkLiveRunClassification = {
  attemptCount: number;
  validationOutcome: ClawBoundBenchmarkLiveValidationOutcome;
  failureClassification: ClawBoundBenchmarkLiveFailureClassification | null;
  productValid: boolean;
  externallyBlocked: boolean;
  retryable: boolean;
  failureReason: string | null;
};

export type ClawBoundBenchmarkLiveWorkflowSummary = {
  totalRuns: number;
  validProductResults: number;
  externallyBlockedRuns: number;
  providerConnections: number;
  providerTimeouts: number;
  authFailures: number;
  harnessFailures: number;
  runtimeFailures: number;
};

export type ClawBoundBenchmarkLiveModelBinding = {
  provider: string;
  model: string;
};

export type ClawBoundBenchmarkLiveModelBindingTelemetry = {
  intended: ClawBoundBenchmarkLiveModelBinding;
  actual: ClawBoundBenchmarkLiveModelBinding | null;
  diverged: boolean;
  divergenceStage: "none" | "before_meaningful_execution" | "after_meaningful_execution";
};

export type ClawBoundBenchmarkLiveEnvironment = {
  activeLiveModel: ClawBoundBenchmarkLiveModelBinding;
  config: OpenClawConfig;
  configPath: string;
  stateDir: string;
  agentDir: string;
  restoreEnv: () => void;
};

export const DEFAULT_STEWARD_BENCHMARK_LIVE_PROVIDER = "minimax";
export const DEFAULT_STEWARD_BENCHMARK_LIVE_MODEL = "MiniMax-M2.5";

const mainAgentDir = resolveOpenClawAgentDir();
const mainAuthStore = path.join(mainAgentDir, "auth-profiles.json");
const mainModelsStore = path.join(mainAgentDir, "models.json");

const AUTH_FAILURE_PATTERN =
  /\b(?:401|403)\b|authentication_error|invalid x-api-key|api\.responses\.write|unauthorized|forbidden/i;
const PROVIDER_CONNECTION_PATTERN =
  /\bconnection error\b|fetch failed|econnreset|socket hang up|network error|tls|service unavailable/i;
const PROVIDER_TIMEOUT_PATTERN = /\btimeout\b|timed out|deadline exceeded/i;
const HARNESS_FAILURE_PATTERN =
  /listen eperm|operation not permitted 127\.0\.0\.1|eaddrinuse|address already in use|missing ui runner|invalid config at .*openclaw\.json|enoent.*session\.jsonl|session\.jsonl/i;

function normalizeText(value: string | null | undefined) {
  return (value ?? "").trim();
}

export function resolveClawBoundBenchmarkConfiguredLiveModelBinding(): ClawBoundBenchmarkLiveModelBinding {
  const configPath = path.join(os.homedir(), ".openclaw", "openclaw.json");
  if (!fs.existsSync(configPath)) {
    return resolveClawBoundBenchmarkLiveModelBinding({});
  }

  try {
    const rawConfig = JSON.parse(fs.readFileSync(configPath, "utf8")) as Record<string, any>;
    return resolveClawBoundBenchmarkLiveModelBinding(rawConfig);
  } catch {
    return resolveClawBoundBenchmarkLiveModelBinding({});
  }
}

export function resolveClawBoundBenchmarkLiveModelBinding(
  rawConfig: Record<string, any>,
): ClawBoundBenchmarkLiveModelBinding {
  const preferred =
    process.env.STEWARD_LIVE_MODEL_PATH ??
    rawConfig?.agents?.defaults?.model?.primary ??
    `${DEFAULT_STEWARD_BENCHMARK_LIVE_PROVIDER}/${DEFAULT_STEWARD_BENCHMARK_LIVE_MODEL}`;
  const [provider, ...modelParts] = String(preferred).split("/");
  const model = modelParts.join("/");
  if (!provider || !model) {
    return {
      provider: DEFAULT_STEWARD_BENCHMARK_LIVE_PROVIDER,
      model: DEFAULT_STEWARD_BENCHMARK_LIVE_MODEL,
    };
  }
  return { provider, model };
}

export function summarizeClawBoundBenchmarkLiveModelBinding(params: {
  intended: ClawBoundBenchmarkLiveModelBinding;
  actual?: ClawBoundBenchmarkLiveModelBinding | null;
  meaningfulExecution: boolean;
}): ClawBoundBenchmarkLiveModelBindingTelemetry {
  const actual = params.actual ?? null;
  const diverged =
    actual !== null &&
    (actual.provider !== params.intended.provider || actual.model !== params.intended.model);

  if (!diverged) {
    return {
      intended: params.intended,
      actual,
      diverged: false,
      divergenceStage: "none",
    };
  }

  return {
    intended: params.intended,
    actual,
    diverged: true,
    divergenceStage: params.meaningfulExecution
      ? "after_meaningful_execution"
      : "before_meaningful_execution",
  };
}

export async function readClawBoundBenchmarkSessionMessages(
  sessionFile: string,
): Promise<Array<Record<string, unknown>>> {
  if (!fs.existsSync(sessionFile)) {
    return [];
  }
  const raw = await fsp.readFile(sessionFile, "utf8");
  return raw
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as { type?: string; message?: Record<string, unknown> })
    .filter((entry) => entry.type === "message")
    .map((entry) => entry.message ?? {});
}

export async function withClawBoundBenchmarkHarnessTimeout<T>(params: {
  timeoutMs: number;
  timeoutMessage: string;
  work: () => Promise<T>;
}): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const pending = params.work();
  pending.catch(() => {
    // Prevent late rejections from surfacing after the timeout race settles.
  });
  try {
    return await Promise.race([
      pending,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => {
          reject(new Error(params.timeoutMessage));
        }, params.timeoutMs);
      }),
    ]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

export async function prepareClawBoundBenchmarkLiveEnvironment(params: {
  tempRoot: string;
  workspaceDir?: string;
  gatewayPort?: number;
  gatewayToken?: string;
  includeSessionConfig?: boolean;
  includeSubagentModel?: boolean;
  controlUiEnabled?: boolean;
}): Promise<ClawBoundBenchmarkLiveEnvironment> {
  const rawConfig = JSON.parse(
    await fsp.readFile(path.join(os.homedir(), ".openclaw", "openclaw.json"), "utf8"),
  ) as Record<string, any>;
  const rawAgentModels = fs.existsSync(mainModelsStore)
    ? (JSON.parse(await fsp.readFile(mainModelsStore, "utf8")) as Record<string, any>)
    : {};
  const activeLiveModel = resolveClawBoundBenchmarkLiveModelBinding(rawConfig);
  const primaryModel = `${activeLiveModel.provider}/${activeLiveModel.model}`;
  const workspaceDir = params.workspaceDir ?? path.join(params.tempRoot, "workspace-main");
  const models = {
    mode: rawConfig.models?.mode ?? "merge",
    providers: {
      ...(rawConfig.models?.providers ?? {}),
      ...(rawAgentModels.providers ?? {}),
    },
  };
  const config: OpenClawConfig = {
    auth: rawConfig.auth,
    models,
  };

  const stateDir = path.join(params.tempRoot, "openclaw-state");
  const configPath = path.join(stateDir, "openclaw.json");
  const agentDir = path.join(stateDir, "agents", "main", "agent");
  await fsp.mkdir(agentDir, { recursive: true });
  if (fs.existsSync(mainAuthStore)) {
    await fsp.copyFile(mainAuthStore, path.join(agentDir, "auth-profiles.json"));
  }
  if (fs.existsSync(mainModelsStore)) {
    await fsp.copyFile(mainModelsStore, path.join(agentDir, "models.json"));
  }

  const isolatedConfig: OpenClawConfig = {
    auth: rawConfig.auth,
    models: {
      mode: models.mode,
    },
    agents: {
      defaults: {
        workspace: workspaceDir,
        model: {
          primary: primaryModel,
        },
        ...(params.includeSubagentModel
          ? {
              subagents: {
                model: primaryModel,
              },
            }
          : {}),
      },
      list: [
        {
          id: "main",
          default: true,
          workspace: workspaceDir,
          model: {
            primary: primaryModel,
          },
          ...(params.includeSubagentModel
            ? {
                subagents: {
                  model: primaryModel,
                },
              }
            : {}),
        },
      ],
    },
    gateway: {
      bind: "loopback",
      ...(typeof params.gatewayPort === "number" ? { port: params.gatewayPort } : {}),
      controlUi: {
        enabled: params.controlUiEnabled ?? false,
      },
      ...(params.gatewayToken
        ? {
            auth: {
              mode: "token",
              token: params.gatewayToken,
            },
          }
        : {}),
    },
    ...(params.includeSessionConfig
      ? {
          session: {
            mainKey: "main",
            scope: "per-sender",
          },
        }
      : {}),
  };

  await fsp.mkdir(stateDir, { recursive: true });
  await fsp.writeFile(configPath, `${JSON.stringify(isolatedConfig, null, 2)}\n`, "utf8");

  const previousEnv = new Map<string, string | undefined>();
  const envKeys = [
    "OPENCLAW_CONFIG_PATH",
    "OPENCLAW_STATE_DIR",
    "OPENCLAW_AGENT_DIR",
    "PI_CODING_AGENT_DIR",
    "OPENCLAW_GATEWAY_PORT",
    "OPENCLAW_GATEWAY_TOKEN",
  ];
  for (const key of envKeys) {
    previousEnv.set(key, process.env[key]);
  }

  process.env.OPENCLAW_CONFIG_PATH = configPath;
  process.env.OPENCLAW_STATE_DIR = stateDir;
  process.env.OPENCLAW_AGENT_DIR = agentDir;
  process.env.PI_CODING_AGENT_DIR = agentDir;
  if (typeof params.gatewayPort === "number") {
    process.env.OPENCLAW_GATEWAY_PORT = String(params.gatewayPort);
  }
  if (params.gatewayToken) {
    process.env.OPENCLAW_GATEWAY_TOKEN = params.gatewayToken;
  }

  return {
    activeLiveModel,
    config,
    configPath,
    stateDir,
    agentDir,
    restoreEnv: () => {
      for (const key of envKeys) {
        const previous = previousEnv.get(key);
        if (previous === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = previous;
        }
      }
    },
  };
}

function inferMeaningfulExecution(params: {
  outputQuality: string;
  actualToolCalls: string[];
  workspaceMutated: boolean;
  finalCheckExitCode: number | null | undefined;
}) {
  if (params.actualToolCalls.length > 0) {
    return true;
  }
  if (params.workspaceMutated) {
    return true;
  }
  if (typeof params.finalCheckExitCode === "number" && params.finalCheckExitCode === 0) {
    return true;
  }
  const text = params.outputQuality;
  if (!text) {
    return false;
  }
  if (
    AUTH_FAILURE_PATTERN.test(text) ||
    PROVIDER_CONNECTION_PATTERN.test(text) ||
    PROVIDER_TIMEOUT_PATTERN.test(text)
  ) {
    return false;
  }
  return true;
}

export function classifyClawBoundBenchmarkLiveRun(
  input: ClawBoundBenchmarkLiveRunClassificationInput,
): ClawBoundBenchmarkLiveRunClassification {
  const outputQuality = normalizeText(input.outputQuality);
  const fatalErrorMessage = normalizeText(input.fatalErrorMessage);
  const combinedText = [fatalErrorMessage, outputQuality].filter(Boolean).join("\n");
  const actualToolCalls = input.actualToolCalls ?? [];
  const workspaceMutated = input.workspaceMutated === true;
  const meaningfulExecution = inferMeaningfulExecution({
    outputQuality,
    actualToolCalls,
    workspaceMutated,
    finalCheckExitCode: input.finalCheckExitCode,
  });

  if (AUTH_FAILURE_PATTERN.test(combinedText)) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "auth_failure",
      failureClassification: "auth_failure",
      productValid: false,
      externallyBlocked: true,
      retryable: false,
      failureReason: "Authentication or authorization failed before a valid product run could complete.",
    };
  }

  if (HARNESS_FAILURE_PATTERN.test(combinedText)) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "harness_failure",
      failureClassification: "harness_failure",
      productValid: false,
      externallyBlocked: false,
      retryable: false,
      failureReason: "The local live benchmark harness failed before the workflow could be judged.",
    };
  }

  if (!meaningfulExecution && PROVIDER_TIMEOUT_PATTERN.test(combinedText)) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "provider_timeout",
      failureClassification: "provider_timeout",
      productValid: false,
      externallyBlocked: true,
      retryable: true,
      failureReason: "The provider path timed out before meaningful execution began.",
    };
  }

  if (!meaningfulExecution && PROVIDER_CONNECTION_PATTERN.test(combinedText)) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "provider_connection",
      failureClassification: "provider_connection",
      productValid: false,
      externallyBlocked: true,
      retryable: true,
      failureReason: "The provider path failed to establish or maintain a usable live connection.",
    };
  }

  if (fatalErrorMessage && PROVIDER_TIMEOUT_PATTERN.test(fatalErrorMessage)) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "provider_timeout",
      failureClassification: "provider_timeout",
      productValid: false,
      externallyBlocked: true,
      retryable: true,
      failureReason: "The provider path timed out after partial execution began.",
    };
  }

  if (fatalErrorMessage && PROVIDER_CONNECTION_PATTERN.test(fatalErrorMessage)) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "provider_connection",
      failureClassification: "provider_connection",
      productValid: false,
      externallyBlocked: true,
      retryable: true,
      failureReason: "The provider path lost connectivity after partial execution began.",
    };
  }

  if (fatalErrorMessage) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "runtime_failure",
      failureClassification: "runtime_failure",
      productValid: false,
      externallyBlocked: false,
      retryable: false,
      failureReason: "The workflow encountered a runtime failure after partial or attempted execution.",
    };
  }

  if (meaningfulExecution) {
    return {
      attemptCount: input.attemptCount,
      validationOutcome: "valid_product_result",
      failureClassification: null,
      productValid: true,
      externallyBlocked: false,
      retryable: false,
      failureReason: null,
    };
  }

  return {
    attemptCount: input.attemptCount,
    validationOutcome: "runtime_failure",
    failureClassification: "runtime_failure",
    productValid: false,
    externallyBlocked: false,
    retryable: false,
    failureReason: "The run did not produce a valid result and was not explained by an external provider or harness failure.",
  };
}

export function summarizeClawBoundBenchmarkLiveRuns(
  runs: ClawBoundBenchmarkLiveRunClassification[],
): ClawBoundBenchmarkLiveWorkflowSummary {
  return {
    totalRuns: runs.length,
    validProductResults: runs.filter((run) => run.validationOutcome === "valid_product_result").length,
    externallyBlockedRuns: runs.filter((run) => run.externallyBlocked).length,
    providerConnections: runs.filter((run) => run.validationOutcome === "provider_connection").length,
    providerTimeouts: runs.filter((run) => run.validationOutcome === "provider_timeout").length,
    authFailures: runs.filter((run) => run.validationOutcome === "auth_failure").length,
    harnessFailures: runs.filter((run) => run.validationOutcome === "harness_failure").length,
    runtimeFailures: runs.filter((run) => run.validationOutcome === "runtime_failure").length,
  };
}
