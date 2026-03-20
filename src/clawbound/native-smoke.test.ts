import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

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
  containsClawBoundKernel: boolean;
  containsAgentsMd: boolean;
  containsProjectContext: boolean;
  containsRetrievedSnippets: boolean;
  systemPromptChars: number;
};

const observedPrompts = new Map<string, PromptSnapshot>();

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

function synthesizeReply(snapshot: PromptSnapshot) {
  if (snapshot.prompt.includes("Explain what this parser function does.")) {
    if (snapshot.containsClawBoundKernel && !snapshot.containsAgentsMd) {
      return "The parser function appears to read input, turn it into parser state, and return a structured result. The bounded prompt stays in answer mode, keeps only the read tool visible, and avoids unnecessary edits or retrieval.";
    }
    return "The parser function appears to consume input and produce a structured parsing result, but the broader host prompt includes extra workspace context that is not required for this explanation.";
  }

  if (snapshot.prompt.includes("Fix the failing parser test without changing public API.")) {
    if (
      snapshot.containsRetrievedSnippets &&
      snapshot.toolNames.includes("read") &&
      snapshot.toolNames.includes("edit") &&
      snapshot.toolNames.includes("exec")
    ) {
      return "Read the parser and failing test first, edit only the narrow implementation or fixture that breaks the test, and use exec to run the focused parser test before finalizing. The bounded prompt stays code-grounded and preserves the public API contract.";
    }
    return "Investigate the failing parser test, preserve the public API, and run the relevant tests before finalizing the change.";
  }

  return "Bounded smoke response.";
}

vi.mock("@mariozechner/pi-ai", async () => {
  const actual = await vi.importActual<typeof import("@mariozechner/pi-ai")>("@mariozechner/pi-ai");
  return {
    ...actual,
    streamSimple: (model: ModelLike, context: ContextLike) => {
      const label = String((globalThis as { __STEWARDSMOKE_LABEL__?: string }).__STEWARDSMOKE_LABEL__ ?? "unknown");
      const systemPrompt = typeof context?.systemPrompt === "string" ? context.systemPrompt : "";
      const prompt = messageContentToText(context.messages.at(-1)?.content);
      const toolNames = (context.tools ?? []).map((tool) => tool.name);
      const snapshot: PromptSnapshot = {
        label,
        prompt,
        systemPrompt,
        toolNames,
        containsClawBoundKernel: systemPrompt.includes("## ClawBound Kernel"),
        containsAgentsMd: systemPrompt.includes("## AGENTS.md"),
        containsProjectContext: systemPrompt.includes("# Project Context"),
        containsRetrievedSnippets: systemPrompt.includes("## Retrieved Snippets"),
        systemPromptChars: systemPrompt.length,
      };
      observedPrompts.set(label, snapshot);

      const stream = new actual.AssistantMessageEventStream();
      const text = synthesizeReply(snapshot);
      queueMicrotask(() => {
        stream.push({
          type: "done",
          reason: "stop",
          message: {
            role: "assistant",
            content: [{ type: "text", text }],
            stopReason: "stop",
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: Math.max(1, Math.round(systemPrompt.length / 100)),
              output: Math.max(1, Math.round(text.length / 50)),
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: Math.max(2, Math.round(systemPrompt.length / 100) + Math.round(text.length / 50)),
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            timestamp: Date.now(),
          },
        });
        stream.end();
      });
      return stream;
    },
  };
});

let runEmbeddedPiAgent: typeof import("/Users/javiswan/Projects/clawbound/src/agents/pi-embedded-runner.ts").runEmbeddedPiAgent;
let ensureOpenClawModelsJson: typeof import("/Users/javiswan/Projects/clawbound/src/agents/models-config.ts").ensureOpenClawModelsJson;
let ensureAgentWorkspace: typeof import("/Users/javiswan/Projects/clawbound/src/agents/workspace.ts").ensureAgentWorkspace;
let tempRoot = "";
let agentDir = "";
let workspaceDir = "";
let sessionCounter = 0;
const reportPath = path.join(os.tmpdir(), "clawbound-native-smoke-report.json");

function makeConfig(modelIds: string[]) {
  return {
    plugins: { enabled: false },
    models: {
      providers: {
        openai: {
          api: "openai-responses",
          apiKey: "sk-test",
          baseUrl: "https://example.com",
          models: modelIds.map((id) => ({
            id,
            name: `Mock ${id}`,
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 16000,
            maxTokens: 2048,
          })),
        },
      },
    },
  };
}

function nextSessionFile() {
  sessionCounter += 1;
  return path.join(workspaceDir, `session-${sessionCounter}.jsonl`);
}

const immediateEnqueue = async <T>(task: () => Promise<T>) => task();

async function seedWorkspace() {
  await ensureAgentWorkspace({
    dir: workspaceDir,
    ensureBootstrapFiles: true,
  });

  await fs.writeFile(
    path.join(workspaceDir, "AGENTS.md"),
    `# AGENTS.md\n- Prefer concise technical explanations.\n- Keep parser fixes narrow and avoid broad rewrites.\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(workspaceDir, "SOUL.md"),
    `# SOUL.md\n- Speak like a disciplined runtime clawbound.\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(workspaceDir, "TOOLS.md"),
    `# TOOLS.md\n- Use tests after edits.\n- Avoid deleting files unless necessary.\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(workspaceDir, "IDENTITY.md"),
    `# IDENTITY.md\n- Name: ClawBound Smoke Agent\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(workspaceDir, "USER.md"),
    `# USER.md\n- User prefers direct engineering answers.\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(workspaceDir, "HEARTBEAT.md"),
    `# HEARTBEAT.md\n- No pending tasks.\n`,
    "utf8",
  );
  await fs.writeFile(
    path.join(workspaceDir, "BOOTSTRAP.md"),
    `# BOOTSTRAP.md\n- This workspace contains parser-related code.\n`,
    "utf8",
  );
}

async function readNativeArtifact(runId: string) {
  const artifactPath = path.join(workspaceDir, ".clawbound", "context-engine", "runs", `${runId}.json`);
  const payload = JSON.parse(await fs.readFile(artifactPath, "utf8")) as Record<string, any>;
  return { artifactPath, payload };
}

async function runScenario(params: {
  label: string;
  prompt: string;
  nativeMode: boolean;
}) {
  const config = makeConfig(["mock-smoke"]);
  await ensureOpenClawModelsJson(config, agentDir);

  const sessionFile = nextSessionFile();
  const runId = `smoke-${params.label}`;
  const previous = process.env.STEWARD_CONTEXT_ENGINE_MODE;
  if (params.nativeMode) {
    process.env.STEWARD_CONTEXT_ENGINE_MODE = "1";
  } else {
    delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
  }
  (globalThis as { __STEWARDSMOKE_LABEL__?: string }).__STEWARDSMOKE_LABEL__ = params.label;
  try {
    const result = await runEmbeddedPiAgent({
      sessionId: `session-${params.label}`,
      sessionKey: `agent:main:${params.label}`,
      sessionFile,
      workspaceDir,
      config,
      prompt: params.prompt,
      provider: "openai",
      model: "mock-smoke",
      timeoutMs: 10_000,
      agentDir,
      enqueue: immediateEnqueue,
      runId,
    });

    const observed = observedPrompts.get(params.label);
    const systemPromptReport = result.meta.systemPromptReport;
    const outputText = result.payloads?.map((payload) => payload.text).filter(Boolean).join("\n") ?? "";
    const inflation = {
      systemPromptChars: systemPromptReport?.systemPrompt.chars ?? 0,
      projectContextChars: systemPromptReport?.systemPrompt.projectContextChars ?? 0,
      injectedWorkspaceFilesCount: systemPromptReport?.injectedWorkspaceFiles.length ?? 0,
      injectedWorkspaceChars:
        systemPromptReport?.injectedWorkspaceFiles.reduce(
          (total, file) => total + (file.injectedChars ?? 0),
          0,
        ) ?? 0,
      skillsPromptChars: systemPromptReport?.skills.promptChars ?? 0,
      observedContainsClawBoundKernel: observed?.containsClawBoundKernel ?? false,
      observedContainsAgentsMd: observed?.containsAgentsMd ?? false,
      observedContainsProjectContext: observed?.containsProjectContext ?? false,
      observedContainsRetrievedSnippets: observed?.containsRetrievedSnippets ?? false,
    };

    let nativeArtifact: { artifactPath: string; payload: Record<string, any> } | null = null;
    if (params.nativeMode) {
      nativeArtifact = await readNativeArtifact(runId);
    }

    return {
      label: params.label,
      task: params.prompt,
      hostPathUsed: params.nativeMode
        ? "embedded-runner / clawbound-bounded-context-engine"
        : "embedded-runner / default-host-path",
      clawboundContextEngineModeActive: params.nativeMode,
      routePlan: nativeArtifact?.payload.routePlan ?? null,
      retrievalPlan: nativeArtifact?.payload.retrievalPlan ?? null,
      noLoad: nativeArtifact?.payload.retrievalPlan?.noLoad ?? null,
      promptPackageSummary: nativeArtifact
        ? {
            kernelVersion: nativeArtifact.payload.promptPackage?.kernelVersion,
            assemblyOrder: nativeArtifact.payload.promptPackage?.assemblyOrder,
            retrievedUnitIds: (nativeArtifact.payload.promptPackage?.retrievedUnits ?? []).map(
              (unit: { id: string }) => unit.id,
            ),
            localContextCount: nativeArtifact.payload.promptPackage?.localContext?.length ?? 0,
            totalEstimatedTokens:
              nativeArtifact.payload.promptPackage?.assemblyStats?.totalEstimatedTokens ?? 0,
            noLoad: nativeArtifact.payload.promptPackage?.noLoad ?? null,
          }
        : null,
      promptInflationSignals: inflation,
      finalToolProfile: nativeArtifact
        ? nativeArtifact.payload.toolProfile
        : {
            profile: "host-default",
            allowedTools: observed?.toolNames ?? [],
            deniedTools: [],
            notes: ["Default host path; no ClawBound-native bounded profile applied."],
          },
      actualToolNamesSeenByModel: observed?.toolNames ?? [],
      runOutput: outputText,
      runOutcome: {
        completed: !(result.meta.aborted ?? false),
        aborted: result.meta.aborted ?? false,
        stopReason: result.meta.stopReason ?? "completed",
      },
      systemPromptCharsSeenByModel: observed?.systemPromptChars ?? 0,
      persistedArtifactsLocation: nativeArtifact?.artifactPath ?? null,
      parity: nativeArtifact?.payload.parity ?? null,
    };
  } finally {
    if (typeof previous === "string") {
      process.env.STEWARD_CONTEXT_ENGINE_MODE = previous;
    } else {
      delete process.env.STEWARD_CONTEXT_ENGINE_MODE;
    }
    delete (globalThis as { __STEWARDSMOKE_LABEL__?: string }).__STEWARDSMOKE_LABEL__;
  }
}

beforeAll(async () => {
  vi.useRealTimers();
  ({ runEmbeddedPiAgent } = await import("/Users/javiswan/Projects/clawbound/src/agents/pi-embedded-runner.ts"));
  ({ ensureOpenClawModelsJson } = await import("/Users/javiswan/Projects/clawbound/src/agents/models-config.ts"));
  ({ ensureAgentWorkspace } = await import("/Users/javiswan/Projects/clawbound/src/agents/workspace.ts"));
  tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-native-smoke-"));
  agentDir = path.join(tempRoot, "agent");
  workspaceDir = path.join(tempRoot, "workspace");
  await fs.mkdir(agentDir, { recursive: true });
  await fs.mkdir(workspaceDir, { recursive: true });
  await seedWorkspace();
});

afterAll(async () => {
  if (tempRoot) {
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
});

describe("ClawBound bounded native smoke", () => {
  it("runs bounded native and baseline host smoke scenarios and writes a validation report", async () => {
    const runs = [];
    runs.push(
      await runScenario({
        label: "answer-host",
        prompt: "Explain what this parser function does.",
        nativeMode: false,
      }),
    );
    runs.push(
      await runScenario({
        label: "answer-native",
        prompt: "Explain what this parser function does.",
        nativeMode: true,
      }),
    );
    runs.push(
      await runScenario({
        label: "codefix-host",
        prompt: "Fix the failing parser test without changing public API.",
        nativeMode: false,
      }),
    );
    runs.push(
      await runScenario({
        label: "codefix-native",
        prompt: "Fix the failing parser test without changing public API.",
        nativeMode: true,
      }),
    );

    const report = {
      createdAt: new Date().toISOString(),
      workspaceDir,
      runs,
    };
    await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

    expect(runs).toHaveLength(4);
    expect(runs.filter((run) => run.clawboundContextEngineModeActive)).toHaveLength(2);
    expect(runs.find((run) => run.label === "answer-native")?.promptPackageSummary?.noLoad).toBe(true);
    expect(runs.find((run) => run.label === "answer-native")?.actualToolNamesSeenByModel).toEqual(["read"]);
    expect(runs.find((run) => run.label === "codefix-native")?.actualToolNamesSeenByModel).toEqual([
      "read",
      "edit",
      "exec",
    ]);
    expect(runs.find((run) => run.label === "codefix-native")?.runOutput).toContain("Read the parser");
    expect(
      runs
        .find((run) => run.label === "codefix-native")
        ?.promptPackageSummary?.retrievedUnitIds?.slice()
        .sort(),
    ).toEqual(["law_002", "pn_001", "pn_002"]);
  });
});
