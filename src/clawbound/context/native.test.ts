import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  applyEmbeddedClawBoundContextEngineOverrides,
  expandClawBoundToolNamesToHostToolNames,
  isClawBoundContextEngineModeEnabled,
  normalizeClawBoundExecCommandForWorkspace,
  planClawBoundContextEngineForEmbeddedAttempt,
  wrapEmbeddedClawBoundSubagentTools,
} from "./native.js";
import { SparseClawBoundContextEngine } from "./engine.js";

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("embedded ClawBound context-engine overrides", () => {
  it("canonicalizes ClawBound tool ids into host tool ids at the filter seam", () => {
    expect(
      expandClawBoundToolNamesToHostToolNames(["read_file", "edit_file", "run_tests", "run_command"]),
    ).toEqual(["read_file", "read", "edit_file", "edit", "run_tests", "exec", "run_command"]);
  });

  it("normalizes bounded npm test invocations to an available focused test script", async () => {
    const workspaceRoot = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-native-exec-normalize-"));
    tempDirs.push(workspaceRoot);
    await fs.writeFile(
      path.join(workspaceRoot, "package.json"),
      JSON.stringify(
        {
          type: "module",
          scripts: {
            "test:parser": "node --test parser.test.js",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    await expect(
      normalizeClawBoundExecCommandForWorkspace({
        command: "npm test -- parser.test.js",
        workspaceDir: workspaceRoot,
      }),
    ).resolves.toBe("npm run test:parser");
  });

  it("keeps the non-ClawBound host path unchanged when the context engine is not active", () => {
    expect(isClawBoundContextEngineModeEnabled({ STEWARD_CONTEXT_ENGINE_MODE: "1" })).toBe(true);
    expect(isClawBoundContextEngineModeEnabled({ STEWARD_CONTEXT_ENGINE_MODE: "0" })).toBe(false);

    const result = applyEmbeddedClawBoundContextEngineOverrides({
      contextEngine: null,
      promptMode: "full",
      extraSystemPrompt: "Host extra prompt",
      skillsPrompt: "Existing skills prompt",
      contextFiles: [{ name: "AGENTS.md", content: "agent instructions" }],
      workspaceNotes: ["Commit your changes."],
      tools: [{ name: "read_file" }, { name: "edit_file" }],
    });

    expect(result.promptMode).toBe("full");
    expect(result.extraSystemPrompt).toBe("Host extra prompt");
    expect(result.skillsPrompt).toBe("Existing skills prompt");
    expect(result.contextFiles).toEqual([{ name: "AGENTS.md", content: "agent instructions" }]);
    expect(result.workspaceNotes).toEqual(["Commit your changes."]);
    expect(result.tools.map((tool) => tool.name)).toEqual(["read_file", "edit_file"]);
  });

  it("suppresses host-owned context inflation and filters tools on the bounded ClawBound path", () => {
    const result = applyEmbeddedClawBoundContextEngineOverrides({
      contextEngine: {
        promptText: "## ClawBound Kernel\nKernel first.",
        routePlan: { executionMode: "answer" },
        retrievalPlan: { noLoad: true },
        toolProfile: {
          profile: "answer-minimal",
          allowedTools: ["read_file"],
          deniedTools: ["edit_file", "message"],
          notes: ["Answer mode stays read-only."],
        },
      },
      promptMode: "full",
      extraSystemPrompt: "Host extra prompt",
      skillsPrompt: "Existing skills prompt",
      contextFiles: [{ name: "AGENTS.md", content: "agent instructions" }],
      workspaceNotes: ["Commit your changes."],
      tools: [{ name: "read" }, { name: "edit" }, { name: "message" }],
    });

    expect(result.promptMode).toBe("minimal");
    expect(result.extraSystemPrompt).toContain("Host extra prompt");
    expect(result.extraSystemPrompt).toContain("## ClawBound Kernel");
    expect(result.skillsPrompt).toBeUndefined();
    expect(result.contextFiles).toEqual([]);
    expect(result.workspaceNotes).toBeUndefined();
    expect(result.tools.map((tool) => tool.name)).toEqual(["read"]);
  });

  it("keeps executor-bounded tool exposure non-empty on the ClawBound path", () => {
    const result = applyEmbeddedClawBoundContextEngineOverrides({
      contextEngine: {
        promptText: "## ClawBound Kernel\nExecute narrowly.",
        routePlan: { executionMode: "executor_then_reviewer" },
        retrievalPlan: { noLoad: false },
        toolProfile: {
          profile: "executor-bounded",
          allowedTools: ["read_file", "edit_file", "run_tests", "run_command"],
          deniedTools: ["delete_file"],
          notes: ["Use only bounded execution and edit tools."],
        },
      },
      promptMode: "full",
      extraSystemPrompt: "Host extra prompt",
      skillsPrompt: "Existing skills prompt",
      contextFiles: [{ name: "AGENTS.md", content: "agent instructions" }],
      workspaceNotes: ["Commit your changes."],
      tools: [{ name: "read" }, { name: "edit" }, { name: "exec" }, { name: "process" }, { name: "write" }],
    });

    expect(result.tools.map((tool) => tool.name)).toEqual(["read", "edit", "exec"]);
  });

  it("keeps delegated child sessions closed even if the native profile text still includes sessions_spawn", () => {
    const result = applyEmbeddedClawBoundContextEngineOverrides({
      contextEngine: {
        promptText: "## ClawBound Kernel\nReview narrowly.",
        routePlan: { executionMode: "reviewer" },
        retrievalPlan: { noLoad: false },
        toolProfile: {
          profile: "review-delegate-bounded",
          allowedTools: ["read_file", "run_tests", "sessions_spawn"],
          deniedTools: ["edit_file", "write_file", "delete_file"],
          notes: ["Child reviewers must not reopen delegation."],
        },
      },
      sessionKey: "agent:main:subagent:test-child",
      promptMode: "full",
      extraSystemPrompt: "Host extra prompt",
      skillsPrompt: "Existing skills prompt",
      contextFiles: [{ name: "AGENTS.md", content: "agent instructions" }],
      workspaceNotes: ["Commit your changes."],
      tools: [{ name: "read" }, { name: "exec" }, { name: "sessions_spawn" }],
    });

    expect(result.tools.map((tool) => tool.name)).toEqual(["read", "exec"]);
  });

  it("wraps sessions_spawn with a persisted bounded ClawBound handoff on the native path", async () => {
    const workspaceRoot = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-native-subagent-"));
    tempDirs.push(workspaceRoot);
    const rootDir = path.join(workspaceRoot, ".clawbound", "context-engine");
    let nextId = 0;
    const engine = new SparseClawBoundContextEngine({
      rootDir,
      idFactory: () => `seed-${String(++nextId).padStart(4, "0")}`,
    });

    const state = (
      await engine.assemble({
        state: await engine.bootstrap(
          {
            hostRunId: "host-run-parent",
            sessionId: "session-parent",
            conversationId: "conversation-parent",
          },
          {
            userInput:
              "Delegate exactly one narrow parser audit to a subagent. Review parser.js and parser.test.js for quoted-field regression risk only.",
          },
        ),
        candidateTools: ["read_file", "run_tests", "sessions_spawn"],
      })
    ).state;

    const execute = vi.fn(async (_toolCallId: string, args: Record<string, unknown>) => ({
      content: [{ type: "text", text: JSON.stringify({ status: "accepted" }) }],
      details: {
        status: "accepted",
        childSessionKey: "agent:main:subagent:test-child",
        runId: "run-child",
        args,
      },
    }));

    const [wrapped] = await wrapEmbeddedClawBoundSubagentTools({
      contextEngineState: state,
      rootDir,
      workspaceDir: workspaceRoot,
      tools: [
        {
          name: "sessions_spawn",
          execute,
        },
      ],
    });

    const result = await wrapped.execute?.("tool-call-1", {
      task: "Audit quoted-field regression risk only.",
    });

    expect(execute).toHaveBeenCalledTimes(1);
    const forwardedArgs = execute.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(typeof forwardedArgs.__clawbound_handoff_text).toBe("string");
    expect(String(forwardedArgs.__clawbound_handoff_text)).toContain("## ClawBound Subagent Handoff");
    expect(result?.details).toMatchObject({
      status: "accepted",
      clawbound: {
        inheritsFullTranscript: false,
      },
    });
    const artifactPath = String((result?.details as { clawbound?: { handoffArtifactPath?: string } })?.clawbound?.handoffArtifactPath ?? "");
    expect(artifactPath).toContain(path.join(".clawbound", "context-engine", "handoffs"));
    const persisted = JSON.parse(await fs.readFile(artifactPath, "utf8")) as {
      inheritsFullTranscript: boolean;
      retrievedUnitIds: string[];
      localContextRefs: string[];
    };
    expect(persisted.inheritsFullTranscript).toBe(false);
    expect(Array.isArray(persisted.retrievedUnitIds)).toBe(true);
    expect(Array.isArray(persisted.localContextRefs)).toBe(true);
  });

  it("normalizes bounded exec test commands before forwarding to the host tool", async () => {
    const workspaceRoot = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-native-exec-tool-"));
    tempDirs.push(workspaceRoot);
    const rootDir = path.join(workspaceRoot, ".clawbound", "context-engine");
    await fs.writeFile(
      path.join(workspaceRoot, "package.json"),
      JSON.stringify(
        {
          type: "module",
          scripts: {
            "test:parser": "node --test parser.test.js",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    let nextId = 0;
    const engine = new SparseClawBoundContextEngine({
      rootDir,
      idFactory: () => `seed-${String(++nextId).padStart(4, "0")}`,
    });
    const state = (
      await engine.assemble({
        state: await engine.bootstrap(
          {
            hostRunId: "host-run-exec",
            sessionId: "session-exec",
            conversationId: "conversation-exec",
          },
          {
            userInput:
              "Fix parseCsvLine so parser.test.js passes without changing public API. Read the relevant files first and run only the focused parser test before finalizing.",
          },
        ),
        candidateTools: ["read_file", "edit_file", "run_tests", "run_command"],
      })
    ).state;

    const execute = vi.fn(async (_toolCallId: string, args: Record<string, unknown>) => ({
      content: [{ type: "text", text: JSON.stringify(args) }],
      details: args,
    }));

    const [wrapped] = await wrapEmbeddedClawBoundSubagentTools({
      contextEngineState: state,
      rootDir,
      workspaceDir: workspaceRoot,
      tools: [
        {
          name: "exec",
          execute,
        },
      ],
    });

    await wrapped.execute?.("tool-call-2", {
      command: "npm test -- parser.test.js",
    });

    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute.mock.calls[0]?.[1]).toMatchObject({
      command: "npm run test:parser",
    });
  });

  it("summarizes focused test exec output before returning it to the model", async () => {
    const workspaceRoot = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-native-exec-summary-"));
    tempDirs.push(workspaceRoot);
    const rootDir = path.join(workspaceRoot, ".clawbound", "context-engine");
    await fs.writeFile(
      path.join(workspaceRoot, "package.json"),
      JSON.stringify(
        {
          type: "module",
          scripts: {
            "test:parser": "node --test parser.test.js",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    let nextId = 0;
    const engine = new SparseClawBoundContextEngine({
      rootDir,
      idFactory: () => `seed-${String(++nextId).padStart(4, "0")}`,
    });
    const state = (
      await engine.assemble({
        state: await engine.bootstrap(
          {
            hostRunId: "host-run-exec-summary",
            sessionId: "session-exec-summary",
            conversationId: "conversation-exec-summary",
          },
          {
            userInput:
              "Fix parseCsvLine so parser.test.js passes without changing public API. Run only the focused parser test before finalizing.",
          },
        ),
        candidateTools: ["read_file", "edit_file", "run_tests"],
      })
    ).state;

    const execute = vi.fn(async () => ({
      content: [
        {
          type: "text",
          text: [
            "✔ preserves empty middle fields while trimming whitespace (1.0ms)",
            "✔ preserves trailing empty fields (0.1ms)",
            "ℹ tests 2",
            "ℹ pass 2",
            "ℹ fail 0",
          ].join("\n"),
        },
      ],
      details: {
        status: "completed",
        exitCode: 0,
        aggregated: [
          "✔ preserves empty middle fields while trimming whitespace (1.0ms)",
          "✔ preserves trailing empty fields (0.1ms)",
          "ℹ tests 2",
          "ℹ pass 2",
          "ℹ fail 0",
        ].join("\n"),
      },
    }));

    const [wrapped] = await wrapEmbeddedClawBoundSubagentTools({
      contextEngineState: state,
      rootDir,
      workspaceDir: workspaceRoot,
      tools: [
        {
          name: "exec",
          execute,
        },
      ],
    });

    const result = await wrapped.execute?.("tool-call-3", {
      command: "npm test -- parser.test.js",
    });

    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute.mock.calls[0]?.[1]).toMatchObject({
      command: "npm run test:parser",
    });
    expect(result).toMatchObject({
      content: [
        {
          type: "text",
          text: expect.stringContaining("Focused test result: passed"),
        },
      ],
      details: {
        aggregated: expect.stringContaining("ℹ tests 2"),
        clawboundExec: {
          kind: "focused_test",
          exitCode: 0,
          counts: {
            tests: 2,
            pass: 2,
            fail: 0,
          },
        },
      },
    });
  });

  it("persists inspectable compact artifacts for the native context-engine path", async () => {
    const workspaceRoot = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-native-compact-"));
    tempDirs.push(workspaceRoot);

    const result = await planClawBoundContextEngineForEmbeddedAttempt({
      env: { STEWARD_CONTEXT_ENGINE_MODE: "1" },
      workspaceDir: workspaceRoot,
      hostRunId: "host-run-compact-artifact",
      sessionId: "session-compact-artifact",
      sessionKey: "agent:main:compact-artifact",
      prompt:
        "Verify in two bullets what changed in parser.js and whether parser.test.js now passes. Do not make further edits.",
      candidateTools: ["read", "edit", "exec"],
      localContext: [
        {
          kind: "diff_summary",
          ref: "working-tree",
          content: "parser.js was trimmed narrowly and parser.test.js is expected to pass.",
        },
      ],
    });

    expect(result).not.toBeNull();
    const compactPath = result?.compactPersistedPath ?? "";
    expect(compactPath).toContain(path.join(".clawbound", "context-engine", "compacts"));
    const compacted = JSON.parse(await fs.readFile(compactPath, "utf8")) as {
      activeContext: { localContextRefs: string[]; retrievedUnitIds: string[] };
      omitted: { reasons: string[] };
      boundedness: { stayedBounded: boolean; reasons: string[] };
    };
    expect(compacted.activeContext.localContextRefs).toEqual(["working-tree"]);
    expect(Array.isArray(compacted.activeContext.retrievedUnitIds)).toBe(true);
    expect(compacted.omitted.reasons).toContain("full_transcript_not_loaded");
    expect(compacted.boundedness.stayedBounded).toBe(true);
    expect(compacted.boundedness.reasons).toContain("no_host_skills_prompt");
  });
});
