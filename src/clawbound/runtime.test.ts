import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ClawBoundRuntime } from "./runtime.js";

const tempDirs: string[] = [];

async function makeRuntime() {
  const rootDir = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-runtime-"));
  tempDirs.push(rootDir);
  let nextId = 0;
  return new ClawBoundRuntime({
    shadowRootDir: rootDir,
    idFactory: () => `seed-${String(++nextId).padStart(4, "0")}`,
  });
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("ClawBoundRuntime.planRun", () => {
  it("plans explain tasks as answer mode with explicit no-load and minimal tools", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-answer",
      sessionId: "session-answer",
      conversationId: "conversation-answer",
      userInput: "Explain what this parser function does.",
    });

    expect(result.runId).toContain("clawbound-run-");
    expect(result.traceId).toContain("clawbound-trace-");
    expect(result.routePlan.taskType).toBe("answer");
    expect(result.routePlan.executionMode).toBe("answer");
    expect(result.retrievalPlan.noLoad).toBe(true);
    expect(result.promptPackage.kernelVersion).toBe("context-kernel-v0");
    expect(result.promptPackage.retrievedUnits).toEqual([]);
    expect(result.promptPackage.assemblyStats.retrievedTokens).toBe(0);
    expect(result.toolProfile.profile).toBe("answer-minimal");
    expect(result.toolProfile.allowedTools).toEqual(["read_file"]);
    expect(result.toolProfile.deniedTools).toContain("message");
    expect(result.parity.fixtureId).toBe("phase1-answer-explain-parser");
    expect(result.parity.checks.route).toBe(true);
    expect(result.parity.checks.noLoad).toBe(true);
    expect(result.parity.checks.retrievalUnits).toBe(true);
    expect(result.parity.checks.toolProfile).toBe(true);
  });

  it("plans bounded code-fix tasks with gated snippet retrieval and a stricter tool profile", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-fix",
      sessionId: "session-fix",
      conversationId: "conversation-fix",
      userInput: "Fix the failing parser test without changing public API.",
      localContext: [
        {
          kind: "diff_summary",
          ref: "working-tree",
          content: "Parser tests are failing and public APIs should stay stable.",
        },
      ],
      candidateTools: ["read_file", "edit_file", "run_tests", "run_command", "message"],
    });

    expect(result.routePlan.taskType).toBe("code_change");
    expect(result.routePlan.executionMode).toBe("executor_then_reviewer");
    expect(result.retrievalPlan.noLoad).toBe(false);
    expect(result.retrievalPlan.loadLaws).toBe(true);
    expect(result.retrievalPlan.loadProjectNotes).toBe(true);
    expect(result.promptPackage.retrievedUnits.map((unit) => unit.id)).toEqual([
      "pn_002",
      "pn_001",
      "law_002",
    ]);
    expect(result.promptPackage.retrievedUnits.length).toBeLessThanOrEqual(
      result.retrievalPlan.maxUnits,
    );
    const retrievedTokens = result.promptPackage.retrievedUnits.reduce(
      (total, unit) => total + unit.tokenEstimate,
      0,
    );
    expect(retrievedTokens).toBeLessThanOrEqual(result.retrievalPlan.maxTokens);
    expect(result.promptPackage.retrievedUnits.every((unit) => unit.tokenEstimate <= 48)).toBe(
      true,
    );
    expect(result.toolProfile.profile).toBe("executor-bounded");
    expect(result.toolProfile.allowedTools).toEqual([
      "read_file",
      "edit_file",
      "run_tests",
      "run_command",
    ]);
    expect(result.parity.fixtureId).toBe("phase1-executor-fix-parser");
    expect(result.parity.checks.route).toBe(true);
    expect(result.parity.checks.noLoad).toBe(true);
    expect(result.parity.checks.retrievalUnits).toBe(true);
    expect(result.parity.checks.toolProfile).toBe(true);
  });

  it("persists native planning artifacts and required event sequence", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-events",
      sessionId: "session-events",
      conversationId: "conversation-events",
      userInput: "Fix the failing parser test without changing public API.",
      localContext: [
        {
          kind: "diff_summary",
          ref: "working-tree",
          content: "Parser tests are failing and public APIs should stay stable.",
        },
      ],
    });

    const persisted = JSON.parse(await fs.readFile(result.persistedPath, "utf8")) as {
      hostRunId: string;
      runId: string;
      traceId: string;
      routePlan: { executionMode: string };
      retrievalPlan: { noLoad: boolean };
      promptPackage: { retrievedUnits: Array<{ id: string }> };
      toolProfile: { profile: string };
      events: Array<{ eventType: string; payload: Record<string, unknown> }>;
    };

    expect(persisted.hostRunId).toBe("host-run-events");
    expect(persisted.runId).toBe(result.runId);
    expect(persisted.traceId).toBe(result.traceId);
    expect(persisted.routePlan.executionMode).toBe("executor_then_reviewer");
    expect(persisted.retrievalPlan.noLoad).toBe(false);
    expect(persisted.promptPackage.retrievedUnits.map((unit) => unit.id)).toEqual([
      "pn_002",
      "pn_001",
      "law_002",
    ]);
    expect(persisted.toolProfile.profile).toBe("executor-bounded");
    expect(persisted.events.map((event) => event.eventType)).toEqual([
      "task_received",
      "route_decided",
      "gate_decided",
      "retrieval_executed",
      "prompt_built",
      "tool_profile_built",
      "run_completed",
    ]);
    expect(persisted.events[3]?.payload.selectedUnitIds).toEqual(["pn_002", "pn_001", "law_002"]);
    expect(persisted.events[4]?.payload.promptPackage).toMatchObject({
      traceId: result.traceId,
      runId: result.runId,
    });
    expect(persisted.events[5]?.payload.toolProfile).toMatchObject({
      profile: "executor-bounded",
    });
  });

  it("allows one bounded subagent spawn for explicit delegated review tasks", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-delegate-review",
      sessionId: "session-delegate-review",
      conversationId: "conversation-delegate-review",
      userInput:
        "Delegate exactly one narrow parser audit to a subagent. Do not edit files yourself. Review parser.js and parser.test.js for quoted-field regression risk only.",
      candidateTools: ["read_file", "run_tests", "sessions_spawn", "edit_file", "run_command"],
    });

    expect(result.routePlan.taskType).toBe("review");
    expect(result.routePlan.executionMode).toBe("reviewer");
    expect(result.toolProfile.profile).toBe("review-delegate-bounded");
    expect(result.toolProfile.allowedTools).toEqual(["read_file", "run_tests", "sessions_spawn"]);
    expect(result.toolProfile.deniedTools).toContain("edit_file");
    expect(result.toolProfile.deniedTools).toContain("write_file");
  });

  it("keeps subagent review sessions closed to nested delegation", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-delegate-review-child",
      sessionId: "session-delegate-review-child",
      sessionKey: "agent:main:subagent:test-child",
      conversationId: "conversation-delegate-review-child",
      userInput:
        "Delegate exactly one narrow parser audit to a subagent. Do not edit files yourself. Review parser.js and parser.test.js for quoted-field regression risk only.",
      candidateTools: ["read_file", "run_tests", "sessions_spawn", "edit_file", "run_command"],
    });

    expect(result.routePlan.taskType).toBe("review");
    expect(result.routePlan.executionMode).toBe("reviewer");
    expect(result.toolProfile.profile).toBe("review-read-only");
    expect(result.toolProfile.allowedTools).toEqual(["read_file", "run_tests"]);
    expect(result.toolProfile.allowedTools).not.toContain("sessions_spawn");
  });

  it("routes verification-like no-edit turns as reviewer instead of executor", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-verify",
      sessionId: "session-verify",
      conversationId: "conversation-verify",
      userInput:
        "Verify in two bullets what changed in parser.js and whether parser.test.js now passes. Do not make further edits.",
      candidateTools: ["read_file", "edit_file", "run_tests", "run_command", "message"],
    });

    expect(result.routePlan.taskType).toBe("review");
    expect(result.routePlan.executionMode).toBe("reviewer");
    expect(result.toolProfile.profile).toBe("review-read-only");
    expect(result.toolProfile.allowedTools).toEqual(["read_file", "run_tests"]);
    expect(result.toolProfile.allowedTools).not.toContain("edit_file");
  });

  it("adds focused verification guidance when a bounded task names an explicit test file", async () => {
    const runtime = await makeRuntime();

    const result = await runtime.planRun({
      hostRunId: "host-run-focused-test-guidance",
      sessionId: "session-focused-test-guidance",
      conversationId: "conversation-focused-test-guidance",
      userInput:
        "Fix parseCsvLine so parser.test.js passes without changing public API. Read the relevant files first and run only the focused parser test before finalizing.",
      candidateTools: ["read_file", "edit_file", "run_tests", "run_command"],
    });

    expect(result.toolProfile.profile).toBe("executor-bounded");
    expect(result.toolProfile.notes.some((note) => note.includes("parser.test.js"))).toBe(true);
    expect(result.toolProfile.notes.some((note) => note.includes("avoid npm test -- <file>"))).toBe(
      true,
    );
  });
});
