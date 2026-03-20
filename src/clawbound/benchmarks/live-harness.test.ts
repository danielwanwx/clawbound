import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  classifyClawBoundBenchmarkLiveRun,
  readClawBoundBenchmarkSessionMessages,
  resolveClawBoundBenchmarkConfiguredLiveModelBinding,
  resolveClawBoundBenchmarkLiveModelBinding,
  summarizeClawBoundBenchmarkLiveModelBinding,
  summarizeClawBoundBenchmarkLiveRuns,
  withClawBoundBenchmarkHarnessTimeout,
} from "./live-harness.js";

describe("ClawBound live benchmark harness classification", () => {
  it("classifies zero-tool-call connection errors as provider_connection", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 2,
      outputQuality: "Connection error.",
      actualToolCalls: [],
      workspaceMutated: false,
      finalCheckExitCode: 1,
    });

    expect(result).toMatchObject({
      validationOutcome: "provider_connection",
      failureClassification: "provider_connection",
      productValid: false,
      externallyBlocked: true,
      retryable: true,
      attemptCount: 2,
    });
  });

  it("classifies fetch failed as provider_connection when execution never started", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 1,
      outputQuality: "fetch failed",
      actualToolCalls: [],
      workspaceMutated: false,
      finalCheckExitCode: 1,
    });

    expect(result.validationOutcome).toBe("provider_connection");
    expect(result.externallyBlocked).toBe(true);
    expect(result.retryable).toBe(true);
  });

  it("classifies timeout-like zero-tool-call aborts as provider_timeout", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 3,
      outputQuality: "gateway timeout after 10000ms",
      actualToolCalls: [],
      workspaceMutated: false,
    });

    expect(result).toMatchObject({
      validationOutcome: "provider_timeout",
      failureClassification: "provider_timeout",
      productValid: false,
      retryable: true,
    });
  });

  it("classifies local harness listener failures as harness_failure", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 1,
      outputQuality: "",
      actualToolCalls: [],
      workspaceMutated: false,
      fatalErrorMessage: "listen EPERM: operation not permitted 127.0.0.1",
    });

    expect(result).toMatchObject({
      validationOutcome: "harness_failure",
      failureClassification: "harness_failure",
      productValid: false,
      externallyBlocked: false,
      retryable: false,
    });
  });

  it("classifies auth errors as auth_failure", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 1,
      outputQuality: "HTTP 401 authentication_error: invalid x-api-key",
      actualToolCalls: [],
      workspaceMutated: false,
    });

    expect(result.validationOutcome).toBe("auth_failure");
    expect(result.externallyBlocked).toBe(true);
    expect(result.retryable).toBe(false);
  });

  it("classifies config-schema mismatches as harness_failure", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 1,
      outputQuality: "",
      actualToolCalls: [],
      workspaceMutated: false,
      fatalErrorMessage:
        "Invalid config at /Users/example/.openclaw/openclaw.json.\ncommands.ownerDisplay: unknown key",
    });

    expect(result.validationOutcome).toBe("harness_failure");
    expect(result.failureClassification).toBe("harness_failure");
  });

  it("classifies unexpected fatal errors as runtime_failure even after partial execution", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 1,
      outputQuality: "Partial assistant output",
      actualToolCalls: ["read"],
      workspaceMutated: false,
      fatalErrorMessage: "child session file missing",
    });

    expect(result).toMatchObject({
      validationOutcome: "runtime_failure",
      failureClassification: "runtime_failure",
      productValid: false,
      externallyBlocked: false,
      retryable: false,
    });
  });

  it("classifies meaningful executions as valid_product_result", () => {
    const result = classifyClawBoundBenchmarkLiveRun({
      attemptCount: 1,
      outputQuality: "Focused test result: passed",
      actualToolCalls: ["read", "edit", "exec"],
      workspaceMutated: true,
      finalCheckExitCode: 0,
    });

    expect(result).toMatchObject({
      validationOutcome: "valid_product_result",
      failureClassification: null,
      productValid: true,
      externallyBlocked: false,
      retryable: false,
    });
  });

  it("counts outcomes honestly in the workflow summary", () => {
    const summary = summarizeClawBoundBenchmarkLiveRuns([
      classifyClawBoundBenchmarkLiveRun({
        attemptCount: 1,
        outputQuality: "Connection error.",
        actualToolCalls: [],
        workspaceMutated: false,
      }),
      classifyClawBoundBenchmarkLiveRun({
        attemptCount: 2,
        outputQuality: "Focused test result: passed",
        actualToolCalls: ["exec"],
        workspaceMutated: true,
        finalCheckExitCode: 0,
      }),
      classifyClawBoundBenchmarkLiveRun({
        attemptCount: 1,
        outputQuality: "",
        actualToolCalls: [],
        workspaceMutated: false,
        fatalErrorMessage: "listen EPERM: operation not permitted 127.0.0.1",
      }),
    ]);

    expect(summary).toMatchObject({
      totalRuns: 3,
      validProductResults: 1,
      externallyBlockedRuns: 1,
      harnessFailures: 1,
    });
  });

  it("resolves the configured live model path conservatively", () => {
    const resolved = resolveClawBoundBenchmarkLiveModelBinding({
      agents: {
        defaults: {
          model: {
            primary: "openai-codex/gpt-5.1",
          },
        },
      },
    });

    expect(resolved).toEqual({
      provider: "openai-codex",
      model: "gpt-5.1",
    });
  });

  it("summarizes provider/model divergence before meaningful execution", () => {
    expect(
      summarizeClawBoundBenchmarkLiveModelBinding({
        intended: { provider: "openai-codex", model: "gpt-5.1" },
        actual: { provider: "minimax", model: "MiniMax-M2.5" },
        meaningfulExecution: false,
      }),
    ).toEqual({
      intended: { provider: "openai-codex", model: "gpt-5.1" },
      actual: { provider: "minimax", model: "MiniMax-M2.5" },
      diverged: true,
      divergenceStage: "before_meaningful_execution",
    });
  });

  it("reads the configured home live model path when the local config exists", () => {
    const resolved = resolveClawBoundBenchmarkConfiguredLiveModelBinding();
    expect(resolved.provider.length > 0).toBe(true);
    expect(resolved.model.length > 0).toBe(true);
  });

  it("returns no session messages when the session artifact was never created", async () => {
    const missing = path.join(os.tmpdir(), `missing-session-${Date.now()}.jsonl`);

    await expect(readClawBoundBenchmarkSessionMessages(missing)).resolves.toEqual([]);
  });

  it("reads message entries from an existing session artifact", async () => {
    const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "clawbound-live-harness-"));
    const sessionFile = path.join(tempDir, "session.jsonl");
    await fsp.writeFile(
      sessionFile,
      [
        JSON.stringify({ type: "message", message: { role: "assistant", content: "ok" } }),
        JSON.stringify({ type: "tool", message: { ignored: true } }),
      ].join("\n"),
      "utf8",
    );

    await expect(readClawBoundBenchmarkSessionMessages(sessionFile)).resolves.toEqual([
      { role: "assistant", content: "ok" },
    ]);
    await fsp.rm(tempDir, { recursive: true, force: true });
  });

  it("fails fast with a harness timeout when work does not settle", async () => {
    await expect(
      withClawBoundBenchmarkHarnessTimeout({
        timeoutMs: 10,
        timeoutMessage: "live harness timeout after 10ms",
        work: () => new Promise<never>(() => {}),
      }),
    ).rejects.toThrow("live harness timeout after 10ms");
  });
});
