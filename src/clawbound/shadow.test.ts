import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { isClawBoundShadowModeEnabled, planClawBoundShadowForEmbeddedAttempt } from "./shadow.js";

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("clawbound shadow mode", () => {
  it("only plans when shadow mode is enabled and persists under the workspace shadow dir", async () => {
    expect(isClawBoundShadowModeEnabled({ STEWARD_SHADOW_MODE: "1" })).toBe(true);
    expect(isClawBoundShadowModeEnabled({ STEWARD_SHADOW_MODE: "0" })).toBe(false);

    const workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-shadow-"));
    tempDirs.push(workspaceDir);

    const result = await planClawBoundShadowForEmbeddedAttempt({
      env: { STEWARD_SHADOW_MODE: "1" },
      workspaceDir,
      hostRunId: "host-run-shadow",
      sessionId: "session-shadow",
      sessionKey: "session-key-shadow",
      prompt: "Explain what this parser function does.",
      candidateTools: ["read", "edit", "exec", "message"],
    });

    expect(result).not.toBeNull();
    expect(result?.persistedPath).toContain(path.join(".clawbound", "shadow", "runs"));
    const persisted = JSON.parse(await fs.readFile(result!.persistedPath, "utf8")) as {
      sourceHost: string;
      hostRunId: string;
      toolProfile: { candidateToolsObserved: string[] };
    };
    expect(persisted.sourceHost).toBe("clawbound-shadow/openclaw-embedded-runner");
    expect(persisted.hostRunId).toBe("host-run-shadow");
    expect(persisted.toolProfile.candidateToolsObserved).toEqual([
      "read_file",
      "edit_file",
      "run_command",
      "message",
    ]);
  });
});
