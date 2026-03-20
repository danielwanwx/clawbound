import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { SparseClawBoundContextEngine } from "./engine.js";

const tempDirs: string[] = [];

async function makeEngine() {
  const rootDir = await fs.mkdtemp(path.join(os.tmpdir(), "clawbound-context-engine-"));
  tempDirs.push(rootDir);
  let nextId = 0;
  return new SparseClawBoundContextEngine({
    rootDir,
    idFactory: () => `seed-${String(++nextId).padStart(4, "0")}`,
  });
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("SparseClawBoundContextEngine", () => {
  it("bootstraps kernel-first with bounded empty active context", async () => {
    const engine = await makeEngine();

    const state = await engine.bootstrap(
      {
        hostRunId: "host-run-bootstrap",
        sessionId: "session-bootstrap",
        sessionKey: "session-key-bootstrap",
        conversationId: "conversation-bootstrap",
        sourceHost: "clawbound-test/bootstrap",
      },
      {
        userInput: "Explain what this parser function does.",
      },
    );

    expect(state.runId).toBe("clawbound-run-seed-0001");
    expect(state.traceId).toBe("clawbound-trace-seed-0002");
    expect(state.kernel.version).toBe("context-kernel-v0");
    expect(state.runtimeDefaults.noLoadAllowed).toBe(true);
    expect(state.activeContext.localContext).toEqual([]);
    expect(state.activeContext.retrievedUnits).toEqual([]);
    expect(state.rawEvents).toEqual([]);
  });

  it("assembles a bounded ClawBound prompt package without silent context inflation", async () => {
    const engine = await makeEngine();

    let state = await engine.bootstrap(
      {
        hostRunId: "host-run-assemble",
        sessionId: "session-assemble",
        conversationId: "conversation-assemble",
      },
      {
        userInput: "Fix the failing parser test without changing public API.",
      },
    );

    state = await engine.ingest({
      state,
      events: [
        {
          type: "local_context",
          item: {
            kind: "diff_summary",
            ref: "working-tree",
            content: "Parser tests are failing and public APIs should stay stable.",
          },
        },
      ],
    });

    const assembled = await engine.assemble({
      state,
      candidateTools: ["read_file", "edit_file", "run_tests", "run_command", "message"],
    });

    expect(assembled.routePlan.executionMode).toBe("executor_then_reviewer");
    expect(assembled.retrievalPlan.noLoad).toBe(false);
    expect(assembled.promptPackage.retrievedUnits.map((unit) => unit.id)).toEqual([
      "pn_002",
      "pn_001",
      "law_002",
    ]);
    expect(assembled.toolProfile.profile).toBe("executor-bounded");
    expect(assembled.promptText).toContain("## ClawBound Kernel");
    expect(assembled.promptText).toContain("## Retrieved Snippets");
    expect(assembled.promptText).not.toContain("skills");
    expect(assembled.state.activeContext.localContext).toHaveLength(1);
    expect(assembled.state.activeContext.retrievedUnits.map((unit) => unit.id)).toEqual([
      "pn_002",
      "pn_001",
      "law_002",
    ]);
    expect(assembled.persistedPath).toContain(path.join("runs", "host-run-assemble.json"));
  });

  it("compacts to bounded trace and retrieval summaries only", async () => {
    const engine = await makeEngine();

    let state = await engine.bootstrap(
      {
        hostRunId: "host-run-compact",
        sessionId: "session-compact",
        conversationId: "conversation-compact",
      },
      {
        userInput: "Fix the failing parser test without changing public API.",
        localContext: [
          {
            kind: "diff_summary",
            ref: "working-tree",
            content: "Parser tests are failing and public APIs should stay stable.",
          },
        ],
      },
    );

    state = (
      await engine.assemble({
        state,
        candidateTools: ["read_file", "edit_file", "run_tests", "run_command", "message"],
      })
    ).state;

    const compacted = await engine.compact(state);

    expect(compacted.traceSummary.executionMode).toBe("executor_then_reviewer");
    expect(compacted.traceSummary.eventTypes).toEqual([
      "task_received",
      "route_decided",
      "gate_decided",
      "retrieval_executed",
      "prompt_built",
      "tool_profile_built",
      "run_completed",
    ]);
    expect(compacted.retrievalSummary.selectedUnitIds).toEqual(["pn_002", "pn_001", "law_002"]);
    expect(compacted.activeContext.localContextRefs).toEqual(["working-tree"]);
    expect(compacted.activeContext.retrievedUnitIds).toEqual(["pn_002", "pn_001", "law_002"]);
    expect(compacted.omitted.reasons).toEqual([
      "full_transcript_not_loaded",
      "host_workspace_inflation_not_loaded",
      "host_skills_prompt_not_loaded",
    ]);
    expect(compacted.boundedness.stayedBounded).toBe(true);
    expect(compacted.boundedness.reasons).toContain("no_host_workspace_inflation");
    expect(compacted.boundedness.reasons).toContain("retrieval_within_budget");
    expect(compacted.transcriptIncluded).toBe(false);
  });

  it("prepares a bounded subagent handoff without full transcript inheritance", async () => {
    const engine = await makeEngine();

    let state = await engine.bootstrap(
      {
        hostRunId: "host-run-subagent",
        sessionId: "session-subagent",
        conversationId: "conversation-subagent",
      },
      {
        userInput: "Fix the failing parser test without changing public API.",
        localContext: [
          {
            kind: "diff_summary",
            ref: "working-tree",
            content: "Parser tests are failing and public APIs should stay stable.",
          },
        ],
      },
    );

    state = (
      await engine.assemble({
        state,
        candidateTools: ["read_file", "edit_file", "run_tests", "run_command", "message"],
      })
    ).state;

    const handoff = await engine.prepareSubagentContext(state, {
      userInput: "Review the parser fix for regression risk only.",
    });

    expect(handoff.inheritsFullTranscript).toBe(false);
    expect(handoff.localContext.length).toBeLessThanOrEqual(1);
    expect(handoff.retrievedUnits.length).toBeLessThanOrEqual(2);
    expect(handoff.text).toContain("## ClawBound Subagent Handoff");
    expect(handoff.text).toContain("Review the parser fix for regression risk only.");
    expect(handoff.text).not.toContain("full transcript");
  });
});
