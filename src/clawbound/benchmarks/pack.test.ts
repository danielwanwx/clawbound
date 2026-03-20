import { describe, expect, it } from "vitest";
import { STEWARD_BENCHMARK_PACK } from "./pack.js";

describe("ClawBound benchmark pack", () => {
  it("freezes the validated workflow inventory into seven reusable benchmark entries", () => {
    expect(STEWARD_BENCHMARK_PACK.version).toBeTruthy();
    expect(STEWARD_BENCHMARK_PACK.workflows).toHaveLength(7);
    expect(STEWARD_BENCHMARK_PACK.workflows.map((workflow) => workflow.id)).toEqual([
      "answer",
      "review",
      "code-fix",
      "delegated-review",
      "multi-turn-continuity",
      "diff-aware-review",
      "multi-file-bounded-fix",
    ]);
  });

  it("records validation mode, prompts, assertions, and artifact expectations for every workflow", () => {
    for (const workflow of STEWARD_BENCHMARK_PACK.workflows) {
      expect(workflow.taskClass).toBeTruthy();
      expect(workflow.taskPrompt.length).toBeGreaterThan(0);
      expect(workflow.fixtureFiles.length).toBeGreaterThan(0);
      expect(workflow.expectedBoundedBehavior.length).toBeGreaterThan(0);
      expect(workflow.keyAssertions.length).toBeGreaterThan(0);
      expect(workflow.validationModes.length).toBeGreaterThan(0);
      expect(workflow.artifactExpectations.length).toBeGreaterThan(0);
      expect(workflow.testFiles.length).toBeGreaterThan(0);
    }
  });
});
