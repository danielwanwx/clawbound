export type ClawBoundBenchmarkValidationMode = "unit" | "mocked_host_boundary" | "live";

export type ClawBoundBenchmarkWorkflow = {
  id: string;
  taskClass: string;
  taskPrompt: string;
  fixtureFiles: string[];
  expectedBoundedBehavior: string[];
  keyAssertions: string[];
  validationModes: ClawBoundBenchmarkValidationMode[];
  artifactExpectations: string[];
  testFiles: string[];
};

export const STEWARD_BENCHMARK_PACK = {
  version: "2026-03-17",
  workflows: [
    {
      id: "answer",
      taskClass: "answer",
      taskPrompt: "Explain what this parser function does.",
      fixtureFiles: ["parser.js"],
      expectedBoundedBehavior: [
        "route to answer mode",
        "explicit no_load=true",
        "zero retrieved units",
        "minimal read-only tool profile",
      ],
      keyAssertions: [
        "native prompt contains ClawBound kernel and no host project inflation",
        "native tool boundary exposes read only",
        "answer stays direct and file-grounded",
      ],
      validationModes: ["unit", "mocked_host_boundary", "live"],
      artifactExpectations: [
        "native run artifact for bounded path",
        "prompt/tool boundary snapshot",
      ],
      testFiles: [
        "src/clawbound/runtime.test.ts",
        "src/clawbound/native-smoke.test.ts",
        "src/clawbound/native-live-sanity.live.test.ts",
      ],
    },
    {
      id: "review",
      taskClass: "review",
      taskPrompt:
        "Review parseCsvLine in parser.js for correctness and edge cases without editing files. Focus on parser.js and parser.test.js. If needed, run the focused parser test for evidence.",
      fixtureFiles: ["parser.js", "parser.test.js"],
      expectedBoundedBehavior: [
        "route to reviewer mode",
        "read/test-only tool profile",
        "no workspace mutation",
      ],
      keyAssertions: [
        "prompt inflation stays materially below host path",
        "model boundary exposes read and exec only",
        "review identifies real edge cases without edits",
      ],
      validationModes: ["live"],
      artifactExpectations: [
        "live report entry with tool call log",
        "native run artifact for bounded review path",
      ],
      testFiles: ["src/clawbound/native-review-live.live.test.ts"],
    },
    {
      id: "code-fix",
      taskClass: "code_change",
      taskPrompt: "Fix the failing parser test without changing public API.",
      fixtureFiles: ["parser.js", "parser.test.js"],
      expectedBoundedBehavior: [
        "route to executor_then_reviewer",
        "law and project-note retrieval stays snippet-bounded",
        "bounded read/edit/exec tool profile",
      ],
      keyAssertions: [
        "native prompt owns active context payload",
        "bounded tools are actually used",
        "task completes without broad host inflation",
      ],
      validationModes: ["unit", "mocked_host_boundary", "live"],
      artifactExpectations: [
        "native run artifact with route/retrieval/tool profile",
        "tool boundary snapshot and focused test result",
      ],
      testFiles: [
        "src/clawbound/runtime.test.ts",
        "src/clawbound/native-smoke.test.ts",
        "src/clawbound/native-live-sanity.live.test.ts",
      ],
    },
    {
      id: "delegated-review",
      taskClass: "delegated_review",
      taskPrompt:
        "Use the sessions_spawn tool exactly once to delegate a narrow review of parseCsvLine quoted-field handling and regression risk. Do not edit files yourself.",
      fixtureFiles: ["parser.js", "parser.test.js"],
      expectedBoundedBehavior: [
        "parent may spawn exactly one bounded subagent",
        "child receives bounded handoff instead of full transcript",
        "child tool profile excludes nested sessions_spawn",
      ],
      keyAssertions: [
        "handoff artifact records retrieved units and bounded context only",
        "prompt inflation remains suppressed on parent and child",
        "delegated review stays non-mutating",
      ],
      validationModes: ["unit", "live"],
      artifactExpectations: [
        "handoff artifact",
        "parent and child run artifacts",
        "child session transcript",
      ],
      testFiles: [
        "src/clawbound/context/native.test.ts",
        "src/clawbound/native-subagent-live.live.test.ts",
      ],
    },
    {
      id: "multi-turn-continuity",
      taskClass: "multi_turn_continuity",
      taskPrompt:
        "Explain parser behavior, review edge cases, make the bounded fix, then verify the focused parser test result across one small continued session.",
      fixtureFiles: ["parser.js", "parser.test.js"],
      expectedBoundedBehavior: [
        "active context stays bounded across turns",
        "verification-like turns stay on reviewer path",
        "compact artifacts explain carried, omitted, and bounded context",
      ],
      keyAssertions: [
        "native prompt size stays in a tight range over turns",
        "silent prompt growth does not occur",
        "continuity artifacts expose omitted reasons and boundedness reasons",
      ],
      validationModes: ["live"],
      artifactExpectations: [
        "session-level live report",
        "per-turn compact artifacts",
      ],
      testFiles: ["src/clawbound/native-multiturn-live.live.test.ts"],
    },
    {
      id: "diff-aware-review",
      taskClass: "diff_review",
      taskPrompt:
        "Review the proposed patch in parser.diff for correctness and regression risk without editing files. Keep scope to parser.diff, parser.js, and parser.test.js.",
      fixtureFiles: ["parser.diff", "parser.js", "parser.test.js"],
      expectedBoundedBehavior: [
        "reviewer path with read/exec only",
        "review stays non-mutating",
        "analysis stays within diff-scoped files",
      ],
      keyAssertions: [
        "native path finds the real regression in the diff",
        "prompt inflation remains materially suppressed",
        "compact artifacts remain interpretable",
      ],
      validationModes: ["live"],
      artifactExpectations: [
        "live diff-review report entry",
        "native run artifact with bounded compact summary",
      ],
      testFiles: ["src/clawbound/native-diff-review-live.live.test.ts"],
    },
    {
      id: "multi-file-bounded-fix",
      taskClass: "multi_file_code_change",
      taskPrompt:
        "Fix parseCsvLine so parser.test.js passes without changing public API. Keep scope to parser.js, csv-utils.js, and parser.test.js. Read the relevant files first and run only the focused parser test before finalizing.",
      fixtureFiles: ["parser.js", "csv-utils.js", "parser.test.js"],
      expectedBoundedBehavior: [
        "executor_then_reviewer route with bounded read/edit/exec",
        "work stays within the intended three-file set",
        "verification uses the focused test command form",
      ],
      keyAssertions: [
        "native path completes the fix and focused test passes",
        "prompt inflation remains materially below host path",
        "compact artifact stays interpretable for the bounded native run",
      ],
      validationModes: ["live"],
      artifactExpectations: [
        "live report with focused test result",
        "native run artifact",
        "compact artifact",
      ],
      testFiles: ["src/clawbound/native-multifile-live.live.test.ts"],
    },
  ] satisfies ClawBoundBenchmarkWorkflow[],
} as const;

export const STEWARD_BENCHMARK_VALIDATION_CATEGORIES = {
  unit: STEWARD_BENCHMARK_PACK.workflows.filter((workflow) =>
    workflow.validationModes.includes("unit"),
  ),
  mockedHostBoundary: STEWARD_BENCHMARK_PACK.workflows.filter((workflow) =>
    workflow.validationModes.includes("mocked_host_boundary"),
  ),
  live: STEWARD_BENCHMARK_PACK.workflows.filter((workflow) =>
    workflow.validationModes.includes("live"),
  ),
} as const;
