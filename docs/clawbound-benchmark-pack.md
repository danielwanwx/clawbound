# ClawBound Benchmark Pack

ClawBound now has a bounded validation surface large enough to freeze into a reusable benchmark pack.
This pack is the current regression and comparison baseline for the host-native sparse runtime path.

## Purpose

- Re-run the currently validated bounded workflows without redefining them each time.
- Preserve what ClawBound has already proven across unit, mocked host-boundary, and live validation.
- Give future phases a concise baseline before comparing provider paths, execution quality, or wider host coverage.

## Pack Source

- Machine-readable inventory: `src/clawbound/benchmarks/pack.ts`
- Integrity test: `src/clawbound/benchmarks/pack.test.ts`

## Workflow Inventory

| Workflow                 | Task Class               | Validation Modes                       | Primary Tests                                                                |
| ------------------------ | ------------------------ | -------------------------------------- | ---------------------------------------------------------------------------- |
| `answer`                 | `answer`                 | `unit`, `mocked_host_boundary`, `live` | `runtime.test.ts`, `native-smoke.test.ts`, `native-live-sanity.live.test.ts` |
| `review`                 | `review`                 | `live`                                 | `native-review-live.live.test.ts`                                            |
| `code-fix`               | `code_change`            | `unit`, `mocked_host_boundary`, `live` | `runtime.test.ts`, `native-smoke.test.ts`, `native-live-sanity.live.test.ts` |
| `delegated-review`       | `delegated_review`       | `unit`, `live`                         | `context/native.test.ts`, `native-subagent-live.live.test.ts`                |
| `multi-turn-continuity`  | `multi_turn_continuity`  | `live`                                 | `native-multiturn-live.live.test.ts`                                         |
| `diff-aware-review`      | `diff_review`            | `live`                                 | `native-diff-review-live.live.test.ts`                                       |
| `multi-file-bounded-fix` | `multi_file_code_change` | `live`                                 | `native-multifile-live.live.test.ts`                                         |

## Validation Categories

### Unit

Use this to validate deterministic planning, context-engine mechanics, compact artifacts, and the benchmark inventory itself.

- `src/clawbound/runtime.test.ts`
- `src/clawbound/context/engine.test.ts`
- `src/clawbound/context/native.test.ts`
- `src/clawbound/benchmarks/pack.test.ts`

### Mocked Host-Boundary

Use this to validate real host prompt/tool assembly with a mocked model boundary.

- `src/clawbound/native-smoke.test.ts`

### Live

Use this to validate the bounded runtime with a real model/provider path.

- `src/clawbound/native-live-sanity.live.test.ts`
- `src/clawbound/native-review-live.live.test.ts`
- `src/clawbound/native-subagent-live.live.test.ts`
- `src/clawbound/native-multiturn-live.live.test.ts`
- `src/clawbound/native-diff-review-live.live.test.ts`
- `src/clawbound/native-multifile-live.live.test.ts`

## How To Run The Pack

### Unit

```bash
npm run test:clawbound-bench:unit
```

### Mocked Host-Boundary

```bash
npm run test:clawbound-bench:mocked
```

### Live

```bash
npm run test:clawbound-bench:live
```

### Full Pack

```bash
npm run test:clawbound-bench:all
```

## Artifact Expectations

- Unit runs should produce persisted native planning or compact artifacts where the relevant test asserts them.
- Mocked host-boundary runs should produce prompt/tool boundary evidence plus bounded native run artifacts for the ClawBound path.
- Live runs should produce report JSON plus per-run session/context-engine artifacts under the temp workspace used by the test.

## What Is Still Out Of Scope

- Broad host rollout
- UI or dashboards
- Memory products
- Semantic retrieval
- DAG compaction
- Large-repo or broad codebase workflows
- Provider benchmarking beyond the bounded validation path
