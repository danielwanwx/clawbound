---
title: "ClawBound Publish Gate"
summary: "Minimum release gate for publishing ClawBound as a research preview."
read_when:
  - Deciding whether ClawBound is ready for a research-preview publish
  - Collecting the evidence bundle for a preview cut
  - Separating gating workflows from experimental workflows
---

# ClawBound Publish Gate

This document defines the minimum gate for publishing ClawBound as a
**research preview**.

It is deliberately narrower than a normal release checklist. It exists to make
the first ClawBound publish honest and reproducible.

## Research-Preview Standard

A ClawBound research preview is publishable only if all of the following are
true:

- the bounded runtime claims in
  [ClawBound Research Preview](./clawbound-research-preview.md)
  remain accurate
- the known-good provider path in
  [ClawBound Known-Good Runbook](./clawbound-known-good-runbook.md)
  still reproduces
- the required benchmark commands pass on the current commit
- the evidence bundle for that commit is preserved

## Required Provider Path

The preview gate requires exactly one known-good live provider path:

- default model: `openai-codex/gpt-5.1`
- auth profile: `openai-codex:default`
- probe state: `status = ok`
- every fresh gating live report must also show `provider = openai-codex` and
  `model = gpt-5.1`

Other configured providers may be disclosed, but they are not part of the
publish gate unless explicitly re-promoted into the gate.

## Required Benchmark Commands

### Deterministic gates

```bash
pnpm test:clawbound-bench:unit
pnpm test:clawbound-bench:mocked
```

### Provider preflight

```bash
openclaw models status --json --probe
```

### Gating live subset

```bash
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts \
  src/clawbound/native-live-sanity.live.test.ts \
  src/clawbound/native-review-live.live.test.ts \
  src/clawbound/native-subagent-live.live.test.ts \
  src/clawbound/native-multiturn-live.live.test.ts \
  src/clawbound/native-multifile-live.live.test.ts
```

## Gating vs Experimental Workflows

### Gating

These workflows must produce usable evidence for a publish cut:

| Workflow                 | Primary live harness                             | Publish expectation    |
| ------------------------ | ------------------------------------------------ | ---------------------- |
| `answer`                 | `src/clawbound/native-live-sanity.live.test.ts`    | `valid_product_result` |
| `code-fix`               | `src/clawbound/native-live-sanity.live.test.ts`    | `valid_product_result` |
| `review`                 | `src/clawbound/native-review-live.live.test.ts`    | `valid_product_result` |
| `delegated-review`       | `src/clawbound/native-subagent-live.live.test.ts`  | `valid_product_result` |
| `multi-turn-continuity`  | `src/clawbound/native-multiturn-live.live.test.ts` | `valid_product_result` |
| `multi-file-bounded-fix` | `src/clawbound/native-multifile-live.live.test.ts` | `valid_product_result` |

### Experimental

These workflows may be included in the preview narrative, but they are not
publish-gating yet:

| Workflow            | Primary live harness                               | Current status                        |
| ------------------- | -------------------------------------------------- | ------------------------------------- |
| `diff-aware-review` | `src/clawbound/native-diff-review-live.live.test.ts` | valuable but still provider-sensitive |

## Required Evidence Bundle

Every preview cut should preserve one evidence bundle tied to a specific commit.

Minimum contents:

- git commit SHA
- timestamp and operator
- `openclaw models status --json --probe` output
- output from:
  - `pnpm test:clawbound-bench:unit`
  - `pnpm test:clawbound-bench:mocked`
  - the gating live subset command
- report JSON paths or copied report JSON files for:
  - `clawbound-native-live-sanity-report.json`
  - `clawbound-native-review-live-report.json`
  - `clawbound-native-subagent-live-report.json`
  - `clawbound-native-multiturn-live-report.json`
  - `clawbound-native-multifile-live-report.json`
- a short summary that states:
  - provider/model used
  - which workflows were gating
  - whether any runs were externally blocked

## Publish-Blocking Conditions

Any of the following block a ClawBound research-preview publish:

- unit benchmark failure
- mocked benchmark failure
- `openai-codex/gpt-5.1` not probeable as `ok`
- missing or invalid `openai-codex:default` auth profile
- any gating live workflow failing to produce `valid_product_result`
- any gating live harness failing before writing a fresh report artifact
- any fresh gating live report resolving to a provider/model other than
  `openai-codex/gpt-5.1`
- any config-schema mismatch or local harness boot failure during the gating
  live subset
- any gating workflow showing boundedness regression
- missing evidence bundle
- docs that overstate ClawBound beyond research-preview scope

For the publish gate, **externally blocked live runs are still blocking**. They
do not count as product failures, but they also do not count as publishable
evidence.

## Required Disclosures In The Publish

The first publish must explicitly disclose:

- ClawBound is a research preview
- the preview is source-first and benchmark-driven
- one provider path is known-good today
- alternative providers may be configured but are not part of the publish gate
- workflow coverage is representative, not broad

## What Does Not Block The First Publish

These do not block the first research-preview publish:

- lack of a second known-good provider path
- lack of broad workflow coverage
- lack of UI work
- lack of memory or semantic retrieval features
- non-gating experimental live workflows remaining provider-sensitive

## Related Docs

- [ClawBound Research Preview](./clawbound-research-preview.md)
- [ClawBound Known-Good Runbook](./clawbound-known-good-runbook.md)
- [ClawBound Benchmark Pack](./clawbound-benchmark-pack.md)
