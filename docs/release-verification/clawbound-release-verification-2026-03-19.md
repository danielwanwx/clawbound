# ClawBound Release Verification 2026-03-19

- Commit: `5167e6ac0241df644e71de28c2b7ae1de110dddb`
- Provider/model: `openai-codex/gpt-5.1`
- Auth profile: `openai-codex:default`
- Operator: Codex

---

## Run 1 (earlier today — did not ship)

### Commands Run

Preflight:

```bash
openclaw models status --json --probe
openclaw config validate
```

Deterministic gates:

```bash
npm run test:clawbound-bench:unit
npm run test:clawbound-bench:mocked
```

Live publish gate:

```bash
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts \
  src/clawbound/native-live-sanity.live.test.ts \
  src/clawbound/native-review-live.live.test.ts \
  src/clawbound/native-subagent-live.live.test.ts \
  src/clawbound/native-multiturn-live.live.test.ts \
  src/clawbound/native-multifile-live.live.test.ts
```

Fragile harness standalone reruns:

```bash
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts src/clawbound/native-subagent-live.live.test.ts
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts src/clawbound/native-multiturn-live.live.test.ts
```

### Results

Preflight:

- `openclaw config validate`: passed
- `openclaw models status --json --probe`:
  - `resolvedDefault = openai-codex/gpt-5.1`
  - `openai-codex:default` OAuth status = `ok`
  - probe status for `openai-codex/gpt-5.1` = `ok`

Deterministic gates:

- unit gate: passed (`38/38`)
- mocked gate: passed (`1/1`)

Full publish-gate confirmation:

- did not pass
- the combined live gate failed immediately on `native-subagent-live`
- fresh failure observed during the live gate:
  - classification: `harness_failure`
  - error: `listen EPERM: operation not permitted 127.0.0.1`
  - provider/model binding: intended `openai-codex/gpt-5.1`
- because the full gate already failed, it was stopped before waiting for the
  rest of the combined run to settle

Fragile harness confirmation:

- `native-subagent-live`: passed standalone
  - report: `/var/folders/yj/7htkqzsj22qgq9ch0tfjt6k80000gn/T/clawbound-native-subagent-live-report.json`
  - createdAt: `2026-03-19T17:58:15.222Z`
  - summary: `validProductResults=2`, `harnessFailures=0`
- `native-multiturn-live`: passed standalone
  - report: `/var/folders/yj/7htkqzsj22qgq9ch0tfjt6k80000gn/T/clawbound-native-multiturn-live-report.json`
  - createdAt: `2026-03-19T17:58:23.464Z`
  - summary: `validProductResults=2`, `runtimeFailures=0`

### Ship Decision (Run 1)

- Full publish gate pass: `no` (EPERM harness failure, not a product failure)
- Subagent pass: `yes` (standalone)
- Multiturn pass: `yes` (standalone)
- Fresh artifacts emitted: `yes` for standalone fragile reruns
- Provider/model binding correct: `yes`
- Recommendation: `Do not ship yet` — combined gate did not pass clean

---

## Run 2 (release-verification rerun)

### Commands Run

Preflight:

```bash
openclaw models status --json --probe
openclaw config validate
```

Deterministic gates:

```bash
npm run test:clawbound-bench:unit
npm run test:clawbound-bench:mocked
```

Full live publish gate (all 5 gating harnesses combined):

```bash
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts \
  src/clawbound/native-live-sanity.live.test.ts \
  src/clawbound/native-review-live.live.test.ts \
  src/clawbound/native-subagent-live.live.test.ts \
  src/clawbound/native-multiturn-live.live.test.ts \
  src/clawbound/native-multifile-live.live.test.ts
```

Fragile harness standalone reruns:

```bash
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts src/clawbound/native-subagent-live.live.test.ts
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts src/clawbound/native-multiturn-live.live.test.ts
```

### Results

Preflight:

- `openclaw config validate`: passed
- `openclaw models status --json --probe`:
  - `resolvedDefault = openai-codex/gpt-5.1`
  - `openai-codex:default` OAuth status = `ok`
  - probe status for `openai-codex/gpt-5.1` = `ok` (latency 3724ms)

Deterministic gates:

- unit gate: passed (`38/38`, 5 test files)
- mocked gate: passed (`1/1`)

Full publish-gate confirmation:

- **passed** — all 5 test files passed, 5/5 tests green
- no EPERM or harness failures
- combined run duration: ~115s
- per-harness results:
  - `native-live-sanity`: passed (101s)
  - `native-review-live`: passed (108s)
  - `native-subagent-live`: passed (97s)
  - `native-multiturn-live`: passed (114s)
  - `native-multifile-live`: passed (88s)

Fragile harness confirmation (standalone reruns):

- `native-subagent-live`: **passed** standalone
  - report: `clawbound-native-subagent-live-report.json`
  - createdAt: `2026-03-19T20:29:59.147Z`
  - summary: `validProductResults=2`, `harnessFailures=0`
  - subagentDelegationOccurred: `true`
  - taskStayedWithinDelegatedScope: `true`
- `native-multiturn-live`: **passed** standalone
  - report: `clawbound-native-multiturn-live-report.json`
  - createdAt: `2026-03-19T20:31:18.402Z`
  - summary: `validProductResults=2`, `runtimeFailures=0`
  - activeContextStayedBounded: `true`
  - silentPromptGrowthDetected: `false`

### Fresh Report Artifacts

All 5 gating reports emitted fresh:

| Report | createdAt | Provider | Model | Runs | Valid | Failures |
|--------|-----------|----------|-------|------|-------|----------|
| live-sanity | 2026-03-19T20:28:16.238Z | openai-codex | gpt-5.1 | 4 | 4 | 0 |
| review-live | 2026-03-19T20:28:22.919Z | openai-codex | gpt-5.1 | 2 | 2 | 0 |
| subagent-live | 2026-03-19T20:29:59.147Z | openai-codex | gpt-5.1 | 2 | 2 | 0 |
| multiturn-live | 2026-03-19T20:31:18.402Z | openai-codex | gpt-5.1 | 2 | 2 | 0 |
| multifile-live | 2026-03-19T20:28:03.269Z | openai-codex | gpt-5.1 | 2 | 2 | 0 |

- **12 total runs, 12 valid_product_result, 0 failures of any kind**
- Provider/model binding: `openai-codex/gpt-5.1` in all reports, no divergence
- Model binding diverged: `false` in all reports

### Evidence Bundle

Preserved at: `docs/release-verification/evidence-2026-03-19-run2/`

Contents:

- `clawbound-native-live-sanity-report.json`
- `clawbound-native-review-live-report.json`
- `clawbound-native-subagent-live-report.json`
- `clawbound-native-multiturn-live-report.json`
- `clawbound-native-multifile-live-report.json`

### Ship Decision (Run 2)

- Full publish gate pass again: **yes**
- Subagent pass again: **yes** (combined + standalone)
- Multiturn pass again: **yes** (combined + standalone)
- Fresh artifacts emitted again: **yes** (all 5 reports)
- Provider/model binding correct again: **yes** (`openai-codex/gpt-5.1`, no divergence)

---

## Final Ship Recommendation

**Ship the ClawBound research preview.**

Rationale:

- The full publish gate passed on the combined run (5/5 harnesses, 12/12 runs)
- Both historically fragile harnesses passed both combined and standalone
- Run 1's EPERM failure was a transient harness/OS-level issue, not a product failure
- All fresh report artifacts confirm `openai-codex/gpt-5.1` binding with zero divergence
- All gating workflows produced `valid_product_result`
- Deterministic gates (unit + mocked) passed on both runs
- Evidence bundle preserved
