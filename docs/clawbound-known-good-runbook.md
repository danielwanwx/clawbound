---
title: "ClawBound Known-Good Runbook"
summary: "Source-based reproduction runbook for the current known-good ClawBound preview path."
read_when:
  - Reproducing the current ClawBound research preview locally
  - Verifying the known-good provider path before running live benchmarks
  - Distinguishing external provider failures from ClawBound runtime failures
---

# ClawBound Known-Good Runbook

This runbook defines the **current known-good reproduction path** for the
ClawBound research preview.

It is intentionally narrow. It does not promise broad provider coverage.

## Known-Good Provider Path

- active default model: `openai-codex/gpt-5.1`
- auth mechanism: `openai-codex` OAuth profile
- expected profile id: `openai-codex:default`
- acceptable fallback posture: optional and non-gating

For the research preview, treat `openai-codex/gpt-5.1` as the only required live
path. Other configured providers may exist, but they are not part of the
preview gate unless explicitly revalidated.

## Preconditions

- Node 22+
- `pnpm`
- a source checkout of this repo
- an interactive terminal for OAuth login if the `openai-codex` profile is not
  already present

## Source Setup

```bash
pnpm install
pnpm build
```

## Provider Auth Setup

If the `openai-codex` OAuth profile is missing, run:

```bash
openclaw models auth login --provider openai-codex
```

Then make the known-good model explicit:

```bash
openclaw models set openai-codex/gpt-5.1
```

If you want the narrowest possible preview path, clear fallback models:

```bash
openclaw models fallbacks clear
```

That step is optional. Fallbacks are not part of the preview gate.

## Probe Commands

### Check configured state

```bash
openclaw models list --json
openclaw models status --json
openclaw config validate
```

Expected signal:

- `openai-codex/gpt-5.1` is listed as `available: true`
- `defaultModel` and `resolvedDefault` are `openai-codex/gpt-5.1`

### Check live provider/auth health

```bash
openclaw models status --json --check
openclaw models status --json --probe
```

Expected signal:

- `openai-codex:default` exists
- its OAuth status is `ok`
- the probe result for `openai-codex/gpt-5.1` is `status: "ok"`

Other providers may show `auth`, `unknown`, or other non-OK states. That does
not block the ClawBound preview as long as the known-good `openai-codex` path is
healthy.

Also verify that the live harnesses do not emit config-schema errors during
startup. A successful standalone `openclaw config validate` is necessary, but it
is not sufficient if a live harness still reports config incompatibility or
fails before writing a fresh report artifact.

## Benchmark Commands

### Deterministic gates

```bash
pnpm test:clawbound-bench:unit
pnpm test:clawbound-bench:mocked
```

Expected outcome:

- both commands exit `0`

### Live preview subset

```bash
OPENCLAW_LIVE_TEST=1 ./node_modules/.bin/vitest run --config vitest.live.config.ts \
  src/clawbound/native-live-sanity.live.test.ts \
  src/clawbound/native-review-live.live.test.ts \
  src/clawbound/native-subagent-live.live.test.ts \
  src/clawbound/native-multiturn-live.live.test.ts \
  src/clawbound/native-multifile-live.live.test.ts
```

Expected outcome:

- each test writes a report JSON under the temp directory used by the harness
- each gating workflow reaches `validationOutcome = valid_product_result`
- each fresh live report records `provider = openai-codex` and `model = gpt-5.1`
- native runs stay bounded on prompt/tool/file-scope indicators

The live preview subset covers:

- answer
- code-fix
- review
- delegated review
- multi-turn continuity
- multi-file bounded fix

## Failure Interpretation

ClawBound live reports use an honest failure taxonomy. Interpret failures like
this:

- `valid_product_result`
  - acceptable preview evidence
- `provider_connection`
  - external provider or transport failure
  - rerun in a stable provider window before drawing product conclusions
- `provider_timeout`
  - external provider delay or availability issue
  - rerun before drawing product conclusions
- `auth_failure`
  - provider auth is missing, expired, or invalid
  - fix auth before rerunning
- `harness_failure`
  - local benchmark harness issue
  - fix the harness before treating results as evidence
- `runtime_failure`
  - meaningful ClawBound-side failure after execution started
  - treat as a product issue until explained

If a gating live harness:

- emits config-schema errors
- fails before writing a fresh report artifact
- or writes a fresh report that resolved to a provider/model other than
  `openai-codex/gpt-5.1`

then the publish cut is blocked even if other parts of the runbook succeed.

## Expected Preview Outcome

A successful research-preview reproduction run produces:

- passing unit and mocked benchmark surfaces
- a healthy `openai-codex/gpt-5.1` probe result
- valid live results for the preview-gating subset
- report artifacts that show bounded prompt ownership, bounded tool exposure,
  and interpretable runtime artifacts

## What This Runbook Does Not Promise

This runbook does not promise:

- broad provider reliability
- provider-independent validation
- broad workflow coverage
- general end-user installation polish

For the authoritative publish requirements, see
[ClawBound Publish Gate](./clawbound-publish-gate.md).
