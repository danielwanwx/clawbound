---
title: "ClawBound Research Preview"
summary: "Release note for the first ClawBound research preview."
---

# ClawBound Research Preview

ClawBound is a bounded sparse-context runtime prototype inside the OpenClaw repo. It is not a replacement for the host runtime and it is not a general agent platform. ClawBound exists to prove a narrower idea: that agent execution can be governed at the runtime layer through bounded prompt assembly, bounded tool exposure, and inspectable artifacts at every decision point.

## What this preview includes

ClawBound provides a native runtime path with:

- Host-owned prompt assembly where the runtime, not the model, controls final context composition
- Gated snippet retrieval where no-load is a legal retrieval outcome
- Bounded tool profiles scoped per task mode (read-only, review, read/edit/exec)
- Bounded subagent handoff with explicit delegation scope and trace
- Bounded multi-turn continuity with active context growth monitoring
- A frozen benchmark pack with deterministic, mocked, and live validation surfaces

## What has been validated

The following workflows have been validated end-to-end on a live provider path:

- Single-turn answer
- Single-turn review
- Single-turn code-fix
- Delegated review (bounded subagent handoff)
- Multi-turn bounded continuity
- Multi-file bounded fix

All gating live harnesses passed in the release verification run. Both historically fragile harnesses (subagent delegation and multi-turn continuity) passed in combined and standalone confirmation runs. Fresh report artifacts confirm correct provider/model binding with no divergence.

## Known-good provider path

- Provider/model: `openai-codex/gpt-5.1`
- Auth: `openai-codex:default` (OAuth)
- Probe status: `ok`

This is the only provider path included in the publish gate. Other providers may be configured but are not part of this release verification.

## What this preview does not include

- Broad provider reliability or provider-independent validation
- Broad workflow coverage beyond the representative set above
- Production readiness or enterprise deployment support
- Memory, semantic retrieval, or long-context innovations
- UI or end-user installation polish

## Where to go next

- [ClawBound Research Preview](../clawbound-research-preview.md) -- positioning and claim boundary
- [Known-Good Runbook](../clawbound-known-good-runbook.md) -- local reproduction steps
- [Publish Gate](../clawbound-publish-gate.md) -- release criteria and blocking conditions
- [Benchmark Pack](../clawbound-benchmark-pack.md) -- frozen benchmark surface
- [Release Verification Evidence](../release-verification/) -- evidence bundle from the verification run
