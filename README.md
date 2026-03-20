# ClawBound

ClawBound is a bounded sparse-context runtime prototype for agent execution governance. It is built on the [OpenClaw](https://github.com/openclaw/openclaw) host runtime.

ClawBound exists to prove a narrow idea: the host runtime, not the model, should own prompt assembly, tool exposure, retrieval decisions, and handoff boundaries during agent execution. Every run produces inspectable artifacts at every decision point.

This is a **research preview**, not a general agent platform, not a developer preview, and not a production-ready product.

## Runtime architecture

![ClawBound Runtime Architecture](docs/assets/clawbound-runtime-architecture.svg)

The diagram above shows the full runtime pipeline. Every run passes through nine stages, from task input to shadow snapshot. Each stage produces an inspectable artifact. Retrieval gating can produce a no-load outcome (no retrieval context sent to the model). Tool profiles are bounded per execution mode. The event trace spans all stages and is persisted alongside the shadow snapshot.

## What ClawBound does

- **Bounded prompt assembly.** The runtime owns final context composition. The model receives a task-shaped prompt, not an unbounded context dump.
- **Bounded tool exposure.** Tool profiles are scoped per task mode. Review tasks get read-only tools. Code-fix tasks get read/edit/exec. The runtime enforces the boundary.
- **Gated retrieval.** Retrieval is optional and snippet-bounded. If nothing meets the relevance gate, the runtime sends no retrieval context. No-load is a legal outcome.
- **Bounded handoff and continuity.** Subagent delegation has explicit scope. Multi-turn sessions have active context monitoring. Silent prompt growth is detected and flagged.
- **Trace-native observability.** Every run produces inspectable artifacts for route, retrieval, prompt, and tool policy decisions.

## Validated workflows

The following workflows have been validated end-to-end on a live provider path:

- Single-turn answer
- Single-turn review
- Single-turn code-fix
- Delegated review (bounded subagent handoff)
- Multi-turn bounded continuity
- Multi-file bounded fix

## Known-good provider path

| Field | Value |
|-------|-------|
| Provider/model | `openai-codex/gpt-5.1` |
| Auth | `openai-codex:default` (OAuth) |
| Probe status | `ok` |

This is the only provider path included in the publish gate. Other providers may be configured but are not part of this release verification.

## What this preview does not include

- Broad provider reliability or provider-independent validation
- Broad workflow coverage beyond the representative set above
- Production readiness or enterprise deployment support
- Memory, semantic retrieval, or long-context innovations
- UI or end-user installation polish

## Source structure

```
src/clawbound/
  runtime.ts              # core bounded runtime
  shadow.ts               # shadow execution path
  context/
    engine.ts             # context assembly engine
    native.ts             # native context path
  benchmarks/
    pack.ts               # frozen benchmark pack
    live-harness.ts       # live validation harness
  native-live-sanity.live.test.ts
  native-review-live.live.test.ts
  native-subagent-live.live.test.ts
  native-multiturn-live.live.test.ts
  native-multifile-live.live.test.ts
```

## Documentation

- [Research Preview](docs/clawbound-research-preview.md) -- positioning and claim boundary
- [Known-Good Runbook](docs/clawbound-known-good-runbook.md) -- local reproduction steps
- [Publish Gate](docs/clawbound-publish-gate.md) -- release criteria and blocking conditions
- [Benchmark Pack](docs/clawbound-benchmark-pack.md) -- frozen benchmark surface
- [Design Principles](docs/clawbound-design-principles.md) -- runtime design rationale
- [Release Verification Evidence](docs/release-verification/) -- evidence bundle

## Preconditions

- Node 22+
- `pnpm`
- A source checkout of the [OpenClaw](https://github.com/openclaw/openclaw) repo (ClawBound lives inside it)
- An `openai-codex` OAuth profile for live validation

## License

MIT
