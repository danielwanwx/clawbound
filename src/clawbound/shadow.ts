import path from "node:path";
import type { ClawBoundLocalContextItem, ClawBoundPlanRunResult } from "./runtime.js";
import { ClawBoundRuntime } from "./runtime.js";

export type EmbeddedClawBoundShadowInput = {
  env?: NodeJS.ProcessEnv;
  workspaceDir: string;
  hostRunId: string;
  sessionId: string;
  sessionKey?: string;
  prompt: string;
  candidateTools?: string[];
  localContext?: ClawBoundLocalContextItem[];
  continuationOf?: string;
};

export function isClawBoundShadowModeEnabled(env: NodeJS.ProcessEnv = process.env) {
  const raw =
    env.STEWARD_SHADOW_MODE ??
    env.OPENCLAW_STEWARD_SHADOW_MODE ??
    env.OPENCLAW_STEWARD_SHADOW ??
    "";
  const normalized = raw.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

export async function planClawBoundShadowForEmbeddedAttempt(
  input: EmbeddedClawBoundShadowInput,
): Promise<ClawBoundPlanRunResult | null> {
  if (!isClawBoundShadowModeEnabled(input.env)) {
    return null;
  }
  const runtime = new ClawBoundRuntime({
    shadowRootDir: path.join(input.workspaceDir, ".clawbound", "shadow"),
  });
  return runtime.planRun({
    hostRunId: input.hostRunId,
    sessionId: input.sessionId,
    sessionKey: input.sessionKey,
    conversationId: input.sessionKey ?? input.sessionId,
    userInput: input.prompt,
    continuationOf: input.continuationOf,
    candidateTools: input.candidateTools,
    localContext: input.localContext,
    sourceHost: "clawbound-shadow/openclaw-embedded-runner",
    emitToHostEvents: true,
  });
}
