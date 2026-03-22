"""ClawBound CLI — native entrypoint.

Proves ClawBound can run independently as a standalone engine.
Zero imports from outside clawbound.

Usage:
    clawbound --provider anthropic --model claude-sonnet-4-20250514 --message "What is 2+2?"

Options:
    --provider    Provider name (anthropic, google, minimax)
    --model       Model identifier
    --message     User message
    --api-key     API key (optional, falls back to env)
    --session-id  Session ID for multi-turn continuity (optional)
    --json        Output full JSON response instead of content only
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Literal

from clawbound.engine import (
    ClawBoundEngine,
    EngineConfig,
    EngineRequest,
    EngineResponse,
    create_engine,
)


# ─── Public types ────────────────────────────────────────────────────────────


class RunOptions:
    __slots__ = ("provider", "model", "message", "api_key", "session_id", "provider_env")

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        message: str,
        api_key: str | None = None,
        session_id: str | None = None,
        provider_env: dict[str, str | None] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.message = message
        self.api_key = api_key
        self.session_id = session_id
        self.provider_env = provider_env


OutputFormat = Literal["text", "json"]


# ─── Arg parsing ─────────────────────────────────────────────────────────────


HELP_TEXT = """Usage: clawbound --provider NAME --model MODEL --message "text" [options]

Options:
  --provider    Provider name (anthropic, google, minimax)
  --model       Model identifier
  --message     User message
  --api-key     API key (optional, falls back to env)
  --session-id  Session ID for multi-turn continuity (optional)
  --json        Output full JSON response instead of content only
  --help        Show this help message
"""


def parse_args(args: list[str]) -> RunOptions:
    """Parse CLI arguments into RunOptions."""
    if "--help" in args or "-h" in args:
        sys.stdout.write(HELP_TEXT)
        sys.exit(0)

    parsed: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            parsed[arg] = args[i + 1]
            i += 2
        else:
            i += 1

    provider = parsed.get("--provider")
    model = parsed.get("--model")
    message = parsed.get("--message")

    if not provider:
        raise ValueError("--provider is required")
    if not model:
        raise ValueError("--model is required")
    if not message:
        raise ValueError("--message is required")

    return RunOptions(
        provider=provider,
        model=model,
        message=message,
        api_key=parsed.get("--api-key"),
        session_id=parsed.get("--session-id"),
    )


# ─── Output formatting ───────────────────────────────────────────────────────


def format_result(response: EngineResponse, fmt: OutputFormat) -> str:
    """Format engine response for output."""
    if fmt == "json":
        data = {
            "content": response.content,
            "run_id": response.run_id,
            "trace_id": response.trace_id,
            "termination": response.termination,
            "iterations": response.iterations,
            "diagnostics": response.diagnostics.model_dump(),
            "duration_ms": response.duration_ms,
        }
        return json.dumps(data, indent=2, default=str)
    return response.content


# ─── Programmatic entrypoint ──────────────────────────────────────────────────


async def run_clawbound(
    options: RunOptions,
    engine: ClawBoundEngine | None = None,
) -> EngineResponse:
    """Run ClawBound directly — no external host, no bridge, no routing.

    Accepts an optional pre-built engine for testing. When omitted,
    creates a real engine from the provided options.
    """
    e = engine or create_engine(EngineConfig(
        provider=options.provider,
        model=options.model,
        api_key=options.api_key,
        provider_env=options.provider_env,
    ))

    return await e.run(EngineRequest(
        message=options.message,
        session_id=options.session_id,
    ))


# ─── CLI main ────────────────────────────────────────────────────────────────


async def async_main(args: list[str] | None = None) -> None:
    """Async CLI main function."""
    if args is None:
        args = sys.argv[1:]

    options = parse_args(args)
    fmt: OutputFormat = "json" if "--json" in args else "text"

    response = await run_clawbound(options)
    output = format_result(response, fmt)

    sys.stdout.write(output + "\n")


def main() -> None:
    """Synchronous entrypoint for console_scripts."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
