"""clawbound — Bounded sparse-context execution engine.

Public API:
    create_engine     - Create a production engine with provider resolution
    create_test_engine - Create a test engine with scripted responses
    run_clawbound     - Run ClawBound directly (programmatic entrypoint)

Types:
    ClawBoundEngine, EngineConfig, EngineRequest, EngineResponse, EngineEvent
"""

from clawbound.engine import (
    ClawBoundEngine,
    EngineConfig,
    EngineEvent,
    EngineRequest,
    EngineResponse,
    create_engine,
    create_test_engine,
)
from clawbound.cli import run_clawbound

__all__ = [
    "ClawBoundEngine",
    "EngineConfig",
    "EngineEvent",
    "EngineRequest",
    "EngineResponse",
    "create_engine",
    "create_test_engine",
    "run_clawbound",
]
