"""RuntimeOrchestrator — composes all modules into end-to-end invocation.

Pipeline: TaskCompiler -> PolicyEngine -> PromptBuilder -> (ToolBroker + SignalProcessor + ActionGate) -> ExecutionLoop
"""

from __future__ import annotations

from typing import Any

from clawbound.contracts.types import (
    KernelAsset,
    LocalContextItem,
    LoopConfig,
    ModelMessage,
    OrchestratorDiagnostics,
    OrchestratorResult,
    ToolDefinition,
)
from clawbound.execution_loop.action_gate import ActionGateImpl
from clawbound.execution_loop.loop import run_loop
from clawbound.policy_engine.engine import PolicyEngineImpl, default_runtime_config
from clawbound.prompt_builder.builder import PromptBuilderImpl
from clawbound.signal_processor.processor import SignalProcessorImpl
from clawbound.task_compiler.compiler import CompileInput, TaskCompilerImpl
from clawbound.tool_broker.broker import ToolBrokerExecuteFn, ToolBrokerImpl

DEFAULT_KERNEL = KernelAsset(
    version="context-kernel-v0",
    content="- Do not fabricate completion.\n- Default to sparse context.",
    token_estimate=10,
)


class ToolRegistration:
    __slots__ = ("definition", "execute_fn")

    def __init__(self, definition: ToolDefinition, execute_fn: ToolBrokerExecuteFn) -> None:
        self.definition = definition
        self.execute_fn = execute_fn


async def run_orchestrator(
    *,
    run_id: str,
    trace_id: str,
    user_message: str,
    model_adapter: Any,
    local_context: tuple[LocalContextItem, ...] | None = None,
    tool_registrations: list[ToolRegistration] | None = None,
    max_iterations: int | None = None,
    initial_messages: tuple[ModelMessage, ...] | None = None,
) -> OrchestratorResult:
    # 1. Compile task
    compiler = TaskCompilerImpl()
    ctx_items = list(local_context) if local_context else []
    task_spec = compiler.compile_from_input(CompileInput(
        trace_id=trace_id,
        user_input=user_message,
        continuation_of=None,
        local_context=ctx_items,
    ))

    # 2. Resolve policy
    policy_engine = PolicyEngineImpl()
    runtime_policy = policy_engine.resolve(task_spec, default_runtime_config())

    # 3. Set up modules
    prompt_builder = PromptBuilderImpl()
    tool_broker = ToolBrokerImpl()
    signal_processor = SignalProcessorImpl()
    action_gate = ActionGateImpl()

    for reg in (tool_registrations or []):
        tool_broker.register(reg.definition, reg.execute_fn)

    # 4. Diagnostic pre-build
    from clawbound.contracts.types import PromptBuildInput
    prompt_envelope = prompt_builder.build(PromptBuildInput(
        run_id=run_id,
        trace_id=trace_id,
        task_spec=task_spec,
        runtime_policy=runtime_policy,
        kernel=DEFAULT_KERNEL,
        local_context=tuple(ctx_items),
        retrieved_units=(),
        no_load=True,
    ))

    tool_defs = tool_broker.resolve_for_turn(runtime_policy.tool_profile)

    # 5. Run execution loop
    iters = max_iterations or runtime_policy.iteration_policy.max_turns
    loop_result = await run_loop(
        run_id=run_id,
        trace_id=trace_id,
        task_spec=task_spec,
        runtime_policy=runtime_policy,
        kernel=DEFAULT_KERNEL,
        local_context=tuple(ctx_items),
        retrieved_units=(),
        user_message=user_message,
        initial_messages=initial_messages,
        config=LoopConfig(
            max_iterations=iters,
            max_consecutive_errors=runtime_policy.iteration_policy.max_consecutive_errors,
            max_transient_retries=runtime_policy.iteration_policy.max_transient_retries,
        ),
        prompt_builder=prompt_builder,
        tool_broker=tool_broker,
        signal_processor=signal_processor,
        model_adapter=model_adapter,
        action_gate=action_gate,
    )

    # 6. Build diagnostics
    diagnostics = OrchestratorDiagnostics(
        task_type=task_spec.task_type,
        execution_mode=task_spec.execution_mode,
        complexity=task_spec.complexity,
        risk=task_spec.risk,
        prompt_token_estimate=prompt_envelope.assembly_stats.total_estimated_tokens,
        tools_resolved=len(tool_defs),
        segments_admitted=prompt_envelope.assembly_stats.segments_admitted,
        segments_rejected=prompt_envelope.assembly_stats.segments_rejected,
    )

    return OrchestratorResult(
        run_id=run_id,
        trace_id=trace_id,
        termination=loop_result.termination,
        final_content=loop_result.final_content,
        iterations=loop_result.iterations,
        diagnostics=diagnostics,
        loop_result=loop_result,
        task_spec=task_spec,
        runtime_policy=runtime_policy,
    )
