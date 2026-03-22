"""ExecutionLoop — runtime-owned turn orchestration.

Consumes the 5 extracted modules through explicit interfaces. Owns:
- turn-state transitions, iteration counting, completion criteria
- tool call sequencing (gate -> broker -> signal)
- retry/repair on transient or invalid responses
- loop trace events for diagnostics

Does NOT own: task classification, policy derivation, prompt content,
tool implementation, signal transformation, provider-specific semantics.
"""

from __future__ import annotations

from typing import Any

from clawbound.contracts.types import (
    FinalAnswer,
    LoopConfig,
    LoopResult,
    LoopStepEvent,
    LoopTermination,
    ModelError,
    ModelMessage,
    PromptBuildInput,
    SignalBundle,
    ToolCalls,
    ToolResult,
)
from clawbound.execution_loop.action_gate import ActionGateImpl
from clawbound.prompt_builder.builder import PromptBuilderImpl
from clawbound.signal_processor.processor import SignalProcessorImpl
from clawbound.tool_broker.broker import ToolBrokerImpl


async def run_loop(
    *,
    run_id: str,
    trace_id: str,
    task_spec: Any,
    runtime_policy: Any,
    kernel: Any,
    local_context: tuple[Any, ...],
    retrieved_units: tuple[Any, ...],
    user_message: str,
    initial_messages: tuple[ModelMessage, ...] | None,
    config: LoopConfig,
    prompt_builder: PromptBuilderImpl,
    tool_broker: ToolBrokerImpl,
    signal_processor: SignalProcessorImpl,
    model_adapter: Any,
    action_gate: ActionGateImpl,
) -> LoopResult:
    # Build system prompt
    prompt_envelope = prompt_builder.build(PromptBuildInput(
        run_id=run_id,
        trace_id=trace_id,
        task_spec=task_spec,
        runtime_policy=runtime_policy,
        kernel=kernel,
        local_context=local_context,
        retrieved_units=retrieved_units,
        no_load=len(retrieved_units) == 0,
    ))

    # Resolve available tools
    tool_defs = tool_broker.resolve_for_turn(runtime_policy.tool_profile)

    # Loop state
    messages: list[ModelMessage] = []
    if initial_messages:
        messages.extend(initial_messages)
    messages.append(ModelMessage(role="user", content=user_message))

    all_tool_results: list[ToolResult] = []
    all_signal_bundles: list[SignalBundle] = []
    events: list[LoopStepEvent] = []

    iteration = 0
    consecutive_errors = 0
    termination: LoopTermination = "max_iterations"
    final_content = ""

    while iteration < config.max_iterations:
        iteration += 1

        events.append(LoopStepEvent(
            iteration=iteration,
            kind="model_request",
            detail={"message_count": len(messages), "tool_count": len(tool_defs)},
        ))

        try:
            from clawbound.contracts.types import ModelRequest
            response = await model_adapter.send(ModelRequest(
                system_prompt=prompt_envelope.system_prompt,
                messages=tuple(messages),
                tool_definitions=tuple(tool_defs),
            ))
        except Exception as err:
            events.append(LoopStepEvent(
                iteration=iteration,
                kind="error",
                detail={"source": "adapter_throw", "error": str(err)},
            ))
            consecutive_errors += 1
            if consecutive_errors >= config.max_consecutive_errors:
                termination = "adapter_failure"
                final_content = f"Adapter threw: {err}"
                break
            continue

        # Route response
        if isinstance(response, FinalAnswer):
            events.append(LoopStepEvent(
                iteration=iteration,
                kind="final_answer",
                detail={"content_length": len(response.content)},
            ))
            termination = "final_answer"
            final_content = response.content
            break

        if isinstance(response, ModelError):
            events.append(LoopStepEvent(
                iteration=iteration,
                kind="error",
                detail={"error": response.error, "is_transient": response.is_transient},
            ))

            if response.is_transient:
                retry_count = sum(
                    1 for e in events if e.kind == "retry" and e.iteration == iteration
                )
                if retry_count < config.max_transient_retries:
                    events.append(LoopStepEvent(
                        iteration=iteration,
                        kind="retry",
                        detail={"retry_count": retry_count + 1, "reason": "transient_error"},
                    ))
                    iteration -= 1
                    continue

            consecutive_errors += 1
            if consecutive_errors >= config.max_consecutive_errors:
                termination = "max_errors"
                final_content = f"Loop terminated after {consecutive_errors} consecutive errors. Last: {response.error}"
                break
            continue

        # Tool calls path
        assert isinstance(response, ToolCalls)
        consecutive_errors = 0

        assistant_content = response.reasoning or ""
        messages.append(ModelMessage(
            role="assistant",
            content=assistant_content,
            tool_calls=response.calls,
        ))

        for call in response.calls:
            gate_decision = action_gate.check(call.tool_name, call.args, runtime_policy)

            if not gate_decision.allowed:
                events.append(LoopStepEvent(
                    iteration=iteration,
                    kind="gate_denied",
                    detail={"tool_name": call.tool_name, "reason": gate_decision.reason},
                ))
                messages.append(ModelMessage(
                    role="tool",
                    content=f'Tool "{call.tool_name}" denied by action gate: {gate_decision.reason}',
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                ))
                continue

            from clawbound.contracts.types import ToolExecuteParams
            tool_result = await tool_broker.execute(ToolExecuteParams(
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                args=call.args,
                policy=runtime_policy.tool_profile,
            ))

            all_tool_results.append(tool_result)
            events.append(LoopStepEvent(
                iteration=iteration,
                kind="tool_execution",
                detail={
                    "tool_name": tool_result.tool_name,
                    "status": tool_result.status,
                    "output_kind": tool_result.output_kind,
                    "duration_ms": tool_result.duration_ms,
                },
            ))

            signal = signal_processor.process(tool_result)
            all_signal_bundles.append(signal)
            events.append(LoopStepEvent(
                iteration=iteration,
                kind="signal_processed",
                detail={
                    "tool_name": signal.tool_name,
                    "signal_kind": signal.structured.kind,
                    "compression_ratio": signal.compression_metrics.compression_ratio,
                },
            ))

            messages.append(ModelMessage(
                role="tool",
                content=signal.compressed_text,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
            ))

    if iteration >= config.max_iterations and termination == "max_iterations":
        final_content = final_content or f"Loop terminated after {config.max_iterations} iterations without final answer."

    return LoopResult(
        run_id=run_id,
        trace_id=trace_id,
        termination=termination,
        final_content=final_content,
        iterations=iteration,
        tool_results=tuple(all_tool_results),
        signal_bundles=tuple(all_signal_bundles),
        events=tuple(events),
    )
