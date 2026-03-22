"""Anthropic Provider Adapter tests — Phase 6.

Covers:
1. Request translation (ModelRequest -> Anthropic format)
2. Message normalization (tool result batching, content blocks)
3. Response normalization (Anthropic response -> ModelResponse)
4. Tool definition translation
5. Error classification (transient, non-transient)
6. Full adapter with mocked httpx transport
7. Integration: loop + AnthropicAdapter with mocked transport

All tests use pure translation functions or httpx MockTransport.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from clawbound.contracts.types import (
    KernelAsset,
    LoopConfig,
    ModelMessage,
    ModelRequest,
    ModelToolCall,
    ToolDefinition,
)
from clawbound.provider_adapter.anthropic import (
    classify_http_error,
    create_anthropic_adapter,
    normalize_messages,
    normalize_response,
    translate_request,
    translate_tool_definitions,
)
from clawbound.provider_adapter.types import AnthropicConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────


TEST_CONFIG = AnthropicConfig(
    api_key="test-key",
    model="claude-3-haiku-20240307",
    max_tokens=1024,
)


def make_request(**overrides: Any) -> ModelRequest:
    defaults: dict[str, Any] = {
        "system_prompt": "You are helpful.",
        "messages": (ModelMessage(role="user", content="Hello"),),
        "tool_definitions": (),
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)


def make_anthropic_response(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    defaults.update(overrides)
    return defaults


def mock_transport(response_body: dict[str, Any], status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=response_body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def mock_transport_error(status: int, body: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def mock_transport_network_error() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("ECONNREFUSED")
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ─── 1. Request translation ──────────────────────────────────────────────────


class TestRequestTranslation:
    def test_translates_basic_request(self):
        result = translate_request(make_request(), TEST_CONFIG)

        assert result["model"] == "claude-3-haiku-20240307"
        assert result["max_tokens"] == 1024
        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_omits_system_when_empty(self):
        result = translate_request(
            make_request(system_prompt=""),
            TEST_CONFIG,
        )
        assert "system" not in result

    def test_omits_tools_when_none(self):
        result = translate_request(make_request(), TEST_CONFIG)
        assert "tools" not in result

    def test_includes_tools_when_provided(self):
        tools = (
            ToolDefinition(name="read_file", category="filesystem", risk_level="read_only", description="Read a file"),
            ToolDefinition(name="run_command", category="execution", risk_level="side_effect"),
        )
        result = translate_request(
            make_request(tool_definitions=tools),
            TEST_CONFIG,
        )

        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "read_file"
        assert result["tools"][0]["description"] == "Read a file"
        assert result["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
        assert result["tools"][1]["description"] == "Tool: run_command"


# ─── 2. Message normalization ────────────────────────────────────────────────


class TestMessageNormalization:
    def test_user_message_to_text_content_block(self):
        messages = (ModelMessage(role="user", content="Hello"),)
        result = normalize_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_assistant_with_tool_calls(self):
        messages = (
            ModelMessage(
                role="assistant",
                content="Let me read the file.",
                tool_calls=(
                    ModelToolCall(tool_call_id="tc-1", tool_name="read_file", args={"path": "/tmp/test.ts"}),
                ),
            ),
        )
        result = normalize_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0] == {"type": "text", "text": "Let me read the file."}
        assert result[0]["content"][1] == {
            "type": "tool_use",
            "id": "tc-1",
            "name": "read_file",
            "input": {"path": "/tmp/test.ts"},
        }

    def test_batches_consecutive_tool_messages(self):
        messages = (
            ModelMessage(role="user", content="Fix the bug."),
            ModelMessage(
                role="assistant",
                content="Reading files.",
                tool_calls=(
                    ModelToolCall(tool_call_id="tc-1", tool_name="read_file", args={"path": "a.ts"}),
                    ModelToolCall(tool_call_id="tc-2", tool_name="read_file", args={"path": "b.ts"}),
                ),
            ),
            ModelMessage(role="tool", content="file a contents", tool_call_id="tc-1", tool_name="read_file"),
            ModelMessage(role="tool", content="file b contents", tool_call_id="tc-2", tool_name="read_file"),
        )
        result = normalize_messages(messages)

        assert len(result) == 3  # user, assistant, user(tool_results)
        assert result[2]["role"] == "user"
        assert len(result[2]["content"]) == 2
        assert result[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tc-1", "content": "file a contents"}
        assert result[2]["content"][1] == {"type": "tool_result", "tool_use_id": "tc-2", "content": "file b contents"}

    def test_multi_turn_with_interleaved_tools(self):
        messages = (
            ModelMessage(role="user", content="Fix the bug."),
            ModelMessage(role="assistant", content="Reading.", tool_calls=(
                ModelToolCall(tool_call_id="tc-1", tool_name="read_file", args={}),
            )),
            ModelMessage(role="tool", content="file contents", tool_call_id="tc-1", tool_name="read_file"),
            ModelMessage(role="assistant", content="Now editing.", tool_calls=(
                ModelToolCall(tool_call_id="tc-2", tool_name="edit_file", args={}),
            )),
            ModelMessage(role="tool", content="edited ok", tool_call_id="tc-2", tool_name="edit_file"),
        )
        result = normalize_messages(messages)

        assert len(result) == 5
        assert [m["role"] for m in result] == ["user", "assistant", "user", "assistant", "user"]

    def test_assistant_without_tool_calls(self):
        messages = (ModelMessage(role="assistant", content="Just a text response."),)
        result = normalize_messages(messages)

        assert len(result) == 1
        assert result[0]["content"] == [{"type": "text", "text": "Just a text response."}]


# ─── 3. Response normalization ────────────────────────────────────────────────


class TestResponseNormalization:
    def test_text_only_as_final_answer(self):
        raw = make_anthropic_response(content=[{"type": "text", "text": "The answer is 42."}])
        result = normalize_response(raw)

        assert result.kind == "final_answer"
        assert result.content == "The answer is 42."

    def test_tool_use_as_tool_calls(self):
        raw = make_anthropic_response(content=[
            {"type": "text", "text": "I need to read this file."},
            {"type": "tool_use", "id": "toolu_abc", "name": "read_file", "input": {"path": "test.ts"}},
        ], stop_reason="tool_use")
        result = normalize_response(raw)

        assert result.kind == "tool_calls"
        assert len(result.calls) == 1
        assert result.calls[0].tool_call_id == "toolu_abc"
        assert result.calls[0].tool_name == "read_file"
        assert result.calls[0].args == {"path": "test.ts"}
        assert result.reasoning == "I need to read this file."

    def test_multiple_tool_use_blocks(self):
        raw = make_anthropic_response(content=[
            {"type": "tool_use", "id": "tc-1", "name": "read_file", "input": {"path": "a.ts"}},
            {"type": "tool_use", "id": "tc-2", "name": "read_file", "input": {"path": "b.ts"}},
        ], stop_reason="tool_use")
        result = normalize_response(raw)

        assert result.kind == "tool_calls"
        assert len(result.calls) == 2
        assert result.reasoning is None

    def test_joins_multiple_text_blocks(self):
        raw = make_anthropic_response(content=[
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ])
        result = normalize_response(raw)

        assert result.kind == "final_answer"
        assert result.content == "First part.\nSecond part."

    def test_empty_content_array(self):
        raw = make_anthropic_response(content=[])
        result = normalize_response(raw)

        assert result.kind == "final_answer"
        assert result.content == ""


# ─── 4. Tool definition translation ──────────────────────────────────────────


class TestToolDefinitionTranslation:
    def test_translates_to_anthropic_format(self):
        tools = (ToolDefinition(name="read_file", category="filesystem", risk_level="read_only", description="Read a file from disk"),)
        result = translate_tool_definitions(tools)

        assert len(result) == 1
        assert result[0] == {
            "name": "read_file",
            "description": "Read a file from disk",
            "input_schema": {"type": "object", "properties": {}},
        }

    def test_fallback_description(self):
        tools = (ToolDefinition(name="custom_tool", category="execution", risk_level="side_effect"),)
        result = translate_tool_definitions(tools)

        assert result[0]["description"] == "Tool: custom_tool"


# ─── 5. Error classification ─────────────────────────────────────────────────


class TestErrorClassification:
    def test_429_transient(self):
        result = classify_http_error(429, '{"error":{"message":"Rate limit exceeded"}}')
        assert result.is_transient is True
        assert "Rate limit exceeded" in result.error

    def test_529_transient(self):
        result = classify_http_error(529, '{"error":{"message":"Overloaded"}}')
        assert result.is_transient is True

    def test_500_transient(self):
        result = classify_http_error(500, "Internal server error")
        assert result.is_transient is True

    def test_503_transient(self):
        result = classify_http_error(503, "")
        assert result.is_transient is True

    def test_400_non_transient(self):
        result = classify_http_error(400, '{"error":{"message":"Invalid request"}}')
        assert result.is_transient is False
        assert "Invalid request" in result.error

    def test_401_non_transient(self):
        result = classify_http_error(401, '{"error":{"message":"Invalid API key"}}')
        assert result.is_transient is False

    def test_403_non_transient(self):
        result = classify_http_error(403, "")
        assert result.is_transient is False

    def test_non_json_error_body(self):
        result = classify_http_error(502, "Bad Gateway")
        assert result.is_transient is True
        assert "Bad Gateway" in result.error


# ─── 6. Full adapter with mocked transport ────────────────────────────────────


class TestAdapterWithMockedTransport:
    async def test_sends_request_returns_final_answer(self):
        client = mock_transport(make_anthropic_response(
            content=[{"type": "text", "text": "The answer is 42."}],
        ))
        adapter = create_anthropic_adapter(TEST_CONFIG, client)

        result = await adapter.send(make_request())
        assert result.kind == "final_answer"
        assert result.content == "The answer is 42."

    async def test_sends_request_returns_tool_calls(self):
        client = mock_transport(make_anthropic_response(
            content=[
                {"type": "text", "text": "Reading file."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "test.ts"}},
            ],
            stop_reason="tool_use",
        ))
        adapter = create_anthropic_adapter(TEST_CONFIG, client)

        result = await adapter.send(make_request())
        assert result.kind == "tool_calls"
        assert len(result.calls) == 1
        assert result.calls[0].tool_name == "read_file"

    async def test_transient_error_for_rate_limiting(self):
        client = mock_transport_error(429, '{"error":{"message":"Rate limited"}}')
        adapter = create_anthropic_adapter(TEST_CONFIG, client)

        result = await adapter.send(make_request())
        assert result.kind == "error"
        assert result.is_transient is True

    async def test_non_transient_error_for_bad_request(self):
        client = mock_transport_error(400, '{"error":{"message":"Invalid model"}}')
        adapter = create_anthropic_adapter(TEST_CONFIG, client)

        result = await adapter.send(make_request())
        assert result.kind == "error"
        assert result.is_transient is False
        assert "Invalid model" in result.error

    async def test_transient_error_for_network_failure(self):
        client = mock_transport_network_error()
        adapter = create_anthropic_adapter(TEST_CONFIG, client)

        result = await adapter.send(make_request())
        assert result.kind == "error"
        assert result.is_transient is True
        assert "ECONNREFUSED" in result.error


# ─── 7. Integration: loop + adapter (mocked) ─────────────────────────────────


class TestLoopAdapterIntegration:
    async def test_full_pipeline_with_mocked_responses(self):
        from clawbound.execution_loop.action_gate import ActionGateImpl
        from clawbound.execution_loop.loop import run_loop
        from clawbound.policy_engine.engine import PolicyEngineImpl, default_runtime_config
        from clawbound.prompt_builder.builder import PromptBuilderImpl
        from clawbound.signal_processor.processor import SignalProcessorImpl
        from clawbound.task_compiler.compiler import CompileInput, TaskCompilerImpl
        from clawbound.tool_broker.broker import ToolBrokerImpl

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, json=make_anthropic_response(
                    content=[
                        {"type": "text", "text": "I will read the file."},
                        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "src/main.ts"}},
                    ],
                    stop_reason="tool_use",
                ))
            return httpx.Response(200, json=make_anthropic_response(
                content=[{"type": "text", "text": "The main module exports the entry point."}],
                stop_reason="end_turn",
            ))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = create_anthropic_adapter(TEST_CONFIG, client)

        broker = ToolBrokerImpl()
        async def read_file_fn(args: dict) -> dict:
            return {"output": "export function main() { }"}

        broker.register(
            ToolDefinition(name="read_file", category="filesystem", risk_level="read_only"),
            read_file_fn,
        )

        compiler = TaskCompilerImpl()
        policy_engine = PolicyEngineImpl()
        user_input = "Explain what the main module does."

        task_spec = compiler.compile_from_input(CompileInput(
            trace_id=str(uuid.uuid4()),
            user_input=user_input,
            continuation_of=None,
            local_context=[],
        ))
        runtime_policy = policy_engine.resolve(task_spec, default_runtime_config())

        result = await run_loop(
            run_id=str(uuid.uuid4()),
            trace_id=task_spec.trace_id,
            task_spec=task_spec,
            runtime_policy=runtime_policy,
            kernel=KernelAsset(version="context-kernel-v0", content="- Be concise.", token_estimate=5),
            local_context=(),
            retrieved_units=(),
            user_message=user_input,
            initial_messages=None,
            config=LoopConfig(max_iterations=10, max_consecutive_errors=3, max_transient_retries=2),
            prompt_builder=PromptBuilderImpl(),
            tool_broker=broker,
            signal_processor=SignalProcessorImpl(),
            model_adapter=adapter,
            action_gate=ActionGateImpl(),
        )

        assert result.termination == "final_answer"
        assert "main module" in result.final_content
        assert result.iterations == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].status == "success"
        assert len(result.signal_bundles) == 1
        assert call_count == 2
