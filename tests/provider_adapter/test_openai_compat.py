"""OpenAI-Compatible Adapter tests — Phase 6.

Covers:
1. Request translation (system prompt, messages, tools)
2. Message normalization (user, assistant, tool, mixed)
3. Tool definition translation
4. Response normalization (final answer, tool calls, empty, malformed args)
5. Error classification (transient, non-transient)
6. Gemini factory (base URL, default model)
7. Adapter.send integration with mocked transport
8. MiniMax factory

All tests use httpx MockTransport — no live provider access.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from clawbound.contracts.types import (
    ModelMessage,
    ModelRequest,
    ModelToolCall,
    ToolDefinition,
)
from clawbound.provider_adapter.openai_compat import (
    GEMINI_OPENAI_BASE_URL,
    MINIMAX_CN_OPENAI_BASE_URL,
    MINIMAX_OPENAI_BASE_URL,
    OpenAICompatAdapter,
    classify_http_error,
    create_gemini_adapter,
    create_minimax_adapter,
    normalize_messages,
    normalize_response,
    translate_request,
    translate_tool_definitions,
)
from clawbound.provider_adapter.types import OpenAICompatConfig


# ─── 1. Request translation ──────────────────────────────────────────────────


class TestTranslateRequest:
    config = OpenAICompatConfig(api_key="k", model="gemini-2.5-flash", base_url="https://test", max_tokens=2048)

    def test_includes_system_prompt_as_system_message(self):
        request = ModelRequest(
            system_prompt="You are a helpful assistant.",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        )
        result = translate_request(request, self.config)

        assert result["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result["messages"][1] == {"role": "user", "content": "hi"}
        assert result["model"] == "gemini-2.5-flash"
        assert result["max_tokens"] == 2048

    def test_omits_system_when_empty(self):
        request = ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        )
        result = translate_request(request, self.config)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_includes_tools_when_present(self):
        request = ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(ToolDefinition(name="read_file", category="filesystem", risk_level="read_only", description="Read a file"),),
        )
        result = translate_request(request, self.config)

        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "read_file"

    def test_omits_tools_when_empty(self):
        request = ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        )
        result = translate_request(request, self.config)

        assert "tools" not in result


# ─── 2. Message normalization ────────────────────────────────────────────────


class TestNormalizeMessages:
    def test_maps_user_messages(self):
        messages = (ModelMessage(role="user", content="hello"),)
        result = normalize_messages(messages)

        assert result == [{"role": "user", "content": "hello"}]

    def test_maps_assistant_text_only(self):
        messages = (ModelMessage(role="assistant", content="sure"),)
        result = normalize_messages(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "sure"
        assert "tool_calls" not in result[0]

    def test_maps_assistant_with_tool_calls(self):
        messages = (
            ModelMessage(
                role="assistant",
                content="Let me check.",
                tool_calls=(
                    ModelToolCall(tool_call_id="tc1", tool_name="read_file", args={"path": "/tmp/a.txt"}),
                ),
            ),
        )
        result = normalize_messages(messages)

        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0] == {
            "id": "tc1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/a.txt"}',
            },
        }

    def test_maps_tool_result_messages(self):
        messages = (ModelMessage(role="tool", tool_call_id="tc1", content="file contents here"),)
        result = normalize_messages(messages)

        assert result[0] == {"role": "tool", "tool_call_id": "tc1", "content": "file contents here"}

    def test_preserves_ordering_in_mixed_conversation(self):
        messages = (
            ModelMessage(role="user", content="read file"),
            ModelMessage(role="assistant", content="", tool_calls=(
                ModelToolCall(tool_call_id="tc1", tool_name="read_file", args={"path": "/a"}),
            )),
            ModelMessage(role="tool", tool_call_id="tc1", content="contents"),
            ModelMessage(role="assistant", content="Here is the file."),
        )
        result = normalize_messages(messages)

        assert len(result) == 4
        assert [m["role"] for m in result] == ["user", "assistant", "tool", "assistant"]


# ─── 3. Tool definition translation ──────────────────────────────────────────


class TestTranslateToolDefinitions:
    def test_translates_to_openai_function_format(self):
        tools = (ToolDefinition(name="run_command", category="execution", risk_level="side_effect", description="Execute a shell command"),)
        result = translate_tool_definitions(tools)

        assert result[0] == {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute a shell command",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_fallback_description(self):
        tools = (ToolDefinition(name="my_tool", category="execution", risk_level="side_effect"),)
        result = translate_tool_definitions(tools)

        assert result[0]["function"]["description"] == "Tool: my_tool"


# ─── 4. Response normalization ────────────────────────────────────────────────


class TestNormalizeResponse:
    def test_final_answer(self):
        raw = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}]}
        result = normalize_response(raw)

        assert result.kind == "final_answer"
        assert result.content == "Hello!"

    def test_tool_calls(self):
        raw = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"/tmp/a"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        result = normalize_response(raw)

        assert result.kind == "tool_calls"
        assert len(result.calls) == 1
        assert result.calls[0].tool_name == "read_file"
        assert result.calls[0].args == {"path": "/tmp/a"}
        assert result.reasoning == "Let me check."

    def test_null_content_in_final_answer(self):
        raw = {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]}
        result = normalize_response(raw)

        assert result.kind == "final_answer"
        assert result.content == ""

    def test_empty_choices(self):
        raw = {"choices": []}
        result = normalize_response(raw)

        assert result.kind == "error"

    def test_malformed_tool_call_arguments(self):
        raw = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "test", "arguments": "not valid json"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        result = normalize_response(raw)

        assert result.kind == "tool_calls"
        assert result.calls[0].args == {"_raw": "not valid json"}


# ─── 5. Error classification ─────────────────────────────────────────────────


class TestClassifyHttpError:
    def test_429_transient(self):
        result = classify_http_error(429, '{"error":{"message":"rate limited"}}')
        assert result.is_transient is True
        assert "rate limited" in result.error

    def test_500_transient(self):
        result = classify_http_error(500, "")
        assert result.is_transient is True

    def test_401_non_transient(self):
        result = classify_http_error(401, '{"error":{"message":"invalid key"}}')
        assert result.is_transient is False
        assert "invalid key" in result.error

    def test_400_non_transient(self):
        result = classify_http_error(400, "bad request")
        assert result.is_transient is False


# ─── 6. Gemini + MiniMax factories ───────────────────────────────────────────


class TestFactories:
    def test_gemini_adapter_is_openai_compat(self):
        adapter = create_gemini_adapter("test-key")
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_gemini_base_url_constant(self):
        assert GEMINI_OPENAI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_minimax_base_url_constants(self):
        assert MINIMAX_OPENAI_BASE_URL == "https://api.minimax.io/v1"
        assert MINIMAX_CN_OPENAI_BASE_URL == "https://api.minimaxi.com/v1"

    def test_minimax_adapter_is_openai_compat(self):
        adapter = create_minimax_adapter("test-key")
        assert isinstance(adapter, OpenAICompatAdapter)


# ─── 7. Adapter.send integration with mocked transport ───────────────────────


class TestAdapterSendIntegration:
    async def test_sends_request_and_normalizes_response(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenAICompatAdapter(
            OpenAICompatConfig(api_key="test", model="gemini-2.5-flash", base_url="https://test.api"),
            client,
        )

        result = await adapter.send(ModelRequest(
            system_prompt="Be helpful.",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        ))

        assert result.kind == "final_answer"
        assert result.content == "Hello!"
        assert captured["body"]["model"] == "gemini-2.5-flash"
        assert len(captured["body"]["messages"]) == 2  # system + user

    async def test_uses_bearer_auth_header(self):
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenAICompatAdapter(
            OpenAICompatConfig(api_key="my-gemini-key", model="test", base_url="https://test.api"),
            client,
        )

        await adapter.send(ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        ))

        assert captured_headers["authorization"] == "Bearer my-gemini-key"

    async def test_posts_to_chat_completions_endpoint(self):
        captured_url = ""

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_url
            captured_url = str(request.url)
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenAICompatAdapter(
            OpenAICompatConfig(api_key="k", model="m", base_url="https://example.com/v1"),
            client,
        )

        await adapter.send(ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        ))

        assert captured_url == "https://example.com/v1/chat/completions"

    async def test_handles_tool_call_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"/tmp/x"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenAICompatAdapter(
            OpenAICompatConfig(api_key="k", model="m", base_url="https://test"),
            client,
        )

        result = await adapter.send(ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="read file"),),
            tool_definitions=(ToolDefinition(name="read_file", category="filesystem", risk_level="read_only"),),
        ))

        assert result.kind == "tool_calls"
        assert result.calls[0].tool_name == "read_file"

    async def test_handles_http_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text='{"error":{"message":"quota exceeded"}}')

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenAICompatAdapter(
            OpenAICompatConfig(api_key="k", model="m", base_url="https://test"),
            client,
        )

        result = await adapter.send(ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        ))

        assert result.kind == "error"
        assert result.is_transient is True
        assert "quota exceeded" in result.error

    async def test_handles_network_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("DNS resolution failed")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = OpenAICompatAdapter(
            OpenAICompatConfig(api_key="k", model="m", base_url="https://test"),
            client,
        )

        result = await adapter.send(ModelRequest(
            system_prompt="",
            messages=(ModelMessage(role="user", content="hi"),),
            tool_definitions=(),
        ))

        assert result.kind == "error"
        assert result.is_transient is True
        assert "DNS resolution failed" in result.error
