"""Anthropic Messages API Adapter.

Thin translation layer between ClawBound's runtime-owned types
(ModelRequest/ModelResponse) and the Anthropic Messages API.

The adapter owns:
- request/response translation
- message format normalization (batching tool results, content blocks)
- tool-call protocol mapping
- transient error classification

Known limitations (v1):
- No streaming support (uses synchronous Messages API)
- ToolDefinition lacks input_schema -> uses empty schema
- No extended thinking support
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from clawbound.contracts.types import (
    FinalAnswer,
    ModelError,
    ModelMessage,
    ModelRequest,
    ModelToolCall,
    ToolCalls,
    ToolDefinition,
)

from .types import AnthropicConfig


# ─── Adapter implementation ──────────────────────────────────────────────────


class AnthropicAdapter:
    def __init__(
        self,
        config: AnthropicConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient()

    async def send(
        self,
        request: ModelRequest,
    ) -> FinalAnswer | ToolCalls | ModelError:
        anthropic_request = translate_request(request, self._config)

        try:
            http_response = await self._client.post(
                f"{self._config.base_url}/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._config.api_key,
                    "anthropic-version": self._config.anthropic_version,
                },
                json=anthropic_request,
            )
        except Exception as e:
            return ModelError(
                error=f"Network error: {e}",
                is_transient=True,
            )

        if http_response.status_code < 200 or http_response.status_code >= 300:
            body = http_response.text
            return classify_http_error(http_response.status_code, body)

        raw = http_response.json()
        return normalize_response(raw)


# ─── Request translation ─────────────────────────────────────────────────────


def translate_request(
    request: ModelRequest,
    config: AnthropicConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": normalize_messages(request.messages),
    }

    if request.system_prompt:
        result["system"] = request.system_prompt

    if request.tool_definitions:
        result["tools"] = translate_tool_definitions(request.tool_definitions)

    return result


def normalize_messages(
    messages: tuple[ModelMessage, ...],
) -> list[dict[str, Any]]:
    """Normalize ClawBound messages to Anthropic message format.

    Key translations:
    - user messages -> Anthropic user with text content block
    - assistant messages -> Anthropic assistant with text + tool_use blocks
    - consecutive tool messages -> batched into ONE Anthropic user message with tool_result blocks
    """
    result: list[dict[str, Any]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg.role == "user":
            result.append({
                "role": "user",
                "content": [{"type": "text", "text": msg.content}],
            })
            i += 1

        elif msg.role == "assistant":
            content_blocks: list[dict[str, Any]] = []

            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})

            for tc in (msg.tool_calls or ()):
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.tool_call_id,
                    "name": tc.tool_name,
                    "input": tc.args,
                })

            result.append({
                "role": "assistant",
                "content": content_blocks,
            })
            i += 1

        elif msg.role == "tool":
            # Group consecutive tool messages into one user message
            tool_results: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].role == "tool":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": messages[i].tool_call_id or "",
                    "content": messages[i].content,
                })
                i += 1
            result.append({
                "role": "user",
                "content": tool_results,
            })
        else:
            i += 1  # skip unknown roles

    return result


def translate_tool_definitions(
    tools: tuple[ToolDefinition, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description or f"Tool: {t.name}",
            "input_schema": {"type": "object", "properties": {}},
        }
        for t in tools
    ]


# ─── Response normalization ───────────────────────────────────────────────────


def normalize_response(raw: dict[str, Any]) -> FinalAnswer | ToolCalls | ModelError:
    content_blocks = raw.get("content", [])

    tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

    if tool_use_blocks:
        text_blocks = [b for b in content_blocks if b.get("type") == "text"]
        reasoning = "\n".join(b["text"] for b in text_blocks) or None

        calls = tuple(
            ModelToolCall(
                tool_call_id=b["id"],
                tool_name=b["name"],
                args=b.get("input", {}),
            )
            for b in tool_use_blocks
        )

        return ToolCalls(calls=calls, reasoning=reasoning)

    # Pure text response
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]
    text_content = "\n".join(b["text"] for b in text_blocks)

    return FinalAnswer(content=text_content)


# ─── Error classification ────────────────────────────────────────────────────


def classify_http_error(status: int, body: str) -> ModelError:
    """Classify HTTP errors as transient or non-transient.

    429 (rate limit), 529 (overloaded), 5xx -> transient
    400, 401, 403, 404 -> non-transient
    """
    is_transient = status == 429 or status == 529 or status >= 500

    error_message = f"HTTP {status}"
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            msg = parsed["error"].get("message")
            if msg:
                error_message = f"HTTP {status}: {msg}"
    except (json.JSONDecodeError, KeyError):
        if body:
            error_message = f"HTTP {status}: {body[:200]}"

    return ModelError(error=error_message, is_transient=is_transient)


# ─── Factory ─────────────────────────────────────────────────────────────────


def create_anthropic_adapter(
    config: AnthropicConfig,
    client: httpx.AsyncClient | None = None,
) -> AnthropicAdapter:
    return AnthropicAdapter(config, client)
