"""OpenAI-Compatible Provider Adapter.

Thin translation layer between ClawBound's runtime-owned types
(ModelRequest/ModelResponse) and any provider that exposes an
OpenAI chat completions compatible endpoint.

Primary use cases: Gemini, MiniMax via their OpenAI compatibility endpoints.
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

from .types import OpenAICompatConfig

# ─── Constants ────────────────────────────────────────────────────────────────

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
MINIMAX_OPENAI_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CN_OPENAI_BASE_URL = "https://api.minimaxi.com/v1"


# ─── Adapter implementation ──────────────────────────────────────────────────


class OpenAICompatAdapter:
    def __init__(
        self,
        config: OpenAICompatConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient()

    async def send(
        self,
        request: ModelRequest,
    ) -> FinalAnswer | ToolCalls | ModelError:
        oai_request = translate_request(request, self._config)

        try:
            http_response = await self._client.post(
                f"{self._config.base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._config.api_key}",
                },
                json=oai_request,
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
    config: OpenAICompatConfig,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []

    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})

    messages.extend(normalize_messages(request.messages))

    result: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": messages,
    }

    if request.tool_definitions:
        result["tools"] = translate_tool_definitions(request.tool_definitions)

    return result


def normalize_messages(
    messages: tuple[ModelMessage, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            result.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            oai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or None,
            }
            if msg.tool_calls:
                oai_msg["tool_calls"] = [
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(oai_msg)

        elif msg.role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "",
                "content": msg.content,
            })

    return result


def translate_tool_definitions(
    tools: tuple[ToolDefinition, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or f"Tool: {t.name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


# ─── Response normalization ───────────────────────────────────────────────────


def normalize_response(raw: dict[str, Any]) -> FinalAnswer | ToolCalls | ModelError:
    choices = raw.get("choices", [])
    if not choices:
        return ModelError(error="Empty response: no choices", is_transient=False)

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")

    if tool_calls and len(tool_calls) > 0:
        calls = tuple(
            ModelToolCall(
                tool_call_id=tc["id"],
                tool_name=tc["function"]["name"],
                args=_safe_parse_args(tc["function"]["arguments"]),
            )
            for tc in tool_calls
        )
        return ToolCalls(
            calls=calls,
            reasoning=message.get("content") or None,
        )

    return FinalAnswer(content=message.get("content") or "")


def _safe_parse_args(raw: str) -> dict[str, Any]:
    try:
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


# ─── Error classification ────────────────────────────────────────────────────


def classify_http_error(status: int, body: str) -> ModelError:
    is_transient = status == 429 or status >= 500

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


# ─── Factories ────────────────────────────────────────────────────────────────


def create_gemini_adapter(
    api_key: str,
    model: str = "gemini-2.5-flash",
    client: httpx.AsyncClient | None = None,
) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        OpenAICompatConfig(api_key=api_key, model=model, base_url=GEMINI_OPENAI_BASE_URL),
        client,
    )


def create_minimax_adapter(
    api_key: str,
    model: str = "MiniMax-M2.7",
    base_url: str = MINIMAX_OPENAI_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        OpenAICompatConfig(api_key=api_key, model=model, base_url=base_url),
        client,
    )
