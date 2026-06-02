"""Ollama local LLM provider.

Ollama runs models locally and requires no API key.
The base URL defaults to http://localhost:11434 but can be overridden via
the TWIN_OLLAMA_URL environment variable or config.json.
"""

import json
import os
from typing import Any, AsyncIterator

import httpx

from twin.config import ModelInfo
from twin.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider.

    Uses the Ollama HTTP API (/api/chat) with an httpx async client.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model: Model name. Defaults to 'llama3'.
            base_url: Ollama API base URL. Defaults to http://localhost:11434.
        """
        self.base_url = (
            base_url
            or os.environ.get("TWIN_OLLAMA_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or "llama3"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation to Ollama and return a normalized LLMResponse.

        Args:
            messages: Conversation history in Anthropic message format.
            tools: Tool definitions (Ollama supports OpenAI-compatible tool format).
            system: System prompt.

        Returns:
            Normalized LLMResponse.
        """
        from twin.llm.openai import _convert_messages, _tool_to_openai

        oai_messages = _convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return _normalize_response(data)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream an Ollama response token-by-token via NDJSON.

        Args:
            messages: Conversation history.
            tools: Tool definitions.
            system: System prompt.

        Yields:
            Text chunks as they arrive.
        """
        from twin.llm.openai import _convert_messages

        oai_messages = _convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """
        Ollama runs locally with no per-token billing.

        Returns:
            Always None.
        """
        return None

    def list_models(self) -> list[ModelInfo]:
        """
        Query the local Ollama API to list installed models.

        Makes a synchronous HTTP call; intended for CLI use only.

        Returns:
            List of ModelInfo for installed Ollama models.
        """
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return [
                ModelInfo(
                    model_id=m["name"],
                    name=m["name"],
                    supports_tools=False,
                )
                for m in data.get("models", [])
            ]
        except Exception:
            return [ModelInfo(model_id=self.model, name=self.model, supports_tools=False)]


def _normalize_response(data: dict[str, Any]) -> LLMResponse:
    """Convert an Ollama /api/chat response to a normalized LLMResponse."""
    msg = data.get("message", {})
    text: str | None = msg.get("content") or None
    tool_calls: list[ToolCall] = []

    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(ToolCall(
            id=fn.get("name", ""),
            name=fn.get("name", ""),
            input=args,
        ))

    prompt_tokens = data.get("prompt_eval_count", 0) or 0
    completion_tokens = data.get("eval_count", 0) or 0
    done_reason = data.get("done_reason", "stop")

    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason="tool_use" if tool_calls else done_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
