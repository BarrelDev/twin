"""Anthropic Claude LLM provider."""

import os
from typing import Any, AsyncIterator

from anthropic import Anthropic, AsyncAnthropic

from twin.config import ModelInfo
from twin.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition

_MAX_TOKENS = 4096

# Approximate pricing per million tokens (input, output).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku": (0.25, 1.25),
}


def _lookup_price(model: str) -> tuple[float, float] | None:
    for prefix, prices in _PRICING.items():
        if model.startswith(prefix):
            return prices
    return None


class Claude(LLMProvider):
    """Anthropic Claude LLM provider.

    Uses a synchronous client for one-time model discovery at initialization
    and an async client for all subsequent LLM calls.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        """
        Initialize the Claude provider.

        Args:
            model: Model identifier. Defaults to the first available model.
            api_key: API key. Falls back to ANTHROPIC_API_KEY env var.

        Raises:
            ValueError: If no API key is found or no models are returned by the API.
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No Anthropic API key found.\n"
                "Run: twin config set-key\n"
                "Or set ANTHROPIC_API_KEY in your environment."
            )

        # Sync client: used once at init to discover available models.
        sync_client = Anthropic(api_key=api_key)
        models_page = sync_client.models.list()
        raw_ids = [m.id for m in models_page.data]
        if not raw_ids:
            raise ValueError("No models available from Anthropic API.")

        self._models = [
            ModelInfo(model_id=m_id, name=m_id, supports_tools=True)
            for m_id in raw_ids
        ]

        if model:
            if model not in raw_ids:
                raise ValueError(
                    f"Model '{model}' not available. "
                    f"Available models: {', '.join(raw_ids)}"
                )
            self.model = model
        else:
            self.model = raw_ids[0]

        # Async client: used for all LLM calls.
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation to Claude and return a normalized LLMResponse.

        Args:
            messages: Conversation history in Anthropic message format.
            tools: Tool definitions to expose to the model.
            system: System prompt.

        Returns:
            Normalized LLMResponse.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": system or "",
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        resp = await self._client.messages.create(**kwargs)
        return _normalize_response(resp)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a Claude response token-by-token.

        Args:
            messages: Conversation history.
            tools: Tool definitions.
            system: System prompt.

        Yields:
            Text chunks as they arrive.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": system or "",
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        async with self._client.messages.stream(**kwargs) as s:
            async for text in s.text_stream:
                yield text

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """
        Estimate cost using approximate Anthropic pricing.

        Args:
            prompt_tokens: Input token count.
            completion_tokens: Output token count.

        Returns:
            Estimated USD cost, or None if the model is not in the pricing table.
        """
        prices = _lookup_price(self.model)
        if prices is None:
            return None
        return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000

    def list_models(self) -> list[ModelInfo]:
        """
        Return Claude models discovered at initialization.

        Returns:
            List of ModelInfo for each available Claude model.
        """
        return self._models


def _normalize_response(resp: Any) -> LLMResponse:
    """Convert an Anthropic Message to a normalized LLMResponse."""
    text: str | None = None
    tool_calls: list[ToolCall] = []
    for block in getattr(resp, "content", []):
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text = block.text
        elif block_type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
    usage = getattr(resp, "usage", None)
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason=getattr(resp, "stop_reason", "end_turn") or "end_turn",
        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
    )
