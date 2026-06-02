"""OpenAI GPT LLM provider."""

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from twin.config import ModelInfo
from twin.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition

_MAX_TOKENS = 4096

# Approximate pricing per million tokens (input, output) as of mid-2025.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-preview": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o3-mini": (1.10, 4.40),
}

_DEFAULT_MODELS = [
    ModelInfo("gpt-4o", "GPT-4o", supports_tools=True),
    ModelInfo("gpt-4o-mini", "GPT-4o Mini", supports_tools=True),
    ModelInfo("gpt-4-turbo", "GPT-4 Turbo", supports_tools=True),
    ModelInfo("o1-preview", "o1-preview", supports_tools=False),
    ModelInfo("o1-mini", "o1-mini", supports_tools=False),
    ModelInfo("o3-mini", "o3-mini", supports_tools=False),
]


def _lookup_price(model: str) -> tuple[float, float] | None:
    for prefix, prices in _PRICING.items():
        if model.startswith(prefix):
            return prices
    return None


class OpenAIProvider(LLMProvider):
    """OpenAI GPT LLM provider."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model identifier. Defaults to 'gpt-4o'.
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model or "gpt-4o"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation to OpenAI and return a normalized LLMResponse.

        Converts Anthropic-style messages to OpenAI format internally.

        Args:
            messages: Conversation history in Anthropic message format.
            tools: Tool definitions to expose.
            system: System prompt.

        Returns:
            Normalized LLMResponse.
        """
        oai_messages = _convert_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        resp = await self._client.chat.completions.create(**kwargs)
        return _normalize_response(resp)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a GPT response token-by-token.

        Args:
            messages: Conversation history.
            tools: Tool definitions.
            system: System prompt.

        Yields:
            Text chunks as they arrive.
        """
        oai_messages = _convert_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": oai_messages,
            "stream": True,
        }
        async with await self._client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """
        Estimate cost using approximate OpenAI pricing.

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
        Return the standard GPT model list.

        Returns:
            List of ModelInfo for commonly used OpenAI models.
        """
        return _DEFAULT_MODELS


def _tool_to_openai(tool: ToolDefinition) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _convert_messages(
    messages: list[dict[str, Any]], system: str | None
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to OpenAI chat format."""
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Check what kind of content blocks we have.
            is_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )
            is_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )

            if role == "assistant" and is_tool_use:
                # Convert to OpenAI assistant message with tool_calls.
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                oai_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": " ".join(text_parts) or None,
                }
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)

            elif role == "user" and is_tool_result:
                # Convert tool_result blocks to OpenAI tool role messages.
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(block.get("content", "")),
                        })
            else:
                # Regular message: join text blocks.
                text = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                result.append({"role": role, "content": text})

    return result


def _normalize_response(resp: Any) -> LLMResponse:
    """Convert an OpenAI ChatCompletion to a normalized LLMResponse."""
    choice = resp.choices[0] if resp.choices else None
    if choice is None:
        return LLMResponse(content=None)

    msg = choice.message
    text = msg.content
    tool_calls: list[ToolCall] = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

    usage = getattr(resp, "usage", None)
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason=choice.finish_reason or "end_turn",
        prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
    )
