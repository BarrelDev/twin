"""Google Gemini LLM provider using the google-genai SDK."""

import json
from typing import Any, AsyncIterator

import google.genai as genai
from google.genai import types as genai_types

from twin.config import ModelInfo
from twin.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition

# Approximate pricing per million tokens (input, output) as of mid-2025.
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}

_DEFAULT_MODELS = [
    ModelInfo("gemini-2.0-flash", "Gemini 2.0 Flash", supports_tools=True),
    ModelInfo("gemini-2.0-pro", "Gemini 2.0 Pro", supports_tools=True),
    ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash", supports_tools=True),
    ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro", supports_tools=True),
]


def _lookup_price(model: str) -> tuple[float, float] | None:
    for prefix, prices in _PRICING.items():
        if model.startswith(prefix):
            return prices
    return None


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """
        Initialize the Gemini provider.

        Args:
            api_key: Google AI API key.
            model: Model identifier. Defaults to 'gemini-2.0-flash'.
        """
        self._client = genai.Client(api_key=api_key)
        self.model = model or "gemini-2.0-flash"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation to Gemini and return a normalized LLMResponse.

        Args:
            messages: Conversation history in Anthropic message format.
            tools: Tool definitions to expose.
            system: System prompt.

        Returns:
            Normalized LLMResponse.
        """
        contents, config = _build_request(messages, system, tools)
        resp = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return _normalize_response(resp)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a Gemini response token-by-token.

        Args:
            messages: Conversation history.
            tools: Tool definitions.
            system: System prompt.

        Yields:
            Text chunks as they arrive.
        """
        contents, config = _build_request(messages, system, tools)
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """
        Estimate cost using approximate Gemini pricing.

        Args:
            prompt_tokens: Input token count.
            completion_tokens: Output token count.

        Returns:
            Estimated USD cost, or None if model not in pricing table.
        """
        prices = _lookup_price(self.model)
        if prices is None:
            return None
        return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000

    def list_models(self) -> list[ModelInfo]:
        """
        Return available Gemini models.

        Returns:
            List of ModelInfo for commonly used Gemini models.
        """
        return _DEFAULT_MODELS


def _build_request(
    messages: list[dict[str, Any]],
    system: str | None,
    tools: list[ToolDefinition] | None,
) -> tuple[list[genai_types.ContentDict], genai_types.GenerateContentConfigDict]:
    """Build Gemini API request contents and config from Anthropic-style messages."""
    contents: list[genai_types.ContentDict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Gemini uses "model" not "assistant".
        gemini_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append({"text": block.get("text", "")})
                elif btype == "tool_use":
                    parts.append({"function_call": {
                        "name": block["name"],
                        "args": block.get("input", {}),
                    }})
                elif btype == "tool_result":
                    parts.append({"function_response": {
                        "name": "",
                        "response": {"content": str(block.get("content", ""))},
                    }})
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

    config: genai_types.GenerateContentConfigDict = {}
    if system:
        config["system_instruction"] = system
    if tools:
        config["tools"] = [
            genai_types.Tool(function_declarations=[
                genai_types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.input_schema,
                )
                for t in tools
            ])
        ]
    return contents, config


def _normalize_response(resp: Any) -> LLMResponse:
    """Convert a Gemini GenerateContentResponse to a normalized LLMResponse."""
    text: str | None = None
    tool_calls: list[ToolCall] = []

    for candidate in getattr(resp, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            if hasattr(part, "text") and part.text:
                text = part.text
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=fc.name,
                    name=fc.name,
                    input=dict(fc.args) if fc.args else {},
                ))

    usage = getattr(resp, "usage_metadata", None)
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason="tool_use" if tool_calls else "end_turn",
        prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
        completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
    )
