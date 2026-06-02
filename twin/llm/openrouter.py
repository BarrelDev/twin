"""OpenRouter LLM provider.

OpenRouter is an OpenAI-compatible API proxy that gives access to many
models (Claude, GPT-4o, Gemini, open-source) via a single API key.
Model name format: 'provider/model', e.g. 'anthropic/claude-sonnet-4-5'.
"""

import json
from typing import Any, AsyncIterator

import httpx

from twin.config import ModelInfo
from twin.llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition
from twin.llm.openai import _convert_messages, _tool_to_openai

_BASE_URL = "https://openrouter.ai/api/v1"
_PRICING_URL = f"{_BASE_URL}/models"


class OpenRouterProvider(LLMProvider):
    """OpenRouter multi-provider LLM proxy."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key.
            model: Model in 'provider/model' format.
                   No sensible default — must be specified by the user.

        Raises:
            ValueError: If no model is specified.
        """
        if not model:
            raise ValueError(
                "OpenRouter requires an explicit model name in 'provider/model' format.\n"
                "Example: twin config set-model anthropic/claude-sonnet-4-5"
            )
        self._api_key = api_key
        self.model = model
        self._price_cache: dict[str, tuple[float, float]] | None = None

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/twin-project/twin",
            "X-Title": "Twin",
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation through OpenRouter and return a normalized LLMResponse.

        Args:
            messages: Conversation history in Anthropic message format.
            tools: Tool definitions to expose.
            system: System prompt.

        Returns:
            Normalized LLMResponse.
        """
        oai_messages = _convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
        }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_BASE_URL}/chat/completions",
                json=payload,
                headers=self._auth_headers(),
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
        Stream an OpenRouter response token-by-token via SSE.

        Args:
            messages: Conversation history.
            tools: Tool definitions.
            system: System prompt.

        Yields:
            Text chunks as they arrive.
        """
        oai_messages = _convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{_BASE_URL}/chat/completions",
                json=payload,
                headers=self._auth_headers(),
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """
        Estimate cost by looking up live pricing from the OpenRouter API.

        Fetches and caches the model pricing on first call.

        Args:
            prompt_tokens: Input token count.
            completion_tokens: Output token count.

        Returns:
            Estimated USD cost, or None if pricing is unavailable.
        """
        if self._price_cache is None:
            self._price_cache = self._fetch_pricing()
        prices = self._price_cache.get(self.model)
        if prices is None:
            return None
        return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000

    def _fetch_pricing(self) -> dict[str, tuple[float, float]]:
        """Fetch model pricing from the OpenRouter API synchronously."""
        try:
            resp = httpx.get(
                _PRICING_URL,
                headers=self._auth_headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            result: dict[str, tuple[float, float]] = {}
            for model in data.get("data", []):
                model_id = model.get("id", "")
                pricing = model.get("pricing", {})
                try:
                    # OpenRouter pricing is in USD per token (not per million).
                    prompt_price = float(pricing.get("prompt", 0)) * 1_000_000
                    completion_price = float(pricing.get("completion", 0)) * 1_000_000
                    result[model_id] = (prompt_price, completion_price)
                except (TypeError, ValueError):
                    continue
            return result
        except Exception:
            return {}

    def list_models(self) -> list[ModelInfo]:
        """
        Fetch the current model list from OpenRouter.

        Returns:
            List of ModelInfo for models available through OpenRouter.
        """
        try:
            resp = httpx.get(
                _PRICING_URL,
                headers=self._auth_headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                ModelInfo(
                    model_id=m.get("id", ""),
                    name=m.get("name", m.get("id", "")),
                    supports_tools=m.get("id", "") not in _TOOL_UNSUPPORTED,
                )
                for m in data.get("data", [])
                if m.get("id")
            ]
        except Exception:
            return [ModelInfo(model_id=self.model, name=self.model, supports_tools=True)]


def _normalize_response(data: dict[str, Any]) -> LLMResponse:
    """Convert an OpenRouter chat completion response to a normalized LLMResponse."""
    choices = data.get("choices", [])
    if not choices:
        return LLMResponse(content=None)

    msg = choices[0].get("message", {})
    text: str | None = msg.get("content") or None
    tool_calls: list[ToolCall] = []

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(
            id=tc.get("id", fn.get("name", "")),
            name=fn.get("name", ""),
            input=args,
        ))

    usage = data.get("usage", {})
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason=choices[0].get("finish_reason", "end_turn"),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )


# Models known not to support tool use via OpenRouter.
_TOOL_UNSUPPORTED: set[str] = {
    "openai/o1-preview",
    "openai/o1-mini",
}
