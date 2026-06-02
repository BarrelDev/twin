"""Tests for all LLM providers using mocked HTTP clients and SDKs."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from twin.config import ModelInfo
from twin.llm.base import LLMResponse, ToolCall, ToolDefinition


# ── OpenAI provider ──────────────────────────────────────────────────────────

class TestOpenAIProvider:

    def _make_provider(self, model: str = "gpt-4o"):
        from twin.llm.openai import OpenAIProvider
        with patch("twin.llm.openai.AsyncOpenAI"):
            return OpenAIProvider(api_key="test-key", model=model)

    def test_initialization(self) -> None:
        provider = self._make_provider()
        assert provider.model == "gpt-4o"

    def test_list_models_returns_model_info(self) -> None:
        provider = self._make_provider()
        models = provider.list_models()
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)

    def test_estimate_cost_known_model(self) -> None:
        provider = self._make_provider("gpt-4o")
        cost = provider.estimate_cost(1000, 500)
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_unknown_model_returns_none(self) -> None:
        from twin.llm.openai import OpenAIProvider
        with patch("twin.llm.openai.AsyncOpenAI"):
            provider = OpenAIProvider(api_key="test-key", model="gpt-99-unknown")
        assert provider.estimate_cost(1000, 500) is None

    @pytest.mark.anyio
    async def test_complete_returns_llm_response(self) -> None:
        from twin.llm.openai import OpenAIProvider

        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Hello from GPT"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("twin.llm.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")

        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from GPT"
        assert result.tool_calls == []

    @pytest.mark.anyio
    async def test_complete_with_tool_call(self) -> None:
        from twin.llm.openai import OpenAIProvider

        mock_client = MagicMock()
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "search_kb"
        tc.function.arguments = json.dumps({"query": "test"})

        choice = MagicMock()
        choice.message.content = None
        choice.message.tool_calls = [tc]
        choice.finish_reason = "tool_calls"
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("twin.llm.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")

        tools = [ToolDefinition("search_kb", "search", {"type": "object"})]
        result = await provider.complete([{"role": "user", "content": "search"}], tools=tools)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_kb"
        assert result.tool_calls[0].input == {"query": "test"}


# ── Gemini provider ──────────────────────────────────────────────────────────

class TestGeminiProvider:

    def _make_provider(self):
        from twin.llm.gemini import GeminiProvider
        with patch("twin.llm.gemini.genai.Client"):
            return GeminiProvider(api_key="test-key")

    def test_initialization(self) -> None:
        provider = self._make_provider()
        assert provider.model == "gemini-2.0-flash"

    def test_list_models_returns_model_info(self) -> None:
        provider = self._make_provider()
        models = provider.list_models()
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)

    def test_estimate_cost_known_model(self) -> None:
        provider = self._make_provider()
        cost = provider.estimate_cost(1000, 500)
        assert isinstance(cost, float)

    def test_estimate_cost_unknown_model_returns_none(self) -> None:
        from twin.llm.gemini import GeminiProvider
        with patch("twin.llm.gemini.genai.Client"):
            provider = GeminiProvider(api_key="test-key", model="gemini-99-unknown")
        assert provider.estimate_cost(1000, 500) is None

    @pytest.mark.anyio
    async def test_complete_returns_llm_response(self) -> None:
        from twin.llm.gemini import GeminiProvider

        mock_client = MagicMock()
        part = MagicMock()
        part.text = "Hello from Gemini"
        part.function_call = None
        candidate = MagicMock()
        candidate.content.parts = [part]
        mock_resp = MagicMock()
        mock_resp.candidates = [candidate]
        mock_resp.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5
        )
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

        with patch("twin.llm.gemini.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="test-key")

        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Gemini"


# ── Ollama provider ──────────────────────────────────────────────────────────

class TestOllamaProvider:

    def test_initialization(self) -> None:
        from twin.llm.ollama import OllamaProvider
        provider = OllamaProvider(model="llama3")
        assert provider.model == "llama3"
        assert provider.base_url == "http://localhost:11434"

    def test_custom_base_url(self) -> None:
        from twin.llm.ollama import OllamaProvider
        provider = OllamaProvider(base_url="http://myserver:11434")
        assert provider.base_url == "http://myserver:11434"

    def test_estimate_cost_always_none(self) -> None:
        from twin.llm.ollama import OllamaProvider
        provider = OllamaProvider()
        assert provider.estimate_cost(1000, 500) is None

    @pytest.mark.anyio
    async def test_complete_returns_llm_response(self) -> None:
        from twin.llm.ollama import OllamaProvider
        import httpx

        response_data = {
            "message": {"role": "assistant", "content": "Hello from Ollama"},
            "prompt_eval_count": 10,
            "eval_count": 5,
            "done_reason": "stop",
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = response_data

        with patch("twin.llm.ollama.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)

            provider = OllamaProvider(model="llama3")
            result = await provider.complete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Ollama"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5


# ── OpenRouter provider ──────────────────────────────────────────────────────

class TestOpenRouterProvider:

    def test_requires_model(self) -> None:
        from twin.llm.openrouter import OpenRouterProvider
        with pytest.raises(ValueError, match="model"):
            OpenRouterProvider(api_key="test-key")

    def test_initialization(self) -> None:
        from twin.llm.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(
            api_key="test-key", model="anthropic/claude-sonnet-4-5"
        )
        assert provider.model == "anthropic/claude-sonnet-4-5"

    @pytest.mark.anyio
    async def test_complete_returns_llm_response(self) -> None:
        from twin.llm.openrouter import OpenRouterProvider
        import httpx

        response_data = {
            "choices": [{
                "message": {"content": "Hello from OpenRouter", "tool_calls": None},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = response_data

        with patch("twin.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)

            provider = OpenRouterProvider(
                api_key="test-key", model="anthropic/claude-sonnet-4-5"
            )
            result = await provider.complete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from OpenRouter"

    def test_estimate_cost_uses_cached_pricing(self) -> None:
        from twin.llm.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(
            api_key="test-key", model="anthropic/claude-sonnet-4-5"
        )
        # Pre-populate the price cache so no HTTP call is made.
        provider._price_cache = {"anthropic/claude-sonnet-4-5": (3.0, 15.0)}
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0)

    def test_estimate_cost_returns_none_for_unknown_model(self) -> None:
        from twin.llm.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(
            api_key="test-key", model="unknown/model-xyz"
        )
        provider._price_cache = {}
        assert provider.estimate_cost(1000, 500) is None
