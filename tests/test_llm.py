"""Tests for the Claude LLM provider."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from twin.config import ModelInfo
from twin.llm import Claude, LLMProvider
from twin.llm.base import LLMResponse, ToolCall, ToolDefinition


def _mock_models_list(model_ids: list[str]):
    mock = MagicMock()
    mock.data = [MagicMock(id=m) for m in model_ids]
    return mock


def _make_anthropic_response(
    text: str = "Hello",
    tool_name: str | None = None,
    tool_id: str = "tu_1",
    tool_input: dict | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    """Build a mock Anthropic Message response."""
    blocks = []
    if text:
        b = MagicMock()
        b.type = "text"
        b.text = text
        blocks.append(b)
    if tool_name:
        b = MagicMock()
        b.type = "tool_use"
        b.id = tool_id
        b.name = tool_name
        b.input = tool_input or {}
        blocks.append(b)
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


class TestClaude:
    """Tests for the Claude LLM provider."""

    def test_missing_api_key_raises_error(self) -> None:
        """Claude raises ValueError when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                Claude()

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_initialization_with_api_key(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """Claude initializes successfully with a valid API key."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(
            ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
        )
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            claude = Claude()
            mock_sync_cls.assert_called_once_with(api_key="test-key-123")
            mock_async_cls.assert_called_once_with(api_key="test-key-123")
            assert claude.model == "claude-opus-4-8"

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_initialization_with_custom_model(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """Claude accepts a custom model parameter."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(
            ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
        )
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude(model="claude-haiku-4-5")
            assert claude.model == "claude-haiku-4-5"

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_initialization_with_invalid_model(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """Claude raises ValueError for an unavailable model."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(
            ["claude-opus-4-8", "claude-sonnet-4-6"]
        )
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="not available"):
                Claude(model="claude-nonexistent")

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_no_models_available_raises_error(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """Claude raises ValueError when the API returns no models."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list([])
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No models available"):
                Claude()

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_list_models_returns_model_info(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """list_models() returns ModelInfo objects."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        available = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
        mock_sync.models.list.return_value = _mock_models_list(available)
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            models = claude.list_models()
            assert len(models) == 3
            assert all(isinstance(m, ModelInfo) for m in models)
            assert models[0].model_id == "claude-opus-4-8"
            assert models[0].supports_tools is True

    @pytest.mark.anyio
    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    async def test_complete_returns_llm_response(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """complete() returns a normalized LLMResponse."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(["claude-opus-4-8"])

        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async
        mock_async.messages.create = AsyncMock(
            return_value=_make_anthropic_response(text="Hello world")
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            result = await claude.complete(
                [{"role": "user", "content": "Hi"}],
                tools=None,
                system="You are helpful.",
            )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    @pytest.mark.anyio
    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    async def test_complete_with_tools_returns_tool_call(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """complete() normalizes tool_use blocks into ToolCall objects."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(["claude-opus-4-8"])

        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async
        mock_async.messages.create = AsyncMock(
            return_value=_make_anthropic_response(
                text="",
                tool_name="search_knowledge_base",
                tool_id="tu_123",
                tool_input={"query": "test"},
                stop_reason="tool_use",
            )
        )

        tools = [ToolDefinition(
            name="search_knowledge_base",
            description="Search KB",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            result = await claude.complete(
                [{"role": "user", "content": "search for X"}],
                tools=tools,
                system="system",
            )

        assert result.tool_calls == [ToolCall(id="tu_123", name="search_knowledge_base", input={"query": "test"})]
        assert result.stop_reason == "tool_use"

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_estimate_cost_known_model(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """estimate_cost() returns a float for a known model."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(["claude-sonnet-4-6"])
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude(model="claude-sonnet-4-6")
            cost = claude.estimate_cost(1000, 500)
            assert isinstance(cost, float)
            assert cost > 0

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_estimate_cost_unknown_model_returns_none(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """estimate_cost() returns None for a model not in the pricing table."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(["claude-future-v99"])
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude(model="claude-future-v99")
            assert claude.estimate_cost(1000, 500) is None

    @patch("twin.llm.anthropic.AsyncAnthropic")
    @patch("twin.llm.anthropic.Anthropic")
    def test_extract_answer_returns_content(
        self, mock_sync_cls: MagicMock, mock_async_cls: MagicMock
    ) -> None:
        """extract_answer() returns the content string from LLMResponse."""
        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.models.list.return_value = _mock_models_list(["claude-opus-4-8"])
        mock_async_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            response = LLMResponse(content="The answer is 42.")
            assert claude.extract_answer(response) == "The answer is 42."

    def test_claude_implements_llm_provider_interface(self) -> None:
        """Claude is a subclass of LLMProvider."""
        assert issubclass(Claude, LLMProvider)
        assert hasattr(Claude, "complete")
        assert hasattr(Claude, "list_models")
        assert hasattr(Claude, "stream")
        assert hasattr(Claude, "estimate_cost")
