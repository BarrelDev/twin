import os
import pytest
from unittest.mock import MagicMock, patch

from twin.llm import Claude, LLMProvider


def _mock_models_list(model_ids: list[str]):
    """Helper to create a mock models list response."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(id=model_id) for model_id in model_ids]
    return mock_response


class TestClaude:
    """Test suite for the Claude LLM provider."""

    def test_missing_api_key_raises_error(self):
        """Claude should raise ValueError if ANTHROPIC_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                Claude()

    @patch("twin.llm.anthropic.Anthropic")
    def test_initialization_with_api_key(self, mock_anthropic_class):
        """Claude should initialize successfully when API key is set."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.models.list.return_value = _mock_models_list(
            ["claude-opus-4.1", "claude-sonnet-4", "claude-haiku-3"]
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            claude = Claude()
            mock_anthropic_class.assert_called_once_with(api_key="test-key-123")
            assert claude.model == "claude-opus-4.1"
            mock_client.models.list.assert_called_once()

    @patch("twin.llm.anthropic.Anthropic")
    def test_initialization_with_custom_model(self, mock_anthropic_class):
        """Claude should accept a custom model parameter."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.models.list.return_value = _mock_models_list(
            ["claude-opus-4.1", "claude-sonnet-4", "claude-haiku-3"]
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude(model="claude-haiku-3")
            assert claude.model == "claude-haiku-3"

    @patch("twin.llm.anthropic.Anthropic")
    def test_initialization_with_invalid_model(self, mock_anthropic_class):
        """Claude should raise ValueError if specified model is not available."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.models.list.return_value = _mock_models_list(
            ["claude-opus-4.1", "claude-sonnet-4"]
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="not available"):
                Claude(model="claude-nonexistent")

    @patch("twin.llm.anthropic.Anthropic")
    def test_no_models_available_raises_error(self, mock_anthropic_class):
        """Claude should raise ValueError if no models are available from the API."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.models.list.return_value = _mock_models_list([])

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="No models available"):
                Claude()

    @patch("twin.llm.anthropic.Anthropic")
    def test_list_models(self, mock_anthropic_class):
        """list_models should return available Claude models from the API."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        available_models = ["claude-opus-4.1", "claude-sonnet-4", "claude-haiku-3"]
        mock_client.models.list.return_value = _mock_models_list(available_models)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            models = claude.list_models()
            assert models == available_models

    @patch("twin.llm.anthropic.Anthropic")
    def test_complete_without_tools(self, mock_anthropic_class):
        """complete() should send messages and system prompt correctly without tools."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.models.list.return_value = _mock_models_list(
            ["claude-opus-4.1", "claude-sonnet-4"]
        )
        mock_response = {"content": [{"type": "text", "text": "Hello"}]}
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            messages = [{"role": "user", "content": "Hi"}]
            system = "You are helpful."

            result = claude.complete(messages, None, system)

            mock_client.messages.create.assert_called_once_with(
                model="claude-opus-4.1",
                max_tokens=1024,
                system="You are helpful.",
                messages=messages,
            )
            assert result == mock_response

    @patch("twin.llm.anthropic.Anthropic")
    def test_complete_with_tools(self, mock_anthropic_class):
        """complete() should include tools in the API call when provided."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.models.list.return_value = _mock_models_list(
            ["claude-opus-4.1", "claude-sonnet-4"]
        )
        mock_response = {"content": [{"type": "tool_use", "id": "tool_123"}]}
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            claude = Claude()
            messages = [{"role": "user", "content": "Use the tool"}]
            tools = [{"name": "test_tool", "description": "A test tool"}]
            system = "You are helpful."

            result = claude.complete(messages, tools, system)

            mock_client.messages.create.assert_called_once_with(
                model="claude-opus-4.1",
                max_tokens=1024,
                system="You are helpful.",
                messages=messages,
                tools=tools,
            )
            assert result == mock_response

    def test_claude_implements_llm_provider_interface(self):
        """Claude should implement the LLMProvider interface."""
        assert issubclass(Claude, LLMProvider)
        assert hasattr(Claude, "complete")
        assert hasattr(Claude, "list_models")
