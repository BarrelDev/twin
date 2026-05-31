"""Tests for the agent runtime."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from twin.agent.runtime import AgentRuntime, AgentOutput
from twin.agent.tools import ToolDefinition, ToolDispatcher
from twin.llm.base import LLMProvider


class TestAgentOutput:
    """Test the AgentOutput dataclass."""

    def test_agent_output_creation(self) -> None:
        """Test creating an AgentOutput."""
        output = AgentOutput(
            final_answer="Test answer",
            tool_calls=2,
            activity_log=[{"event": "test"}],
        )
        assert output.final_answer == "Test answer"
        assert output.tool_calls == 2
        assert len(output.activity_log) == 1


class TestAgentRuntime:
    """Test the AgentRuntime class."""

    @pytest.fixture
    def mock_llm(self) -> Mock:
        """Create a mock LLM provider."""
        return Mock(spec=LLMProvider)

    @pytest.fixture
    def mock_dispatcher(self) -> Mock:
        """Create a mock tool dispatcher."""
        dispatcher = Mock(spec=ToolDispatcher)
        tool = ToolDefinition(
            name="search_knowledge_base",
            description="Search KB",
            input_schema={"type": "object"},
        )
        dispatcher.get_tool_definitions.return_value = [tool]
        return dispatcher

    @pytest.fixture
    def runtime(self, mock_llm: Mock, mock_dispatcher: Mock) -> AgentRuntime:
        """Create an AgentRuntime instance with mocks."""
        return AgentRuntime(
            llm=mock_llm,
            tool_dispatcher=mock_dispatcher,
            max_iterations=5,
        )

    def test_runtime_initialization(
        self, mock_llm: Mock, mock_dispatcher: Mock
    ) -> None:
        """Test AgentRuntime initialization."""
        runtime = AgentRuntime(mock_llm, mock_dispatcher, max_iterations=10)
        assert runtime._llm == mock_llm
        assert runtime._tool_dispatcher == mock_dispatcher
        assert runtime._max_iterations == 10

    def test_runtime_default_max_iterations(
        self, mock_llm: Mock, mock_dispatcher: Mock
    ) -> None:
        """Test that default max_iterations is 5."""
        runtime = AgentRuntime(mock_llm, mock_dispatcher)
        assert runtime._max_iterations == 5

    def test_execute_with_final_answer_no_tools(
        self, runtime: AgentRuntime, mock_llm: Mock
    ) -> None:
        """Test execution when LLM returns a final answer without tool calls."""
        mock_response = Mock()
        mock_response.content = [Mock(text="This is the final answer")]
        mock_llm.complete.return_value = mock_response
        mock_llm.extract_answer.return_value = "This is the final answer"

        output = runtime.execute("What is Python?")

        assert output.final_answer == "This is the final answer"
        assert output.tool_calls == 0
        assert len(output.activity_log) > 0
        mock_llm.complete.assert_called_once()

    def test_execute_logs_activity(
        self, runtime: AgentRuntime, mock_llm: Mock
    ) -> None:
        """Test that execution logs are recorded."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_llm.complete.return_value = mock_response
        mock_llm.extract_answer.return_value = "Answer"

        output = runtime.execute("Test task")

        assert len(output.activity_log) >= 1
        log_events = [e["event_type"] for e in output.activity_log]
        assert "llm_response" in log_events
        assert "final_answer" in log_events

    def test_execute_with_single_tool_call(
        self,
        runtime: AgentRuntime,
        mock_llm: Mock,
        mock_dispatcher: Mock,
    ) -> None:
        """Test execution with one tool call."""
        # First response: tool call
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_knowledge_base"
        tool_block.input = {"query": "test"}

        tool_call_response = Mock()
        tool_call_response.content = [tool_block]

        # Second response: final answer
        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]

        mock_llm.complete.side_effect = [tool_call_response, final_response]
        mock_llm.extract_answer.return_value = "Final answer"
        mock_dispatcher.dispatch.return_value = "Tool result"

        output = runtime.execute("What is Python?")

        assert output.tool_calls == 1
        assert output.final_answer == "Final answer"
        mock_dispatcher.dispatch.assert_called_once_with(
            "search_knowledge_base", {"query": "test"}
        )

    def test_execute_with_multiple_tool_calls(
        self,
        runtime: AgentRuntime,
        mock_llm: Mock,
        mock_dispatcher: Mock,
    ) -> None:
        """Test execution with multiple tool calls."""
        # Responses: tool call, tool call, final answer
        block1 = Mock()
        block1.type = "tool_use"
        block1.name = "search_knowledge_base"
        block1.input = {"query": "q1"}

        block2 = Mock()
        block2.type = "tool_use"
        block2.name = "search_knowledge_base"
        block2.input = {"query": "q2"}

        responses = [
            Mock(content=[block1]),
            Mock(content=[block2]),
            Mock(content=[Mock(text="Final answer")]),
        ]

        mock_llm.complete.side_effect = responses
        mock_llm.extract_answer.return_value = "Final answer"
        mock_dispatcher.dispatch.side_effect = ["Result 1", "Result 2"]

        output = runtime.execute("Multi-step task")

        assert output.tool_calls == 2
        assert output.final_answer == "Final answer"
        assert mock_dispatcher.dispatch.call_count == 2

    def test_execute_respects_max_iterations(
        self,
        runtime: AgentRuntime,
        mock_llm: Mock,
        mock_dispatcher: Mock,
    ) -> None:
        """Test that execution stops at max iterations."""
        # Return tool calls every iteration
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_knowledge_base"
        tool_block.input = {"query": "q"}

        tool_response = Mock(content=[tool_block])
        mock_llm.complete.return_value = tool_response
        mock_llm.extract_answer.return_value = "Fallback answer"
        mock_dispatcher.dispatch.return_value = "Result"

        runtime._max_iterations = 2
        output = runtime.execute("Task")

        # Should make 2 complete calls (2 iterations), then stop
        assert mock_llm.complete.call_count == 2
        assert output.tool_calls <= 2

    def test_execute_with_tool_call_error(
        self,
        runtime: AgentRuntime,
        mock_llm: Mock,
        mock_dispatcher: Mock,
    ) -> None:
        """Test execution handles tool dispatch errors gracefully."""
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_knowledge_base"
        tool_block.input = {"query": "q"}

        tool_response = Mock(content=[tool_block])
        final_response = Mock(content=[Mock(text="Answer")])

        mock_llm.complete.side_effect = [tool_response, final_response]
        mock_llm.extract_answer.return_value = "Answer"
        mock_dispatcher.dispatch.side_effect = ValueError("Unknown tool")

        with pytest.raises(ValueError):
            runtime.execute("Task")

    def test_format_tool_definitions(self, runtime: AgentRuntime) -> None:
        """Test tool definition formatting."""
        tool_defs = [
            ToolDefinition(
                name="search_kb",
                description="Search knowledge base",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

        formatted = runtime._format_tool_definitions(tool_defs)

        assert len(formatted) == 1
        assert formatted[0]["name"] == "search_kb"
        assert formatted[0]["description"] == "Search knowledge base"
        assert "input_schema" in formatted[0]

    def test_has_tool_call_detection(self, runtime: AgentRuntime) -> None:
        """Test tool call detection."""
        response_with_tool = Mock()
        response_with_tool.content = [Mock(type="tool_use")]

        response_without_tool = Mock()
        response_without_tool.content = [Mock(type="text")]

        assert runtime._has_tool_call(response_with_tool) is True
        assert runtime._has_tool_call(response_without_tool) is False

    def test_extract_tool_call(self, runtime: AgentRuntime) -> None:
        """Test extracting tool call from response."""
        block = Mock()
        block.type = "tool_use"
        block.name = "search_kb"
        block.input = {"q": "test"}
        block.id = "toolu_abc123"
        response = Mock(content=[block])

        tool_name, tool_input, tool_use_id = runtime._extract_tool_call(response)

        assert tool_name == "search_kb"
        assert tool_input == {"q": "test"}
        assert tool_use_id == "toolu_abc123"

    def test_extract_tool_call_not_found(self, runtime: AgentRuntime) -> None:
        """Test error when extracting non-existent tool call."""
        response = Mock(content=[Mock(type="text")])

        with pytest.raises(ValueError, match="No tool call found"):
            runtime._extract_tool_call(response)

    def test_extract_tool_call_from_multiple_blocks(
        self, runtime: AgentRuntime
    ) -> None:
        """Test extracting tool call from multiple content blocks."""
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_kb"
        tool_block.input = {"q": "test"}
        tool_block.id = "toolu_xyz789"

        blocks = [
            Mock(type="text"),
            tool_block,
            Mock(type="text"),
        ]
        response = Mock(content=blocks)

        tool_name, tool_input, tool_use_id = runtime._extract_tool_call(response)

        assert tool_name == "search_kb"
        assert tool_input == {"q": "test"}
        assert tool_use_id == "toolu_xyz789"

    def test_agent_output_includes_tool_calls_count(
        self,
        runtime: AgentRuntime,
        mock_llm: Mock,
        mock_dispatcher: Mock,
    ) -> None:
        """Test that output correctly counts tool calls."""
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_knowledge_base"
        tool_block.input = {"query": "q"}

        tool_response = Mock(content=[tool_block])
        final_response = Mock(content=[Mock(text="Answer")])

        mock_llm.complete.side_effect = [tool_response, final_response]
        mock_llm.extract_answer.return_value = "Answer"
        mock_dispatcher.dispatch.return_value = "Result"

        output = runtime.execute("Task")

        assert output.tool_calls == 1
