"""Tests for the agent runtime."""

import pytest
from unittest.mock import AsyncMock, Mock

from twin.agent.runtime import AgentRuntime, AgentOutput
from twin.agent.tools import ToolDefinition, ToolDispatcher
from twin.llm.base import LLMProvider, LLMResponse, ToolCall


class TestAgentOutput:
    """Tests for the AgentOutput dataclass."""

    def test_agent_output_creation(self) -> None:
        output = AgentOutput(
            final_answer="Test answer",
            tool_calls=2,
            activity_log=[{"event": "test"}],
        )
        assert output.final_answer == "Test answer"
        assert output.tool_calls == 2
        assert len(output.activity_log) == 1


class TestAgentRuntime:
    """Tests for the AgentRuntime class."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLMProvider with async complete() that returns LLMResponse."""
        m = Mock(spec=LLMProvider)
        m.complete = AsyncMock(
            return_value=LLMResponse(content="Default answer", tool_calls=[])
        )
        m.extract_answer = Mock(side_effect=lambda r: r.content or "")
        return m

    @pytest.fixture
    def mock_dispatcher(self):
        """Mock ToolDispatcher that returns a canned search result."""
        dispatcher = Mock(spec=ToolDispatcher)
        tool = ToolDefinition(
            name="search_knowledge_base",
            description="Search KB",
            input_schema={"type": "object"},
        )
        dispatcher.get_tool_definitions.return_value = [tool]
        dispatcher.dispatch.return_value = "Tool result"
        return dispatcher

    @pytest.fixture
    def runtime(self, mock_llm, mock_dispatcher) -> AgentRuntime:
        return AgentRuntime(llm=mock_llm, tool_dispatcher=mock_dispatcher, max_iterations=5)

    def test_runtime_initialization(self, mock_llm, mock_dispatcher) -> None:
        runtime = AgentRuntime(mock_llm, mock_dispatcher, max_iterations=10)
        assert runtime._llm is mock_llm
        assert runtime._tool_dispatcher is mock_dispatcher
        assert runtime._max_iterations == 10

    def test_runtime_default_max_iterations(self, mock_llm, mock_dispatcher) -> None:
        runtime = AgentRuntime(mock_llm, mock_dispatcher)
        assert runtime._max_iterations == 5

    @pytest.mark.anyio
    async def test_execute_with_final_answer_no_tools(
        self, runtime: AgentRuntime, mock_llm
    ) -> None:
        """execute() returns the final answer when no tool calls are made."""
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(content="This is the final answer", tool_calls=[])
        )
        output = await runtime.execute("What is Python?")
        assert output.final_answer == "This is the final answer"
        assert output.tool_calls == 0
        assert len(output.activity_log) > 0

    @pytest.mark.anyio
    async def test_execute_logs_activity(self, runtime: AgentRuntime, mock_llm) -> None:
        """execute() logs both llm_response and final_answer events."""
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(content="Answer", tool_calls=[])
        )
        output = await runtime.execute("Test task")
        log_events = [e["event_type"] for e in output.activity_log]
        assert "llm_response" in log_events
        assert "final_answer" in log_events

    @pytest.mark.anyio
    async def test_execute_with_single_tool_call(
        self, runtime: AgentRuntime, mock_llm, mock_dispatcher
    ) -> None:
        """execute() dispatches a tool call and then returns the final answer."""
        mock_llm.complete = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="search_knowledge_base", input={"query": "test"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Final answer", tool_calls=[]),
        ])
        output = await runtime.execute("What is Python?")
        assert output.tool_calls == 1
        assert output.final_answer == "Final answer"
        mock_dispatcher.dispatch.assert_called_once_with(
            "search_knowledge_base", {"query": "test"}
        )

    @pytest.mark.anyio
    async def test_execute_with_multiple_tool_calls(
        self, runtime: AgentRuntime, mock_llm, mock_dispatcher
    ) -> None:
        """execute() supports multiple sequential tool calls."""
        mock_llm.complete = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="search_knowledge_base", input={"query": "q1"})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_2", name="search_knowledge_base", input={"query": "q2"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Final answer", tool_calls=[]),
        ])
        mock_dispatcher.dispatch.side_effect = ["Result 1", "Result 2"]
        output = await runtime.execute("Multi-step task")
        assert output.tool_calls == 2
        assert output.final_answer == "Final answer"
        assert mock_dispatcher.dispatch.call_count == 2

    @pytest.mark.anyio
    async def test_execute_respects_max_iterations(
        self, runtime: AgentRuntime, mock_llm, mock_dispatcher
    ) -> None:
        """execute() stops after max_iterations even without a final answer."""
        tool_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc", name="search_knowledge_base", input={"query": "q"})],
            stop_reason="tool_use",
        )
        mock_llm.complete = AsyncMock(return_value=tool_response)
        runtime._max_iterations = 2
        output = await runtime.execute("Task")
        assert mock_llm.complete.call_count == 2
        assert output.tool_calls <= 2

    @pytest.mark.anyio
    async def test_execute_with_tool_call_error(
        self, runtime: AgentRuntime, mock_llm, mock_dispatcher
    ) -> None:
        """execute() propagates ValueError raised by the tool dispatcher."""
        mock_llm.complete = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc", name="search_knowledge_base", input={"query": "q"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Answer", tool_calls=[]),
        ])
        mock_dispatcher.dispatch.side_effect = ValueError("Unknown tool")
        with pytest.raises(ValueError):
            await runtime.execute("Task")

    @pytest.mark.anyio
    async def test_agent_output_includes_tool_calls_count(
        self, runtime: AgentRuntime, mock_llm, mock_dispatcher
    ) -> None:
        """AgentOutput.tool_calls accurately counts dispatched tool calls."""
        mock_llm.complete = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc", name="search_knowledge_base", input={"query": "q"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Answer", tool_calls=[]),
        ])
        output = await runtime.execute("Task")
        assert output.tool_calls == 1

    # ── _has_tool_call / _extract_tool_call ─────────────────────────────────

    def test_has_tool_call_true_when_tool_calls_present(
        self, runtime: AgentRuntime
    ) -> None:
        response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc", name="search", input={})],
        )
        assert runtime._has_tool_call(response) is True

    def test_has_tool_call_false_when_no_tool_calls(
        self, runtime: AgentRuntime
    ) -> None:
        response = LLMResponse(content="text", tool_calls=[])
        assert runtime._has_tool_call(response) is False

    def test_extract_tool_call_returns_first_call(
        self, runtime: AgentRuntime
    ) -> None:
        tc = ToolCall(id="tc_abc", name="search_kb", input={"q": "test"})
        response = LLMResponse(content=None, tool_calls=[tc])
        result = runtime._extract_tool_call(response)
        assert result.id == "tc_abc"
        assert result.name == "search_kb"
        assert result.input == {"q": "test"}

    def test_extract_tool_call_raises_when_empty(
        self, runtime: AgentRuntime
    ) -> None:
        with pytest.raises(ValueError, match="No tool call"):
            runtime._extract_tool_call(LLMResponse(content="text", tool_calls=[]))
