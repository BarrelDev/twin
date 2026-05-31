"""Tests for the agent activity log."""

import pytest
from datetime import datetime

from twin.agent.log import AgentLog, LogEntry


class TestLogEntry:
    """Test the LogEntry dataclass."""

    def test_log_entry_creation(self) -> None:
        """Test creating a LogEntry."""
        entry = LogEntry(
            timestamp="2026-05-30T10:00:00Z",
            iteration=0,
            event_type="llm_response",
            details={"stop_reason": "end_turn"},
        )
        assert entry.timestamp == "2026-05-30T10:00:00Z"
        assert entry.iteration == 0
        assert entry.event_type == "llm_response"
        assert entry.details["stop_reason"] == "end_turn"


class TestAgentLog:
    """Test the AgentLog class."""

    @pytest.fixture
    def log(self) -> AgentLog:
        """Create a fresh AgentLog instance."""
        return AgentLog()

    def test_log_initialization(self, log: AgentLog) -> None:
        """Test that a new log is empty."""
        assert log.entry_count() == 0
        assert log.get_log() == []
        assert log.get_last_entry() is None

    def test_log_llm_response(self, log: AgentLog) -> None:
        """Test logging an LLM response."""
        mock_response = type("Response", (), {
            "stop_reason": "end_turn",
            "content": [],
        })()

        log.log_llm_response(0, mock_response)

        assert log.entry_count() == 1
        entry = log.get_last_entry()
        assert entry.iteration == 0
        assert entry.event_type == "llm_response"
        assert entry.details["stop_reason"] == "end_turn"

    def test_log_tool_call(self, log: AgentLog) -> None:
        """Test logging a tool call."""
        tool_input = {"query": "test query", "k": 5}

        log.log_tool_call(0, "search_knowledge_base", tool_input)

        assert log.entry_count() == 1
        entry = log.get_last_entry()
        assert entry.event_type == "tool_call"
        assert entry.details["tool_name"] == "search_knowledge_base"
        assert entry.details["tool_input"] == tool_input

    def test_log_tool_result(self, log: AgentLog) -> None:
        """Test logging a tool result."""
        result = "Found 3 relevant chunks about decorators"

        log.log_tool_result(0, result)

        assert log.entry_count() == 1
        entry = log.get_last_entry()
        assert entry.event_type == "tool_result"
        assert entry.details["result"] == result

    def test_log_final_answer(self, log: AgentLog) -> None:
        """Test logging the final answer."""
        answer = "Decorators are functions that modify other functions."

        log.log_final_answer(1, answer, reason="tool_result")

        assert log.entry_count() == 1
        entry = log.get_last_entry()
        assert entry.event_type == "final_answer"
        assert entry.details["answer"] == answer
        assert entry.details["termination_reason"] == "tool_result"

    def test_log_final_answer_max_iterations(self, log: AgentLog) -> None:
        """Test logging final answer due to max iterations."""
        answer = "Final answer."

        log.log_final_answer(4, answer, reason="max_iterations")

        entry = log.get_last_entry()
        assert entry.details["termination_reason"] == "max_iterations"

    def test_full_agent_execution_log(self, log: AgentLog) -> None:
        """Test a complete sequence of log entries."""
        # Simulate a full agent execution
        mock_response = type("Response", (), {
            "stop_reason": "tool_use",
            "content": [type("Block", (), {"type": "tool_use"})()],
        })()

        log.log_llm_response(0, mock_response)
        log.log_tool_call(0, "search_knowledge_base", {"query": "test"})
        log.log_tool_result(0, "Found results")
        log.log_final_answer(1, "Answer based on results")

        assert log.entry_count() == 4
        entries = log.get_log()
        assert entries[0]["event_type"] == "llm_response"
        assert entries[1]["event_type"] == "tool_call"
        assert entries[2]["event_type"] == "tool_result"
        assert entries[3]["event_type"] == "final_answer"

    def test_log_preserves_chronological_order(self, log: AgentLog) -> None:
        """Test that entries are logged in chronological order."""
        for i in range(3):
            log.log_tool_call(i, "search_knowledge_base", {"query": f"q{i}"})

        entries = log.get_log()
        assert entries[0]["details"]["tool_input"]["query"] == "q0"
        assert entries[1]["details"]["tool_input"]["query"] == "q1"
        assert entries[2]["details"]["tool_input"]["query"] == "q2"

    def test_get_log_returns_dicts(self, log: AgentLog) -> None:
        """Test that get_log returns list of dicts, not LogEntry objects."""
        log.log_tool_call(0, "search_knowledge_base", {"query": "test"})

        entries = log.get_log()
        assert isinstance(entries, list)
        assert isinstance(entries[0], dict)
        assert "timestamp" in entries[0]
        assert "iteration" in entries[0]
        assert "event_type" in entries[0]
        assert "details" in entries[0]

    def test_log_timestamps_are_iso_formatted(self, log: AgentLog) -> None:
        """Test that log timestamps are ISO 8601 formatted."""
        log.log_tool_call(0, "search_knowledge_base", {"query": "test"})

        entry = log.get_last_entry()
        # Should end with 'Z' and contain ISO format
        assert entry.timestamp.endswith("Z")
        assert "T" in entry.timestamp  # ISO format includes T

    def test_multiple_iterations(self, log: AgentLog) -> None:
        """Test logging across multiple iterations."""
        for iteration in range(3):
            log.log_llm_response(iteration, type("R", (), {
                "stop_reason": "tool_use",
                "content": [],
            })())
            log.log_tool_call(iteration, "search_knowledge_base", {"query": f"q{iteration}"})

        assert log.entry_count() == 6
        entries = log.get_log()
        iterations = [e["iteration"] for e in entries]
        assert iterations == [0, 0, 1, 1, 2, 2]

    def test_get_last_entry_after_multiple_logs(self, log: AgentLog) -> None:
        """Test that get_last_entry returns the most recent entry."""
        log.log_tool_call(0, "search_knowledge_base", {"query": "first"})
        log.log_tool_call(1, "search_knowledge_base", {"query": "second"})

        last = log.get_last_entry()
        assert last.details["tool_input"]["query"] == "second"
