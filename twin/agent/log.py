"""Activity logging for the agent runtime.

Tracks all steps in the agent loop: LLM calls, tool calls, results, and
final answers. Used for debugging, transparency, and audit trails.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass
class LogEntry:
    """A single entry in the agent activity log."""

    timestamp: str
    """ISO 8601 timestamp of when this entry was created."""

    iteration: int
    """Which iteration of the agent loop this occurred in."""

    event_type: str
    """Type of event: 'llm_response', 'tool_call', 'tool_result', 'final_answer'."""

    details: dict[str, Any]
    """Event-specific details (content, tool name, result, etc.)."""


class AgentLog:
    """
    Records all activity during agent execution.

    Maintains a chronological log of:
    - LLM responses at each iteration
    - Tool calls and their inputs
    - Tool execution results
    - Final answer generation
    """

    def __init__(self) -> None:
        """Initialize an empty activity log."""
        self._entries: list[LogEntry] = []

    def log_llm_response(self, iteration: int, response: Any) -> None:
        """
        Log an LLM response.

        Args:
            iteration: The iteration number of the agent loop.
            response: The response object from the LLM provider.
        """
        entry = LogEntry(
            timestamp=self._now_iso(),
            iteration=iteration,
            event_type="llm_response",
            details={
                "stop_reason": getattr(response, "stop_reason", None),
                "has_tool_call": self._has_tool_call(response),
            },
        )
        self._entries.append(entry)

    def log_tool_call(
        self, iteration: int, tool_name: str, tool_input: dict
    ) -> None:
        """
        Log a tool invocation.

        Args:
            iteration: The iteration number of the agent loop.
            tool_name: Name of the tool being called.
            tool_input: Input parameters for the tool.
        """
        entry = LogEntry(
            timestamp=self._now_iso(),
            iteration=iteration,
            event_type="tool_call",
            details={
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
        )
        self._entries.append(entry)

    def log_tool_result(self, iteration: int, result: str) -> None:
        """
        Log the result of a tool call.

        Args:
            iteration: The iteration number of the agent loop.
            result: The result returned by the tool.
        """
        entry = LogEntry(
            timestamp=self._now_iso(),
            iteration=iteration,
            event_type="tool_result",
            details={
                "result": result,
            },
        )
        self._entries.append(entry)

    def log_final_answer(
        self, iteration: int, answer: str, reason: str = "tool_result"
    ) -> None:
        """
        Log the final answer.

        Args:
            iteration: The iteration number when the final answer was generated.
            answer: The final answer text.
            reason: Why the loop terminated ('tool_result' or 'max_iterations').
        """
        entry = LogEntry(
            timestamp=self._now_iso(),
            iteration=iteration,
            event_type="final_answer",
            details={
                "answer": answer,
                "termination_reason": reason,
            },
        )
        self._entries.append(entry)

    def get_log(self) -> list[dict]:
        """
        Return the complete activity log as a list of dicts.

        Returns:
            List of log entry dicts in chronological order.
        """
        return [asdict(entry) for entry in self._entries]

    def get_last_entry(self) -> LogEntry | None:
        """
        Get the most recent log entry.

        Returns:
            The last LogEntry, or None if the log is empty.
        """
        return self._entries[-1] if self._entries else None

    def entry_count(self) -> int:
        """
        Return the total number of log entries.

        Returns:
            Number of entries in the log.
        """
        return len(self._entries)

    def _now_iso(self) -> str:
        """
        Get current timestamp in ISO 8601 format.

        Returns:
            Current time as ISO 8601 string.
        """
        return datetime.utcnow().isoformat() + "Z"

    def _has_tool_call(self, response: Any) -> bool:
        """
        Check if a response contains a tool call (provider-agnostic).

        Args:
            response: Response object from the LLM.

        Returns:
            True if the response contains a tool call.
        """
        if not hasattr(response, "content"):
            return False

        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                return True

        return False
