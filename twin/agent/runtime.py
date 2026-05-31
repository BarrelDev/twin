"""Agent runtime for multi-step reasoning over the knowledge base.

The agent runtime executes a task by repeatedly deciding whether to call tools
(e.g., KB search) or return a final answer. It coordinates LLM interactions
with tool execution and logs activity for transparency.
"""

from dataclasses import dataclass

from twin.llm.base import LLMProvider
from twin.agent.tools import ToolDispatcher, ToolDefinition
from twin.agent.log import AgentLog
from twin.rag.prompts import SystemPrompts


@dataclass
class AgentOutput:
    """Output from the agent runtime."""

    final_answer: str
    """The agent's final answer to the task."""

    tool_calls: int
    """Number of tool calls made during execution."""

    activity_log: list[dict]
    """Detailed log of each agent step and tool call."""


class AgentRuntime:
    """
    Orchestrates multi-step reasoning using an LLM and tool access.

    The runtime executes a task by maintaining a conversation with an LLM,
    routing tool calls through the dispatcher, and terminating when the LLM
    returns a final answer or the iteration limit is reached.
    """

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(
        self,
        llm: LLMProvider,
        tool_dispatcher: ToolDispatcher,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """
        Initialize the agent runtime.

        Args:
            llm: LLMProvider instance for generating responses.
            tool_dispatcher: ToolDispatcher for routing and executing tool calls.
            max_iterations: Maximum number of tool calls allowed (default: 5).
        """
        self._llm = llm
        self._tool_dispatcher = tool_dispatcher
        self._max_iterations = max_iterations
        self._log = AgentLog()

    def execute(self, task: str) -> AgentOutput:
        """
        Execute a task using the agent loop.

        The loop:
        1. Sends the task and conversation history to the LLM.
        2. Checks if the response contains tool calls or a final answer.
        3. If tools are called, executes them and adds results to the conversation.
        4. Repeats until a final answer is returned or iteration limit is reached.

        Args:
            task: The user's task description.

        Returns:
            AgentOutput with the final answer, tool call count, and activity log.
        """
        messages = [{"role": "user", "content": task}]
        tool_calls_made = 0

        for iteration in range(self._max_iterations):
            # Call the LLM with current conversation state
            tool_definitions = self._tool_dispatcher.get_tool_definitions()
            tool_dicts = self._format_tool_definitions(tool_definitions)

            response = self._llm.complete(
                messages=messages,
                tools=tool_dicts,
                system=SystemPrompts.AGENT_SYSTEM,
            )

            self._log.log_llm_response(iteration, response)

            # Check if response contains a tool call or final answer
            if self._has_tool_call(response):
                # Extract tool call and execute it
                tool_name, tool_input = self._extract_tool_call(response)
                tool_calls_made += 1

                self._log.log_tool_call(iteration, tool_name, tool_input)

                # Execute the tool
                tool_result = self._tool_dispatcher.dispatch(tool_name, tool_input)

                self._log.log_tool_result(iteration, tool_result)

                # Add assistant response and tool result to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {"role": "user", "content": f"Tool result: {tool_result}"}
                )
            else:
                # Final answer received
                final_answer = self._llm.extract_answer(response)
                self._log.log_final_answer(iteration, final_answer)

                return AgentOutput(
                    final_answer=final_answer,
                    tool_calls=tool_calls_made,
                    activity_log=self._log.get_log(),
                )

        # Max iterations reached without final answer
        final_answer = self._llm.extract_answer(response)
        self._log.log_final_answer(
            self._max_iterations - 1, final_answer, reason="max_iterations"
        )

        return AgentOutput(
            final_answer=final_answer,
            tool_calls=tool_calls_made,
            activity_log=self._log.get_log(),
        )

    def _has_tool_call(self, response: dict) -> bool:
        """
        Check if the LLM response contains a tool call.

        Args:
            response: Response object from the LLM.

        Returns:
            True if the response contains a tool call, False otherwise.
        """
        # TODO: Implement provider-specific tool call detection
        # For Anthropic, check if response has tool_use content blocks
        if not hasattr(response, "content"):
            return False

        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                return True

        return False

    def _extract_tool_call(self, response: dict) -> tuple[str, dict]:
        """
        Extract tool name and input from an LLM response.

        Args:
            response: Response object from the LLM containing a tool call.

        Returns:
            Tuple of (tool_name, tool_input_dict).

        Raises:
            ValueError: If no tool call is found in the response.
        """
        # TODO: Implement provider-specific tool call extraction
        # For Anthropic, extract from tool_use content block
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                return tool_name, tool_input

        raise ValueError("No tool call found in response")

    def _format_tool_definitions(
        self, tool_defs: list[ToolDefinition]
    ) -> list[dict]:
        """
        Convert ToolDefinition objects to provider-specific format.

        Args:
            tool_defs: List of ToolDefinition objects.

        Returns:
            List of tool definition dicts in provider format (e.g., Anthropic).
        """
        # TODO: Make this provider-specific
        # For now, format as Anthropic expects
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tool_defs
        ]
